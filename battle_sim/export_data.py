"""The engine's data, normalised once by Python and written out for a second engine to read."""

import argparse
import ast
import json
from dataclasses import is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from loguru import logger

from battle_sim.database.loader import get_all_moves, get_all_species, get_all_z_moves
from battle_sim.mechanics.abilities import ABILITY_BINDERS
from battle_sim.mechanics.items import ITEM_BINDERS
from battle_sim.models.moves import Move, MoveEffect
from battle_sim.models.species import BaseSpecies
from battle_sim.models.type_matchups import TYPE_CHART
from battle_sim.utils import Nature, Type, field_values

SCHEMA_VERSION = 1


def _plain(value: object) -> object:
    if isinstance(value, Enum):
        return value.name
    if is_dataclass(value) and not isinstance(value, type):
        return {"kind": type(value).__name__} | {name: _plain(item) for name, item in field_values(value).items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(_plain(k)): _plain(v) for k, v in value.items()}
    return value


def effect_json(effect: MoveEffect) -> dict[str, object]:
    """One effect, with its dataclass name as `kind` and nothing else."""
    written = {name: _plain(value) for name, value in field_values(effect).items()}
    inner = written.pop("kind", None)
    written["kind"] = type(effect).__name__
    if inner is not None:
        written["variant"] = inner
    # Stat changes are logged in dict order, so that order is part of the mechanics.
    stages = written.get("stages")
    if isinstance(stages, dict):
        written["stages"] = [[stat, change] for stat, change in stages.items()]
    return written


def move_json(move: Move) -> dict[str, object]:
    """One move as the engine understands it: its effects already resolved, not Showdown's raw JSON."""
    written = {name: _plain(value) for name, value in field_values(move).items()}
    written["effects"] = [effect_json(effect) for effect in move.effects]
    return written


def species_json(species: BaseSpecies) -> dict[str, object]:
    from battle_sim.teams import ability_from_showdown, item_from_showdown
    from battle_sim.utils import Ability

    out = {name: _plain(value) for name, value in field_values(species).items()}
    # `fused_item` is `required_item` resolved once here to the enum name every other item field uses.
    item = item_from_showdown(species.required_item) if species.required_item else None
    out["fused_item"] = item.name if item is not None else None
    # A forme swap needs the new forme's ability in this engine's own name, as with `fused_item`.
    ability = ability_from_showdown(species.regular_abilities[0]) if species.regular_abilities else Ability.NONE
    out["regular_ability"] = ability.name if ability is not Ability.NONE else None
    return out


# The modules that hold rules; policy and analysis modules name moves without changing them.
RULE_MODULES = ("engine", "mechanics", "maths", "formes.py", "zmoves.py")


def coded_move_names() -> list[str]:
    """Every move the Python special-cases by name, read out of the source rather than listed here."""
    known = {move.name for move in get_all_moves().values()}
    found: set[str] = set()
    for file in _rule_files():
        tree = ast.parse(file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in known:
                found.add(node.value)
    return sorted(found)


def _rule_files() -> list[Path]:
    roots = [Path(__file__).parent / part for part in RULE_MODULES]
    return [f for root in roots for f in ([root] if root.suffix else sorted(root.rglob("*.py")))]


def _named_in_rules(enum_name: str) -> set[str]:
    """Every `Ability.X` / `Item.X` the rules mention, by attribute name."""
    found: set[str] = set()
    for file in _rule_files():
        for node in ast.walk(ast.parse(file.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == enum_name:
                found.add(node.attr)
    return found - {"NONE"}


def live_behaviour() -> tuple[list[str], list[str]]:
    """The abilities and items that actually do something in the Python engine."""
    from battle_sim.formes import mega_stones

    abilities = {ability.name for ability in ABILITY_BINDERS} | _named_in_rules("Ability")
    # Mega stones reach a mechanic through a runtime lookup the AST sweep cannot see, so add them directly.
    items = {item.name for item in ITEM_BINDERS} | _named_in_rules("Item") | {item.name for item in mega_stones()}
    return sorted(abilities), sorted(items)


def rules_json() -> dict[str, Any]:
    """The tables that are neither moves nor species: the type chart and what each nature does."""
    # noqa: SLF001 -- the whole point of this block is to export these
    from battle_sim.formes import _ULTRA_BURST, _forme_by_base_and_item, _forme_by_base_and_move

    abilities, items = live_behaviour()
    return {
        "coded_moves": coded_move_names(),
        "live_abilities": abilities,
        "live_items": items,
        # Mega Rayquaza needs no item; it is gated on knowing Dragon Ascent instead.
        "move_gated_formes": [
            {"base_species": base, "move": move, "forme": forme}
            for (base, move), forme in _forme_by_base_and_move().items()
        ],
        # The stone/orb table `mega_forme` reads, and the Ultra Burst pairing it checks first.
        "mega_formes": [
            {"base_species": base, "item": item.name, "forme": forme}
            for (base, item), forme in _forme_by_base_and_item().items()
        ],
        "ultra_burst_formes": [
            {"base_species": species, "item": item.name, "forme": forme}
            for (species, item), forme in _ULTRA_BURST.items()
        ],
        "types": [t.name for t in Type],
        "type_chart": {
            attacker.name: {defender.name: TYPE_CHART[attacker].get(defender, 1.0) for defender in Type}
            for attacker in Type
        },
        "natures": {nature.name: {"up": nature.value.UP.name, "down": nature.value.DOWN.name} for nature in Nature},
    }


def export(directory: Path) -> dict[str, int]:
    directory.mkdir(parents=True, exist_ok=True)
    moves = {move_id: move_json(move) for move_id, move in sorted(get_all_moves().items())}
    species = {name: species_json(entry) for name, entry in sorted(get_all_species().items())}
    # Z-moves are kept out of `get_all_moves()`, so they get their own export of the same shape.
    z_moves = {crystal_id: move_json(move) for crystal_id, move in sorted(get_all_z_moves().items())}
    (directory / "rules.json").write_text(
        json.dumps({"schema": SCHEMA_VERSION} | rules_json(), indent=1, sort_keys=True)
    )
    for name, payload in (("moves", moves), ("species", species), ("zmoves", z_moves)):
        (directory / f"{name}.json").write_text(
            json.dumps({"schema": SCHEMA_VERSION, name: payload}, indent=1, sort_keys=True)
        )
    return {"moves": len(moves), "species": len(species), "zmoves": len(z_moves), "types": len(Type)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("rust/data"))
    args = parser.parse_args()
    counts = export(args.out)
    logger.info(f"wrote {counts['moves']} moves, {counts['species']} species, {counts['types']} types to {args.out}")


if __name__ == "__main__":
    main()
