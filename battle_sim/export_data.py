"""The engine's data, normalised once by Python and written out for a second engine to read.

The obvious alternative — have the Rust engine parse Showdown's vendored `pokedex.json` and
`moves.json` itself — means reimplementing `database/loader.py`: which of Showdown's fields map to
which effect, which `volatileStatus` values are modelled and which are deliberately skipped, which
moves get bespoke `CodedMoveKind` logic. That is several hundred lines of interpretation, and every
line of it is a chance for the two engines to disagree about what a move *is* before either of them
has simulated anything.

So Python stays the interpreter and exports what it decided. The Rust side reads a flat schema it
cannot misread, and a change to the loader is a re-export rather than a second edit in another
language. When the loader skips something it says so here too, in `skipped`, so the other engine can
refuse to play a move nobody has modelled instead of quietly treating it as a no-op.

    uv run python -m battle_sim.export_data --out rust/data
"""

import argparse
import ast
import json
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from battle_sim.database.loader import get_all_moves, get_all_species
from battle_sim.mechanics.abilities import ABILITY_BINDERS
from battle_sim.mechanics.items import ITEM_BINDERS
from battle_sim.models.type_matchups import TYPE_CHART
from battle_sim.utils import Nature, Type

SCHEMA_VERSION = 1


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.name
    if is_dataclass(value) and not isinstance(value, type):
        return {"kind": type(value).__name__} | {f.name: _plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(_plain(k)): _plain(v) for k, v in value.items()}
    return value


def effect_json(effect: Any) -> dict[str, Any]:
    """One effect, with its dataclass name as `kind` and nothing else.

    Several of these dataclasses have a field *also* called `kind` — WeatherEffect's is SUN,
    SideConditionEffect's is SPIKES — and merging the class name with the fields let the field
    quietly win. A reader then could not tell a weather from a terrain from a hazard, and two
    unrelated dataclasses shared one discriminator space. The inner value goes out as `variant`.
    """
    written = {f.name: _plain(getattr(effect, f.name)) for f in fields(effect)}
    inner = written.pop("kind", None)
    written["kind"] = type(effect).__name__
    if inner is not None:
        written["variant"] = inner
    # A stat change emits one log entry per stat *in the order the dict was built*, so the order is
    # part of the mechanics rather than a detail of how it was written down. Sorted keys (which this
    # file writes, for stable diffs) would silently reorder them, so the pairs go out as a list
    # where the order survives being sorted around them.
    if isinstance(written.get("stages"), dict):
        written["stages"] = [[stat, change] for stat, change in written["stages"].items()]
    return written


def move_json(move: Any) -> dict[str, Any]:
    """One move as the engine understands it: its effects already resolved, not Showdown's raw JSON."""
    written = {f.name: _plain(getattr(move, f.name)) for f in fields(move)}
    written["effects"] = [effect_json(effect) for effect in move.effects]
    return written


def species_json(species: Any) -> dict[str, Any]:
    from battle_sim.teams import item_from_showdown

    out = {f.name: _plain(getattr(species, f.name)) for f in fields(species)}
    # `required_item` is Showdown's display name ("Charizardite X"); Rust never needs to parse a
    # display name to get from one to `Item`, since Python already can. `fused_item` is that lookup
    # done once, here, as the enum member's own name — the same string every other item field in
    # this export already uses. `is_fused_to`'s two special cases (Arceus's plates, Giratina's two
    # names for one orb) are not carried over: every item either of them can name is a *live* one —
    # it has a real binder — so a Pokemon holding it is refused before `is_fused_to` would ever be
    # asked, on either engine.
    item = item_from_showdown(species.required_item) if species.required_item else None
    out["fused_item"] = item.name if item is not None else None
    return out


# The modules that hold rules. Anything here that names a move by hand is doing something to it
# that the move's own data does not say.
#
# Policy and analysis modules are left out on purpose. They name moves constantly — `human_policy`
# alone names 44 — without changing what any of them do, and refusing those would shrink the
# comparable slice for nothing. `replay_state` was in this list by mistake and cost the engine all
# four entry hazards: it is a position-evaluator feature extractor that reconstructs a board from a
# replay log, and the only reason it says "Stealth Rock" is to count one.
RULE_MODULES = ("engine", "mechanics", "maths", "formes.py", "zmoves.py")


def coded_move_names() -> list[str]:
    """Every move the Python special-cases by name, read out of the source rather than listed here.

    These are the moves whose behaviour is not in their data: Revenge doubles when its user was
    hit, Gyro Ball reads the speed difference, Weather Ball changes type, Last Resort simply fails
    until its user has spent its other moves. A second engine reading only the effect list gets all
    of them wrong, and quietly — Revenge came back at 96 against the Python's 150, and Last Resort
    resolved a whole extra move, which put the two engines a draw apart on the tape for good.

    Written as a source sweep, and this is the point: a hand-maintained list was already missing
    `coded_move_fails` and `move_type_override` when those were nowhere near the tables it copied
    from. A sweep cannot fall behind the code it reads. It over-approximates — a name mentioned in
    a log string is refused too — and that is the correct direction to be wrong in.
    """
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
    """The abilities and items that actually do something in the Python engine.

    A second engine that reads `ability: "INTIMIDATE"` off a Pokemon and does nothing with it is
    not playing the same game — and will not say so. That is the failure this project exists to
    prevent, so this list is exported and the Rust engine refuses any Pokemon carrying something it
    has not itself implemented.

    The binder tables are not enough on their own. Sniper is read straight out of the damage
    formula, Clear Body out of the stat-drop code, Chlorophyll out of the speed calculation — 85
    abilities and 59 items have live behaviour without ever touching the event bus, and a check
    built on the tables alone would have waved every one of them through. So the tables are unioned
    with a sweep of the same rule modules `coded_move_names` reads, for the same reason: a
    hand-maintained list falls behind the code, and a sweep cannot.
    """
    from battle_sim.formes import mega_stones

    abilities = {ability.name for ability in ABILITY_BINDERS} | _named_in_rules("Ability")
    # `mega_stones()` union in on its own: `_forme_by_base_and_item()` builds its table by calling
    # `item_from_showdown(species.required_item)` at runtime, a data-driven lookup the AST sweep
    # cannot see through the way it sees a literal `Item.GARCHOMPITE`. Ninety-six items reach a real
    # mechanic this way -- `_resolve_mega_evolution` swaps species, ability and ends up changing turn
    # order, all before either engine has looked at a single move -- and none of the sweep's other
    # tables mention them by name anywhere, so without this union a Rust battle would carry a mega
    # stone in total silence: never refused, never evolving, just wrong for the rest of the battle.
    items = {item.name for item in ITEM_BINDERS} | _named_in_rules("Item") | {item.name for item in mega_stones()}
    return sorted(abilities), sorted(items)


def rules_json() -> dict[str, Any]:
    """The tables that are neither moves nor species: the type chart and what each nature does.

    Exported for the same reason as the rest — a hand-copied type chart in a second language is a
    transcription error waiting to change one matchup by a factor of two, silently, forever.
    """
    from battle_sim.formes import _forme_by_base_and_move  # noqa: SLF001 -- the whole point is to export it

    abilities, items = live_behaviour()
    return {
        "coded_moves": coded_move_names(),
        "live_abilities": abilities,
        "live_items": items,
        # Mega Rayquaza needs no item at all — gated on knowing Dragon Ascent instead, the one
        # entry `_forme_by_base_and_move` has ever needed. Not a `live_items` fact, since nothing
        # here is an item; a Pokemon matching one of these pairs auto-Mega-Evolves regardless.
        "move_gated_formes": [{"base_species": base, "move": move} for base, move in _forme_by_base_and_move()],
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
    (directory / "rules.json").write_text(
        json.dumps({"schema": SCHEMA_VERSION} | rules_json(), indent=1, sort_keys=True)
    )
    for name, payload in (("moves", moves), ("species", species)):
        (directory / f"{name}.json").write_text(
            json.dumps({"schema": SCHEMA_VERSION, name: payload}, indent=1, sort_keys=True)
        )
    return {"moves": len(moves), "species": len(species), "types": len(Type)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("rust/data"))
    args = parser.parse_args()
    counts = export(args.out)
    print(f"wrote {counts['moves']} moves, {counts['species']} species, {counts['types']} types to {args.out}")


if __name__ == "__main__":
    main()
