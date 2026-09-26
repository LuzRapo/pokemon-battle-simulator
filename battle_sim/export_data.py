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


def move_json(move: Any) -> dict[str, Any]:
    """One move as the engine understands it: its effects already resolved, not Showdown's raw JSON."""
    written = {f.name: _plain(getattr(move, f.name)) for f in fields(move)}
    for effect in written.get("effects", []):
        # A stat change emits one log entry per stat *in the order the dict was built*, so the order
        # is part of the mechanics rather than a detail of how it was written down. Sorted keys
        # (which this file writes, for stable diffs) would silently reorder them, so the pairs go
        # out as a list where the order survives being sorted around them.
        if isinstance(effect, dict) and isinstance(effect.get("stages"), dict):
            effect["stages"] = [[stat, change] for stat, change in effect["stages"].items()]
    return written


def species_json(species: Any) -> dict[str, Any]:
    return {f.name: _plain(getattr(species, f.name)) for f in fields(species)}


# The modules that hold rules. Anything here that names a move by hand is doing something to it
# that the move's own data does not say. Policy and UI modules are left out on purpose: they name
# moves constantly (`human_policy` alone names 44) without changing what any of them do, and
# refusing those would shrink the comparable slice for nothing.
RULE_MODULES = ("engine", "mechanics", "maths", "formes.py", "zmoves.py", "replay_state.py")


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
    roots = [Path(__file__).parent / part for part in RULE_MODULES]
    files = [f for root in roots for f in ([root] if root.suffix else sorted(root.rglob("*.py")))]
    known = {move.name for move in get_all_moves().values()}
    found: set[str] = set()
    for file in files:
        tree = ast.parse(file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in known:
                found.add(node.value)
    return sorted(found)


def rules_json() -> dict[str, Any]:
    """The tables that are neither moves nor species: the type chart and what each nature does.

    Exported for the same reason as the rest — a hand-copied type chart in a second language is a
    transcription error waiting to change one matchup by a factor of two, silently, forever.
    """
    return {
        "coded_moves": coded_move_names(),
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
