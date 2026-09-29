"""The Gen 7 Anything Goes sets `tools.build_ag_sets` mined from the replay logs.

Up to two real sets per species, for the ~90 species that appear often enough in high-ladder AG
games to mine one from. Regenerate the data file with `uv run python -m tools.build_ag_sets`.
"""

import json
from functools import cache
from pathlib import Path

from battle_sim.differential import decode_spec
from battle_sim.models.spec import PokemonSpec

SETS = Path(__file__).parent / "data" / "ag_sets.json"


@cache
def ag_sets() -> dict[str, tuple[PokemonSpec, ...]]:
    """Species name -> its mined sets, most-played first."""
    raw: dict[str, list[dict[str, object]]] = json.loads(SETS.read_text())
    return {species: tuple(decode_spec(entry["set"]) for entry in entries) for species, entries in raw.items()}  # type: ignore[arg-type]
