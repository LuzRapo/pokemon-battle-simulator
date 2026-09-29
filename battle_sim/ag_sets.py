"""The Gen 7 Anything Goes sets `tools.build_ag_sets` mined from the replay logs, and Mirror teams.

Up to two real sets per species, for the ~90 species that appear often enough in high-ladder AG
games to mine one from. Regenerate the data file with `uv run python -m tools.build_ag_sets`.

`mirror_team` is the one team both sides of a Mirror battle play. It lives here, not in the bot,
because self-play training has to deal the exact teams the bot deals.
"""

import json
import random
from functools import cache
from pathlib import Path

from battle_sim.database.loader import get_species, normalize_id
from battle_sim.differential import decode_spec
from battle_sim.formes import forme_item_kind
from battle_sim.models.spec import PokemonSpec
from battle_sim.utils import Item

SETS = Path(__file__).parent / "data" / "ag_sets.json"
TEAM_SIZE = 6


@cache
def ag_sets() -> dict[str, tuple[PokemonSpec, ...]]:
    """Species name -> its mined sets, most-played first."""
    raw: dict[str, list[dict[str, object]]] = json.loads(SETS.read_text())
    return {species: tuple(decode_spec(entry["set"]) for entry in entries) for species, entries in raw.items()}  # type: ignore[arg-type]


def mirror_pool() -> tuple[str, ...]:
    """Every species eligible for a Mirror team, name-sorted. Arceus appears once per type it was
    played as, each forme with its own plate."""
    return tuple(sorted(ag_sets()))


def _base_species(name: str) -> str:
    return get_species(normalize_id(name)).base_species or name


def mirror_team(rng: random.Random) -> tuple[PokemonSpec, ...]:
    """Six real Gen 7 AG sets, no two of the same base species — one Arceus at most, however many
    of its formes the pool holds — as the one team both sides of a Mirror battle play."""
    chosen: list[str] = []
    for species in rng.sample(mirror_pool(), len(mirror_pool())):
        if _base_species(species) not in {_base_species(taken) for taken in chosen}:
            chosen.append(species)
        if len(chosen) == TEAM_SIZE:
            break
    return capped_forme_items([rng.choice(ag_sets()[species]) for species in chosen])


def capped_forme_items(specs: list[PokemonSpec]) -> tuple[PokemonSpec, ...]:
    """One Mega Stone and one Primal orb at most, the first of each kept and any later one stripped.

    A side Mega Evolves once and Primal Reverts once per battle — separate allowances, so an orb
    never costs the Mega its slot — and a second of either kind would be a dead item that merely
    looks like an upgrade.
    """
    seen: set[str] = set()
    capped: list[PokemonSpec] = []
    for spec in specs:
        kind = forme_item_kind(spec.item)
        if kind is not None:
            if kind in seen:
                spec = spec.model_copy(update={"item": Item.NONE})
            seen.add(kind)
        capped.append(spec)
    return tuple(capped)
