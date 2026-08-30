"""Mid-battle species swaps: Mega Evolution and Primal Reversion.

A swap replaces the species-derived half of a Pokemon — name, base stats, types, ability, weight —
and leaves everything the trainer chose alone: moves, item, EVs/IVs/nature/level. Base HP is
identical across every Gen-7 mega/primal forme and its base species (verified for all 49 pairs), so
current HP carries straight over instead of being rescaled.

The stone -> forme table is derived from the vendored pokedex's own `requiredItem` field rather than
hand-maintained: that field is what makes a forme reachable, so deriving from it cannot drift out of
step with the species data. Those formes are all flagged `isNonstandard: "Past"` (Mega Evolution
does not exist in Gen 9), which `scope.py` deliberately keeps in scope — it excludes only
`{"CAP", "Custom"}`. `in_scope_species` still bars them as *starting* species; they are only ever
reached mid-battle through the stone, exactly as Palafin-Hero is reached through its ability.
"""

from functools import cache

from battle_sim.database.loader import get_all_species, get_species, normalize_id
from battle_sim.models.pokemon import Pokemon
from battle_sim.teams import ability_from_showdown, item_from_showdown
from battle_sim.utils import Item

_MEGA_FORME_PREFIXES = ("Mega", "Primal")


@cache
def _forme_by_base_and_item() -> dict[tuple[str, Item], str]:
    """(base species key, held item) -> the forme name that pairing reaches."""
    table: dict[tuple[str, Item], str] = {}
    for species in get_all_species().values():
        if species.required_item is None or species.base_species is None:
            continue
        if not species.name.removeprefix(species.base_species).strip("-").startswith(_MEGA_FORME_PREFIXES):
            continue
        item = item_from_showdown(species.required_item)
        if item is not None:
            table[normalize_id(species.base_species), item] = species.name
    return table


def mega_forme(species: str, item: Item) -> str | None:
    """The forme this species reaches while holding this item, or None if the pairing does nothing."""
    return _forme_by_base_and_item().get((normalize_id(species), item))


@cache
def mega_stones() -> frozenset[Item]:
    """Every item that reaches a Mega Evolution or Primal Reversion forme."""
    return frozenset(item for _, item in _forme_by_base_and_item())


def apply_forme(pokemon: Pokemon, forme: str) -> None:
    """Swap a live Pokemon onto another forme's species data, in place.

    Current HP keeps its *fraction* of the maximum. No Gen-7 mega or primal forme changes base HP,
    so for those this is exactly a no-op on HP; it matters only for the handful of other formes that
    do (Zygarde-Complete's Power Construct doubles it).

    Ability event handlers are NOT rewired here — a caller holding a live battle must follow this
    with `rewire_active`, while a reconstructed state (which never bound handlers) must not.
    """
    species = get_species(forme)
    fraction = pokemon.live_stats.HP / pokemon.stat_totals.HP
    pokemon.name = species.name
    pokemon.base_stats = species.base_stats
    pokemon.types = species.types
    pokemon.ability = ability_from_showdown(species.regular_abilities[0])
    pokemon.weight_kg = species.weight_kg
    pokemon.refresh_stats()
    pokemon.live_stats.HP = max(1, round(fraction * pokemon.stat_totals.HP)) if fraction > 0 else 0
