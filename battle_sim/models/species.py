from dataclasses import dataclass

from battle_sim.models.stats import BaseStats
from battle_sim.models.type_matchups import TypePair


@dataclass(frozen=True)
class BaseSpecies:
    name: str
    dex_number: int
    types: TypePair
    base_stats: BaseStats
    regular_abilities: tuple[str, ...]
    hidden_ability: str | None
    height_m: float
    weight_kg: float
    fully_evolved: bool
    is_legendary_or_mythical: bool = False  # Showdown's own Sub-/Restricted-Legendary or Mythical tag
    base_species: str | None = None  # the forme this one transforms from, for Mega/Primal formes
    required_item: str | None = None  # the Mega Stone / orb that reaches this forme
    required_move: str | None = None  # the move that reaches it instead, for Mega Rayquaza
    evolutions: tuple[str, ...] = ()  # what it becomes; several, for the likes of Eevee
    pre_evolution: str | None = None  # what it came from, which is also where its early moves live
