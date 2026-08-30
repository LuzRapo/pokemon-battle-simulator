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
    base_species: str | None = None  # the forme this one transforms from, for Mega/Primal formes
    required_item: str | None = None  # the Mega Stone / orb that reaches this forme
