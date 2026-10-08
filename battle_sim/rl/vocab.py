"""The integer ids a network embeds, one sorted table each for species, abilities, items and moves."""

from functools import cache

from battle_sim.database.loader import get_all_moves, get_all_species, normalize_id
from battle_sim.utils import Ability, Item

Vocabulary = dict[str, list[str]]


@cache
def vocabulary() -> Vocabulary:
    """Species and moves by their normalised id, abilities and items by their enum name."""
    return {
        "species": sorted(get_all_species()),
        "moves": sorted(get_all_moves()),
        "abilities": sorted(a.name for a in Ability if a is not Ability.NONE),
        "items": sorted(i.name for i in Item if i is not Item.NONE),
    }


@cache
def _indices(table: str) -> dict[str, int]:
    return {name: index + 1 for index, name in enumerate(vocabulary()[table])}


def species_id(name: str) -> int:
    return _indices("species").get(normalize_id(name), 0)


def move_id(name: str) -> int:
    return _indices("moves").get(normalize_id(name), 0)


def ability_id(ability: Ability) -> int:
    return _indices("abilities").get(ability.name, 0)


def item_id(item: Item) -> int:
    return _indices("items").get(item.name, 0)
