import pytest

from battle_sim.database.loader import get_species
from battle_sim.formes import _forme_by_base_and_item, apply_forme, mega_forme, mega_stones
from battle_sim.models.spec import PokemonSpec
from battle_sim.teams import ability_from_showdown, build_pokemon
from battle_sim.utils import Ability, Item, Nature, Type


def _tyranitar(item: Item) -> PokemonSpec:
    return PokemonSpec(
        species="Tyranitar",
        level=80,
        moves=["Stone Edge", "Crunch", "Earthquake", "Fire Punch"],
        ability=Ability.SAND_STREAM,
        item=item,
        nature=Nature.ADAMANT,
    )


@pytest.mark.parametrize(
    ("species", "item", "expected"),
    [
        ("Tyranitar", Item.TYRANITARITE, "Tyranitar-Mega"),
        ("Charizard", Item.CHARIZARDITE_X, "Charizard-Mega-X"),
        ("Charizard", Item.CHARIZARDITE_Y, "Charizard-Mega-Y"),
        ("Groudon", Item.RED_ORB, "Groudon-Primal"),
        ("Kyogre", Item.BLUE_ORB, "Kyogre-Primal"),
    ],
)
def test_stone_reaches_its_forme(species: str, item: Item, expected: str):
    assert mega_forme(species, item) == expected


@pytest.mark.parametrize(
    ("species", "item"),
    [
        ("Charizard", Item.LEFTOVERS),  # not a stone
        ("Charizard", Item.TYRANITARITE),  # someone else's stone
        ("Tyranitar", Item.NONE),
    ],
)
def test_wrong_pairing_reaches_nothing(species: str, item: Item):
    assert mega_forme(species, item) is None


def test_mega_swaps_species_data_and_refreshes_cached_stats():
    pokemon = build_pokemon(_tyranitar(Item.TYRANITARITE))
    before = pokemon.stat_totals.ATTACK  # read first, so the cached_property is populated pre-swap

    apply_forme(pokemon, "Tyranitar-Mega")

    mega = get_species("Tyranitar-Mega")
    assert pokemon.name == "Tyranitar-Mega"
    assert pokemon.base_stats == mega.base_stats
    assert pokemon.weight_kg == mega.weight_kg
    assert before < pokemon.stat_totals.ATTACK  # a stale cache would still report the base forme's total


def test_mega_keeps_what_the_trainer_chose():
    pokemon = build_pokemon(_tyranitar(Item.TYRANITARITE))
    moves = [move.name for move in pokemon.moves if move is not None]
    pokemon.live_stats.HP = 150

    apply_forme(pokemon, "Tyranitar-Mega")

    assert [move.name for move in pokemon.moves if move is not None] == moves
    assert pokemon.item is Item.TYRANITARITE
    assert pokemon.live_stats.HP == 150
    assert pokemon.stat_totals.HP == build_pokemon(_tyranitar(Item.TYRANITARITE)).stat_totals.HP


def test_mega_can_change_typing_and_ability():
    spec = PokemonSpec(
        species="Charizard",
        level=80,
        moves=["Flamethrower", "Air Slash", "Roost", "Dragon Pulse"],
        ability=Ability.BLAZE,
        item=Item.CHARIZARDITE_X,
        nature=Nature.ADAMANT,
    )
    pokemon = build_pokemon(spec)
    assert Type.FLYING in pokemon.types

    apply_forme(pokemon, "Charizard-Mega-X")

    assert pokemon.types == (Type.FIRE, Type.DRAGON)
    assert pokemon.ability is Ability.TOUGH_CLAWS


def test_every_stone_forme_is_constructible():
    """Guards the whole table: a forme we can name but not build would crash mid-battle."""
    table = _forme_by_base_and_item()
    assert len(table) == 49  # every Gen-7 mega and primal forme, keyed by the stone that reaches it
    for (base, item), forme in table.items():
        species = get_species(forme)
        assert species.regular_abilities, f"{forme} has no ability to assign"
        assert ability_from_showdown(species.regular_abilities[0]) is not Ability.NONE, f"{forme} ability unmapped"
        assert species.base_stats.HP == get_species(base).base_stats.HP, f"{forme} changes base HP"
        assert item is not Item.NONE


def test_mega_stones_are_exactly_the_items_the_forme_table_uses():
    assert mega_stones() == {item for _, item in _forme_by_base_and_item()}
    assert Item.CHARIZARDITE_X in mega_stones()
    assert Item.LEFTOVERS not in mega_stones()


def test_forme_that_changes_max_hp_keeps_the_hp_fraction():
    """Zygarde's Power Construct doubles base HP — the only corpus forme that does."""
    spec = PokemonSpec(
        species="Zygarde",
        level=80,
        moves=["Thousand Arrows", "Dragon Dance", "Extreme Speed", "Substitute"],
        ability=Ability.AURA_BREAK,
        item=Item.NONE,
        nature=Nature.ADAMANT,
    )
    pokemon = build_pokemon(spec)
    pokemon.live_stats.HP = pokemon.stat_totals.HP // 2

    apply_forme(pokemon, "Zygarde-Complete")

    assert pokemon.name == "Zygarde-Complete"
    assert abs(pokemon.live_stats.HP - pokemon.stat_totals.HP // 2) <= 1


def test_fainted_stays_fainted_across_a_forme_change():
    pokemon = build_pokemon(_tyranitar(Item.TYRANITARITE))
    pokemon.live_stats.HP = 0

    apply_forme(pokemon, "Tyranitar-Mega")

    assert pokemon.is_fainted()
