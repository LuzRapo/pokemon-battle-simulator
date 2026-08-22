import random

import pytest

from battle_sim.database.scope import in_scope_species
from battle_sim.models.species import BaseSpecies
from battle_sim.models.stats import BaseStats
from battle_sim.setgen import _BULKY_ITEMS, _FAST_OFFENSE_ITEMS, _SLOW_OFFENSE_ITEMS, _infer_item, random_set
from battle_sim.teams import build_pokemon
from battle_sim.utils import Ability, Item, Type

_STAT_NAMES = ("HP", "ATTACK", "DEFENCE", "SP_ATTACK", "SP_DEFENCE", "SPEED")


def _species(base_stats: BaseStats, fully_evolved: bool = True) -> BaseSpecies:
    return BaseSpecies(
        name="TestMon",
        dex_number=1,
        types=(Type.NORMAL, None),
        base_stats=base_stats,
        regular_abilities=("Levitate",),
        hidden_ability=None,
        height_m=1.0,
        weight_kg=1.0,
        fully_evolved=fully_evolved,
    )


def test_default_level_is_50():
    spec = random_set("Ditto", random.Random(0))
    assert spec.level == 50


def test_explicit_level_overrides_the_default():
    spec = random_set("Ditto", random.Random(0), level=25)
    assert spec.level == 25


def test_evs_are_legal_and_randomized_across_calls():
    rng = random.Random(0)
    spreads = [random_set("Ditto", rng).effort_values for _ in range(20)]
    for evs in spreads:
        assert evs.total <= 510
        assert all(0 <= getattr(evs, stat) <= 252 for stat in _STAT_NAMES)
    assert len({tuple(getattr(evs, stat) for stat in _STAT_NAMES) for evs in spreads}) > 1


def test_ivs_are_legal_and_randomized_across_calls():
    rng = random.Random(0)
    spreads = [random_set("Ditto", rng).individual_values for _ in range(20)]
    for ivs in spreads:
        assert all(0 <= getattr(ivs, stat) <= 31 for stat in _STAT_NAMES)
    assert len({tuple(getattr(ivs, stat) for stat in _STAT_NAMES) for ivs in spreads}) > 1


def test_nature_is_randomized_across_calls():
    rng = random.Random(0)
    natures = {random_set("Ditto", rng).nature for _ in range(20)}
    assert len(natures) > 1


def test_randbats_role_species_gets_a_real_ability_and_item():
    spec = random_set("Ditto", random.Random(0))
    assert spec.ability is Ability.IMPOSTER
    assert spec.item is not Item.NONE
    assert 1 <= len(spec.moves) <= 4


def test_species_outside_randbats_falls_back_to_a_legal_heuristic_set():
    spec = random_set("Cosmog", random.Random(0))
    assert spec.ability is not Ability.NONE
    assert 1 <= len(spec.moves) <= 4


def test_unknown_species_raises():
    with pytest.raises(KeyError):
        random_set("Not-A-Real-Species", random.Random(0))


def test_mega_forme_is_rejected():
    with pytest.raises(KeyError):
        random_set("Charizard-Mega-X", random.Random(0))


@pytest.mark.parametrize("species", ["Rayquaza", "Wobbuffet", "Unown", "Arceus-Bug", "Abomasnow"])
def test_generated_set_builds_a_real_pokemon(species: str):
    spec = random_set(species, random.Random(0))
    build_pokemon(spec)


def test_every_in_scope_species_generates_a_buildable_set():
    rng = random.Random(1)
    for species in in_scope_species():
        spec = random_set(species, rng)
        assert spec.moves
        build_pokemon(spec)


def test_infer_item_gives_eviolite_to_not_fully_evolved_species():
    stats = BaseStats(HP=60, ATTACK=60, DEFENCE=60, SP_ATTACK=60, SP_DEFENCE=60, SPEED=60)
    species = _species(stats, fully_evolved=False)
    for seed in range(10):
        assert _infer_item(species, ["tackle"], random.Random(seed)) is Item.EVIOLITE


def test_infer_item_picks_a_bulky_item_for_a_tank_stat_spread():
    stats = BaseStats(HP=150, ATTACK=30, DEFENCE=120, SP_ATTACK=30, SP_DEFENCE=120, SPEED=30)
    species = _species(stats)
    for seed in range(20):
        assert _infer_item(species, ["tackle"], random.Random(seed)) in _BULKY_ITEMS


def test_infer_item_picks_a_fast_offense_item_for_a_fast_attacker():
    stats = BaseStats(HP=70, ATTACK=130, DEFENCE=60, SP_ATTACK=50, SP_DEFENCE=60, SPEED=120)
    species = _species(stats)
    for seed in range(20):
        assert _infer_item(species, ["tackle"], random.Random(seed)) in (Item.CHOICE_BAND, *_FAST_OFFENSE_ITEMS)


def test_infer_item_picks_a_wallbreaker_item_for_a_slow_attacker():
    stats = BaseStats(HP=90, ATTACK=140, DEFENCE=80, SP_ATTACK=40, SP_DEFENCE=80, SPEED=40)
    species = _species(stats)
    for seed in range(20):
        assert _infer_item(species, ["tackle"], random.Random(seed)) in (Item.CHOICE_BAND, *_SLOW_OFFENSE_ITEMS)


def test_infer_item_breaks_a_stat_tie_using_the_movepools_physical_special_split():
    """Equal Attack/Sp. Atk: three physical moves vs one special move should read as physical."""
    stats = BaseStats(HP=90, ATTACK=100, DEFENCE=70, SP_ATTACK=100, SP_DEFENCE=70, SPEED=40)
    species = _species(stats)
    physical_leaning = ["Tackle", "Close Combat", "Earthquake", "Ember"]
    special_leaning = ["Ember", "Ice Beam", "Thunderbolt", "Tackle"]
    physical_choices = {_infer_item(species, physical_leaning, random.Random(s)) for s in range(30)}
    special_choices = {_infer_item(species, special_leaning, random.Random(s)) for s in range(30)}
    assert Item.CHOICE_BAND in physical_choices
    assert Item.CHOICE_SPECS in special_choices
    assert Item.CHOICE_SPECS not in physical_choices
    assert Item.CHOICE_BAND not in special_choices
