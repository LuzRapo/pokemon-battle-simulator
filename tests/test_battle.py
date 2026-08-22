import pytest
from pydantic import ValidationError

from battle_sim.maths.rng import RNG
from battle_sim.mechanics.battle import BattleState, FieldState, SideState
from battle_sim.utils import BattleFormat, ExtraStatus, Hazards, PseudoWeather, Terrain, Weather


def test_field_state_defaults():
    field = FieldState()
    assert field.weather is Weather.NONE
    assert field.weather_turns_left == 0
    assert field.terrain is Terrain.NONE
    assert field.terrain_turns_left == 0
    assert field.pseudo_weather == {}


def test_field_state_holds_weather_terrain_and_pseudo_weather():
    field = FieldState(
        weather=Weather.RAIN,
        weather_turns_left=5,
        terrain=Terrain.GRASSY,
        terrain_turns_left=8,
        pseudo_weather={PseudoWeather.TRICK_ROOM: 5, PseudoWeather.GRAVITY: 4},
    )
    assert field.weather is Weather.RAIN
    assert field.terrain is Terrain.GRASSY
    assert field.pseudo_weather[PseudoWeather.TRICK_ROOM] == 5
    assert field.pseudo_weather[PseudoWeather.GRAVITY] == 4


def test_side_state_defaults_to_first_active(garchomp_factory):
    side = SideState(team=[garchomp_factory("A"), garchomp_factory("B")])
    assert side.active == [0]
    assert side.active_pokemon.nickname == "A"
    assert [p.nickname for p in side.bench] == ["B"]


def test_side_state_respects_explicit_active(garchomp_factory):
    side = SideState(team=[garchomp_factory("A"), garchomp_factory("B"), garchomp_factory("C")], active=[2])
    assert side.active_pokemon.nickname == "C"
    assert [p.nickname for p in side.bench] == ["A", "B"]


def test_side_state_rejects_empty_team():
    with pytest.raises(ValidationError):
        SideState(team=[])


def test_side_state_rejects_oversized_team(garchomp_factory):
    with pytest.raises(ValidationError):
        SideState(team=[garchomp_factory(f"G{i}") for i in range(7)])


def test_side_state_rejects_active_index_out_of_range(garchomp_factory):
    with pytest.raises(ValidationError):
        SideState(team=[garchomp_factory("A")], active=[5])


def test_side_state_rejects_negative_active_index(garchomp_factory):
    with pytest.raises(ValidationError):
        SideState(team=[garchomp_factory("A")], active=[-1])


def test_side_state_rejects_duplicate_active_indices(garchomp_factory):
    with pytest.raises(ValidationError):
        SideState(team=[garchomp_factory("A"), garchomp_factory("B")], active=[0, 0])


def test_side_state_carries_hazards_and_screens(garchomp_factory):
    side = SideState(
        team=[garchomp_factory("A")],
        hazards={Hazards.STEALTH_ROCK: 1, Hazards.SPIKES: 2},
        screens={Hazards.LIGHT_SCREEN: 5, Hazards.REFLECT: 3},
        tailwind_turns=4,
    )
    assert side.hazards == {Hazards.STEALTH_ROCK: 1, Hazards.SPIKES: 2}
    assert side.screens == {Hazards.LIGHT_SCREEN: 5, Hazards.REFLECT: 3}
    assert side.tailwind_turns == 4


def test_side_state_rejects_negative_tailwind(garchomp_factory):
    with pytest.raises(ValidationError):
        SideState(team=[garchomp_factory("A")], tailwind_turns=-1)


def test_battle_state_defaults(garchomp_factory):
    sides = (
        SideState(team=[garchomp_factory("A")]),
        SideState(team=[garchomp_factory("B")]),
    )
    state = BattleState(sides=sides)
    assert state.turn == 0
    assert state.format is BattleFormat.SINGLES
    assert state.field.weather is Weather.NONE
    assert isinstance(state.rng, RNG)


def test_battle_state_threads_rng_for_reproducibility(garchomp_factory):
    sides = (
        SideState(team=[garchomp_factory("A")]),
        SideState(team=[garchomp_factory("B")]),
    )
    state = BattleState(sides=sides, rng=RNG(seed=42))
    expected = RNG(seed=42).random_probability()
    assert state.rng.random_probability() == expected


def test_battle_state_singles_requires_exactly_one_active(garchomp_factory):
    side1 = SideState(team=[garchomp_factory("A"), garchomp_factory("B")], active=[0, 1])
    side2 = SideState(team=[garchomp_factory("C")])
    with pytest.raises(ValidationError, match="active pokemon"):
        BattleState(sides=(side1, side2))


def test_battle_state_rejects_negative_turn(garchomp_factory):
    sides = (
        SideState(team=[garchomp_factory("A")]),
        SideState(team=[garchomp_factory("B")]),
    )
    with pytest.raises(ValidationError):
        BattleState(sides=sides, turn=-1)


def test_pokemon_volatiles_default_empty(garchomp_factory):
    chomp = garchomp_factory("A")
    assert chomp.volatiles == {}


def test_pokemon_volatiles_are_mutable(garchomp_factory):
    chomp = garchomp_factory("A")
    chomp.volatiles[ExtraStatus.CONFUSION] = 3
    chomp.volatiles[ExtraStatus.LEECH_SEED] = 1
    assert chomp.volatiles[ExtraStatus.CONFUSION] == 3
    assert chomp.volatiles[ExtraStatus.LEECH_SEED] == 1


def test_pokemon_volatiles_independent_between_instances(garchomp_factory):
    a = garchomp_factory("A")
    b = garchomp_factory("B")
    a.volatiles[ExtraStatus.TAUNT] = 4
    assert b.volatiles == {}


def test_battle_state_round_trip_through_sides_index(garchomp_factory):
    side0 = SideState(team=[garchomp_factory("A"), garchomp_factory("B")])
    side1 = SideState(team=[garchomp_factory("C"), garchomp_factory("D")], active=[1])
    state = BattleState(sides=(side0, side1))
    assert state.sides[0].active_pokemon.nickname == "A"
    assert state.sides[1].active_pokemon.nickname == "D"
