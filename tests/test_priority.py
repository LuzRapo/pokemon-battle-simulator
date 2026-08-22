import pytest

from battle_sim.database.loader import get_move
from battle_sim.database.sample_moves import EARTHQUAKE, ROCK_SLIDE
from battle_sim.maths.rng import RNG
from battle_sim.mechanics.battle import BattleState, FieldState, SideState
from battle_sim.mechanics.priority import effective_speed, order_actions
from battle_sim.models.actions import Action, ActionType
from battle_sim.models.moves import MoveSet, MoveSlot
from battle_sim.models.pokemon import Pokemon
from battle_sim.models.stats import BaseStats, EVs, IVs
from battle_sim.utils import Nature, PseudoWeather, Status, Target, Type


@pytest.fixture
def make_chomp():
    def _make(nickname: str, speed_stages: int = 0, status: Status = Status.NONE) -> Pokemon:
        quick_attack = get_move("Quick Attack")
        tackle = get_move("Tackle")
        pokemon = Pokemon(
            name="Garchomp",
            nickname=nickname,
            level=78,
            base_stats=BaseStats(HP=108, ATTACK=130, DEFENCE=95, SP_ATTACK=80, SP_DEFENCE=85, SPEED=102),
            effort_values=EVs(HP=74, ATTACK=190, DEFENCE=91, SP_ATTACK=48, SP_DEFENCE=84, SPEED=23),
            individual_values=IVs(HP=24, ATTACK=12, DEFENCE=30, SP_ATTACK=16, SP_DEFENCE=23, SPEED=5),
            types=(Type.DRAGON, Type.GROUND),
            moves=MoveSet(tackle, quick_attack, EARTHQUAKE, ROCK_SLIDE),
            nature=Nature.ADAMANT,
            status=status,
        )
        pokemon.stat_stages.SPEED = speed_stages
        return pokemon

    return _make


def _state(side0: SideState, side1: SideState, rng: RNG | None = None, field: FieldState | None = None) -> BattleState:
    return BattleState(
        sides=(side0, side1),
        rng=rng if rng is not None else RNG(seed=0),
        field=field if field is not None else FieldState(),
    )


TACKLE_FIRST = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.FIRST)
QUICK_ATTACK_SECOND = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.SECOND)


def test_effective_speed_base(make_chomp):
    pokemon = make_chomp("A")
    side = SideState(team=[pokemon])
    assert effective_speed(pokemon, side, FieldState()) == 171


def test_effective_speed_paralysis_halves(make_chomp):
    pokemon = make_chomp("A", status=Status.PARALYSIS)
    side = SideState(team=[pokemon])
    assert effective_speed(pokemon, side, FieldState()) == 85


def test_effective_speed_tailwind_doubles(make_chomp):
    pokemon = make_chomp("A")
    side = SideState(team=[pokemon], tailwind_turns=3)
    assert effective_speed(pokemon, side, FieldState()) == 342


def test_effective_speed_paralysis_then_tailwind(make_chomp):
    pokemon = make_chomp("A", status=Status.PARALYSIS)
    side = SideState(team=[pokemon], tailwind_turns=3)
    assert effective_speed(pokemon, side, FieldState()) == 170


def test_effective_speed_stage_modifier(make_chomp):
    pokemon = make_chomp("A", speed_stages=2)
    side = SideState(team=[pokemon])
    assert effective_speed(pokemon, side, FieldState()) == 342


def test_higher_priority_acts_first(make_chomp):
    side0 = SideState(team=[make_chomp("A")])
    side1 = SideState(team=[make_chomp("B", speed_stages=2)])
    state = _state(side0, side1)

    ordered = order_actions({0: QUICK_ATTACK_SECOND, 1: TACKLE_FIRST}, state)
    assert [side_index for side_index, _ in ordered] == [0, 1]


def test_equal_priority_higher_speed_first(make_chomp):
    side0 = SideState(team=[make_chomp("A")])
    side1 = SideState(team=[make_chomp("B", speed_stages=2)])
    state = _state(side0, side1)

    ordered = order_actions({0: TACKLE_FIRST, 1: TACKLE_FIRST}, state)
    assert [side_index for side_index, _ in ordered] == [1, 0]


def test_trick_room_inverts_speed(make_chomp):
    side0 = SideState(team=[make_chomp("A")])
    side1 = SideState(team=[make_chomp("B", speed_stages=2)])
    state = _state(side0, side1, field=FieldState(pseudo_weather={PseudoWeather.TRICK_ROOM: 5}))

    ordered = order_actions({0: TACKLE_FIRST, 1: TACKLE_FIRST}, state)
    assert [side_index for side_index, _ in ordered] == [0, 1]


def test_paralysis_lowers_effective_speed(make_chomp):
    side0 = SideState(team=[make_chomp("A", speed_stages=2)])
    side1 = SideState(team=[make_chomp("B", status=Status.PARALYSIS)])
    state = _state(side0, side1)

    ordered = order_actions({0: TACKLE_FIRST, 1: TACKLE_FIRST}, state)
    assert [side_index for side_index, _ in ordered] == [0, 1]


def test_tailwind_raises_effective_speed(make_chomp):
    side0 = SideState(team=[make_chomp("A")], tailwind_turns=4)
    side1 = SideState(team=[make_chomp("B", speed_stages=1)])
    state = _state(side0, side1)

    ordered = order_actions({0: TACKLE_FIRST, 1: TACKLE_FIRST}, state)
    assert [side_index for side_index, _ in ordered] == [0, 1]


def test_switch_acts_before_move(make_chomp):
    side0 = SideState(team=[make_chomp("A"), make_chomp("A2")])
    side1 = SideState(team=[make_chomp("B", speed_stages=6)])
    state = _state(side0, side1)

    switch = Action(action=ActionType.SWITCH_OUT, switch_in=side0.team[1])
    ordered = order_actions({0: switch, 1: QUICK_ATTACK_SECOND}, state)
    assert ordered[0][0] == 0


def test_switch_vs_switch_orders_by_speed(make_chomp):
    side0 = SideState(team=[make_chomp("A"), make_chomp("A2")])
    side1 = SideState(team=[make_chomp("B", speed_stages=2), make_chomp("B2")])
    state = _state(side0, side1)

    switch0 = Action(action=ActionType.SWITCH_OUT, switch_in=side0.team[1])
    switch1 = Action(action=ActionType.SWITCH_OUT, switch_in=side1.team[1])
    ordered = order_actions({0: switch0, 1: switch1}, state)
    assert [side_index for side_index, _ in ordered] == [1, 0]


def test_speed_tie_deterministic_with_same_seed(make_chomp):
    def run(seed: int) -> list[int]:
        side0 = SideState(team=[make_chomp("A")])
        side1 = SideState(team=[make_chomp("B")])
        state = _state(side0, side1, rng=RNG(seed=seed))
        return [side_index for side_index, _ in order_actions({0: TACKLE_FIRST, 1: TACKLE_FIRST}, state)]

    assert run(42) == run(42)


def test_speed_tie_resolution_can_flip(make_chomp):
    orderings: set[tuple[int, ...]] = set()
    for seed in range(20):
        side0 = SideState(team=[make_chomp("A")])
        side1 = SideState(team=[make_chomp("B")])
        state = _state(side0, side1, rng=RNG(seed=seed))
        ordered = order_actions({0: TACKLE_FIRST, 1: TACKLE_FIRST}, state)
        orderings.add(tuple(side_index for side_index, _ in ordered))
    assert orderings == {(0, 1), (1, 0)}


def test_speed_tie_is_approximately_fair(make_chomp):
    side0_wins = 0
    trials = 2000
    for seed in range(trials):
        side0 = SideState(team=[make_chomp("A")])
        side1 = SideState(team=[make_chomp("B")])
        state = _state(side0, side1, rng=RNG(seed=seed))
        ordered = order_actions({0: TACKLE_FIRST, 1: TACKLE_FIRST}, state)
        if ordered[0][0] == 0:
            side0_wins += 1
    rate = side0_wins / trials
    assert 0.45 < rate < 0.55, f"Speed-tie fairness rate {rate} outside 45-55% over {trials} trials"


def test_priority_beats_speed_under_trick_room(make_chomp):
    side0 = SideState(team=[make_chomp("A", speed_stages=2)])
    side1 = SideState(team=[make_chomp("B")])
    state = _state(side0, side1, field=FieldState(pseudo_weather={PseudoWeather.TRICK_ROOM: 5}))

    ordered = order_actions({0: QUICK_ATTACK_SECOND, 1: TACKLE_FIRST}, state)
    assert ordered[0][0] == 0
