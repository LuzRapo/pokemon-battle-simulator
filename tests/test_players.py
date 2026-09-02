"""The analysis layer and the reference players."""

import random
from collections.abc import Sequence
from dataclasses import dataclass

from battle_sim.analysis import damage_range, strongest_hit
from battle_sim.database.loader import get_move
from battle_sim.maths.rng import RNG
from battle_sim.mechanics.battle import BattleState, FieldState, SideState
from battle_sim.models.actions import Action, ActionType
from battle_sim.models.moves import MoveSet, MoveSlot
from battle_sim.models.pokemon import Pokemon
from battle_sim.models.spec import PokemonSpec
from battle_sim.models.stats import BaseStats, EVs, IVs
from battle_sim.observation import BeliefSampler
from battle_sim.players import BasicPlayer, FixedOrderPlayer, RandomPlayer
from battle_sim.runner import run_battle
from battle_sim.teams import parse_showdown_team
from battle_sim.utils import Ability, Hazards, Nature, Outcome, Target, Type
from tests.test_teams import USER_SAMPLE_TEAM

TACKLE = get_move("Tackle")
EMBER = get_move("Ember")
RECOVER = get_move("Recover")
STEALTH_ROCK = get_move("Stealth Rock")
SWORDS_DANCE = get_move("Swords Dance")
SPLASH = get_move("Splash")


def _mk(
    nickname: str = "X",
    types: tuple[Type, Type | None] = (Type.NORMAL, None),
    base_stats: BaseStats | None = None,
    moves: MoveSet | None = None,
) -> Pokemon:
    return Pokemon(
        name="TestMon",
        nickname=nickname,
        level=50,
        base_stats=base_stats or BaseStats(HP=100, ATTACK=100, DEFENCE=100, SP_ATTACK=100, SP_DEFENCE=100, SPEED=100),
        effort_values=EVs(),
        individual_values=IVs(),
        types=types,
        moves=moves or MoveSet(TACKLE, EMBER, RECOVER, SPLASH),
        nature=Nature.HARDY,
    )


def _battle(side0: list[Pokemon], side1: list[Pokemon]) -> BattleState:
    return BattleState(sides=(SideState(team=side0), SideState(team=side1)), rng=RNG(seed=0), field=FieldState())


def _use(slot: MoveSlot, target: Target = Target.SINGLE_OPPONENT) -> Action:
    return Action(action=ActionType.USE_MOVE, target=target, move=slot)


def test_damage_range_is_ordered_and_positive():
    state = _battle([_mk("A")], [_mk("B")])
    low, high = damage_range(TACKLE, state.sides[0].active_pokemon, state.sides[1].active_pokemon, state)
    assert 0 < low <= high


def test_damage_range_is_zero_for_status_moves_and_immunities():
    state = _battle([_mk("A")], [_mk("B", types=(Type.GHOST, None))])
    me, ghost = state.sides[0].active_pokemon, state.sides[1].active_pokemon
    assert damage_range(SWORDS_DANCE, me, ghost, state) == (0, 0)
    assert damage_range(TACKLE, me, ghost, state) == (0, 0)  # normal vs ghost


def test_damage_range_respects_unaware_in_both_directions():
    state = _battle([_mk("A")], [_mk("B")])
    me, them = state.sides[0].active_pokemon, state.sides[1].active_pokemon
    flat = damage_range(TACKLE, me, them, state)
    me.stat_stages.ATTACK = 4
    assert damage_range(TACKLE, me, them, state) > flat  # boosts work against an aware defender
    them.ability = Ability.UNAWARE
    assert damage_range(TACKLE, me, them, state) == flat  # ...and are ignored by Unaware
    me.ability = Ability.MOLD_BREAKER
    assert damage_range(TACKLE, me, them, state) > flat  # Mold Breaker pierces Unaware
    me.ability = Ability.UNAWARE
    me.stat_stages.ATTACK = 0
    them.ability = Ability.NONE
    them.stat_stages.DEFENCE = 4
    assert damage_range(TACKLE, me, them, state) == flat  # the Unaware attacker ignores defence boosts


def test_strongest_hit_reports_the_biggest_threat():
    state = _battle([_mk("A")], [_mk("B", types=(Type.GRASS, None))])
    me, grass = state.sides[0].active_pokemon, state.sides[1].active_pokemon
    _, ember_high = damage_range(EMBER, me, grass, state)
    assert strongest_hit(me, grass, state) == ember_high  # super-effective Ember outhits Tackle


def test_basic_player_prefers_super_effective_damage():
    state = _battle([_mk("A")], [_mk("B", types=(Type.GRASS, None))])
    actions = [_use(MoveSlot.FIRST), _use(MoveSlot.SECOND)]
    chosen = BasicPlayer().choose_action(state, 0, actions)
    assert chosen.move is MoveSlot.SECOND  # Ember


def test_basic_player_takes_the_guaranteed_ko():
    weak = _mk("B")
    weak.live_stats.HP = 5
    state = _battle([_mk("A")], [weak])
    actions = [_use(MoveSlot.FIRST), _use(MoveSlot.THIRD, Target.SELF), _use(MoveSlot.FOURTH, Target.SELF)]
    chosen = BasicPlayer().choose_action(state, 0, actions)
    assert chosen.move is MoveSlot.FIRST


def test_basic_player_heals_when_hurt_and_safe():
    hurt = _mk("A")
    hurt.live_stats.HP = hurt.stat_totals.HP // 4
    pacifist = _mk("B", moves=MoveSet(SPLASH, SPLASH, SPLASH, SPLASH))
    tank = BaseStats(HP=255, ATTACK=10, DEFENCE=230, SP_ATTACK=10, SP_DEFENCE=230, SPEED=10)
    state = _battle([hurt], [_mk("B", base_stats=tank, moves=MoveSet(SPLASH, SPLASH, SPLASH, SPLASH))])
    actions = [_use(MoveSlot.FIRST), _use(MoveSlot.SECOND), _use(MoveSlot.THIRD, Target.SELF)]
    chosen = BasicPlayer().choose_action(state, 0, actions)
    assert chosen.move is MoveSlot.THIRD  # Recover beats chipping a wall
    assert pacifist is not None


def test_basic_player_lays_rocks_against_a_wall():
    rocks_moves = MoveSet(STEALTH_ROCK, TACKLE, EMBER, SPLASH)
    tank = BaseStats(HP=255, ATTACK=10, DEFENCE=230, SP_ATTACK=10, SP_DEFENCE=230, SPEED=10)
    state = _battle([_mk("A", moves=rocks_moves)], [_mk("B", base_stats=tank)])
    actions = [_use(MoveSlot.FIRST, Target.OPPONENT_SIDE), _use(MoveSlot.SECOND), _use(MoveSlot.THIRD)]
    chosen = BasicPlayer().choose_action(state, 0, actions)
    assert chosen.move is MoveSlot.FIRST
    state.sides[1].hazards[Hazards.STEALTH_ROCK] = 1  # once laid, attack instead
    chosen = BasicPlayer().choose_action(state, 0, actions)
    assert chosen.move is not MoveSlot.FIRST


def test_basic_player_picks_the_best_replacement():
    fire_attacker = _mk("F", types=(Type.FIRE, None), moves=MoveSet(EMBER, EMBER, EMBER, EMBER))
    fainted = _mk("Down")
    fainted.live_stats.HP = 0
    grass = _mk("Leafy", types=(Type.GRASS, None))
    water = _mk("Wet", types=(Type.WATER, None))
    state = _battle([fainted, grass, water], [fire_attacker])
    actions = [
        Action(action=ActionType.SWITCH_OUT, switch_in=grass),
        Action(action=ActionType.SWITCH_OUT, switch_in=water),
    ]
    chosen = BasicPlayer().choose_action(state, 0, actions)
    assert chosen.switch_in is water  # resists Ember; grass invites disaster


def test_basic_player_choose_order_is_a_permutation():
    specs = parse_showdown_team(USER_SAMPLE_TEAM).specs
    order = BasicPlayer().choose_order(specs, specs)
    assert sorted(order) == list(range(6))


def test_random_player_choose_order_is_a_permutation():
    specs = parse_showdown_team(USER_SAMPLE_TEAM).specs
    order = RandomPlayer(seed=3).choose_order(specs, specs)
    assert sorted(order) == list(range(6))


def test_fixed_order_player_ignores_the_inners_own_choice():
    specs = parse_showdown_team(USER_SAMPLE_TEAM).specs
    wrapped = FixedOrderPlayer(RandomPlayer(seed=1))
    assert wrapped.choose_order(specs, specs) == list(range(6))


def test_fixed_order_player_still_delegates_actions_to_the_inner_player():
    state = _battle([_mk("A")], [_mk("B")])
    inner = BasicPlayer()
    wrapped = FixedOrderPlayer(inner)
    actions = [_use(MoveSlot.FIRST)]
    assert wrapped.choose_action(state, 0, actions) == inner.choose_action(state, 0, actions)


def test_fixed_order_player_forwards_the_belief_sampler_to_a_determinizing_inner():
    calls: list[BeliefSampler] = []

    @dataclass
    class _Determinizing:
        def choose_order(self, own: Sequence[PokemonSpec], opponent: Sequence[PokemonSpec]) -> Sequence[int]:
            return list(range(len(own)))

        def choose_action(self, state: BattleState, side_index: int, actions: Sequence[Action]) -> Action:
            return actions[0]

        def bind_belief_sampler(self, sampler: BeliefSampler) -> None:
            calls.append(sampler)

    def sentinel(rng: random.Random) -> BattleState:
        raise NotImplementedError  # never actually called — only identity matters here

    FixedOrderPlayer(_Determinizing()).bind_belief_sampler(sentinel)
    assert calls == [sentinel]


def test_fixed_order_player_bind_belief_sampler_is_a_no_op_for_a_plain_inner():
    def sentinel(rng: random.Random) -> BattleState:
        raise NotImplementedError

    FixedOrderPlayer(RandomPlayer()).bind_belief_sampler(sentinel)  # must not raise


def test_basic_player_crushes_random_play():
    specs = parse_showdown_team(USER_SAMPLE_TEAM).specs
    wins = 0
    for seed in range(30):
        result = run_battle(specs, specs, BasicPlayer(), RandomPlayer(seed=seed), seed=seed)
        wins += result.outcome is Outcome.P1_WIN
    assert wins >= 24  # observed: 100% in mirrors; 80% is the regression floor
