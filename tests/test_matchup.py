import math

from battle_sim.analysis import entry_hazard_chip, exchange_edge, survival_turns
from battle_sim.database.loader import get_move
from battle_sim.matchup import MatchupPlayer
from battle_sim.maths.rng import RNG
from battle_sim.mechanics.battle import BattleState, FieldState, SideState
from battle_sim.models.actions import Action, ActionType
from battle_sim.models.moves import MoveSet, MoveSlot
from battle_sim.models.pokemon import Pokemon
from battle_sim.models.stats import BaseStats, EVs, IVs
from battle_sim.players import RandomPlayer
from battle_sim.runner import run_battle
from battle_sim.teams import parse_showdown_team
from battle_sim.utils import Hazards, Item, Nature, Outcome, Status, Target, Type
from tests.test_teams import USER_SAMPLE_TEAM

TACKLE = get_move("Tackle")
EMBER = get_move("Ember")
RECOVER = get_move("Recover")
SPLASH = get_move("Splash")
STEALTH_ROCK = get_move("Stealth Rock")
THUNDER_WAVE = get_move("Thunder Wave")
SWORDS_DANCE = get_move("Swords Dance")

TANK = BaseStats(HP=255, ATTACK=10, DEFENCE=230, SP_ATTACK=10, SP_DEFENCE=230, SPEED=10)
FAST = BaseStats(HP=100, ATTACK=100, DEFENCE=100, SP_ATTACK=100, SP_DEFENCE=100, SPEED=200)


def _mk(
    nickname: str = "X",
    types: tuple[Type, Type | None] = (Type.NORMAL, None),
    base_stats: BaseStats | None = None,
    moves: MoveSet | None = None,
    item: Item = Item.NONE,
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
        item=item,
    )


def _battle(side0: list[Pokemon], side1: list[Pokemon]) -> BattleState:
    return BattleState(sides=(SideState(team=side0), SideState(team=side1)), rng=RNG(seed=0), field=FieldState())


def _use(slot: MoveSlot, target: Target = Target.SINGLE_OPPONENT) -> Action:
    return Action(action=ActionType.USE_MOVE, target=target, move=slot)


def _switch(pokemon: Pokemon) -> Action:
    return Action(action=ActionType.SWITCH_OUT, switch_in=pokemon)


# -- Analysis primitives ----------------------------------------------------------


def test_survival_turns_is_infinite_against_a_harmless_attacker():
    pacifist = _mk("P", moves=MoveSet(SPLASH, SPLASH, SPLASH, SPLASH))
    state = _battle([pacifist], [_mk("B")])
    assert math.isinf(survival_turns(state.sides[1].active_pokemon, pacifist, state))


def test_exchange_edge_favors_the_super_effective_faster_side():
    fire = _mk("F", types=(Type.FIRE, None), base_stats=FAST, moves=MoveSet(EMBER, EMBER, EMBER, EMBER))
    grass = _mk("G", types=(Type.GRASS, None))
    state = _battle([fire], [grass])
    assert exchange_edge(fire, grass, state) > 0
    assert exchange_edge(grass, fire, state) < 0


def test_entry_hazard_chip_counts_rocks_and_spikes_but_not_boots():
    side = SideState(team=[_mk("A")], hazards={Hazards.STEALTH_ROCK: 1, Hazards.SPIKES: 1})
    plain = _mk("In")
    assert entry_hazard_chip(plain, side) == int(plain.stat_totals.HP / 8 + plain.stat_totals.HP / 8)
    booted = _mk("Boots", item=Item.HEAVY_DUTY_BOOTS)
    assert entry_hazard_chip(booted, side) == 0
    flying = _mk("Bird", types=(Type.NORMAL, Type.FLYING))
    assert entry_hazard_chip(flying, side) == int(flying.stat_totals.HP / 8 * 2)  # double rocks, no spikes


# -- MatchupPlayer decisions -------------------------------------------------------


def test_takes_the_guaranteed_ko():
    weak = _mk("B")
    weak.live_stats.HP = 5
    state = _battle([_mk("A")], [weak])
    actions = [_use(MoveSlot.FIRST), _use(MoveSlot.THIRD, Target.SELF), _use(MoveSlot.FOURTH, Target.SELF)]
    chosen = MatchupPlayer().choose_action(state, 0, actions)
    assert chosen.move is MoveSlot.FIRST


def test_prefers_super_effective_damage():
    state = _battle([_mk("A")], [_mk("B", types=(Type.GRASS, None))])
    actions = [_use(MoveSlot.FIRST), _use(MoveSlot.SECOND)]
    chosen = MatchupPlayer().choose_action(state, 0, actions)
    assert chosen.move is MoveSlot.SECOND  # Ember


def test_paralyzes_the_faster_wall_it_cannot_break():
    fast_tank = BaseStats(HP=255, ATTACK=10, DEFENCE=230, SP_ATTACK=10, SP_DEFENCE=230, SPEED=200)
    me = _mk("A", moves=MoveSet(TACKLE, THUNDER_WAVE, RECOVER, SPLASH))
    state = _battle([me], [_mk("Wall", base_stats=fast_tank)])
    actions = [_use(MoveSlot.FIRST), _use(MoveSlot.SECOND)]
    chosen = MatchupPlayer().choose_action(state, 0, actions)
    assert chosen.move is MoveSlot.SECOND
    state.sides[1].active_pokemon.status = Status.PARALYSIS
    chosen = MatchupPlayer().choose_action(state, 0, actions)  # already statused: chip instead
    assert chosen.move is MoveSlot.FIRST


def test_lays_rocks_against_a_wall_with_a_bootless_bench():
    me = _mk("A", moves=MoveSet(STEALTH_ROCK, TACKLE, EMBER, SPLASH))
    state = _battle([me], [_mk("Wall", base_stats=TANK), _mk("B2"), _mk("B3")])
    actions = [_use(MoveSlot.FIRST, Target.OPPONENT_SIDE), _use(MoveSlot.SECOND), _use(MoveSlot.THIRD)]
    chosen = MatchupPlayer().choose_action(state, 0, actions)
    assert chosen.move is MoveSlot.FIRST
    state.sides[1].hazards[Hazards.STEALTH_ROCK] = 1  # already up: attack instead
    chosen = MatchupPlayer().choose_action(state, 0, actions)
    assert chosen.move is not MoveSlot.FIRST


def test_heals_when_hurt_behind_a_wall():
    hurt = _mk("A")
    hurt.live_stats.HP = hurt.stat_totals.HP // 4
    state = _battle([hurt], [_mk("Wall", base_stats=TANK, moves=MoveSet(TACKLE, SPLASH, SPLASH, SPLASH))])
    actions = [_use(MoveSlot.FIRST), _use(MoveSlot.THIRD, Target.SELF)]
    chosen = MatchupPlayer().choose_action(state, 0, actions)
    assert chosen.move is MoveSlot.THIRD  # Recover buys more exchange turns than chipping earns


def test_sets_up_on_fodder():
    sword_user = _mk("A", moves=MoveSet(SWORDS_DANCE, TACKLE, SPLASH, SPLASH))
    fodder = _mk("Fodder", base_stats=TANK, moves=MoveSet(SPLASH, SPLASH, SPLASH, SPLASH))
    state = _battle([sword_user], [fodder])
    actions = [_use(MoveSlot.FIRST, Target.SELF), _use(MoveSlot.SECOND)]
    chosen = MatchupPlayer().choose_action(state, 0, actions)
    assert chosen.move is MoveSlot.FIRST


def test_switches_out_of_a_hopeless_matchup():
    helpless = _mk("A", moves=MoveSet(TACKLE, SPLASH, SPLASH, SPLASH))  # normal: no damage into ghost
    answer = _mk("Answer", types=(Type.WATER, None), moves=MoveSet(EMBER, EMBER, EMBER, EMBER))  # resists Ember
    ghost = _mk("Ghost", types=(Type.GHOST, None), base_stats=FAST, moves=MoveSet(EMBER, EMBER, EMBER, EMBER))
    state = _battle([helpless, answer], [ghost])
    actions = [_use(MoveSlot.FIRST), _switch(answer)]
    chosen = MatchupPlayer().choose_action(state, 0, actions)
    assert chosen.action is ActionType.SWITCH_OUT


def test_picks_the_best_forced_replacement():
    fire_attacker = _mk("F", types=(Type.FIRE, None), moves=MoveSet(EMBER, EMBER, EMBER, EMBER))
    fainted = _mk("Down")
    fainted.live_stats.HP = 0
    grass = _mk("Leafy", types=(Type.GRASS, None))
    water = _mk("Wet", types=(Type.WATER, None))
    state = _battle([fainted, grass, water], [fire_attacker])
    chosen = MatchupPlayer().choose_action(state, 0, [_switch(grass), _switch(water)])
    assert chosen.switch_in is water


def test_choose_order_is_a_permutation():
    specs = parse_showdown_team(USER_SAMPLE_TEAM).specs
    order = MatchupPlayer().choose_order(specs, specs)
    assert sorted(order) == list(range(6))


def test_crushes_random_play():
    specs = parse_showdown_team(USER_SAMPLE_TEAM).specs
    wins = 0
    for seed in range(30):
        result = run_battle(specs, specs, MatchupPlayer(), RandomPlayer(seed=seed), seed=seed)
        wins += result.outcome is Outcome.P1_WIN
    assert wins >= 24


def test_runner_reports_survivors():
    specs = parse_showdown_team(USER_SAMPLE_TEAM).specs
    result = run_battle(specs, specs, MatchupPlayer(), RandomPlayer(seed=1), seed=1)
    assert len(result.survivors) == 2
    assert all(0 <= count <= 6 for count in result.survivors)
    if result.outcome is Outcome.P1_WIN:
        assert result.survivors[0] > 0 and result.survivors[1] == 0
