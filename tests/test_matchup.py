import math

import pytest

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
from battle_sim.utils import Ability, ExtraStatus, Hazards, Item, Nature, Outcome, Status, Target, Type
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


def test_a_guaranteed_fail_scores_as_a_no_op_not_a_fantasy_ko():
    """Dream Eater into an awake target always fails — it must not outscore a move that actually
    connects just because its raw power looks like a lethal hit."""
    dream_eater = get_move("Dream Eater")
    charge_beam = get_move("Charge Beam")
    attacker = _mk("A", moves=MoveSet(dream_eater, charge_beam, SPLASH, SPLASH))
    defender = _mk("B")  # awake by default
    state = _battle([attacker], [defender])

    scored = {
        action.move: score
        for score, action in MatchupPlayer().score_actions(state, 0, [_use(MoveSlot.FIRST), _use(MoveSlot.SECOND)])
    }
    assert scored[MoveSlot.FIRST] < scored[MoveSlot.SECOND]  # Dream Eater must lose to Charge Beam, a real attack

    chosen = MatchupPlayer().choose_action(state, 0, [_use(MoveSlot.FIRST), _use(MoveSlot.SECOND)])
    assert chosen.move is MoveSlot.SECOND


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


def test_a_move_about_to_be_reflected_scores_as_a_no_op() -> None:
    """Will-O-Wisp into Magic Bounce comes straight back and burns its own user.

    Without this the effect features price it as though the burn stuck to the target, so a Pokemon
    with nothing better to do throws status into the mirror turn after turn — while already burned
    by the last one. Same treatment as a guaranteed fail: below anything that actually connects.
    """
    will_o_wisp = get_move("Will-O-Wisp")
    charge_beam = get_move("Charge Beam")
    attacker = _mk("A", moves=MoveSet(will_o_wisp, charge_beam, SPLASH, SPLASH))
    mirror = _mk("B")
    mirror.ability = Ability.MAGIC_BOUNCE
    state = _battle([attacker], [mirror])

    scored = {
        action.move: score
        for score, action in MatchupPlayer().score_actions(state, 0, [_use(MoveSlot.FIRST), _use(MoveSlot.SECOND)])
    }
    assert scored[MoveSlot.FIRST] < scored[MoveSlot.SECOND]
    assert (
        MatchupPlayer().choose_action(state, 0, [_use(MoveSlot.FIRST), _use(MoveSlot.SECOND)]).move is MoveSlot.SECOND
    )


def test_the_same_move_is_worth_more_against_anything_else() -> None:
    """The control: the burn is only worthless because it is coming back, not in general."""
    will_o_wisp = get_move("Will-O-Wisp")
    charge_beam = get_move("Charge Beam")

    def burn_score(mirrored: bool) -> float:
        attacker = _mk("A", moves=MoveSet(will_o_wisp, charge_beam, SPLASH, SPLASH))
        defender = _mk("B")
        if mirrored:
            defender.ability = Ability.MAGIC_BOUNCE
        state = _battle([attacker], [defender])
        scored = MatchupPlayer().score_actions(state, 0, [_use(MoveSlot.FIRST), _use(MoveSlot.SECOND)])
        return next(score for score, action in scored if action.move is MoveSlot.FIRST)

    assert burn_score(mirrored=False) > burn_score(mirrored=True)


def test_hazards_are_worthless_against_a_last_pokemon() -> None:
    """Reported from a live mirror match: the AI, 3-vs-1 ahead, spent turns 21, 22 and 23 setting
    Spikes against an opponent who had nothing left to switch in, and lost the game it was winning.

    Hazards only ever charge a Pokemon on the way in, and the one already out has come in. The
    feature counted every unfainted member instead of the bench, so a lone survivor still scored.
    """
    setter = _mk("Setter", moves=MoveSet(STEALTH_ROCK, TACKLE, RECOVER, SPLASH))
    alone = _mk("Alone")
    state = _battle([setter], [alone])
    player = MatchupPlayer()

    scored = {
        action.move: score
        for score, action in player.score_actions(state, 0, [_use(MoveSlot.FIRST), _use(MoveSlot.SECOND)])
    }
    assert scored[MoveSlot.SECOND] > scored[MoveSlot.FIRST]  # hit them rather than set a toll nobody pays


def test_hazards_are_still_worth_setting_while_a_bench_remains() -> None:
    """The other half of the same rule: with somebody left to come in, the toll is real."""
    setter = _mk("Setter", moves=MoveSet(STEALTH_ROCK, TACKLE, RECOVER, SPLASH))
    state = _battle([setter], [_mk("Active"), _mk("Waiting"), _mk("AlsoWaiting")])
    player = MatchupPlayer()

    scored = {action.move: score for score, action in player.score_actions(state, 0, [_use(MoveSlot.FIRST)])}
    assert scored[MoveSlot.FIRST] > 0.0


def test_boots_holders_are_not_counted_as_hazard_targets() -> None:
    """They walk over hazards, so a bench made only of them is no reason to set any."""
    setter = _mk("Setter", moves=MoveSet(STEALTH_ROCK, TACKLE, RECOVER, SPLASH))
    booted = [_mk("Active"), _mk("Booted", item=Item.HEAVY_DUTY_BOOTS)]
    bare = [_mk("Active"), _mk("Bare")]
    player = MatchupPlayer()

    def hazard_score(bench: list[Pokemon]) -> float:
        state = _battle([setter], bench)
        return next(score for score, action in player.score_actions(state, 0, [_use(MoveSlot.FIRST)]))

    assert hazard_score(booted) < hazard_score(bare)


BRUISER = BaseStats(HP=100, ATTACK=255, DEFENCE=100, SP_ATTACK=100, SP_DEFENCE=100, SPEED=100)


def _bruiser() -> Pokemon:
    """A foe that hits hard enough for `heal_turns` to stay off its cap, so the gradient is visible."""
    return _mk("Bruiser", base_stats=BRUISER, moves=MoveSet(get_move("Hyper Beam"), TACKLE, RECOVER, SPLASH))


def _heal_turns(player: MatchupPlayer, state: BattleState) -> float:
    features, _ = player.feature_actions(state, 0, [_use(MoveSlot.FIRST, Target.SELF)])[0]
    assert features is not None
    return features.heal_turns


def test_recovery_is_worthless_when_poison_outpaces_it() -> None:
    """From a live mirror match: a badly poisoned Moltres used Roost eleven turns running while the
    toxic counter climbed past what Roost restores, dealt no damage at all, and lost a won game.

    Recovery is only worth the HP it keeps, so once the toll meets the heal it is worth nothing.
    """
    healer = _mk("Healer", moves=MoveSet(RECOVER, TACKLE, EMBER, SPLASH))
    healer.live_stats.HP = healer.stat_totals.HP // 2
    state = _battle([healer], [_mk("Foe")])
    player = MatchupPlayer()

    healthy = _heal_turns(player, state)
    assert healthy > 0.0

    healer.status = Status.TOXIC
    healer.status_turns = 8  # deep into a counter: one tick now dwarfs what Recover gives back
    assert _heal_turns(player, state) == 0.0


def test_recovery_still_counts_when_it_stays_ahead_of_the_drain() -> None:
    """A fresh burn is a tax, not a losing race, so healing through it is still worth something —
    just less than healing clean."""
    healer = _mk("Healer", moves=MoveSet(RECOVER, TACKLE, EMBER, SPLASH))
    healer.live_stats.HP = healer.stat_totals.HP // 2
    state = _battle([healer], [_bruiser()])
    player = MatchupPlayer()
    clean = _heal_turns(player, state)

    healer.status = Status.BURN
    burned = _heal_turns(player, state)
    assert 0.0 < burned < clean


def test_leftovers_can_keep_a_recovery_race_winnable() -> None:
    """The drain nets both ways: a berry ticking up offsets a tick down."""
    plain = _mk("Plain", moves=MoveSet(RECOVER, TACKLE, EMBER, SPLASH))
    fed = _mk("Fed", moves=MoveSet(RECOVER, TACKLE, EMBER, SPLASH), item=Item.LEFTOVERS)
    player = MatchupPlayer()
    scores = []
    for healer in (plain, fed):
        healer.live_stats.HP = healer.stat_totals.HP // 2
        healer.status = Status.POISON
        scores.append(_heal_turns(player, _battle([healer], [_bruiser()])))
    assert scores[1] > scores[0]


def test_a_status_that_cannot_land_is_worth_nothing() -> None:
    """Toxic at a Steel type starts a timer that never starts. It used to price as though it stuck,
    which is how a status move gets thrown at something it can never touch, turn after turn."""
    poisoner = _mk("Poisoner", moves=MoveSet(get_move("Toxic"), TACKLE, RECOVER, SPLASH))
    player = MatchupPlayer()

    def timer_value(target_types: tuple[Type, Type | None]) -> float:
        state = _battle([poisoner], [_mk("Target", types=target_types)])
        features, _ = player.feature_actions(state, 0, [_use(MoveSlot.FIRST)])[0]
        assert features is not None
        return features.timer_value

    assert timer_value((Type.NORMAL, None)) > 0.0
    assert timer_value((Type.STEEL, None)) == 0.0
    assert timer_value((Type.POISON, None)) == 0.0


# -- Denial and Leech Seed ----------------------------------------------------------
# Haze, Taunt, Encore, Disable and Leech Seed produced the same feature vector as Splash, so the
# scorer could not tell any of them from doing nothing and never played one. Worst of all where it
# matters most: `sweep_threat` teaches the search to fear a Pokemon mid-setup, and these are the
# moves that answer one.


def _denial(move_name: str, their_boost: int = 0, my_boost: int = 0, volatiles: dict | None = None) -> float:
    move = get_move(move_name)
    mine = _mk("Mine", moves=MoveSet(move, EMBER, SWORDS_DANCE, SPLASH))
    theirs = _mk("Theirs", moves=MoveSet(SWORDS_DANCE, TACKLE, RECOVER, SPLASH))
    theirs.stat_stages.ATTACK = their_boost
    mine.stat_stages.ATTACK = my_boost
    for volatile, value in (volatiles or {}).items():
        theirs.volatiles[volatile] = value
    state = _battle([mine], [theirs])
    features, _ = MatchupPlayer().feature_actions(state, 0, [_use(MoveSlot.FIRST, move.target)])[0]
    assert features is not None
    return features.setup_denial


def test_haze_is_worth_more_the_further_ahead_they_are() -> None:
    assert _denial("Haze", their_boost=0) == 0.0
    assert 0 < _denial("Haze", their_boost=2) < _denial("Haze", their_boost=6)


def test_haze_is_worthless_when_we_are_the_ones_who_set_up() -> None:
    """It clears both sides, so hazing away our own sweep to undo their +2 pays less than it costs."""
    assert _denial("Haze", their_boost=2, my_boost=2) == 0.0
    assert _denial("Haze", their_boost=0, my_boost=4) == 0.0


@pytest.mark.parametrize("move_name", ["Taunt", "Encore", "Disable"])
def test_denial_moves_are_worth_something_against_a_threat(move_name: str) -> None:
    assert _denial(move_name, their_boost=2) > 0.0


@pytest.mark.parametrize(
    ("move_name", "volatile"),
    [("Taunt", ExtraStatus.TAUNT), ("Encore", ExtraStatus.ENCORE), ("Disable", ExtraStatus.DISABLE)],
)
def test_a_second_helping_of_the_same_denial_buys_nothing(move_name: str, volatile: ExtraStatus) -> None:
    assert _denial(move_name, their_boost=2, volatiles={volatile: 3}) == 0.0


def test_leech_seed_reads_as_the_timer_it_is() -> None:
    def seed(defender_types: tuple, volatiles: dict | None = None) -> float:
        mine = _mk("Mine", moves=MoveSet(get_move("Leech Seed"), EMBER, RECOVER, SPLASH))
        theirs = _mk("Theirs", types=defender_types, moves=MoveSet(TACKLE, SPLASH, RECOVER, EMBER))
        for volatile, value in (volatiles or {}).items():
            theirs.volatiles[volatile] = value
        state = _battle([mine], [theirs])
        features, _ = MatchupPlayer().feature_actions(state, 0, [_use(MoveSlot.FIRST)])[0]
        assert features is not None
        return features.timer_value

    assert seed((Type.NORMAL, None)) > 0.0
    assert seed((Type.GRASS, None)) == 0.0  # Grass cannot be seeded
    assert seed((Type.NORMAL, None), {ExtraStatus.LEECH_SEED: 1}) == 0.0  # already seeded
