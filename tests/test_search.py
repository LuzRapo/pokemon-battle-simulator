import random
from pathlib import Path

import pytest

from battle_sim.database.loader import get_move
from battle_sim.engine import legal_actions, step
from battle_sim.evolution import load_weights
from battle_sim.matchup import MatchupPlayer, MatchupWeights
from battle_sim.maths.rng import RNG
from battle_sim.mechanics.battle import BattleState, SideState
from battle_sim.models.actions import Action, ActionType
from battle_sim.models.moves import MoveSet, MoveSlot
from battle_sim.models.pokemon import Pokemon
from battle_sim.models.stats import BaseStats, EVs, IVs
from battle_sim.observation import SetPrior
from battle_sim.players import BasicPlayer
from battle_sim.runner import run_battle
from battle_sim.search import (
    PositionWeights,
    SearchPlayer,
    SearchProfile,
    clone_for_search,
    column_equilibrium,
    evaluate_position,
    exploit_mixture,
    gated_mixture,
    solve_zero_sum,
)
from battle_sim.teams import build_pokemon, parse_showdown_team
from battle_sim.utils import Ability, Hazards, Item, Nature, Outcome, Type
from tests.test_teams import USER_SAMPLE_TEAM

TEAM = parse_showdown_team(USER_SAMPLE_TEAM).specs


def _battle_state() -> BattleState:
    return BattleState(
        sides=(
            SideState(team=[build_pokemon(spec) for spec in TEAM]),
            SideState(team=[build_pokemon(spec) for spec in TEAM]),
        ),
        rng=RNG(seed=1),
    )


def test_clone_is_independent_of_the_original():
    state = _battle_state()
    state.sides[0].hazards[Hazards.STEALTH_ROCK] = 1
    clone = clone_for_search(state, seed=5)
    actions = {0: legal_actions(clone, 0)[0], 1: legal_actions(clone, 1)[0]}
    step(clone, actions)
    assert state.turn == 0  # the original battle never moved
    assert clone.turn == 1
    hp = [p.live_stats.HP for p in state.sides[0].team] + [p.live_stats.HP for p in state.sides[1].team]
    assert all(h == p.live_stats.HP for p, h in zip(state.sides[0].team + state.sides[1].team, hp, strict=True))
    clone.sides[0].hazards[Hazards.SPIKES] = 3
    assert Hazards.SPIKES not in state.sides[0].hazards


def test_clone_carries_the_public_position():
    state = _battle_state()
    state.sides[1].team[0].apply_damage(50)
    state.sides[0].hazards[Hazards.STEALTH_ROCK] = 1
    clone = clone_for_search(state, seed=0)
    assert clone.sides[1].team[0].live_stats.HP == state.sides[1].team[0].live_stats.HP
    assert clone.sides[0].hazards == state.sides[0].hazards
    assert clone.sides[0].team[0] is not state.sides[0].team[0]


def test_evaluate_position_prefers_material_and_wins():
    state = _battle_state()
    even = evaluate_position(state, 0, PositionWeights())
    state.sides[1].team[1].apply_damage(10**6)  # one of theirs faints
    ahead = evaluate_position(state, 0, PositionWeights())
    assert ahead > even
    state.outcome = Outcome.P1_WIN
    won = evaluate_position(state, 0, PositionWeights())
    assert won > ahead + 50
    assert evaluate_position(state, 1, PositionWeights()) < -50


def test_solve_zero_sum_finds_the_dominant_strategy():
    strategy, value = solve_zero_sum([[3.0, 2.0], [1.0, 0.0]])  # row 0 dominates
    assert strategy[0] > 0.95
    assert value > 1.9


def test_solve_zero_sum_mixes_on_matching_pennies():
    strategy, value = solve_zero_sum([[1.0, -1.0], [-1.0, 1.0]])
    assert 0.3 < strategy[0] < 0.7
    assert abs(value) < 0.2


def test_gated_mixture_keeps_ties_and_zeroes_worse_rows():
    pennies = gated_mixture([[1.0, -1.0], [-1.0, 1.0]])
    assert 0.3 < pennies[0] < 0.7  # a genuine 50/50 stays mixed
    dominated = gated_mixture([[3.0, 2.0], [2.5, 1.5], [1.0, 0.0]])  # row 0 dominates
    assert dominated[0] == 1.0
    assert dominated[1] == dominated[2] == 0.0


def test_search_player_battles_deterministically():
    prior = SetPrior.from_teams([TEAM])
    profile = SearchProfile(budget=50)
    first = run_battle(TEAM, TEAM, SearchPlayer(profile=profile), BasicPlayer(), seed=11, prior=prior)
    second = run_battle(TEAM, TEAM, SearchPlayer(profile=profile), BasicPlayer(), seed=11, prior=prior)
    assert (first.outcome, first.turns, first.survivors) == (second.outcome, second.turns, second.survivors)


def test_zero_remaining_budget_falls_back_to_the_myopic_choice():
    state = _battle_state()
    actions = legal_actions(state, 0)
    searcher = SearchPlayer(profile=SearchProfile(budget=0))
    myopic = MatchupPlayer()
    assert searcher.choose_action(state, 0, actions) == myopic.choose_action(state, 0, actions)


def test_genome_prior_breaks_a_near_tie_toward_the_real_attack():
    """A full-HP-adjacent Rest and a real attack can come out near-indistinguishable to the
    search's own payoff estimate once the position looks lost regardless — genome_prior is what's
    supposed to hand that near-tie to the myopic scorer, which correctly knows one of them does
    nothing. Reproduces a real observed game (a Snorlax that spent three straight turns on a
    failing Rest against a Quiver Dance Volcarona it could no longer meaningfully out-heal) with
    the actual deployed champion genome — an untrained/default genome has no real ranking for
    genome_prior to blend in, so this needs the real weights to mean anything.
    """
    weights = load_weights(Path("champions/random-a-myopic-tuned.json"))
    snorlax = Pokemon(
        name="Snorlax",
        nickname="Snorlax",
        level=50,
        base_stats=BaseStats(HP=160, ATTACK=110, DEFENCE=65, SP_ATTACK=65, SP_DEFENCE=110, SPEED=30),
        effort_values=EVs(HP=252, DEFENCE=252, SP_DEFENCE=4),
        individual_values=IVs(),
        types=(Type.NORMAL, None),
        moves=MoveSet(get_move("Amnesia"), get_move("Snore"), get_move("Rest"), get_move("Body Slam")),
        nature=Nature.BOLD,
        ability=Ability.IMMUNITY,
        item=Item.CHESTO_BERRY,
    )
    volcarona = Pokemon(
        name="Volcarona",
        nickname="Volcarona",
        level=50,
        base_stats=BaseStats(HP=85, ATTACK=60, DEFENCE=65, SP_ATTACK=135, SP_DEFENCE=105, SPEED=100),
        effort_values=EVs(ATTACK=52, DEFENCE=145, SP_ATTACK=80, SP_DEFENCE=84, SPEED=103),
        individual_values=IVs(HP=26, ATTACK=27, DEFENCE=23, SP_ATTACK=13, SP_DEFENCE=6, SPEED=5),
        types=(Type.BUG, Type.FIRE),
        moves=MoveSet(get_move("Giga Drain"), get_move("Flamethrower"), get_move("Roost"), get_move("Quiver Dance")),
        nature=Nature.LONELY,
        ability=Ability.FLAME_BODY,
        item=Item.SITRUS_BERRY,
    )
    volcarona.stat_stages.SP_ATTACK = 3
    volcarona.stat_stages.SP_DEFENCE = 3
    volcarona.stat_stages.SPEED = 3
    volcarona.live_stats.HP = 119  # not full, but bulky enough that chip damage barely moves the estimate

    state = BattleState(sides=(SideState(team=[volcarona]), SideState(team=[snorlax])), rng=RNG(seed=0))
    actions = [
        Action(action=ActionType.USE_MOVE, target=move.target, move=slot)
        for slot, move in zip(MoveSlot, snorlax.moves.to_list(), strict=False)
    ]
    body_slam = actions[3]

    for seed in range(8):
        player = SearchPlayer(weights, profile=SearchProfile(budget=30))
        player._rng = random.Random(seed)
        assert player.choose_action(state, 1, actions) == body_slam


def test_search_player_runs_at_a_large_budget():
    prior = SetPrior.from_teams([TEAM])
    player = SearchPlayer(profile=SearchProfile(budget=700))
    result = run_battle(TEAM, TEAM, player, MatchupPlayer(), seed=3, prior=prior, max_turns=120)
    assert result.turns <= 120


def test_root_mixture_sampling_floors_dust_and_actually_mixes():
    dominant = SearchPlayer()
    assert all(dominant._sample(["a", "b"], [0.97, 0.03]) == "a" for _ in range(32))  # dust never sampled
    coin = SearchPlayer()
    draws = [coin._sample(["a", "b"], [0.5, 0.5]) for _ in range(32)]
    assert set(draws) == {"a", "b"}  # a 50/50 is played as one
    replay = SearchPlayer()
    assert draws == [replay._sample(["a", "b"], [0.5, 0.5]) for _ in range(32)]  # same seeded stream


def test_choose_order_is_a_seeded_permutation():
    order = SearchPlayer().choose_order(TEAM, TEAM)
    assert sorted(order) == list(range(6))
    assert list(order) == list(SearchPlayer().choose_order(TEAM, TEAM))


def test_seeds_lead_differently_so_a_repeat_opponent_cannot_learn_one_opening():
    """The lead is drawn from a mixture, so distinct seeds must actually reach distinct leads.

    Regression: `choose_order` reset the stream to `Random(0)` on every battle, which collapsed the
    whole mixture onto one sample. A human replaying the same trainer met the identical lead every
    single time, and could plan the whole game around it.
    """
    leads = {SearchPlayer(seed=seed).choose_order(TEAM, TEAM)[0] for seed in range(24)}
    assert len(leads) > 1
    replayed = SearchPlayer(seed=7).choose_order(TEAM, TEAM)
    assert list(replayed) == list(SearchPlayer(seed=7).choose_order(TEAM, TEAM))  # a seed still replays


def test_tie_band_mixes_near_equal_actions_only():
    band = SearchPlayer(profile=SearchProfile(budget=0, tie_band=0.10), seed=3)
    assert {band._pick(["a", "b"], [0.50, 0.45]) for _ in range(32)} == {"a", "b"}  # inside the band
    assert all(band._pick(["a", "b"], [0.80, 0.20]) == "a" for _ in range(32))  # a real preference stands
    strict = SearchPlayer(profile=SearchProfile(budget=0), seed=3)
    assert all(strict._pick(["a", "b"], [0.50, 0.45]) == "a" for _ in range(32))  # band 0.0 is argmax


def test_determinization_battles_deterministically():
    prior = SetPrior.from_teams([TEAM])
    profile = SearchProfile(budget=60, determinizations=3)
    first = run_battle(TEAM, TEAM, SearchPlayer(profile=profile), BasicPlayer(), seed=4, prior=prior)
    second = run_battle(TEAM, TEAM, SearchPlayer(profile=profile), BasicPlayer(), seed=4, prior=prior)
    assert (first.outcome, first.turns, first.survivors) == (second.outcome, second.turns, second.survivors)


def test_exploiting_a_predictable_opponent_beats_answering_their_equilibrium():
    """Row 1 is the equilibrium-safe answer; row 0 punishes a column player who always plays left."""
    matrix = [[4.0, -4.0], [1.0, 1.0]]
    assert gated_mixture(matrix)[1] == 1.0
    predictable = exploit_mixture(matrix, predicted=[1.0, 0.0], exploit_p=1.0)
    assert predictable[0] == 1.0


def test_exploit_p_of_zero_answers_the_equilibrium_mixture():
    matrix = [[4.0, -4.0], [1.0, 1.0]]
    assert exploit_mixture(matrix, predicted=[1.0, 0.0], exploit_p=0.0) == [0.0, 1.0]


def test_column_equilibrium_is_a_distribution_over_their_actions():
    column = column_equilibrium([[1.0, -1.0], [-1.0, 1.0]])
    assert len(column) == 2
    assert sum(column) == pytest.approx(1.0)
    assert 0.3 < column[0] < 0.7


def test_the_predicted_column_is_a_distribution_shaped_by_the_opponent_model():
    """What the opponent model buys: a read on which of their replies is actually coming."""
    state = _battle_state()
    actions = legal_actions(state, 1)[:4]
    attacking = SearchPlayer(opponent_model=MatchupWeights(hko_progress=10.0))._predicted_column(state, 0, actions)
    averse = SearchPlayer(opponent_model=MatchupWeights(hko_progress=-10.0))._predicted_column(state, 0, actions)
    assert sum(attacking) == pytest.approx(1.0)
    assert all(weight > 0 for weight in attacking)  # a softmax never rules a reply out entirely
    hardest_hit = max(range(len(actions)), key=lambda i: attacking[i])
    assert averse[hardest_hit] < attacking[hardest_hit]
