from battle_sim.engine import legal_actions, step
from battle_sim.matchup import MatchupPlayer, MatchupWeights
from battle_sim.maths.rng import RNG
from battle_sim.mechanics.battle import BattleState, SideState
from battle_sim.observation import SetPrior
from battle_sim.players import BasicPlayer
from battle_sim.runner import run_battle
from battle_sim.search import (
    SearchPlayer,
    SearchProfile,
    clone_for_search,
    evaluate_position,
    gated_mixture,
    solve_zero_sum,
)
from battle_sim.teams import build_pokemon, parse_showdown_team
from battle_sim.utils import Hazards, Outcome
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
    even = evaluate_position(state, 0, MatchupWeights())
    state.sides[1].team[1].apply_damage(10**6)  # one of theirs faints
    ahead = evaluate_position(state, 0, MatchupWeights())
    assert ahead > even
    state.outcome = Outcome.P1_WIN
    won = evaluate_position(state, 0, MatchupWeights())
    assert won > ahead + 50
    assert evaluate_position(state, 1, MatchupWeights()) < -50


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


def test_determinization_battles_deterministically():
    prior = SetPrior.from_teams([TEAM])
    profile = SearchProfile(budget=60, determinizations=3)
    first = run_battle(TEAM, TEAM, SearchPlayer(profile=profile), BasicPlayer(), seed=4, prior=prior)
    second = run_battle(TEAM, TEAM, SearchPlayer(profile=profile), BasicPlayer(), seed=4, prior=prior)
    assert (first.outcome, first.turns, first.survivors) == (second.outcome, second.turns, second.survivors)
