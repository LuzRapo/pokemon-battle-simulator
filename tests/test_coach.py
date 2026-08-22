import pytest

from battle_sim.coach import grade_choice, grade_label
from battle_sim.engine import legal_actions
from battle_sim.maths.rng import RNG
from battle_sim.mechanics.battle import BattleState, SideState
from battle_sim.search import SearchPlayer, SearchProfile
from battle_sim.teams import build_pokemon, parse_showdown_team
from tests.test_teams import USER_SAMPLE_TEAM

TEAM = parse_showdown_team(USER_SAMPLE_TEAM).specs


def _battle_state() -> BattleState:
    return BattleState(
        sides=(
            SideState(team=[build_pokemon(spec) for spec in TEAM]),
            SideState(team=[build_pokemon(spec) for spec in TEAM]),
        ),
        rng=RNG(seed=4),
    )


def test_grade_label_buckets():
    assert grade_label(0.0) == "Best"
    assert grade_label(0.4) == "Good"
    assert grade_label(1.0) == "Inaccuracy"
    assert grade_label(5.0) == "Blunder"


def test_grading_the_annotators_best_action_costs_nothing():
    state = _battle_state()
    actions = legal_actions(state, 0)
    values = SearchPlayer(profile=SearchProfile(budget=60)).action_values(state, 0, actions)
    assert len(values) == len(actions)
    best = actions[max(range(len(values)), key=lambda i: values[i])]
    verdict = grade_choice(SearchPlayer(profile=SearchProfile(budget=60)), state, 0, actions, best)
    assert verdict.loss == 0.0
    assert verdict.label in ("Best", "Brilliant")
    assert verdict.best == best


def test_grading_a_worse_action_reports_its_exact_gap():
    state = _battle_state()
    actions = legal_actions(state, 0)
    values = SearchPlayer(profile=SearchProfile(budget=60)).action_values(state, 0, actions)
    worst = actions[min(range(len(values)), key=lambda i: values[i])]
    verdict = grade_choice(SearchPlayer(profile=SearchProfile(budget=60)), state, 0, actions, worst)
    assert verdict.loss == pytest.approx(max(values) - min(values))
    assert verdict.label == grade_label(verdict.loss)


def test_action_values_handles_the_forced_switch_branch():
    state = _battle_state()
    state.sides[0].team[state.sides[0].active[0]].apply_damage(10**6)
    actions = legal_actions(state, 0)  # only switches remain
    values = SearchPlayer(profile=SearchProfile(budget=60)).action_values(state, 0, actions)
    assert len(values) == len(actions)
    assert all(isinstance(value, float) for value in values)
