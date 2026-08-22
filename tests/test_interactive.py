import pytest

from battle_sim.interactive import AI, HUMAN, InteractiveBattle, Phase
from battle_sim.matchup import MatchupPlayer, MatchupWeights
from battle_sim.observation import SetPrior
from battle_sim.runner import run_battle
from battle_sim.search import SearchPlayer, SearchProfile
from battle_sim.teams import build_pokemon, export_to_showdown, parse_showdown_team
from tests.test_teams import USER_SAMPLE_TEAM

TEAM = parse_showdown_team(USER_SAMPLE_TEAM).specs
PRIOR = SetPrior.from_teams([TEAM])


def _drive(battle: InteractiveBattle, human: MatchupPlayer, max_steps: int = 600) -> None:
    for _ in range(max_steps):
        if battle.phase is Phase.OVER:
            return
        battle.submit(human.choose_action(battle.view(), HUMAN, battle.options()))
    raise AssertionError("battle did not finish")


def test_scripted_human_reproduces_run_battle():
    """Driving the controller with a myopic 'human' must match the runner decision for decision."""
    human, ai = MatchupPlayer(), MatchupPlayer(MatchupWeights(ko_now=2.0, hazard_value=1.2))
    reference = run_battle(
        TEAM, TEAM, MatchupPlayer(), MatchupPlayer(MatchupWeights(ko_now=2.0, hazard_value=1.2)), seed=5, prior=PRIOR
    )

    preview_of_ai = [PRIOR.preview(spec.species, spec.level) for spec in TEAM]
    order = list(human.choose_order(TEAM, preview_of_ai))
    battle = InteractiveBattle(TEAM, TEAM, ai, prior=PRIOR, seed=5, human_order=order)
    _drive(battle, human)

    survivors = tuple(sum(1 for p in side.team if not p.is_fainted()) for side in battle.state.sides)
    assert battle.state.outcome == reference.outcome
    assert survivors == reference.survivors
    assert len(battle.logs) == len(reference.logs)


def test_view_keeps_the_information_boundary():
    battle = InteractiveBattle(TEAM, TEAM, MatchupPlayer(), prior=PRIOR, seed=0, human_order=list(range(6)))
    view = battle.view()
    assert view.sides[HUMAN] is battle.state.sides[HUMAN]
    assert view.sides[AI] is not battle.state.sides[AI]
    assert all(b is not t for b, t in zip(view.sides[AI].team, battle.state.sides[AI].team, strict=True))


def test_search_boss_battles_to_completion():
    boss = SearchPlayer(profile=SearchProfile(budget=30, determinizations=2))
    battle = InteractiveBattle(TEAM, TEAM, boss, prior=PRIOR, seed=3, human_order=list(range(6)))
    _drive(battle, MatchupPlayer())
    assert battle.phase is Phase.OVER
    assert battle.state.outcome is not None


def test_submit_after_the_end_crashes():
    battle = InteractiveBattle(TEAM, TEAM, MatchupPlayer(), prior=PRIOR, seed=1, human_order=list(range(6)))
    _drive(battle, MatchupPlayer())
    with pytest.raises(ValueError, match="over"):
        battle.submit(battle.options()[0])


def test_transcript_carries_header_teams_log_and_result():
    battle = InteractiveBattle(TEAM, TEAM, MatchupPlayer(), prior=PRIOR, seed=2, human_order=list(range(6)))
    _drive(battle, MatchupPlayer())
    text = battle.transcript(header="== test header ==")
    assert text.startswith("== test header ==")
    assert "== Human team ==" in text and "== Boss team ==" in text
    assert "== Battle ==" in text and "-- turn" in text
    assert "== Result:" in text
    battle_lines = text.split("== Battle ==")[1].split("== Result:")[0].strip().splitlines()
    rendered = [line for line in battle_lines if not line.startswith("-- turn")]
    assert len(rendered) == sum(len(log) for log in battle.logs)


def test_transcript_team_sheets_show_original_items():
    """Consumed or knocked-off items must still appear on the sheets: they render from the specs."""
    battle = InteractiveBattle(TEAM, TEAM, MatchupPlayer(), prior=PRIOR, seed=2, human_order=list(range(6)))
    _drive(battle, MatchupPlayer())
    original = export_to_showdown([build_pokemon(spec) for spec in TEAM]).strip()
    assert original in battle.transcript()


def test_rejects_a_bad_lead_order():
    with pytest.raises(ValueError, match="permutation"):
        InteractiveBattle(TEAM, TEAM, MatchupPlayer(), prior=PRIOR, seed=0, human_order=[0, 0, 1, 2, 3, 4])
