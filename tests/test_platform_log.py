from pathlib import Path

from battle_sim.interactive import HUMAN, InteractiveBattle, Phase
from battle_sim.matchup import MatchupPlayer
from battle_sim.observation import SetPrior
from battle_sim.platform_log import Origin, human_teams, log_battle, log_team, read_teams
from battle_sim.teams import parse_showdown_team
from battle_sim.utils import Outcome
from tests.test_teams import USER_SAMPLE_TEAM

TEAM = parse_showdown_team(USER_SAMPLE_TEAM).specs
PRIOR = SetPrior.from_teams([TEAM])


def test_a_logged_team_round_trips(tmp_path: Path):
    log = tmp_path / "teams.jsonl"
    log_team(log, "sam", Origin.HUMAN, tuple(TEAM))
    restored = read_teams(log)
    assert len(restored) == 1
    assert [spec.species for spec in restored[0]] == [spec.species for spec in TEAM]
    assert [spec.moves for spec in restored[0]] == [spec.moves for spec in TEAM]


def test_human_and_generated_teams_stay_separable(tmp_path: Path):
    """The two need different priors: a person picks moves they rate, the generator samples them."""
    log = tmp_path / "teams.jsonl"
    log_team(log, "sam", Origin.HUMAN, tuple(TEAM))
    log_team(log, "boss", Origin.GENERATED, tuple(TEAM))
    assert len(read_teams(log)) == 2
    assert len(human_teams(log)) == 1


def test_reading_a_log_that_does_not_exist_yet_is_empty(tmp_path: Path):
    assert read_teams(tmp_path / "nothing.jsonl") == []


def test_battles_and_teams_share_one_log(tmp_path: Path):
    log = tmp_path / "platform.jsonl"
    log_team(log, "sam", Origin.HUMAN, tuple(TEAM))
    log_battle(log, "sam", "boss", seed=3, outcome=Outcome.P1_WIN, turns=17)
    assert len(log.read_text().splitlines()) == 2
    assert len(read_teams(log)) == 1  # battle lines are skipped, not mistaken for teams


def test_a_played_battle_records_both_teams_and_its_result(tmp_path: Path):
    log = tmp_path / "platform.jsonl"
    battle = InteractiveBattle(
        TEAM, TEAM, MatchupPlayer(), prior=PRIOR, seed=1, human_order=list(range(6)), record=log, trainer="sam"
    )
    human = MatchupPlayer()
    while battle.phase is not Phase.OVER:
        battle.submit(human.choose_action(battle.view(), HUMAN, battle.options()))
    assert len(human_teams(log)) == 1  # the person's team, tagged apart from the generated boss
    assert len(read_teams(log)) == 2
    assert '"kind": "battle"' in log.read_text()


def test_a_battle_never_started_still_records_the_team(tmp_path: Path):
    log = tmp_path / "platform.jsonl"
    InteractiveBattle(TEAM, TEAM, MatchupPlayer(), prior=PRIOR, seed=1, human_order=list(range(6)), record=log)
    assert len(read_teams(log)) == 2


def test_logging_is_off_unless_a_record_is_asked_for(tmp_path: Path):
    InteractiveBattle(TEAM, TEAM, MatchupPlayer(), prior=PRIOR, seed=1, human_order=list(range(6)))
    assert list(tmp_path.iterdir()) == []
