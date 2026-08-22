import random
from pathlib import Path

import pytest

from battle_sim.database.loader import get_species
from battle_sim.matchup import MatchupPlayer, MatchupWeights
from battle_sim.players import BasicPlayer
from battle_sim.runner import Player
from battle_sim.team_search import (
    TeamSearchConfig,
    crossover_teams,
    draft_team,
    evolve_teams,
    instantiate,
    load_saved_team,
    mutate_team,
    sample_slate,
    save_team,
    set_universe,
)
from battle_sim.teams import parse_showdown_team
from tests.test_teams import USER_SAMPLE_TEAM

TEAM = parse_showdown_team(USER_SAMPLE_TEAM).specs


def _species_of(team) -> set[str]:
    return {get_species(spec.species).name for spec in team}


def test_set_universe_dedupes_across_teams():
    universe = set_universe([TEAM, TEAM])
    assert len(universe) == 6
    assert all(spec.nickname is None for spec in universe)


def test_draft_team_respects_species_clause():
    universe = set_universe([TEAM])
    team = draft_team(universe, random.Random(0))
    assert len(team) == 6
    assert len(_species_of(team)) == 6


def test_draft_team_crashes_without_six_species():
    universe = set_universe([TEAM])[:5]
    with pytest.raises(ValueError, match="6 distinct species"):
        draft_team(universe, random.Random(0))


def test_operators_preserve_the_species_clause():
    universe = set_universe([TEAM])
    rng = random.Random(1)
    a, b = draft_team(universe, rng), draft_team(universe, rng)
    child = mutate_team(crossover_teams(a, b, rng), universe, rng)
    assert len(_species_of(child)) == 6


def test_slate_instantiates_same_seed_side_pairs():
    slate = sample_slate([TEAM], count=4, rng=random.Random(0), opponent_count=2)
    matchups = instantiate(TEAM, slate)
    assert len(matchups) == 4
    assert [m.candidate_is_p1 for m in matchups] == [True, False, True, False]
    assert matchups[0].seed == matchups[1].seed
    assert matchups[0].opponent_index == matchups[1].opponent_index
    assert all(m.candidate_team == TEAM for m in matchups)


def test_slate_rejects_odd_counts():
    with pytest.raises(ValueError, match="even"):
        sample_slate([TEAM], count=3, rng=random.Random(0), opponent_count=1)


def test_evolve_teams_is_deterministic_and_bounded():
    config = TeamSearchConfig(population=4, generations=2, matchups_per_eval=2, elites=1, tournament=2, seed=5)
    opponents: list[Player] = [BasicPlayer(), MatchupPlayer()]
    history_a = list(evolve_teams(MatchupWeights(), [TEAM], config, opponents))
    history_b = list(evolve_teams(MatchupWeights(), [TEAM], config, opponents))
    assert history_a == history_b
    assert len(history_a) == 2
    assert all(0.0 <= stats.best_fitness <= 1.0 for stats in history_a)
    assert all(stats.mean_fitness <= stats.best_fitness for stats in history_a)


def test_team_roundtrips_through_json(tmp_path: Path):
    universe = set_universe([TEAM])
    team = draft_team(universe, random.Random(2))
    path = tmp_path / "team.json"
    save_team(team, path)
    assert load_saved_team(path) == team
