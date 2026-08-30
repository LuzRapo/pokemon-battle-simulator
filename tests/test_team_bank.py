import random
from pathlib import Path

from battle_sim.database.loader import get_species
from battle_sim.evolution import load_teams
from battle_sim.observation import SetPrior
from battle_sim.team_bank import draft_random_team, write_team_bank


def test_draft_random_team_has_six_distinct_legal_species():
    team = draft_random_team(random.Random(0))
    assert len(team) == 6
    assert len(set(team)) == 6
    for species in team:
        get_species(species)  # raises KeyError if not a real, loadable species


def test_draft_random_team_is_seeded():
    assert draft_random_team(random.Random(0)) == draft_random_team(random.Random(0))


def test_written_bank_round_trips_through_load_teams(tmp_path: Path):
    write_team_bank(tmp_path, count=10, rng=random.Random(0))
    teams = load_teams(tmp_path)
    assert len(teams) == 10
    assert all(len(team) == 6 for team in teams)


def test_written_bank_builds_a_set_prior_covering_many_species(tmp_path: Path):
    write_team_bank(tmp_path, count=200, rng=random.Random(0))
    teams = load_teams(tmp_path)
    prior = SetPrior.from_teams(teams)
    species_seen = {get_species(spec.species).name for team in teams for spec in team}
    assert len(species_seen) > 100  # 200 teams * 6 slots, drawn from 950 species: broad coverage
    for species in list(species_seen)[:20]:
        assert prior.preview(species, level=50).species
