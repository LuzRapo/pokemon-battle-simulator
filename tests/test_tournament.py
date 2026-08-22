import random

import pytest

from battle_sim.matchup import MatchupPlayer, MatchupWeights
from battle_sim.observation import SetPrior
from battle_sim.search import SearchProfile
from battle_sim.teams import parse_showdown_team
from battle_sim.tournament import build_players, duel_margin, sample_pairs
from tests.test_teams import USER_SAMPLE_TEAM

TEAM = parse_showdown_team(USER_SAMPLE_TEAM).specs
PRIOR = SetPrior.from_teams([TEAM])


def test_build_players_pairs_each_genome_with_a_searching_twin():
    players = build_players([("stock", MatchupWeights())], SearchProfile(budget=9))
    assert [label for label, _ in players] == ["Basic", "stock", "stock+S"]


def test_sample_pairs_rejects_odd_counts():
    with pytest.raises(ValueError, match="even"):
        sample_pairs([TEAM], count=5, rng=random.Random(0))


def test_identical_myopic_players_duel_to_exactly_half():
    spec = sample_pairs([TEAM], count=2, rng=random.Random(0))[0]
    assert duel_margin(MatchupPlayer(), MatchupPlayer(), spec, PRIOR) == 0.5


def test_duel_margin_is_deterministic():
    spec = sample_pairs([TEAM], count=2, rng=random.Random(1))[0]
    first = duel_margin(MatchupPlayer(), MatchupPlayer(MatchupWeights(ko_now=0.0)), spec, PRIOR)
    second = duel_margin(MatchupPlayer(), MatchupPlayer(MatchupWeights(ko_now=0.0)), spec, PRIOR)
    assert first == second
    assert 0.0 <= first <= 1.0
