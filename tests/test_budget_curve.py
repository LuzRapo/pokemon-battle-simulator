import random

import pytest

from battle_sim.budget_curve import SidePair, pair_margins, summarize
from battle_sim.evolution import sample_matchups
from battle_sim.matchup import MatchupWeights
from battle_sim.observation import SetPrior
from battle_sim.search import SearchProfile
from battle_sim.teams import parse_showdown_team
from tests.test_teams import USER_SAMPLE_TEAM

TEAM = parse_showdown_team(USER_SAMPLE_TEAM).specs
PRIOR = SetPrior.from_teams([TEAM])


def _pairs(count: int) -> list[SidePair]:
    matchups = sample_matchups([TEAM], count=count, rng=random.Random(0))
    return [matchups[i : i + 2] for i in range(0, len(matchups), 2)]


def test_summarize_mean_and_interval():
    margin, ci = summarize([0.4, 0.6])
    assert margin == pytest.approx(0.5)
    assert ci == pytest.approx(1.96 * 0.1414213562 / 1.4142135624, rel=1e-6)


def test_myopic_cell_against_myopic_self_cancels_exactly():
    """With no search profile both sides are the same myopic player, so every side pair scores exactly 0.5."""
    margins = pair_margins(MatchupWeights(), search=None, pairs=_pairs(4), prior=PRIOR, workers=1, tick=lambda: None)
    assert margins == [0.5, 0.5]
    assert summarize(margins) == (0.5, 0.0)


def test_search_cell_is_deterministic_and_ticks_per_pair():
    pairs = _pairs(2)
    search = SearchProfile(budget=50)
    ticks = 0

    def tick() -> None:
        nonlocal ticks
        ticks += 1

    first = pair_margins(MatchupWeights(), search=search, pairs=pairs, prior=PRIOR, workers=1, tick=tick)
    second = pair_margins(MatchupWeights(), search=search, pairs=pairs, prior=PRIOR, workers=1, tick=tick)
    assert first == second
    assert ticks == 2
    assert all(0.0 <= margin <= 1.0 for margin in first)
