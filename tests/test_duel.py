import random
from pathlib import Path

from battle_sim.duel import Side, duel, pair_margin, sample_pairings
from battle_sim.evolution import Progress, load_teams
from battle_sim.matchup import MatchupWeights
from battle_sim.observation import SetPrior

TEAMS = load_teams(Path("sample_teams"))[:8]
PRIOR = SetPrior.from_teams(TEAMS)


def _side(belief: int = 12, budget: int = 0) -> Side:
    return Side(weights=MatchupWeights(), budget=budget, belief=belief)


def test_identical_sides_draw_exactly():
    """The whole point of same-seed side pairs: with nothing to separate them, nothing separates them."""
    pairings = sample_pairings(TEAMS, 4, random.Random(0))
    verdict = duel(_side(), _side(), pairings, PRIOR, workers=1, progress=Progress(total=len(pairings)))
    assert verdict.margin == 0.5
    assert verdict.ci == 0.0


def test_a_margin_is_symmetric_between_the_duellists():
    pairing = sample_pairings(TEAMS, 1, random.Random(1))[0]
    forward = pair_margin(_side(belief=12), _side(belief=1), pairing, PRIOR)
    reverse = pair_margin(_side(belief=1), _side(belief=12), pairing, PRIOR)
    assert forward + reverse == 1.0


def test_a_pairing_replays_identically():
    pairing = sample_pairings(TEAMS, 1, random.Random(2))[0]
    first = pair_margin(_side(), _side(belief=1), pairing, PRIOR)
    again = pair_margin(_side(), _side(belief=1), pairing, PRIOR)
    assert first == again


def test_sigma_reports_distance_from_break_even():
    pairings = sample_pairings(TEAMS, 6, random.Random(3))
    verdict = duel(_side(), _side(belief=1), pairings, PRIOR, workers=1, progress=Progress(total=len(pairings)))
    assert verdict.pairs == 6
    assert 0.0 <= verdict.margin <= 1.0
