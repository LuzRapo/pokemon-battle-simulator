import random
from typing import Final

from battle_sim.calibration import sample_positions, summarise
from battle_sim.observation import SetPrior
from battle_sim.setgen import random_set

# A small pool drawn many times, so each species turns up with several sets for the prior to weigh.
SPECIES: Final = (
    "Landorus-Therian",
    "Ferrothorn",
    "Garchomp",
    "Tapu Koko",
    "Toxapex",
    "Magearna",
    "Heatran",
    "Clefable",
    "Greninja",
    "Tyranitar",
    "Rotom-Wash",
    "Zapdos",
)
_RNG = random.Random(0)
TEAMS = [tuple(random_set(species, _RNG) for species in _RNG.sample(SPECIES, 6)) for _ in range(24)]
PRIOR = SetPrior.from_teams(TEAMS)


def test_every_position_is_read_at_each_reveal_count():
    samples = sample_positions(TEAMS, PRIOR, count=5, seed=0)
    assert len(samples) == 5 * 4
    assert sorted({s.reveals for s in samples}) == [0, 1, 2, 3]


def test_belief_converges_on_truth_as_moves_are_revealed():
    """The estimator may start wide, but reveals have to narrow it — otherwise it ignores evidence."""
    rows = summarise(sample_positions(TEAMS, PRIOR, count=60, seed=1))
    assert rows[-1].posterior_error < rows[0].posterior_error


def test_sampling_is_seeded():
    first = sample_positions(TEAMS, PRIOR, count=4, seed=7)
    again = sample_positions(TEAMS, PRIOR, count=4, seed=7)
    assert [s.posterior for s in first] == [s.posterior for s in again]
