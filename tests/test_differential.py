"""The harness that will judge a second engine, judged first.

Every failure this reports will be read as "the new engine is wrong". So the expensive mistake is a
harness that cannot reproduce *this* engine — every accusation it makes would be noise, and the
noise would be indistinguishable from the signal. These tests are about that: the tape replays, the
digest notices, and the comparator points at the first thing that actually parted.
"""

import copy
import random
from pathlib import Path

import pytest

from battle_sim.differential import (
    Divergence,
    Scenario,
    TapeExhausted,
    TapeRNG,
    build_state,
    compare,
    find_action,
    name_action,
    record,
    self_check,
    trace,
    write,
)
from battle_sim.engine import legal_actions
from battle_sim.models.spec import PokemonSpec
from battle_sim.setgen import random_set

TEAM_SIZE = 3


def _team(seed: int, species: tuple[str, ...]) -> list[PokemonSpec]:
    rng = random.Random(seed)
    return [random_set(name, rng) for name in species]


def _teams(seed: int = 0) -> tuple[list[PokemonSpec], list[PokemonSpec]]:
    return (
        _team(seed, ("Garchomp", "Blissey", "Heatran")),
        _team(seed + 1, ("Tentacruel", "Dragonite", "Ferrothorn")),
    )


def _random_chooser(seed: int):  # type: ignore[no-untyped-def]
    """Plays legal nonsense, which is what a differential test wants: it visits the odd corners a
    good policy carefully avoids, and those are where two engines disagree."""
    rng = random.Random(seed)

    def choose(state, side_index):  # type: ignore[no-untyped-def]
        return rng.choice(legal_actions(state, side_index))

    return choose


# -- the tape ----------------------------------------------------------------------------------


def test_a_replayed_tape_returns_what_was_recorded() -> None:
    recorder = TapeRNG(seed=7)
    rolled = [recorder.random_probability(), recorder.random_integer(0, 100), recorder.random_probability()]

    replay = TapeRNG.replaying(recorder.tape)

    assert [replay.random_probability(), replay.random_integer(0, 100), replay.random_probability()] == rolled


def test_a_replay_that_runs_past_the_recording_says_so() -> None:
    """Not a shortage of randomness — a divergence. One engine reached a roll the other never made."""
    replay = TapeRNG.replaying([0.5])
    replay.random_probability()

    with pytest.raises(TapeExhausted, match="1 draws"):
        replay.random_probability()


def test_the_tape_makes_a_battle_reproducible_without_the_prng() -> None:
    """The whole point: a second engine never has to reimplement CPython's Mersenne Twister."""
    scenario, expected = record(_teams(), _random_chooser(1), seed=3)

    assert scenario.tape, "a battle with no randomness would prove nothing"
    assert self_check(scenario, expected) is None


# -- replaying this engine against itself ---------------------------------------------------------


@pytest.mark.parametrize("seed", range(6))
def test_this_engine_reproduces_its_own_recordings(seed: int) -> None:
    """If this fails, the harness is unusable: it cannot tell a real divergence from its own noise."""
    scenario, expected = record(_teams(seed), _random_chooser(seed), seed=seed)

    assert self_check(scenario, expected) is None


def test_a_recording_survives_a_round_trip_through_json() -> None:
    """The second engine reads these off disk, so the file is the contract, not the object."""
    scenario, expected = record(_teams(2), _random_chooser(2), seed=2)

    reloaded = Scenario.from_json(scenario.to_json())

    assert compare(expected, trace(reloaded)) is None


def test_scenarios_and_traces_are_written_as_a_pair(tmp_path: Path) -> None:
    scenario, expected = record(_teams(4), _random_chooser(4), seed=4)

    write(scenario, expected, tmp_path, "battle-0001")

    assert (tmp_path / "battle-0001.scenario.json").exists()
    assert (tmp_path / "battle-0001.trace.json").exists()
    assert compare(expected, trace(Scenario.from_json((tmp_path / "battle-0001.scenario.json").read_text()))) is None


# -- actions are named, not numbered --------------------------------------------------------------


def test_an_action_is_named_by_what_it_is() -> None:
    scenario, _ = record(_teams(5), _random_chooser(5), seed=5)

    assert all(
        first.startswith(("move:", "switch:", "zmove:")) and second.startswith(("move:", "switch:", "zmove:"))
        for first, second in scenario.actions
    )


def test_an_action_the_engine_will_not_offer_is_a_loud_failure() -> None:
    """The bug this exists to catch: two engines disagreeing about what is legal. By index that is
    silent — a different move gets picked and the battle quietly diverges."""
    scenario, _ = record(_teams(6), _random_chooser(6), seed=6)
    state, _ = build_state(scenario)

    with pytest.raises(LookupError, match="not legal"):
        find_action("move:Fissure Of The Damned", state, 0)


def test_naming_round_trips_through_finding() -> None:
    scenario, _ = record(_teams(7), _random_chooser(7), seed=7)
    state, _ = build_state(scenario)

    for action in legal_actions(state, 0):
        named = name_action(action, state, 0)
        assert name_action(find_action(named, state, 0), state, 0) == named


# -- the comparator ------------------------------------------------------------------------------


def test_identical_traces_do_not_diverge() -> None:
    _, expected = record(_teams(8), _random_chooser(8), seed=8)

    assert compare(expected, copy.deepcopy(list(expected))) is None


def test_a_changed_event_is_caught_and_located() -> None:
    _, expected = record(_teams(9), _random_chooser(9), seed=9)
    tampered = copy.deepcopy(list(expected))
    tampered[0] = dict(tampered[0], events=[{"type": "NotAThingThatHappened"}])

    found = compare(expected, tampered)

    assert isinstance(found, Divergence) and found.turn == 1 and found.where.startswith("event")


def test_a_changed_state_is_caught_even_when_the_log_agrees() -> None:
    """The reason the digest exists. A volatile's counter, a spent PP and an eaten item all change
    what the next turn does while narrating nothing at all."""
    _, expected = record(_teams(10), _random_chooser(10), seed=10)
    tampered = copy.deepcopy(list(expected))
    state = tampered[0]["state"]
    state["sides"][0]["team"][0] = dict(state["sides"][0]["team"][0], hp=99999)

    found = compare(expected, tampered)

    assert isinstance(found, Divergence) and found.turn == 1
    assert "hp" in found.where and "sides[0]" in found.where, found.where


def test_the_first_divergence_is_the_one_reported() -> None:
    """After two engines part company every later turn differs as a consequence. Only the first is
    a cause, and only the first is worth a human's time."""
    _, expected = record(_teams(11), _random_chooser(11), seed=11)
    assert len(expected) >= 3, "need a few turns to have a second divergence to ignore"
    tampered = copy.deepcopy(list(expected))
    tampered[2] = dict(tampered[2], actions=["move:Wrong", "move:Wrong"])
    tampered[1] = dict(tampered[1], actions=["move:AlsoWrong", "move:AlsoWrong"])

    found = compare(expected, tampered)

    assert isinstance(found, Divergence) and found.turn == 2


def test_a_shorter_battle_is_a_divergence() -> None:
    _, expected = record(_teams(12), _random_chooser(12), seed=12)

    found = compare(expected, list(expected)[:-1])

    assert isinstance(found, Divergence) and "length" in found.where
