"""The species rating run and its fit — mostly the log, which is the thing a 12-hour run cannot lose.

Battles are not played here: `play_pairing` is the only part that needs the engine and it is covered
by a single smoke test. Everything else is about the sampling being even and deterministic, and the
log surviving being resumed, truncated, or written by several workers at once.
"""

import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from battle_sim.species_fit import fit, read_log
from battle_sim.species_rating import (
    TEAM_SIZE,
    Pairing,
    completed_pairings,
    deal_pairings,
    play_pairing,
)

POOL = [f"Mon{i}" for i in range(40)]


def _pairings(count: int, seed: int = 0, start: int = 0) -> list[Pairing]:
    return list(deal_pairings(POOL, count, random.Random(seed), start=start))


# -- sampling ------------------------------------------------------------------------


def test_a_pairing_is_two_full_teams() -> None:
    pairing = _pairings(1)[0]
    assert len(pairing.team_a) == TEAM_SIZE
    assert len(pairing.team_b) == TEAM_SIZE


def test_no_team_ever_repeats_a_species() -> None:
    """The engine refuses a side whose nicknames collide, so a repeat is a battle lost. A team can
    span a bag refill, which is exactly when one would happen."""
    for pairing in _pairings(500):
        assert len(set(pairing.team_a)) == TEAM_SIZE
        assert len(set(pairing.team_b)) == TEAM_SIZE


def test_dealing_keeps_appearances_even() -> None:
    """The whole reason for a bag: independent draws would leave some species barely seen, and a
    species barely seen is a rating nobody can trust."""
    counts: dict[str, int] = dict.fromkeys(POOL, 0)
    for pairing in _pairings(300):
        for name in (*pairing.team_a, *pairing.team_b):
            counts[name] += 1
    spread = max(counts.values()) - min(counts.values())
    assert spread <= 2  # a bag can only ever be one deal out of step


def test_the_sequence_is_reproducible() -> None:
    assert _pairings(20) == _pairings(20)


def test_resuming_continues_the_same_sequence() -> None:
    """A resumed run must not re-roll: pairing 12 has to be the same pairing it would have been."""
    whole = _pairings(20)
    resumed = _pairings(20, start=12)
    assert resumed == whole[12:]


def test_seeds_differ_between_pairings() -> None:
    seeds = {pairing.seed for pairing in _pairings(50)}
    assert len(seeds) > 45  # not a constant, and not obviously colliding


# -- resume accounting ---------------------------------------------------------------


def _row(index: int, margin: float = 0.5) -> dict[str, Any]:
    return {"index": index, "margin": margin, "a": POOL[:TEAM_SIZE], "b": POOL[TEAM_SIZE : 2 * TEAM_SIZE]}


def _write(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def test_a_missing_log_starts_from_the_beginning(tmp_path: Path) -> None:
    assert completed_pairings(tmp_path / "nothing.jsonl") == 0


def test_resume_counts_one_past_the_highest_pairing(tmp_path: Path) -> None:
    log = tmp_path / "battles.jsonl"
    _write(log, [_row(0), _row(0), _row(1)])
    assert completed_pairings(log) == 2


def test_a_truncated_final_line_does_not_block_a_resume(tmp_path: Path) -> None:
    """A run killed mid-write leaves half a line. Refusing to resume over it would mean the log
    costing us the very data it exists to protect."""
    log = tmp_path / "battles.jsonl"
    _write(log, [_row(0), _row(1)])
    with log.open("a") as handle:
        handle.write('{"index": 2, "marg')

    assert completed_pairings(log) == 2
    assert len(read_log(log)) == 2  # and the fit simply ignores the fragment


def test_a_log_of_only_failures_still_resumes(tmp_path: Path) -> None:
    log = tmp_path / "battles.jsonl"
    _write(log, [{"index": 0, "error": "boom"}, {"index": 1, "error": "boom"}])
    assert completed_pairings(log) == 2
    assert read_log(log) == []


# -- the fit -------------------------------------------------------------------------


def _synthetic(strengths: dict[str, float], battles: int, seed: int = 0) -> list[dict[str, Any]]:
    """Battles whose margins come from known strengths, so the fit has a right answer to find."""
    rng = random.Random(seed)
    names = list(strengths)
    rows = []
    for _ in range(battles):
        drawn = rng.sample(names, 2 * TEAM_SIZE)
        a, b = drawn[:TEAM_SIZE], drawn[TEAM_SIZE:]
        edge = sum(strengths[n] for n in a) - sum(strengths[n] for n in b)
        margin = 0.5 + 0.05 * edge + rng.gauss(0, 0.02)
        rows.append({"a": a, "b": b, "margin": max(0.0, min(1.0, margin))})
    return rows


def test_the_fit_recovers_strengths_it_was_given() -> None:
    strengths = {f"Mon{i}": (i - 20) / 10 for i in range(40)}
    result = fit(_synthetic(strengths, 4000))
    truth = np.array([strengths[name] for name in result.names])
    assert np.corrcoef(result.coefficients, truth)[0, 1] > 0.95


def test_the_fit_ranks_a_strong_species_above_a_weak_one() -> None:
    strengths = {f"Mon{i}": 0.0 for i in range(40)} | {"Mon0": 2.0, "Mon39": -2.0}
    ratings = fit(_synthetic(strengths, 3000)).elo()
    assert ratings["Mon0"] > ratings["Mon39"]


def test_ratings_come_back_on_an_elo_like_scale() -> None:
    strengths = dict.fromkeys(POOL, 0.0)
    ratings = fit(_synthetic(strengths, 500)).elo()
    assert all(1000 < value < 2000 for value in ratings.values())


def test_every_species_in_the_log_is_rated_and_counted() -> None:
    result = fit(_synthetic(dict.fromkeys(POOL, 0.0), 400))
    assert set(result.names) == set(POOL)
    assert sum(result.appearances.values()) == 400 * 2 * TEAM_SIZE


def test_a_species_seen_once_is_pulled_toward_the_field_not_off_the_scale() -> None:
    """What the ridge penalty is for: one lucky battle must not mint a 3000-rated Pokemon."""
    rows = _synthetic(dict.fromkeys(POOL, 0.0), 300)
    rows.append({"a": ["Newcomer", *POOL[:5]], "b": POOL[5:11], "margin": 1.0})
    ratings = fit(rows).elo()
    assert abs(ratings["Newcomer"] - 1500) < 400


# -- the one part that needs the engine ----------------------------------------------


def test_playing_a_pairing_produces_two_logged_battles() -> None:
    pairing = Pairing(
        index=0,
        seed=7,
        team_a=("Pikachu", "Bulbasaur", "Squirtle", "Charmander", "Rattata", "Zubat"),
        team_b=("Machop", "Geodude", "Gastly", "Onix", "Krabby", "Voltorb"),
    )
    records = play_pairing((pairing, 1))

    assert len(records) == 2
    assert {record["side"] for record in records} == {0, 1}
    for record in records:
        assert 0.0 <= record["margin"] <= 1.0
        assert record["turns"] > 0
        json.dumps(record)  # every record has to survive the round trip to the log
