"""The Rust engine, checked against this one.

Skipped entirely when the crate has not been built, so the Python suite stays runnable on a machine
with no Rust toolchain. When it *is* built, these are not optional: a disagreement here is the
second engine playing a different game, which is the only way this project can fail quietly.

Stats come first because everything else is downstream of them. A single stat off by one changes a
damage roll, which changes whether something survives, which changes the battle — and traced back
from the far end it would look like a mysterious divergence on turn forty rather than a truncation
in the wrong place.
"""

import json
import subprocess
from pathlib import Path

import pytest

from battle_sim.database.loader import get_all_species
from battle_sim.maths.stats import calculate_total_hp, calculate_total_stat
from battle_sim.utils import Nature, Stats

RUST = Path(__file__).resolve().parent.parent / "rust"
BINARY = RUST / "target" / "release" / "statcheck"
DATA = RUST / "data"

# The three cases `statcheck` computes, and why: a neutral nature with perfect IVs, one that raises
# and lowers, and an EV spread whose values are not multiples of four — the formula truncates three
# separate times and this is where a misplaced floor shows up.
CASES = (
    ("HARDY", 50, dict.fromkeys(Stats, 31), dict.fromkeys(Stats, 0)),
    (
        "JOLLY",
        50,
        dict.fromkeys(Stats, 31),
        {Stats.HP: 4, Stats.ATTACK: 252, Stats.DEFENCE: 0, Stats.SP_ATTACK: 0, Stats.SP_DEFENCE: 0, Stats.SPEED: 252},
    ),
    (
        "MODEST",
        100,
        {Stats.HP: 13, Stats.ATTACK: 7, Stats.DEFENCE: 29, Stats.SP_ATTACK: 30, Stats.SP_DEFENCE: 3, Stats.SPEED: 19},
        {Stats.HP: 74, Stats.ATTACK: 11, Stats.DEFENCE: 6, Stats.SP_ATTACK: 251, Stats.SP_DEFENCE: 85, Stats.SPEED: 83},
    ),
)
_ORDER = (Stats.HP, Stats.ATTACK, Stats.DEFENCE, Stats.SP_ATTACK, Stats.SP_DEFENCE, Stats.SPEED)

needs_rust = pytest.mark.skipif(
    not BINARY.exists(), reason="the Rust crate is not built; run `cargo build --release` in rust/"
)


@pytest.fixture(scope="module")
def rust_stats() -> dict[str, dict[str, list[int]]]:
    result = subprocess.run([str(BINARY), str(DATA)], capture_output=True, text=True, check=True)
    parsed: dict[str, dict[str, list[int]]] = json.loads(result.stdout)
    return parsed


def _python_totals(base, level: int, ivs, evs, nature: Nature) -> list[int]:  # type: ignore[no-untyped-def]
    return [
        calculate_total_hp(base.HP, ivs[Stats.HP], evs[Stats.HP], level),
        *(
            calculate_total_stat(getattr(base, stat.name), ivs[stat], evs[stat], level, nature, stat)
            for stat in _ORDER[1:]
        ),
    ]


@needs_rust
@pytest.mark.parametrize("nature_name,level,ivs,evs", CASES)
def test_both_engines_compute_the_same_stats(rust_stats, nature_name, level, ivs, evs) -> None:  # type: ignore[no-untyped-def]
    theirs = rust_stats[f"{nature_name}-{level}"]
    nature = Nature[nature_name]
    mismatched: list[str] = []
    for key, species in get_all_species().items():
        mine = _python_totals(species.base_stats, level, ivs, evs, nature)
        if theirs.get(key) != mine:
            mismatched.append(f"{key}: python {mine} rust {theirs.get(key)}")
    assert not mismatched, f"{len(mismatched)} species disagree, first few: {mismatched[:5]}"


@needs_rust
def test_every_species_is_present_in_both(rust_stats) -> None:  # type: ignore[no-untyped-def]
    """A species the Rust loader silently dropped would look like agreement on everything it does
    have. The count has to match too."""
    assert set(rust_stats["HARDY-50"]) == set(get_all_species())


@needs_rust
def test_shedinja_is_the_special_case_it_always_is(rust_stats) -> None:  # type: ignore[no-untyped-def]
    """One hit point at any level, any IVs, any EVs — the one branch in the HP formula."""
    for case in ("HARDY-50", "JOLLY-50", "MODEST-100"):
        assert rust_stats[case]["shedinja"][0] == 1
