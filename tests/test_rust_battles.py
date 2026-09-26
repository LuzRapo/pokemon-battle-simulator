"""Whole battles, played by both engines, compared turn for turn.

This is the test the rest of the project exists to make possible. Everything before it — the tape,
the named actions, the digest, the data export — was scaffolding for one question: given the same
teams, the same choices and the same rolls, do the two engines produce the same battle?

The Rust engine knows a deliberately small slice of the game so far: plain damaging moves and
switches. Scenarios are therefore *restricted* to that slice rather than filtered afterwards, and
the binary refuses anything outside it with exit code 2 rather than guessing. A refusal is not a
failure here — a wrong answer is. The two are kept distinguishable on purpose, because the day this
project goes wrong is the day "unsupported" starts being reported as "agrees".
"""

import json
import random
import subprocess
from pathlib import Path

import pytest

from battle_sim.database.loader import get_all_moves
from battle_sim.differential import Scenario, compare, record
from battle_sim.engine import legal_actions
from battle_sim.models.actions import ActionType
from battle_sim.models.spec import PokemonSpec
from battle_sim.models.stats import EVs, IVs
from battle_sim.utils import Ability, Item, Nature

RUST = Path(__file__).resolve().parent.parent / "rust"
BINARY = RUST / "target" / "release" / "replay"
DATA = RUST / "data"

needs_rust = pytest.mark.skipif(
    not BINARY.exists(), reason="the Rust crate is not built; run `cargo build --release` in rust/"
)

# Species with no ability the engine binds behaviour for, so nothing modifies damage on either
# side. Abilities are the bulk of what is still unported; until they land, a scenario containing a
# live one is not a fair comparison, it is a comparison of one engine against three-quarters of
# another.
PLAIN_SPECIES = ("Rhydon", "Machamp", "Kangaskhan", "Tauros", "Dewgong", "Golem")


def _plain_move_names() -> list[str]:
    """Damaging moves with a fixed power, no secondary, and nothing clever attached.

    "Nothing clever" includes the moves whose power is computed from the board — Revenge doubles
    when its user was hit, Gyro Ball reads the speed difference — because that logic lives in
    `engine/power.py` rather than in the effect list. Reading only the effects was how Revenge got
    into a scenario and came back at 96 against the Python's 150.
    """
    special = set(json.loads((DATA / "rules.json").read_text())["special_power_moves"])
    wanted = []
    for move in get_all_moves().values():
        effects = move.effects
        if len(effects) != 1 or type(effects[0]).__name__ != "DamageEffect":
            continue
        if not getattr(effects[0], "power", None) or getattr(effects[0], "multi_hit", None):
            continue
        if any((move.self_switch, move.healing, move.force_switch, move.recharges, move.charge)):
            continue
        if getattr(effects[0], "drain_percent", None) or getattr(effects[0], "recoil_percent", None):
            continue
        if move.name in special:
            continue
        wanted.append(move.name)
    return sorted(wanted)


PLAIN_MOVES = _plain_move_names()


def _team(rng: random.Random, size: int = 2) -> list[PokemonSpec]:
    return [
        PokemonSpec(
            species=rng.choice(PLAIN_SPECIES),
            nickname=f"P{index}",
            level=50,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            effort_values=EVs(),
            individual_values=IVs(),
            moves=rng.sample(PLAIN_MOVES, 2),
        )
        for index in range(size)
    ]


def _chooser(rng: random.Random):  # type: ignore[no-untyped-def]
    """Only moves, so the ported slice is what gets exercised. Switching is supported but a random
    switch mostly tests the harness rather than the engine."""

    def choose(state, side_index):  # type: ignore[no-untyped-def]
        options = [a for a in legal_actions(state, side_index) if a.action is ActionType.USE_MOVE]
        return rng.choice(options or legal_actions(state, side_index))

    return choose


def _rust_trace(scenario: Scenario, tmp_path: Path) -> list[dict] | str:
    """The Rust engine's answer, or the reason it declined to give one."""
    path = tmp_path / "scenario.json"
    path.write_text(scenario.to_json())
    result = subprocess.run([str(BINARY), str(path), str(DATA)], capture_output=True, text=True)
    if result.returncode == 2:
        return result.stderr.strip()
    assert result.returncode == 0, f"replay failed ({result.returncode}): {result.stderr}"
    parsed: list[dict] = json.loads(result.stdout)
    return parsed


@needs_rust
@pytest.mark.parametrize("seed", range(25))
def test_both_engines_play_the_same_battle(seed: int, tmp_path: Path) -> None:
    rng = random.Random(seed)
    scenario, expected = record((_team(rng), _team(rng)), _chooser(rng), seed=seed, max_turns=30)

    theirs = _rust_trace(scenario, tmp_path)

    if isinstance(theirs, str):
        pytest.skip(f"outside the ported slice: {theirs}")
    divergence = compare(expected, theirs)
    assert divergence is None, f"seed {seed}\n{divergence}"


@needs_rust
def test_a_move_it_has_not_learned_is_refused_rather_than_guessed(tmp_path: Path) -> None:
    """The failure mode this project cannot afford is a quiet wrong answer. Everything unported has
    to come back as a refusal, loudly, with the reason."""
    rng = random.Random(99)
    team = [
        PokemonSpec(
            species="Rhydon",
            nickname="P0",
            level=50,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Toxic", "Earthquake"],  # Toxic is a status move: not in the slice
        )
    ]
    scenario, _ = record((team, team), _chooser(rng), seed=1, max_turns=4)

    theirs = _rust_trace(scenario, tmp_path)

    assert isinstance(theirs, str) and "Toxic" in theirs, theirs


@needs_rust
def test_the_tape_running_out_is_reported_as_a_divergence(tmp_path: Path) -> None:
    """If the Rust engine asks for more randomness than the Python used, the two have taken
    different paths — and that must never be papered over with a fresh number."""
    rng = random.Random(7)
    scenario, _ = record((_team(rng), _team(rng)), _chooser(rng), seed=7, max_turns=30)
    starved = Scenario(teams=scenario.teams, actions=scenario.actions, tape=scenario.tape[:1], seed=scenario.seed)

    theirs = _rust_trace(starved, tmp_path)

    assert isinstance(theirs, str) and "asked for" in theirs, theirs
