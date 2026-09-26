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


# The attachments that take a move out of the ported slice: a second use, a heal, a forced switch,
# a charge or recharge turn, a user that blows itself up.
_UNPORTED_ATTACHMENTS = ("self_switch", "healing", "force_switch", "recharges", "charge", "self_destructs")


def _attached(move: object) -> bool:
    return any(getattr(move, name, False) for name in _UNPORTED_ATTACHMENTS)


def _plain_damage(effect: object) -> bool:
    """A damage effect with a fixed power and none of the attachments that are still unported."""
    if not getattr(effect, "power", None) or getattr(effect, "multi_hit", None):
        return False
    return not (getattr(effect, "drain_percent", None) or getattr(effect, "recoil_percent", None))


def _plain_move_names() -> list[str]:
    """Damaging moves with a fixed power, no secondary, and nothing clever attached.

    "Nothing clever" means, among other things, not appearing in `coded_moves` — the sweep of every
    move the Python names by hand somewhere in its rules. That list is where Revenge lives (power
    doubles when its user was hit), and Last Resort (fails outright until its user has spent its
    other moves), and neither fact is anywhere in the move's own data.
    """
    coded = set(json.loads((DATA / "rules.json").read_text())["coded_moves"])
    wanted = []
    for move in get_all_moves().values():
        effects = move.effects
        if len(effects) != 1 or type(effects[0]).__name__ != "DamageEffect" or not _plain_damage(effects[0]):
            continue
        if _attached(move):
            continue
        if move.name in coded:
            continue
        wanted.append(move.name)
    return sorted(wanted)


def _status_and_stage_move_names() -> list[str]:
    """The wider slice: moves that also inflict a real status or move stat stages.

    Volatiles (confusion, Leech Seed, Substitute) are excluded — they are the next milestone, not
    this one. A move whose status is a volatile is refused by the binary anyway; keeping it out of
    the generator is what makes these runs actually compare something rather than skip.
    """
    real_statuses = {"BURN", "FREEZE", "PARALYSIS", "POISON", "TOXIC", "SLEEP"}

    def ported(effect: object) -> bool:
        kind = type(effect).__name__
        if kind == "StatStageChangeEffect":
            return True
        if kind == "InflictStatusEffect":
            return getattr(getattr(effect, "status", None), "name", None) in real_statuses
        return kind == "DamageEffect" and _plain_damage(effect)

    coded = set(json.loads((DATA / "rules.json").read_text())["coded_moves"])
    return sorted(
        move.name
        for move in get_all_moves().values()
        if move.effects
        and move.name not in coded
        and not _attached(move)
        and all(ported(effect) for effect in move.effects)
    )


def _ported() -> dict[str, list[str]]:
    """What the Rust engine says it has implemented.

    Asked of the binary rather than kept here, so the generator cannot drift out of step with the
    engine — which is exactly how the coded-move list went stale and let Last Resort through.
    """
    if not BINARY.exists():
        return {"abilities": [], "items": [], "coded_moves": []}
    out = subprocess.run([str(BINARY), "--ported"], capture_output=True, text=True, check=True).stdout
    listed: dict[str, list[str]] = json.loads(out)
    return listed


PORTED = _ported()
PLAIN_MOVES = _plain_move_names()
STATUS_MOVES = _status_and_stage_move_names()


def _team(
    rng: random.Random, size: int = 2, pool: list[str] | None = None, abilities: bool = False
) -> list[PokemonSpec]:
    moves = pool if pool is not None else PLAIN_MOVES
    # NONE stays in the draw so that "no ability" keeps being tested alongside the ported ones,
    # and so a mixed board — one side with an ability, one without — happens often.
    choices = [Ability.NONE, *(Ability[name] for name in PORTED["abilities"])] if abilities else [Ability.NONE]
    return [
        PokemonSpec(
            species=rng.choice(PLAIN_SPECIES),
            nickname=f"P{index}",
            level=50,
            ability=rng.choice(choices),
            item=Item.NONE,
            nature=Nature.HARDY,
            effort_values=EVs(),
            individual_values=IVs(),
            moves=rng.sample(moves, 2),
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


UNPORTED, DIVERGED = 2, 3


def _rust_trace(scenario: Scenario, tmp_path: Path) -> list[dict] | str:
    """The Rust engine's answer, or the reason it declined to give one.

    A string back means exit 2: the scenario needs something unported, and a caller may skip. Exit
    3 never comes back — it means the tape showed the two engines had already parted, and it is
    raised here rather than returned, because the moment a real divergence can be skipped the
    whole harness stops being worth running. Five of them were sitting in a green run as skips.
    """
    path = tmp_path / "scenario.json"
    path.write_text(scenario.to_json())
    result = subprocess.run([str(BINARY), str(path), str(DATA)], capture_output=True, text=True)
    if result.returncode == UNPORTED:
        return result.stderr.strip()
    assert result.returncode != DIVERGED, f"the engines took different paths: {result.stderr.strip()}"
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
@pytest.mark.parametrize("seed", range(25))
def test_status_and_stat_stages_agree_too(seed: int, tmp_path: Path) -> None:
    """The wider slice: burns, poisons, sleeps and stat drops, which between them are most of what
    the move database actually does."""
    rng = random.Random(1000 + seed)
    teams = (_team(rng, pool=STATUS_MOVES), _team(rng, pool=STATUS_MOVES))
    scenario, expected = record(teams, _chooser(rng), seed=seed, max_turns=40)

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
            moves=["Leech Seed", "Earthquake"],  # a volatile: the next milestone, not this one
        )
    ]
    scenario, _ = record((team, team), _chooser(rng), seed=1, max_turns=4)

    theirs = _rust_trace(scenario, tmp_path)

    assert isinstance(theirs, str) and "Leech Seed" in theirs, theirs


@needs_rust
@pytest.mark.parametrize("seed", range(25))
def test_the_ported_abilities_agree_too(seed: int, tmp_path: Path) -> None:
    """The same battles, with an ability on nearly every Pokemon.

    Which abilities is not decided here — the binary is asked. A list kept on this side would be
    free to fall behind the engine, and quietly testing forty-six of forty-seven is the kind of
    green run this project is built to distrust.
    """
    rng = random.Random(2000 + seed)
    teams = (
        _team(rng, size=3, pool=STATUS_MOVES, abilities=True),
        _team(rng, size=3, pool=STATUS_MOVES, abilities=True),
    )
    scenario, expected = record(teams, _chooser(rng), seed=seed, max_turns=60)

    theirs = _rust_trace(scenario, tmp_path)

    if isinstance(theirs, str):
        pytest.skip(f"outside the ported slice: {theirs}")
    divergence = compare(expected, theirs)
    assert divergence is None, f"seed {seed}\n{divergence}"


@needs_rust
def test_every_ported_ability_reaches_a_battle() -> None:
    """A guard against testing nothing.

    The ability tests would pass just as happily if the generator never picked half these names, so
    this asserts the draw actually reaches all of them. It is the cheap half of the check; the
    expensive half is that turning the ability dispatch off makes 187 of 200 swept battles diverge.
    """
    rng = random.Random(0)
    drawn = {spec.ability.name for _ in range(400) for spec in _team(rng, size=3, abilities=True)}

    missing = set(PORTED["abilities"]) - drawn
    assert not missing, f"never generated: {sorted(missing)}"


@needs_rust
@pytest.mark.parametrize(
    ("ability", "item", "expected"),
    [(Ability.INTIMIDATE, Item.NONE, "INTIMIDATE"), (Ability.NONE, Item.LEFTOVERS, "LEFTOVERS")],
)
def test_live_abilities_and_items_are_refused_rather_than_ignored(
    ability: Ability, item: Item, expected: str, tmp_path: Path
) -> None:
    """Reading an ability off a Pokemon and doing nothing with it is a wrong answer in silence.

    Both of these are wired to the Python's event bus, so a battle containing one is not comparable
    until the Rust engine implements it. The engine has to say so.
    """
    rng = random.Random(3)
    team = [
        PokemonSpec(
            species="Rhydon",
            nickname="P0",
            level=50,
            ability=ability,
            item=item,
            nature=Nature.HARDY,
            moves=["Earthquake", "Rock Slide"],
        )
    ]
    scenario, _ = record((team, team), _chooser(rng), seed=3, max_turns=4)

    theirs = _rust_trace(scenario, tmp_path)

    assert isinstance(theirs, str) and expected in theirs, theirs


@needs_rust
def test_a_pokemon_on_the_bench_is_checked_too(tmp_path: Path) -> None:
    """The lead is comparable; the one behind it is not. Finding that out on turn nine would mean
    eight turns had already been reported as agreement."""
    rng = random.Random(4)
    plain = PokemonSpec(
        species="Rhydon", nickname="P0", level=50, ability=Ability.NONE, item=Item.NONE,
        nature=Nature.HARDY, moves=["Earthquake", "Rock Slide"],
    )
    benched = PokemonSpec(
        species="Rhydon", nickname="P1", level=50, ability=Ability.INTIMIDATE, item=Item.NONE,
        nature=Nature.HARDY, moves=["Earthquake", "Rock Slide"],
    )
    scenario, _ = record(([plain, benched], [plain, benched]), _chooser(rng), seed=4, max_turns=4)

    theirs = _rust_trace(scenario, tmp_path)

    assert isinstance(theirs, str) and "INTIMIDATE" in theirs, theirs


@needs_rust
def test_the_tape_running_out_is_reported_as_a_divergence(tmp_path: Path) -> None:
    """If the Rust engine asks for more randomness than the Python used, the two have taken
    different paths — and that must never be papered over with a fresh number."""
    rng = random.Random(7)
    scenario, _ = record((_team(rng), _team(rng)), _chooser(rng), seed=7, max_turns=30)
    starved = Scenario(teams=scenario.teams, actions=scenario.actions, tape=scenario.tape[:1], seed=scenario.seed)

    path = tmp_path / "starved.json"
    path.write_text(starved.to_json())
    result = subprocess.run([str(BINARY), str(path), str(DATA)], capture_output=True, text=True)

    assert result.returncode == DIVERGED, f"exit {result.returncode}: {result.stderr}"
    assert "asked for" in result.stderr, result.stderr
