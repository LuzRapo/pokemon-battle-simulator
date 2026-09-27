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
from collections import Counter
from pathlib import Path

import pytest

from battle_sim.database.loader import get_all_moves
from battle_sim.differential import Scenario, compare, record
from battle_sim.engine import legal_actions
from battle_sim.models.actions import Action, ActionType
from battle_sim.models.moves import MoveSlot
from battle_sim.models.spec import PokemonSpec
from battle_sim.models.stats import EVs, IVs
from battle_sim.utils import Ability, Item, Nature, Target

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


# The attachments that take a move out of the ported slice: a second use, a forced switch, a
# recharge turn, a user that blows itself up. Not `healing` — see `unsupported` in turn.rs. Not
# `charge` any more either — all 17 two-turn moves are ported.
_UNPORTED_ATTACHMENTS: tuple[str, ...] = ()


def _attached(move: object) -> bool:
    return any(getattr(move, name, False) for name in _UNPORTED_ATTACHMENTS)


def _plain_damage(effect: object) -> bool:
    """A damage effect with a fixed power and none of the attachments that are still unported."""
    return bool(getattr(effect, "power", None))


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
    """The wider slice: moves that also inflict a status, move stat stages, or land a volatile.

    Which volatiles count is asked of the binary rather than listed here — flinch and confusion
    today, more later — so widening the engine widens the generator without a second edit. A move
    carrying a volatile the engine does not know is refused, and a refusal is a skipped battle
    rather than a compared one, which is exactly the waste this keeps in step.
    """
    landable = {"BURN", "FREEZE", "PARALYSIS", "POISON", "TOXIC", "SLEEP", *PORTED["volatiles"]}

    def ported(move, effect: object) -> bool:  # type: ignore[no-untyped-def]
        kind = type(effect).__name__
        if kind == "StatStageChangeEffect":
            return True
        if kind == "InflictStatusEffect":
            return getattr(getattr(effect, "status", None), "name", None) in landable
        if kind in ("FixedDamageEffect", "HealEffect", "WeatherEffect", "TerrainEffect", "PseudoWeatherEffect"):
            return True
        if kind in ("SideConditionEffect", "RemoveHazardsEffect"):
            return True
        if kind == "CodedEffect":
            return getattr(getattr(effect, "kind", None), "name", None) in PORTED["coded_kinds"]
        # A coded move the engine has learned may carry no listed power at all — its formula
        # supplies one — so the plain-damage test is waived for those.
        return kind == "DamageEffect" and (_plain_damage(effect) or move.name in PORTED["coded_moves"])

    coded = set(json.loads((DATA / "rules.json").read_text())["coded_moves"]) - set(PORTED["coded_moves"])
    return sorted(
        move.name
        for move in get_all_moves().values()
        if move.name not in coded
        and not _attached(move)
        and all(ported(move, effect) for effect in move.effects)
    )


def _ported() -> dict[str, list[str]]:
    """What the Rust engine says it has implemented.

    Asked of the binary rather than kept here, so the generator cannot drift out of step with the
    engine — which is exactly how the coded-move list went stale and let Last Resort through.
    """
    if not BINARY.exists():
        return {"abilities": [], "items": [], "coded_moves": [], "coded_kinds": []}
    out = subprocess.run([str(BINARY), "--ported"], capture_output=True, text=True, check=True).stdout
    listed: dict[str, list[str]] = json.loads(out)
    return listed


PORTED = _ported()


def _still_unported(kind: str, enum: type) -> str:
    """Some live ability or item the engine has *not* implemented, whichever one that happens to be.

    Named rather than hardcoded because a hardcoded example goes stale the moment it is ported —
    this test was written against Intimidate and started failing the day Intimidate landed, which
    is a test that stops testing rather than one that fails honestly.
    """
    live = set(json.loads((DATA / "rules.json").read_text())[f"live_{kind}"])
    remaining = sorted(name for name in live - set(PORTED[kind]) if name in enum.__members__)
    assert remaining, f"every live {kind[:-1]} is ported; this test needs rewriting"
    return remaining[0]
PLAIN_MOVES = _plain_move_names()
STATUS_MOVES = _status_and_stage_move_names()


def _team(
    rng: random.Random,
    size: int = 2,
    pool: list[str] | None = None,
    abilities: bool = False,
    items: bool = False,
) -> list[PokemonSpec]:
    moves = pool if pool is not None else PLAIN_MOVES
    # NONE stays in each draw so that "nothing equipped" keeps being tested alongside the ported
    # ones, and so a mixed board — one side carrying something, one not — happens often.
    choices = [Ability.NONE, *(Ability[name] for name in PORTED["abilities"])] if abilities else [Ability.NONE]
    held = [Item.NONE, *(Item[name] for name in PORTED["items"])] if items else [Item.NONE]
    return [
        PokemonSpec(
            species=rng.choice(PLAIN_SPECIES),
            nickname=f"P{index}",
            level=50,
            ability=rng.choice(choices),
            item=rng.choice(held),
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


def _switching_chooser(rng: random.Random):  # type: ignore[no-untyped-def]
    """Like `_chooser`, but switches sometimes — which some mechanics need in order to happen."""

    def choose(state, side_index):  # type: ignore[no-untyped-def]
        options = legal_actions(state, side_index)
        moves = [a for a in options if a.action is ActionType.USE_MOVE]
        if moves and rng.random() < 0.3 and len(options) > len(moves):
            return rng.choice([a for a in options if a.action is not ActionType.USE_MOVE])
        return rng.choice(moves or options)

    return choose


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


# `test_a_move_it_has_not_learned_is_refused_rather_than_guessed` lived here, built around
# `_some_coded_move_this_engine_has_not_learned()`. It was rewritten twice as the gap it depended
# on moved — an unported volatile, then a move refused by name, then a move refused by
# `CodedMoveKind` — and retired outright when Transform closed that last gap: `--coverage` reports
# 843/843 moves playable, so there is no longer a move-shaped way to build this scenario. The same
# invariant (an unported *anything* comes back as a loud refusal, never a quiet wrong answer) is
# still exercised, now the only way left to exercise it: by ability and by item, in
# `test_live_abilities_and_items_are_refused_rather_than_ignored` below.


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
def test_the_ported_items_agree_too(tmp_path: Path) -> None:
    """Held items, drawn the same way the abilities are."""
    failures = []
    for seed in range(25):
        rng = random.Random(3000 + seed)
        teams = (
            _team(rng, size=3, pool=STATUS_MOVES, abilities=True, items=True),
            _team(rng, size=3, pool=STATUS_MOVES, abilities=True, items=True),
        )
        scenario, expected = record(teams, _chooser(rng), seed=seed, max_turns=60)
        theirs = _rust_trace(scenario, tmp_path)
        if isinstance(theirs, str):
            continue
        divergence = compare(expected, theirs)
        if divergence is not None:
            failures.append(f"seed {seed}: {divergence}")
    assert not failures, "\n".join(failures)


@needs_rust
def test_the_ported_volatiles_actually_land_in_the_swept_battles() -> None:
    """Flinches and confusions have to happen, not merely be permitted.

    Agreement is cheap if a feature never fires — a battle in which nobody ever flinched agrees
    about flinching perfectly. So this counts the entries the Python's own trace carries, which is
    the same trace the Rust engine is compared against.
    """
    # A hundred and twenty, not forty: a flinch only *costs* a turn when its victim had not moved
    # yet, and as the move pool widened forty battles started inflicting nine flinches without one
    # of them ever landing on somebody still waiting.
    seen: Counter[str] = Counter()
    for seed in range(120):
        rng = random.Random(9000 + seed)
        teams = (
            _team(rng, size=3, pool=STATUS_MOVES, abilities=True, items=True),
            _team(rng, size=3, pool=STATUS_MOVES, abilities=True, items=True),
        )
        _, expected = record(teams, _chooser(rng), seed=seed, max_turns=60)
        for turn in expected:
            for event in turn["events"]:
                if event["type"] == "VolatileInflicted":
                    seen[event["volatile"]] += 1
                elif event["type"] in ("ConfusionSelfHit", "CantAct"):
                    seen[event.get("reason") or "self_hit"] += 1

    for wanted in ("FLINCH", "CONFUSION", "PROTECT", "flinch", "confused", "self_hit"):
        assert seen[wanted] > 0, f"{wanted} never happened across 120 battles: {dict(seen)}"


def _inflicts(effect: object, volatile: str) -> bool:
    """A move effect that lands this particular volatile."""
    return type(effect).__name__ == "InflictStatusEffect" and getattr(
        getattr(effect, "status", None), "name", None
    ) == volatile


def _moves_where(predicate) -> list[str]:  # type: ignore[no-untyped-def]
    """The slice's moves carrying an effect that satisfies a predicate."""
    by_name = {move.name: move for move in get_all_moves().values()}
    return [name for name in STATUS_MOVES if any(predicate(e) for e in by_name[name].effects)]


@needs_rust
@pytest.mark.parametrize(
    ("event", "predicate"),
    [
        ("MultiHitSummary", lambda e: type(e).__name__ == "DamageEffect" and e.multi_hit is not None),
        ("Drained", lambda e: type(e).__name__ == "DamageEffect" and e.drain_percent is not None),
        ("RecoilDamage", lambda e: type(e).__name__ == "DamageEffect" and e.recoil_percent is not None),
        ("DamageDealt", lambda e: type(e).__name__ == "FixedDamageEffect"),
        ("Healed", lambda e: type(e).__name__ == "HealEffect"),
        ("WeatherChanged", lambda e: type(e).__name__ == "WeatherEffect"),
        ("TerrainChanged", lambda e: type(e).__name__ == "TerrainEffect"),
        ("HazardSet", lambda e: type(e).__name__ == "SideConditionEffect"),
        ("HazardDamage", lambda e: type(e).__name__ == "SideConditionEffect"),
        ("ResidualDamage", lambda e: type(e).__name__ == "WeatherEffect"),
        ("Protected", lambda e: _inflicts(e, "PROTECT")),
        ("LeechSeedSap", lambda e: _inflicts(e, "LEECH_SEED")),
        ("TrapSqueezed", lambda e: _inflicts(e, "PARTIALLY_TRAPPED")),
    ],
)
def test_the_rarer_move_classes_agree_when_the_teams_are_built_for_them(
    event: str, predicate, tmp_path: Path
) -> None:
    """Multi-hit, drain and recoil, drawn deliberately rather than hoped for.

    Eleven of the slice's 578 moves drain or recoil, so a random draw finds them rarely enough that
    forty battles went by without one — a green run saying nothing. Restricting the pool makes the
    class certain to appear, and the assertion is that it appeared *and* both engines agreed.
    """
    pool = _moves_where(predicate)
    assert pool, "no move in the slice carries this effect"

    def team(rng: random.Random) -> list[PokemonSpec]:
        # One move from the class and one ordinary attack each. A team of nothing but healing moves
        # never takes damage, so it never heals either, and the class goes untested while the run
        # stays green.
        return [
            PokemonSpec(
                species=rng.choice(PLAIN_SPECIES),
                nickname=f"P{index}",
                level=50,
                ability=Ability.NONE,
                item=Item.NONE,
                nature=Nature.HARDY,
                moves=[rng.choice(pool), rng.choice(PLAIN_MOVES)],
            )
            for index in range(3)
        ]

    seen = 0
    for seed in range(12):
        rng = random.Random(11000 + seed)
        teams = (team(rng), team(rng))
        # Switching on purpose: an entry hazard is set by one move and paid for by a different
        # Pokemon arriving, so a chooser that never switches can never see the second half of it.
        scenario, expected = record(teams, _switching_chooser(rng), seed=seed, max_turns=60)
        theirs = _rust_trace(scenario, tmp_path)
        assert not isinstance(theirs, str), f"seed {seed} was refused: {theirs}"
        divergence = compare(expected, theirs)
        assert divergence is None, f"seed {seed}\n{divergence}"
        seen += sum(entry["type"] == event for turn in expected for entry in turn["events"])

    assert seen > 0, f"{event} never happened even with a team built for it"


@needs_rust
def test_charge_moves_actually_charge_and_agree(tmp_path: Path) -> None:
    """All 17 two-turn moves, forced to actually charge rather than merely being legal.

    A random draw from the full move pool lets a charge sit unused for turns at a time — the
    generator would happily agree about a battle where nobody ever charged anything. Restricting
    the pool to charges, the semi-invulnerability exceptions, and one plain attack each makes both
    the charging turn *and* the release turn certain to appear, on both sides of the reach rule.
    """
    charges = [
        "Fly", "Bounce", "Dig", "Dive", "Sky Drop", "Phantom Force", "Shadow Force", "Solar Beam",
        "Solar Blade", "Meteor Beam", "Electro Shot", "Freeze Shock", "Geomancy", "Ice Burn",
        "Razor Wind", "Skull Bash", "Sky Attack",
    ]
    reachers = ["Earthquake", "Surf", "Gust", "Thunder"]

    def team(rng: random.Random) -> list[PokemonSpec]:
        return [
            PokemonSpec(
                species=rng.choice(PLAIN_SPECIES),
                nickname=f"P{index}",
                level=50,
                ability=Ability.NONE,
                item=Item.NONE,
                nature=Nature.HARDY,
                moves=[rng.choice(charges), rng.choice(reachers)],
            )
            for index in range(3)
        ]

    charging_turns = 0
    for seed in range(30):
        rng = random.Random(21000 + seed)
        teams = (team(rng), team(rng))
        scenario, expected = record(teams, _chooser(rng), seed=seed, max_turns=60)
        theirs = _rust_trace(scenario, tmp_path)
        assert not isinstance(theirs, str), f"seed {seed} was refused: {theirs}"
        divergence = compare(expected, theirs)
        assert divergence is None, f"seed {seed}\n{divergence}"
        charging_turns += sum(entry["type"] == "ChargingUp" for turn in expected for entry in turn["events"])

    assert charging_turns > 0, "no battle ever reached a charging turn"


@needs_rust
def test_rampage_moves_lock_in_and_confuse_themselves_out(tmp_path: Path) -> None:
    """Outrage and its three relatives: the lock, the free turns, and the confusion on the way out.

    None of `LOCKED_MOVE` taking hold is ever logged — that is the Python's own behaviour, matched
    on purpose — so what this can actually assert firing is the fatigue confusion at the end of the
    run, which the residual only reaches once the lock has counted all the way down.
    """
    pool = ["Outrage", "Petal Dance", "Raging Fury", "Thrash", "Tackle"]

    def team(rng: random.Random) -> list[PokemonSpec]:
        return [
            PokemonSpec(
                species=rng.choice(PLAIN_SPECIES),
                nickname=f"P{index}",
                level=50,
                ability=Ability.NONE,
                item=Item.NONE,
                nature=Nature.HARDY,
                moves=[rng.choice(pool), "Tackle"],
            )
            for index in range(3)
        ]

    fatigue = 0
    for seed in range(30):
        rng = random.Random(22000 + seed)
        teams = (team(rng), team(rng))
        scenario, expected = record(teams, _chooser(rng), seed=seed, max_turns=60)
        theirs = _rust_trace(scenario, tmp_path)
        assert not isinstance(theirs, str), f"seed {seed} was refused: {theirs}"
        divergence = compare(expected, theirs)
        assert divergence is None, f"seed {seed}\n{divergence}"
        fatigue += sum(
            entry["type"] == "VolatileInflicted" and entry["volatile"] == "CONFUSION"
            for turn in expected
            for entry in turn["events"]
        )

    assert fatigue > 0, "no rampage ever ran its course and confused its user"


@needs_rust
def test_rollout_and_ice_ball_escalate_and_lock(tmp_path: Path) -> None:
    """The other use of `LOCKED_MOVE`: a run that doubles its power and stops clean, no confusion.

    Forced onto its own rolling_hits bug once already — the reset that clears the counter for any
    move outside the escalating set was written to look only for Fury Cutter, so a landed Rollout
    zeroed its own count before its own power reads it a few lines later. Every hit after the first
    came out at base power. This pool makes that regression concrete and fixed at 800/800.
    """
    pool = ["Rollout", "Ice Ball", "Tackle"]

    def team(rng: random.Random) -> list[PokemonSpec]:
        return [
            PokemonSpec(
                species=rng.choice(PLAIN_SPECIES),
                nickname=f"P{index}",
                level=50,
                ability=Ability.NONE,
                item=Item.NONE,
                nature=Nature.HARDY,
                moves=[rng.choice(pool), "Tackle"],
            )
            for index in range(3)
        ]

    escalated = 0
    for seed in range(30):
        rng = random.Random(23000 + seed)
        teams = (team(rng), team(rng))
        scenario, expected = record(teams, _chooser(rng), seed=seed, max_turns=60)
        theirs = _rust_trace(scenario, tmp_path)
        assert not isinstance(theirs, str), f"seed {seed} was refused: {theirs}"
        divergence = compare(expected, theirs)
        assert divergence is None, f"seed {seed}\n{divergence}"
        hits = [e["amount"] for turn in expected for e in turn["events"] if e["type"] == "DamageDealt"]
        escalated += sum(1 for a, b in zip(hits, hits[1:], strict=False) if b > a)

    assert escalated > 0, "no consecutive hit ever came out harder than the one before it"


@needs_rust
def test_move_restriction_volatiles_fire_and_agree(tmp_path: Path) -> None:
    """Taunt, Encore and Disable: three different ways of taking a move off the table.

    Disable's block is checked before any PP is spent or `MoveUsed` is logged; Taunt's is checked
    after. Getting either one on the wrong side of the PP-spend line was worth a vacuity check on
    its own (272/300 and 194/300 of a restriction-heavy batch turned red with each disabled), so
    this asserts the three distinguishable outcomes — `DisabledBlocked`, `TauntBlocked`, and an
    `ENCORE` volatile actually landing — all happen and all agree.
    """
    pool = ["Taunt", "Encore", "Disable", "Tackle", "Growl"]

    def team(rng: random.Random) -> list[PokemonSpec]:
        return [
            PokemonSpec(
                species=rng.choice(PLAIN_SPECIES),
                nickname=f"P{index}",
                level=50,
                ability=Ability.NONE,
                item=Item.NONE,
                nature=Nature.HARDY,
                moves=[rng.choice(pool), rng.choice(pool)],
            )
            for index in range(3)
        ]

    seen: Counter[str] = Counter()
    for seed in range(40):
        rng = random.Random(24000 + seed)
        teams = (team(rng), team(rng))
        scenario, expected = record(teams, _switching_chooser(rng), seed=seed, max_turns=60)
        theirs = _rust_trace(scenario, tmp_path)
        assert not isinstance(theirs, str), f"seed {seed} was refused: {theirs}"
        divergence = compare(expected, theirs)
        assert divergence is None, f"seed {seed}\n{divergence}"
        for turn in expected:
            for e in turn["events"]:
                if e["type"] in ("DisabledBlocked", "TauntBlocked", "DisableApplied"):
                    seen[e["type"]] += 1
                elif e["type"] == "VolatileInflicted" and e.get("volatile") == "ENCORE":
                    seen["Encored"] += 1

    for wanted in ("DisabledBlocked", "TauntBlocked", "DisableApplied", "Encored"):
        assert seen[wanted] > 0, f"{wanted} never happened across 40 battles: {dict(seen)}"


@needs_rust
def test_destiny_bond_takes_its_attacker_down_too(tmp_path: Path) -> None:
    """A turn where the Destiny Bond user faints has to take its attacker with it.

    Vacuity-checked directly: disabling the KO turned 105/300 of a Destiny-Bond-heavy batch red, so
    this asserts on the concrete signature of it firing — a single turn logging two `Fainted`
    entries — rather than trusting that mere agreement means the feature ran.
    """
    pool = ["Destiny Bond", "Tackle", "Protect"]

    def team(rng: random.Random) -> list[PokemonSpec]:
        return [
            PokemonSpec(
                species=rng.choice(PLAIN_SPECIES),
                nickname=f"P{index}",
                level=50,
                ability=Ability.NONE,
                item=Item.NONE,
                nature=Nature.HARDY,
                moves=[rng.choice(pool), "Tackle"],
            )
            for index in range(3)
        ]

    double_faints = 0
    for seed in range(40):
        rng = random.Random(25000 + seed)
        teams = (team(rng), team(rng))
        scenario, expected = record(teams, _chooser(rng), seed=seed, max_turns=60)
        theirs = _rust_trace(scenario, tmp_path)
        assert not isinstance(theirs, str), f"seed {seed} was refused: {theirs}"
        divergence = compare(expected, theirs)
        assert divergence is None, f"seed {seed}\n{divergence}"
        double_faints += sum(
            sum(1 for e in turn["events"] if e["type"] == "Fainted") >= 2 for turn in expected
        )

    assert double_faints > 0, "no turn ever fainted both sides at once"


@needs_rust
def test_identify_bypasses_a_ghost_or_dark_immunity(tmp_path: Path) -> None:
    """Foresight/Odor Sleuth and Miracle Eye actually let their moves land, not just resolve.

    A real regression, not a hypothetical one: the immunity gate in `resolve_move` and the type
    multiplier `damage::calculate_hit` computes for the formula are two separate calls into the
    type chart, and fixing only the first left an identified Sableye immune to Tackle for damage
    purposes while the gate had already let it through -- 265/300 of this exact batch diverged
    (`MoveFailed` where Python dealt damage) before both call sites read the bypass. Sableye is
    Dark/Ghost, so the same battle exercises both moves' bypass at once.
    """

    def team_a(rng: random.Random) -> list[PokemonSpec]:
        return [
            PokemonSpec(
                species="Machamp",
                nickname="A0",
                level=50,
                ability=Ability.NONE,
                item=Item.NONE,
                nature=Nature.HARDY,
                moves=[rng.choice(["Foresight", "Miracle Eye"]), "Tackle", "Psychic"],
            )
        ]

    def team_b(rng: random.Random) -> list[PokemonSpec]:
        return [
            PokemonSpec(
                species="Sableye",
                nickname="B0",
                level=50,
                ability=Ability.NONE,
                item=Item.NONE,
                nature=Nature.HARDY,
                moves=["Tackle"],
            )
        ]

    landed = 0
    for seed in range(30):
        rng = random.Random(26000 + seed)
        teams = (team_a(rng), team_b(rng))
        scenario, expected = record(teams, _chooser(rng), seed=seed, max_turns=30)
        theirs = _rust_trace(scenario, tmp_path)
        assert not isinstance(theirs, str), f"seed {seed} was refused: {theirs}"
        divergence = compare(expected, theirs)
        assert divergence is None, f"seed {seed}\n{divergence}"
        landed += sum(
            1 for turn in expected for e in turn["events"] if e["type"] == "DamageDealt" and e["side"] == 1
        )

    assert landed > 0, "an identified Sableye never actually took damage from the bypassed type"


@needs_rust
def test_substitute_soaks_hits_and_blocks_secondaries(tmp_path: Path) -> None:
    """The whole Substitute lifecycle: standing up, soaking, breaking, and blocking what a status
    or a stage drop would otherwise have done to the Pokemon behind it.

    Two features, each vacuity-checked in isolation: disabling the per-hit soak turned 241/300 of
    this exact batch red, and disabling the status/stage block (an opponent-targeted effect must
    not even draw for its probability while a substitute stands) turned 191/300 red.
    """
    pool = ["Substitute", "Tackle", "Thunder Fang", "Fury Attack", "Will-O-Wisp", "Growl"]

    def team(rng: random.Random) -> list[PokemonSpec]:
        return [
            PokemonSpec(
                species=rng.choice(PLAIN_SPECIES),
                nickname=f"P{index}",
                level=50,
                ability=Ability.NONE,
                item=Item.NONE,
                nature=Nature.HARDY,
                moves=[rng.choice(pool), rng.choice(pool)],
            )
            for index in range(4)
        ]

    seen: Counter[str] = Counter()
    for seed in range(40):
        rng = random.Random(27000 + seed)
        teams = (team(rng), team(rng))
        scenario, expected = record(teams, _switching_chooser(rng), seed=seed, max_turns=60)
        theirs = _rust_trace(scenario, tmp_path)
        assert not isinstance(theirs, str), f"seed {seed} was refused: {theirs}"
        divergence = compare(expected, theirs)
        assert divergence is None, f"seed {seed}\n{divergence}"
        for turn in expected:
            for e in turn["events"]:
                if e["type"] in ("SubstituteTookHit", "SubstituteBroke", "SubstituteAlready", "SubstituteTooWeak"):
                    seen[e["type"]] += 1

    for wanted in ("SubstituteTookHit", "SubstituteBroke", "SubstituteAlready", "SubstituteTooWeak"):
        assert seen[wanted] > 0, f"{wanted} never happened across 40 battles: {dict(seen)}"


@needs_rust
def test_sleep_talk_calls_a_real_move_through_the_sleep(tmp_path: Path) -> None:
    """Sleep Talk: acting through a status that would otherwise refuse every other move outright.

    Vacuity-checked directly — disabling the substitution turned 300/300 of this exact matchup red,
    since without it a sleeping Sleep-Talker just sits there logging `CantAct` instead.
    """
    team_a = [
        PokemonSpec(
            species="Gyarados", nickname="A0", level=50, ability=Ability.NONE, item=Item.NONE,
            nature=Nature.HARDY, moves=["Spore"],
        )
    ]
    team_b = [
        PokemonSpec(
            species="Gyarados", nickname="B0", level=50, ability=Ability.NONE, item=Item.NONE,
            nature=Nature.HARDY, moves=["Sleep Talk", "Roost", "King's Shield", "Tackle"],
        )
    ]

    called_through_sleep = 0
    for seed in range(30):
        rng = random.Random(28000 + seed)
        scenario, expected = record((team_a, team_b), _chooser(rng), seed=seed, max_turns=40)
        theirs = _rust_trace(scenario, tmp_path)
        assert not isinstance(theirs, str), f"seed {seed} was refused: {theirs}"
        divergence = compare(expected, theirs)
        assert divergence is None, f"seed {seed}\n{divergence}"
        for turn in expected:
            names = [e["move"] for e in turn["events"] if e["type"] == "MoveUsed" and e["side"] == 1]
            if len(names) >= 2:
                called_through_sleep += 1

    assert called_through_sleep > 0, "Sleep Talk never actually called a second move"


@needs_rust
def test_roost_grounds_its_user_for_the_rest_of_the_turn(tmp_path: Path) -> None:
    """A roosted Flying type loses its immunity to Ground moves for the turn it roosted.

    Vacuity-checked directly against `Pokemon::battle_types` specifically (not `identify_bypass`,
    which shares the same two-call-site shape): reverting only the immunity gate's site turned
    300/300 of this exact matchup red, which means both sites still have to agree independently.
    """
    team_a = [
        PokemonSpec(
            species="Gyarados", nickname="A0", level=50, ability=Ability.NONE, item=Item.NONE,
            nature=Nature.HARDY, moves=["Earthquake"],
        )
    ]
    team_b = [
        PokemonSpec(
            species="Gyarados", nickname="B0", level=50, ability=Ability.NONE, item=Item.NONE,
            nature=Nature.HARDY, moves=["Roost"],
        )
    ]

    landed, immune = 0, 0
    for seed in range(30):
        rng = random.Random(29000 + seed)
        scenario, expected = record((team_a, team_b), _chooser(rng), seed=seed, max_turns=20)
        theirs = _rust_trace(scenario, tmp_path)
        assert not isinstance(theirs, str), f"seed {seed} was refused: {theirs}"
        divergence = compare(expected, theirs)
        assert divergence is None, f"seed {seed}\n{divergence}"
        for turn in expected:
            for e in turn["events"]:
                if e["type"] == "DamageDealt" and e["side"] == 1:
                    landed += 1
                if e["type"] == "NoEffect" and e["side"] == 1:
                    immune += 1

    assert landed > 0, "Earthquake never once landed on a roosted Gyarados"
    assert immune > 0, "Earthquake was never once blocked by an un-roosted Gyarados's Flying type"


@needs_rust
def test_the_first_batch_of_coded_moves_fire_and_agree(tmp_path: Path) -> None:
    """Rest, Haze, Court Change, Perish Song and Curse's Ghost half: five of the twelve
    `CodedMoveKind`s this port has learned, chosen because each asserts on a distinct log entry
    rather than merely on agreement -- a green run over a battle where none of them actually fired
    would say nothing. Pain Split, Strength Sap, Belly Drum, Tidy Up and the cure moves are exercised
    by this same pool but not asserted on individually here.

    Perish Song's residual (a silent four-turn countdown to a mutual faint) is vacuity-checked on
    its own: disabling it turned 300/300 of a Perish-Song-only matchup red. So is Curse's per-turn
    chip, at 300/300 red against a dedicated Gengar-vs-Machamp matchup.
    """
    pool = [
        "Rest", "Moonlight", "Pain Split", "Strength Sap", "Belly Drum", "Haze", "Court Change",
        "Curse", "Tidy Up", "Perish Song", "Heal Bell", "Take Heart", "Tackle", "Substitute", "Screech",
    ]

    def team(rng: random.Random) -> list[PokemonSpec]:
        return [
            PokemonSpec(
                species=rng.choice(PLAIN_SPECIES),
                nickname=f"P{index}",
                level=50,
                ability=Ability.NONE,
                item=Item.NONE,
                nature=Nature.HARDY,
                moves=[rng.choice(pool), rng.choice(pool)],
            )
            for index in range(4)
        ]

    seen: Counter[str] = Counter()
    for seed in range(40):
        rng = random.Random(30000 + seed)
        teams = (team(rng), team(rng))
        scenario, expected = record(teams, _switching_chooser(rng), seed=seed, max_turns=100)
        theirs = _rust_trace(scenario, tmp_path)
        assert not isinstance(theirs, str), f"seed {seed} was refused: {theirs}"
        divergence = compare(expected, theirs)
        assert divergence is None, f"seed {seed}\n{divergence}"
        for turn in expected:
            for e in turn["events"]:
                if e["type"] in ("AllStatsReset", "CourtChanged"):
                    seen[e["type"]] += 1
                if e["type"] == "VolatileInflicted" and e.get("volatile") == "PERISH":
                    seen["PERISH"] += 1
                if e["type"] == "StatusInflicted" and e.get("status") == "SLEEP":
                    seen["Rest"] += 1

    for wanted in ("AllStatsReset", "CourtChanged", "PERISH", "Rest"):
        assert seen[wanted] > 0, f"{wanted} never happened across 40 battles: {dict(seen)}"


@needs_rust
def test_item_and_ability_manipulation_moves_fire_and_agree(tmp_path: Path) -> None:
    """Knock Off's item removal, Trick, Skill Swap, Entrainment and Worry Seed.

    A mega stone (or Primal orb, Rusted Sword/Shield, or Z-Crystal) held by a Pokemon that could
    use one is a real, separate regression this batch's own testing turned up: `is_fused_to`
    refuses to remove or trade one away, but a Rust battle carrying one at all used to be silently
    wrong regardless — Python auto-Mega-Evolves it before a single move is ordered, a mechanic this
    port has never touched, and nothing marked the item "live" because the Python builds its
    forme-by-item table from a runtime `item_from_showdown` lookup the AST sweep that finds live
    items cannot see through. `live_behaviour()` now unions in `formes.mega_stones()` directly, and
    a Pokemon whose species+moveset alone would trigger Mega Rayquaza (no item needed) is refused by
    its own separate check. Vacuity-checked directly: disabling the Z-Crystal half of
    `is_fused_to` turned 300/300 of a dedicated matchup red.
    """
    team_a = [
        PokemonSpec(
            species="Rhydon", nickname="A0", level=50, ability=Ability.NONE, item=Item.LEFTOVERS,
            nature=Nature.HARDY, moves=["Knock Off", "Trick", "Skill Swap", "Role Play"],
        )
    ]
    team_b = [
        PokemonSpec(
            species="Machamp", nickname="B0", level=50, ability=Ability.NONE, item=Item.SITRUS_BERRY,
            nature=Nature.HARDY, moves=["Knock Off", "Trick", "Entrainment", "Worry Seed"],
        )
    ]

    seen: Counter[str] = Counter()
    for seed in range(40):
        rng = random.Random(31000 + seed)
        scenario, expected = record((team_a, team_b), _chooser(rng), seed=seed, max_turns=40)
        theirs = _rust_trace(scenario, tmp_path)
        assert not isinstance(theirs, str), f"seed {seed} was refused: {theirs}"
        divergence = compare(expected, theirs)
        assert divergence is None, f"seed {seed}\n{divergence}"
        for turn in expected:
            for e in turn["events"]:
                if e["type"] in ("ItemRemoved", "ItemsSwapped", "AbilitiesSwapped", "AbilityChanged"):
                    seen[e["type"]] += 1

    for wanted in ("ItemRemoved", "ItemsSwapped", "AbilitiesSwapped", "AbilityChanged"):
        assert seen[wanted] > 0, f"{wanted} never happened across 40 battles: {dict(seen)}"


@needs_rust
def test_a_mega_stone_holder_is_refused_not_played_wrong(tmp_path: Path) -> None:
    """The regression `test_item_and_ability_manipulation_moves_fire_and_agree` names: a Pokemon
    holding a Mega Stone must be refused, not silently kept in its base forme all battle."""
    team = [
        PokemonSpec(
            species="Garchomp", nickname="A0", level=50, ability=Ability.NONE, item=Item.GARCHOMPITE,
            nature=Nature.HARDY, moves=["Tackle"],
        )
    ]
    rng = random.Random(32000)
    scenario, _ = record((team, team), _chooser(rng), seed=1, max_turns=4)
    theirs = _rust_trace(scenario, tmp_path)
    assert isinstance(theirs, str) and "GARCHOMPITE" in theirs, theirs


@needs_rust
def test_trick_room_inverts_the_speed_sort(tmp_path: Path) -> None:
    """The four pseudo-weather rooms. Gravity, Magic Room and Wonder Room have no gameplay effect
    anywhere in this codebase beyond standing up and ticking down — a deliberate simplification,
    not a gap in the port — so only Trick Room's speed-sort inversion is asserted on directly.

    Vacuity-checked: disabling the inversion (leaving the field state itself intact) turned
    177/300 of this exact matchup red.
    """
    team_a = [
        PokemonSpec(
            species="Dewgong", nickname="A0", level=50, ability=Ability.NONE, item=Item.NONE,
            nature=Nature.HARDY, moves=["Trick Room", "Tackle"],
        )
    ]
    team_b = [
        PokemonSpec(
            species="Golem", nickname="B0", level=1, ability=Ability.NONE, item=Item.NONE,
            nature=Nature.HARDY, moves=["Tackle", "Trick Room"],
        )
    ]

    slow_mon_moved_first = 0
    for seed in range(40):
        rng = random.Random(33000 + seed)
        scenario, expected = record((team_a, team_b), _chooser(rng), seed=seed, max_turns=30)
        theirs = _rust_trace(scenario, tmp_path)
        assert not isinstance(theirs, str), f"seed {seed} was refused: {theirs}"
        divergence = compare(expected, theirs)
        assert divergence is None, f"seed {seed}\n{divergence}"
        for turn in expected:
            movers = [e["side"] for e in turn["events"] if e["type"] == "MoveUsed"]
            if movers and movers[0] == 1:
                slow_mon_moved_first += 1

    assert slow_mon_moved_first > 0, "the level-1 Golem never once moved first"


@needs_rust
def test_two_rooms_fading_together_log_in_cast_order_not_alphabetical(tmp_path: Path) -> None:
    """A real divergence the sweep found while this batch was being tested: when Trick Room and
    Wonder Room expire on the same residual pass, Python's `PseudoWeatherEnded` lines come out in
    the order they were *cast*, because a Python `dict` preserves insertion order — a `BTreeMap`
    on the Rust side logged them alphabetically instead, which agreed whenever the two happened to
    coincide and diverged the rest of the time. `Field::pseudo_weather` (and `Side::hazards` and
    `Side::screens`, the same shape) are `OrderedCounts` now, not `BTreeMap`, for exactly this
    reason. Both cast orders are exercised here since the pool doesn't favour either.
    """
    def team(first: str, second: str) -> list[PokemonSpec]:
        return [
            PokemonSpec(
                species="Rhydon", nickname="P0", level=50, ability=Ability.NONE, item=Item.NONE,
                nature=Nature.HARDY, moves=[first, second, "Tackle"],
            )
        ]

    # Scripted rather than random: both sides cast their own room on turn 0 (so both timers start
    # together), then only Tackle after that (so neither timer is refreshed) -- random play almost
    # never holds still for the five turns both durations need to actually align, which is exactly
    # why the sweep needed a much larger, noisier battle to find this in the first place. Each side
    # always casts its own first-listed move on turn 0: which room that is depends only on which
    # team a side is holding, so swapping the two teams between sides (rather than reindexing the
    # chooser) is what actually flips which name gets cast -- and so logged -- first.
    def choose(state, side_index):  # type: ignore[no-untyped-def]
        options = legal_actions(state, side_index)
        wanted = MoveSlot.FIRST if state.turn == 0 else MoveSlot.THIRD
        for action in options:
            if action.action is ActionType.USE_MOVE and action.move is wanted:
                return action
        return options[0]

    both_faded_together = 0
    team_trick_room_first = team("Trick Room", "Wonder Room")
    team_wonder_room_first = team("Wonder Room", "Trick Room")
    for teams in ((team_trick_room_first, team_wonder_room_first), (team_wonder_room_first, team_trick_room_first)):
        scenario, expected = record(teams, choose, seed=1, max_turns=15)
        theirs = _rust_trace(scenario, tmp_path)
        assert not isinstance(theirs, str), f"refused: {theirs}"
        divergence = compare(expected, theirs)
        assert divergence is None, str(divergence)
        for turn in expected:
            fades = [e["kind"] for e in turn["events"] if e["type"] == "PseudoWeatherEnded"]
            if len(fades) >= 2:
                both_faded_together += 1

    assert both_faded_together > 0, "Trick Room and Wonder Room never once faded on the same turn"


@needs_rust
def test_wish_and_the_delayed_moves_fire_and_agree(tmp_path: Path) -> None:
    """Wish, Healing Wish/Lunar Dance, Revival Blessing, and Future Sight/Doom Desire.

    Shed Tail is exercised by the same pool but asserted on separately below, since its own
    regression (a same-turn forced switch this engine wasn't performing at all) is worth its own
    failure message rather than folding into this one.
    """
    pool = ["Wish", "Healing Wish", "Revival Blessing", "Future Sight", "Doom Desire", "Tackle", "Explosion"]

    def team(rng: random.Random) -> list[PokemonSpec]:
        return [
            PokemonSpec(
                species=rng.choice(PLAIN_SPECIES),
                nickname=f"P{index}",
                level=50,
                ability=Ability.NONE,
                item=Item.NONE,
                nature=Nature.HARDY,
                moves=[rng.choice(pool), rng.choice(pool)],
            )
            for index in range(4)
        ]

    seen: Counter[str] = Counter()
    for seed in range(40):
        rng = random.Random(35000 + seed)
        teams = (team(rng), team(rng))
        scenario, expected = record(teams, _switching_chooser(rng), seed=seed, max_turns=100)
        theirs = _rust_trace(scenario, tmp_path)
        if isinstance(theirs, str):
            # Future Sight/Doom Desire landing after their attacker switched away is refused
            # rather than guessed at (see resolve_future_sight); a battle built to force switching
            # hits that refusal often, and that is the correct engine, not a broken generator.
            continue
        divergence = compare(expected, theirs)
        assert divergence is None, f"seed {seed}\n{divergence}"
        for turn in expected:
            for e in turn["events"]:
                if e["type"] in ("WishMade", "Revived", "FutureAttackQueued", "FutureAttackLands"):
                    seen[e["type"]] += 1

    for wanted in ("WishMade", "Revived", "FutureAttackQueued", "FutureAttackLands"):
        assert seen[wanted] > 0, f"{wanted} never happened across 40 battles: {dict(seen)}"


@needs_rust
def test_shed_tail_forces_an_immediate_switch(tmp_path: Path) -> None:
    """A real regression the sweep found: Shed Tail's `needs_switch` is not an AI-side-only
    concern the way it first looked — `_resolve_pending_switches` runs right after *every* action
    resolves, so the replacement arrives the same turn, immediately after the substitute is left
    behind. Vacuity-checked directly: without the same `send_out_replacement` call an ordinary
    pivot already uses, 94/300 of this exact matchup diverged.
    """
    pool = ["Shed Tail", "Tackle"]

    def team(rng: random.Random) -> list[PokemonSpec]:
        return [
            PokemonSpec(
                species=rng.choice(PLAIN_SPECIES),
                nickname=f"P{index}",
                level=50,
                ability=Ability.NONE,
                item=Item.NONE,
                nature=Nature.HARDY,
                moves=[rng.choice(pool), rng.choice(pool)],
            )
            for index in range(4)
        ]

    immediate_switches = 0
    for seed in range(40):
        rng = random.Random(36000 + seed)
        teams = (team(rng), team(rng))
        scenario, expected = record(teams, _chooser(rng), seed=seed, max_turns=60)
        theirs = _rust_trace(scenario, tmp_path)
        assert not isinstance(theirs, str), f"seed {seed} was refused: {theirs}"
        divergence = compare(expected, theirs)
        assert divergence is None, f"seed {seed}\n{divergence}"
        for turn in expected:
            names = [e["type"] for e in turn["events"]]
            for i, name in enumerate(names):
                if name == "MoveUsed" and turn["events"][i]["move"] == "Shed Tail" and "Switched" in names[i:]:
                    immediate_switches += 1

    assert immediate_switches > 0, "Shed Tail never once forced a same-turn switch"


@needs_rust
def test_transform_copies_the_target_and_reverts_on_switch_out(tmp_path: Path) -> None:
    """The last move in the database: base stats (HP's excepted), nature/EVs/IVs, types, ability,
    moveset (5 PP each) and stat stages all copied off the target, restored on switch-out. Rhydon
    and Machamp differ in every one of those, so a divergence in any of them shows up either in the
    next hit's damage (stats), in which named actions even remain legal (moveset), or in the digest
    directly (types, ability, pp, stages) -- and the restore is exercised by switching the
    transformed side out and back in in the same sweep. Vacuity-checked directly: skipping the copy
    (leaving the `Transformed` log line in place) turned the very first of 40 seeds red with a
    replay refusal, not a quiet pass -- a still-Rhydon Pokemon holding a name from Machamp's
    moveset has nowhere to put it.
    """
    pool_a = ["Transform", "Tackle"]
    pool_b = ["Karate Chop", "Rock Slide", "Bulk Up", "Tackle"]

    def team(rng: random.Random, pool: list[str]) -> list[PokemonSpec]:
        return [
            PokemonSpec(
                species=species, nickname=f"P{index}", level=50, ability=Ability.NONE, item=Item.NONE,
                nature=Nature.HARDY, moves=[rng.choice(pool), rng.choice(pool)],
            )
            for index, species in enumerate(["Rhydon", "Rhydon"])
        ]

    seen: Counter[str] = Counter()
    for seed in range(40):
        rng = random.Random(37000 + seed)
        team_a = team(rng, pool_a)
        team_b = [
            PokemonSpec(
                species="Machamp", nickname=f"P{index}", level=50, ability=Ability.NONE, item=Item.NONE,
                nature=Nature.HARDY, moves=[rng.choice(pool_b), rng.choice(pool_b)],
            )
            for index in range(2)
        ]
        scenario, expected = record((team_a, team_b), _switching_chooser(rng), seed=seed, max_turns=60)
        theirs = _rust_trace(scenario, tmp_path)
        assert not isinstance(theirs, str), f"seed {seed} was refused: {theirs}"
        divergence = compare(expected, theirs)
        assert divergence is None, f"seed {seed}\n{divergence}"
        for turn in expected:
            for e in turn["events"]:
                if e["type"] in ("Transformed", "MoveFailed"):
                    seen[e["type"]] += 1

    assert seen["Transformed"] > 0, "Transform never once landed across 40 battles"


@needs_rust
def test_priority_abilities_reorder_moves_and_agree(tmp_path: Path) -> None:
    """Prankster, Gale Wings and Triage each add to a move's priority; Mycelium Might instead
    forces a status move to the back of its own bracket. None of the four touch a stat, so a bug
    here would show up as the wrong side moving first turn one, not a wrong number -- which is
    what each matchup is built to expose: the ability-carrying side is far slower by base speed
    alone, so only the ability can put it first.

    Vacuity-checked directly: disabling `priority_bonus`'s three additions turned the first of 20
    seeds in the very first matchup into a tape divergence rather than a quiet pass, and disabling
    the Mycelium Might override on its own did the same to the fourth matchup's first seed.
    """

    def mon(species: str, nickname: str, ability: Ability, moves: list[str]) -> PokemonSpec:
        return PokemonSpec(
            species=species, nickname=nickname, level=50, ability=ability, item=Item.NONE,
            nature=Nature.HARDY, moves=moves,
        )

    # (team_a, team_b, which side the ability should put first)
    matchups = [
        # Prankster: Growl (status, priority 0) outruns Tauros's Tackle from a Pokemon 70 base
        # speed points slower.
        ([mon("Rhydon", "A0", Ability.PRANKSTER, ["Growl"])], [mon("Tauros", "B0", Ability.NONE, ["Tackle"])], 0),
        # Gale Wings: Peck at full HP gets the same +1 -- true on turn one only, since taking a hit
        # spends the "full HP" condition, which is exactly why only turn one is asserted on.
        ([mon("Rhydon", "A0", Ability.GALE_WINGS, ["Peck"])], [mon("Tauros", "B0", Ability.NONE, ["Tackle"])], 0),
        # Triage: Drain Punch's +3 is the largest of the three, and unconditional.
        ([mon("Rhydon", "A0", Ability.TRIAGE, ["Drain Punch"])], [mon("Tauros", "B0", Ability.NONE, ["Tackle"])], 0),
        # Mycelium Might: the reverse direction. Tauros is the faster side by 70 base speed points,
        # but its own status move sorts *last* in the bracket, so Rhydon's ordinary Growl goes
        # first despite being far slower.
        ([mon("Tauros", "A0", Ability.MYCELIUM_MIGHT, ["Growl"])], [mon("Rhydon", "B0", Ability.NONE, ["Growl"])], 1),
    ]

    for index, (team_a, team_b, favoured_side) in enumerate(matchups):
        favoured_side_moved_first = 0
        for seed in range(20):
            rng = random.Random(38000 + index * 100 + seed)
            scenario, expected = record((team_a, team_b), _chooser(rng), seed=seed, max_turns=5)
            theirs = _rust_trace(scenario, tmp_path)
            assert not isinstance(theirs, str), f"matchup {index} seed {seed} was refused: {theirs}"
            divergence = compare(expected, theirs)
            assert divergence is None, f"matchup {index} seed {seed}\n{divergence}"
            movers = [e["side"] for e in expected[0]["events"] if e["type"] == "MoveUsed"]
            if movers and movers[0] == favoured_side:
                favoured_side_moved_first += 1
        assert favoured_side_moved_first > 0, f"matchup {index} never once put side {favoured_side} first"


@needs_rust
def test_surge_surfer_and_unburden_double_speed_and_agree(tmp_path: Path) -> None:
    """The two abilities in `effective_speed` that key off something other than status, weather or
    an item currently held: Surge Surfer off Electric Terrain, Unburden off having *just* lost an
    item. Each matchup is built so the ability-carrying side is slower until its condition kicks
    in, and faster once it does.

    Vacuity-checked directly: disabling both doublers turned the first of 20 seeds red on turn one
    -- Kangaskhan moved first instead of Machamp, a tape divergence rather than a quiet pass.
    """
    # Surge Surfer: Machamp (55 speed) starts behind Kangaskhan (90), but Electric Surge sets the
    # terrain the instant both leads switch in -- before turn zero is even ordered -- so Machamp's
    # doubled 110 already outruns Kangaskhan on the very first turn.
    team_a = [
        PokemonSpec(
            species="Machamp", nickname="A0", level=50, ability=Ability.SURGE_SURFER, item=Item.NONE,
            nature=Nature.HARDY, moves=["Tackle"],
        )
    ]
    team_b = [
        PokemonSpec(
            species="Kangaskhan", nickname="B0", level=50, ability=Ability.ELECTRIC_SURGE, item=Item.NONE,
            nature=Nature.HARDY, moves=["Tackle"],
        )
    ]
    surge_surfer_moved_first = 0
    for seed in range(20):
        rng = random.Random(39000 + seed)
        scenario, expected = record((team_a, team_b), _chooser(rng), seed=seed, max_turns=3)
        theirs = _rust_trace(scenario, tmp_path)
        assert not isinstance(theirs, str), f"seed {seed} was refused: {theirs}"
        divergence = compare(expected, theirs)
        assert divergence is None, f"seed {seed}\n{divergence}"
        movers = [e["side"] for e in expected[0]["events"] if e["type"] == "MoveUsed"]
        if movers and movers[0] == 0:
            surge_surfer_moved_first += 1
    assert surge_surfer_moved_first > 0, "Machamp never once outran Kangaskhan under Electric Terrain"

    # Unburden: Rhydon (40 speed) starts behind Machamp (55) and holding Leftovers. Machamp's Knock
    # Off lands turn one (it is faster), and from turn two on Rhydon's doubled 80 outruns it.
    team_a = [
        PokemonSpec(
            species="Rhydon", nickname="A0", level=50, ability=Ability.UNBURDEN, item=Item.LEFTOVERS,
            nature=Nature.HARDY, moves=["Tackle"],
        )
    ]
    team_b = [
        PokemonSpec(
            species="Machamp", nickname="B0", level=50, ability=Ability.NONE, item=Item.NONE,
            nature=Nature.HARDY, moves=["Knock Off"],
        )
    ]
    unburden_moved_first_after_the_knock_off = 0
    for seed in range(20):
        rng = random.Random(39500 + seed)
        scenario, expected = record((team_a, team_b), _chooser(rng), seed=seed, max_turns=4)
        theirs = _rust_trace(scenario, tmp_path)
        assert not isinstance(theirs, str), f"seed {seed} was refused: {theirs}"
        divergence = compare(expected, theirs)
        assert divergence is None, f"seed {seed}\n{divergence}"
        for turn in expected[1:]:
            movers = [e["side"] for e in turn["events"] if e["type"] == "MoveUsed"]
            if movers and movers[0] == 0:
                unburden_moved_first_after_the_knock_off += 1
    assert unburden_moved_first_after_the_knock_off > 0, "Rhydon never once outran Machamp after being knocked off"


@needs_rust
def test_quick_draw_occasionally_wins_the_bracket_and_agrees(tmp_path: Path) -> None:
    """Quick Draw: a 30% chance, drawn fresh every move, to go first within the priority bracket
    regardless of speed. Rhydon (40 speed) is far slower than Tauros (110), so any turn it moves
    first is the ability, not the stat -- and since the draw is unconditional (Quick Claw and
    Custap Berry share the same function in the Python, ahead of and behind this branch), a
    scenario carrying neither still owes the tape exactly one probability per Quick Draw user per
    move, agreement or not.

    Vacuity-checked directly: keeping the draw but discarding its result turned seed 0 red on turn
    two -- the first turn happened to roll the 70% miss on both sides of a coincidence, the second
    did not.
    """
    team_a = [
        PokemonSpec(
            species="Rhydon", nickname="A0", level=50, ability=Ability.QUICK_DRAW, item=Item.NONE,
            nature=Nature.HARDY, moves=["Tackle"],
        )
    ]
    team_b = [
        PokemonSpec(
            species="Tauros", nickname="B0", level=50, ability=Ability.NONE, item=Item.NONE,
            nature=Nature.HARDY, moves=["Tackle"],
        )
    ]
    rhydon_moved_first = 0
    for seed in range(60):
        rng = random.Random(39900 + seed)
        scenario, expected = record((team_a, team_b), _chooser(rng), seed=seed, max_turns=5)
        theirs = _rust_trace(scenario, tmp_path)
        assert not isinstance(theirs, str), f"seed {seed} was refused: {theirs}"
        divergence = compare(expected, theirs)
        assert divergence is None, f"seed {seed}\n{divergence}"
        for turn in expected:
            movers = [e["side"] for e in turn["events"] if e["type"] == "MoveUsed"]
            if movers and movers[0] == 0:
                rhydon_moved_first += 1
    assert rhydon_moved_first > 0, "Quick Draw never once won the bracket across 60 seeds x 5 turns"


@needs_rust
def test_sturdy_survives_an_otherwise_lethal_hit_from_full_hp(tmp_path: Path) -> None:
    """Sturdy: an OHKO from full HP is clamped to leave exactly one hit point, the same clamp
    Endure already gets but keyed off full HP rather than a volatile. A level 1 Rhydon has too
    little HP for anything else to matter, so a level 50 Machamp's Earthquake -- guaranteed to hit
    and super effective against Ground/Rock -- overkills it every single time regardless of the
    damage roll, leaving Sturdy as the only thing between it and fainting.

    (Fixed-damage moves -- Fissure, Seismic Toss, Super Fang and the rest -- turned out to bypass
    both Sturdy and Endure entirely in this codebase: `_apply_fixed_damage` has no `ON_BEFORE_HIT`
    emit and no `_land_hit` call, unlike an ordinary hit. Not a gap this port introduces, so not
    fixed here, but worth a moment's suspicion for a first test built around Fissure that agreed
    across 60 seeds and asserted zero saves -- a mismeasurement of a real effect, not a bug.)

    Vacuity-checked directly: disabling the clamp turned seed 0 into a real divergence -- Python's
    `SurvivedAtOneHp` where this engine, undisabled, agrees, against a bare `DamageDealt` for 13
    (a fainting blow) once the clamp was skipped.
    """
    team_a = [
        PokemonSpec(
            species="Rhydon", nickname="A0", level=1, ability=Ability.STURDY, item=Item.NONE,
            nature=Nature.HARDY, moves=["Splash"],
        )
    ]
    team_b = [
        PokemonSpec(
            species="Machamp", nickname="B0", level=50, ability=Ability.NONE, item=Item.NONE,
            nature=Nature.HARDY, moves=["Earthquake"],
        )
    ]
    sturdy_saves = 0
    for seed in range(20):
        rng = random.Random(40000 + seed)
        scenario, expected = record((team_a, team_b), _chooser(rng), seed=seed, max_turns=1)
        theirs = _rust_trace(scenario, tmp_path)
        assert not isinstance(theirs, str), f"seed {seed} was refused: {theirs}"
        divergence = compare(expected, theirs)
        assert divergence is None, f"seed {seed}\n{divergence}"
        for turn in expected:
            for e in turn["events"]:
                if e["type"] == "SurvivedAtOneHp" and e["cause"] == "sturdy":
                    sturdy_saves += 1
    assert sturdy_saves > 0, "Sturdy never once saved the level 1 Rhydon across 20 seeds"


def _mon(species: str, nickname: str, ability: Ability, moves: list[str]) -> PokemonSpec:
    return PokemonSpec(
        species=species, nickname=nickname, level=50, ability=ability, item=Item.NONE,
        nature=Nature.HARDY, moves=moves,
    )


@needs_rust
def test_serene_grace_and_shield_dust_tune_secondary_chances(tmp_path: Path) -> None:
    """Rock Smash's Defense-drop secondary is a flat 50% -- Serene Grace on the attacker doubles it
    to a guaranteed drop, and Shield Dust on the defender blocks it outright, no draw at all. Both
    land at the same `tune_stage_secondary` gate, ahead of the substitute check, so a single
    always-hitting move with a coin-flip secondary turns each ability into a deterministic
    assertion rather than a seed sweep.

    Vacuity-checked directly: skipping the tuning (calling `apply_stages` with the raw 50%
    probability instead) turned every one of 10 seeds in the Serene Grace matchup into a tape
    divergence -- Python's guaranteed drop against this engine's coin flip -- and turned the
    Shield Dust matchup's drop-blocking into a divergence the same way once the block was removed.
    """
    boosted = [_mon("Rhydon", "A0", Ability.SERENE_GRACE, ["Rock Smash"])]
    plain_target = [_mon("Tauros", "B0", Ability.NONE, ["Splash"])]
    for seed in range(10):
        rng = random.Random(41000 + seed)
        scenario, expected = record((boosted, plain_target), _chooser(rng), seed=seed, max_turns=1)
        theirs = _rust_trace(scenario, tmp_path)
        assert not isinstance(theirs, str), f"seed {seed} was refused: {theirs}"
        divergence = compare(expected, theirs)
        assert divergence is None, f"seed {seed}\n{divergence}"
        drops = [e for e in expected[0]["events"] if e["type"] == "StatStageChanged" and e["stat"] == "DEFENCE"]
        assert drops, f"seed {seed}: Serene Grace never doubled Rock Smash's drop to a guaranteed one"

    dusted_target = [_mon("Tauros", "B0", Ability.SHIELD_DUST, ["Splash"])]
    for seed in range(10):
        rng = random.Random(41100 + seed)
        scenario, expected = record((boosted, dusted_target), _chooser(rng), seed=seed, max_turns=1)
        theirs = _rust_trace(scenario, tmp_path)
        assert not isinstance(theirs, str), f"seed {seed} was refused: {theirs}"
        divergence = compare(expected, theirs)
        assert divergence is None, f"seed {seed}\n{divergence}"
        drops = [e for e in expected[0]["events"] if e["type"] == "StatStageChanged" and e["stat"] == "DEFENCE"]
        assert not drops, f"seed {seed}: Shield Dust failed to block Rock Smash's secondary"


@needs_rust
def test_skill_link_always_rolls_max_hits_and_agrees(tmp_path: Path) -> None:
    """Skill Link: Bullet Seed's 2-5 hit range always lands at the top, drawn from the tape not at
    all -- the same shape as Loaded Dice's still-unported roll. Bullet Seed always hits, so the hit
    count is the only thing a seed could vary, and Skill Link removes even that.

    Vacuity-checked directly: falling through to the ordinary tape draw turned seed 0 into a tape
    divergence -- Python's unconditional 5 against a rolled 2, since Skill Link takes no draw at
    all in the Python and this engine, undisabled, doesn't either.
    """
    team_a = [_mon("Rhydon", "A0", Ability.SKILL_LINK, ["Bullet Seed"])]
    team_b = [_mon("Tauros", "B0", Ability.NONE, ["Splash"])]
    for seed in range(10):
        rng = random.Random(41200 + seed)
        scenario, expected = record((team_a, team_b), _chooser(rng), seed=seed, max_turns=1)
        theirs = _rust_trace(scenario, tmp_path)
        assert not isinstance(theirs, str), f"seed {seed} was refused: {theirs}"
        divergence = compare(expected, theirs)
        assert divergence is None, f"seed {seed}\n{divergence}"
        summary = next(e for e in expected[0]["events"] if e["type"] == "MultiHitSummary")
        assert summary["hits"] == 5, f"seed {seed}: Skill Link rolled {summary['hits']} hits instead of 5"


@needs_rust
def test_scrappy_and_minds_eye_hit_ghosts_with_normal_and_fighting(tmp_path: Path) -> None:
    """Scrappy and Mind's Eye: a Normal or Fighting move from either ability bypasses a Ghost
    type's usual immunity to both. Gengar is pure enough (Ghost/Poison) that a landed Tackle is
    unambiguous, and Tackle always hits, so there is nothing for a seed to vary -- the control case
    (no ability) is the same matchup asserting the opposite, so the effect is measured, not assumed.

    Vacuity-checked directly: dropping the Ghost union from `effective_bypass` turned every one of
    10 seeds in both ability matchups into a tape divergence -- this engine's NoEffect against
    Python's landed hit.
    """
    target = [_mon("Gengar", "B0", Ability.NONE, ["Splash"])]
    for ability in (Ability.SCRAPPY, Ability.MINDS_EYE):
        team_a = [_mon("Rhydon", "A0", ability, ["Tackle"])]
        for seed in range(10):
            rng = random.Random(41300 + seed)
            scenario, expected = record((team_a, target), _chooser(rng), seed=seed, max_turns=1)
            theirs = _rust_trace(scenario, tmp_path)
            assert not isinstance(theirs, str), f"{ability} seed {seed} was refused: {theirs}"
            divergence = compare(expected, theirs)
            assert divergence is None, f"{ability} seed {seed}\n{divergence}"
            events = expected[0]["events"]
            assert any(e["type"] == "DamageDealt" for e in events), f"{ability} seed {seed}: Tackle never landed"
            assert not any(e["type"] == "NoEffect" for e in events), f"{ability} seed {seed}: Tackle was refused"

    # Control: the same matchup with no ability at all is the immunity the other two bypass.
    plain_team_a = [_mon("Rhydon", "A0", Ability.NONE, ["Tackle"])]
    scenario, expected = record((plain_team_a, target), _chooser(random.Random(0)), seed=0, max_turns=1)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"control was refused: {theirs}"
    divergence = compare(expected, theirs)
    assert divergence is None, f"control\n{divergence}"
    events = expected[0]["events"]
    assert any(e["type"] == "NoEffect" for e in events), "control: Tackle should have no effect on Gengar"


def _round_robin_chooser():
    """Cycles a Pokemon through its four move slots in order, side by side independently.

    A moveset shorter than four moves is padded by repeating the first one into every empty slot
    (see the charge-move redirect notes elsewhere in this file), and each padded slot carries its
    *own* PP pool rather than sharing one -- a random chooser mostly just rotates between four full
    pools instead of ever emptying one. Round-robining deliberately, in a fixed order, empties all
    four in lockstep instead, which is the only way to force Struggle on a demand.
    """
    slots = [MoveSlot.FIRST, MoveSlot.SECOND, MoveSlot.THIRD, MoveSlot.FOURTH]
    counters = {0: 0, 1: 0}

    def choose(state, side_index):  # type: ignore[no-untyped-def]
        slot = slots[counters[side_index] % 4]
        counters[side_index] += 1
        return Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=slot)

    return choose


@needs_rust
def test_pressure_doubles_the_pp_cost_of_a_move_that_faces_it(tmp_path: Path) -> None:
    """Pressure: a move that faces its holder spends 2 PP instead of 1. Cross Chop's 5 PP, spent
    round-robin across the four padded copies of it Rhydon's one-move set carries, reaches 0 in
    all four after 3 rounds (12 turns: 5, 3, 1 -> 0) under Pressure rather than 5 rounds (20 turns:
    5, 4, 3, 2, 1 -> 0), so the 13th turn is forced into Struggle. Gengar is Ghost/Poison, immune
    to Fighting, so it takes no damage across the run and the immunity gate short-circuits before
    any accuracy roll -- the whole scenario is deterministic, no seed needed.

    Vacuity-checked directly: dropping the cost back to a flat 1 turned turn 13 into a divergence
    -- this engine kept swinging Cross Chop a turn after Python's Pressure-aware engine, at 5 uses
    per slot instead of 3, had forced Struggle nowhere near yet.
    """
    team_a = [_mon("Rhydon", "A0", Ability.NONE, ["Cross Chop"])]
    team_b = [_mon("Gengar", "B0", Ability.PRESSURE, ["Splash"])]
    scenario, expected = record((team_a, team_b), _round_robin_chooser(), seed=0, max_turns=13)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"was refused: {theirs}"
    divergence = compare(expected, theirs)
    assert divergence is None, f"{divergence}"
    moves_used = [e["move"] for turn in expected for e in turn["events"] if e["type"] == "MoveUsed" and e["side"] == 0]
    assert moves_used == ["Cross Chop"] * 12 + ["Struggle"], moves_used


@needs_rust
def test_steadfast_gains_speed_from_flinching(tmp_path: Path) -> None:
    """Steadfast: any flinch, not just an ability's own retaliation, raises its holder's Speed by
    one stage the instant the flinch volatile lands. Fake Out is a guaranteed hit with a guaranteed
    flinch, so every seed sees exactly one.

    Vacuity-checked directly: removing the stage-change call at the flinch site turned seed 0 into
    a plain digest mismatch -- Python's SPEED +1 that this engine, disabled, never sent.
    """
    team_a = [_mon("Rhydon", "A0", Ability.NONE, ["Fake Out"])]
    team_b = [_mon("Tauros", "B0", Ability.STEADFAST, ["Splash"])]
    for seed in range(10):
        rng = random.Random(41500 + seed)
        scenario, expected = record((team_a, team_b), _chooser(rng), seed=seed, max_turns=1)
        theirs = _rust_trace(scenario, tmp_path)
        assert not isinstance(theirs, str), f"seed {seed} was refused: {theirs}"
        divergence = compare(expected, theirs)
        assert divergence is None, f"seed {seed}\n{divergence}"
        boosts = [e for e in expected[0]["events"] if e["type"] == "StatStageChanged" and e["source"] == "steadfast"]
        assert boosts, f"seed {seed}: Steadfast never raised Speed after flinching"


@needs_rust
def test_synchronize_mirrors_a_status_back_onto_its_inflictor(tmp_path: Path) -> None:
    """Synchronize: a burn/paralysis/poison/toxic landing on its holder reflects straight back onto
    whoever inflicted it, but only when the inflictor isn't already statused, and without
    re-triggering anything on the way back (the reflected copy is applied with no inflictor of its
    own). Thunder Wave is the simplest single-target status to force, at 90% accuracy, so this is a
    seed sweep rather than a single deterministic case.

    Vacuity-checked directly: dropping the reflect call left seed 0's attacker unstatused where
    Python's inflictor also ends up paralysed -- a plain digest mismatch, not a refusal.
    """
    team_a = [_mon("Tauros", "A0", Ability.NONE, ["Thunder Wave"])]
    team_b = [_mon("Machamp", "B0", Ability.SYNCHRONIZE, ["Splash"])]
    reflected = 0
    for seed in range(20):
        rng = random.Random(41600 + seed)
        scenario, expected = record((team_a, team_b), _chooser(rng), seed=seed, max_turns=1)
        theirs = _rust_trace(scenario, tmp_path)
        assert not isinstance(theirs, str), f"seed {seed} was refused: {theirs}"
        divergence = compare(expected, theirs)
        assert divergence is None, f"seed {seed}\n{divergence}"
        statuses = [e for e in expected[0]["events"] if e["type"] == "StatusInflicted"]
        if {e["side"] for e in statuses} == {0, 1}:
            reflected += 1
    assert reflected > 0, "Synchronize never once reflected paralysis back onto its inflictor across 20 seeds"


@needs_rust
def test_corrosion_lets_a_poison_status_through_a_steel_type(tmp_path: Path) -> None:
    """Corrosion: the *inflictor's* ability, not the target's, is what lets Toxic poison a Steel
    type that would otherwise be flatly immune -- the only exception `_STATUS_TYPE_IMMUNITY` has.
    Steelix (Steel/Ground) has no poison-immunity clause of its own beyond the Steel half, so a
    landed Toxic here is unambiguously Corrosion's doing. Toxic is 90% accurate, so this is a seed
    sweep.

    Vacuity-checked directly: keeping the type-immunity check unconditional turned seed 0's Toxic
    into a silent no-op where Python's Corrosion-aware engine lands it -- a digest mismatch on the
    first status draw.
    """
    team_a = [_mon("Tauros", "A0", Ability.CORROSION, ["Toxic"])]
    team_b = [_mon("Steelix", "B0", Ability.NONE, ["Splash"])]
    poisoned = 0
    for seed in range(20):
        rng = random.Random(41700 + seed)
        scenario, expected = record((team_a, team_b), _chooser(rng), seed=seed, max_turns=1)
        theirs = _rust_trace(scenario, tmp_path)
        assert not isinstance(theirs, str), f"seed {seed} was refused: {theirs}"
        divergence = compare(expected, theirs)
        assert divergence is None, f"seed {seed}\n{divergence}"
        if any(e["type"] == "StatusInflicted" and e["status"] == "TOXIC" for e in expected[0]["events"]):
            poisoned += 1
    assert poisoned > 0, "Corrosion never once poisoned the Steel type across 20 seeds"


def _damage_dealt(events: list[dict]) -> int:
    return next(e["amount"] for e in events if e["type"] == "DamageDealt")


@needs_rust
def test_water_bubble_doubles_its_own_water_and_halves_fire_taken(tmp_path: Path) -> None:
    """Water Bubble is a pure `ON_DAMAGE_CALC` ability like Blaze or Heatproof -- no new hook
    needed, just two more match arms in the existing dispatch -- plus burn immunity
    (`_STATUS_ABILITY_IMMUNITY`), which was already written into `ability_blocks_status` for
    Water Veil's sake and had simply never been reachable because Water Bubble itself wasn't on
    a `PORTED` list yet.

    Same seed, same move, only the ability differs, so any damage difference is the modifier and
    nothing else: Water Gun always hits with no secondary, so the two engines' shared tape draws
    the same crit/damage roll in the boosted and unboosted battles alike.

    Vacuity-checked directly: dropping the two new match arms out of `abilities::handle` turned
    both the offense and defense matchups into a plain digest mismatch (this engine's un-doubled
    or un-halved `DamageDealt` amount against Python's), and dropping "WATER_BUBBLE" back out of
    `ability_blocks_status`'s reach (by leaving it off the `PORTED` array) turned every one of 20
    burn-immunity seeds into a divergence the moment a burn actually landed in Python and this
    engine refused the whole scenario as carrying a live, unported ability instead.
    """
    bubbled = [_mon("Tauros", "A0", Ability.WATER_BUBBLE, ["Water Gun"])]
    plain_attacker = [_mon("Tauros", "A0", Ability.NONE, ["Water Gun"])]
    target = [_mon("Rhydon", "B0", Ability.NONE, ["Splash"])]

    scenario, boosted = record((bubbled, target), _chooser(random.Random(0)), seed=0, max_turns=1)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"boosted case was refused: {theirs}"
    assert compare(boosted, theirs) is None
    scenario, plain = record((plain_attacker, target), _chooser(random.Random(0)), seed=0, max_turns=1)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"control case was refused: {theirs}"
    assert compare(plain, theirs) is None
    boosted_amount = _damage_dealt(boosted[0]["events"])
    plain_amount = _damage_dealt(plain[0]["events"])
    assert boosted_amount > plain_amount * 1.5, f"{boosted_amount} was not roughly double {plain_amount}"

    attacker = [_mon("Rhydon", "A0", Ability.NONE, ["Flamethrower"])]
    bubbled_defender = [_mon("Tauros", "B0", Ability.WATER_BUBBLE, ["Splash"])]
    plain_defender = [_mon("Tauros", "B0", Ability.NONE, ["Splash"])]

    scenario, reduced = record((attacker, bubbled_defender), _chooser(random.Random(1)), seed=0, max_turns=1)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"reduced case was refused: {theirs}"
    assert compare(reduced, theirs) is None
    scenario, full = record((attacker, plain_defender), _chooser(random.Random(1)), seed=0, max_turns=1)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"control case was refused: {theirs}"
    assert compare(full, theirs) is None
    reduced_amount = _damage_dealt(reduced[0]["events"])
    full_amount = _damage_dealt(full[0]["events"])
    assert reduced_amount < full_amount * 0.6, f"{reduced_amount} was not roughly half {full_amount}"

    burned, burn_landed_at_all = 0, 0
    for seed in range(20):
        rng = random.Random(41800 + seed)
        scenario, expected = record((attacker, bubbled_defender), _chooser(rng), seed=seed, max_turns=1)
        theirs = _rust_trace(scenario, tmp_path)
        assert not isinstance(theirs, str), f"seed {seed} was refused: {theirs}"
        assert compare(expected, theirs) is None, f"seed {seed}"
        if any(e["type"] == "StatusInflicted" and e["status"] == "BURN" for e in expected[0]["events"]):
            burned += 1
        rng = random.Random(41900 + seed)
        scenario, expected = record((attacker, plain_defender), _chooser(rng), seed=seed, max_turns=1)
        theirs = _rust_trace(scenario, tmp_path)
        assert not isinstance(theirs, str), f"control seed {seed} was refused: {theirs}"
        assert compare(expected, theirs) is None, f"control seed {seed}"
        if any(e["type"] == "StatusInflicted" and e["status"] == "BURN" for e in expected[0]["events"]):
            burn_landed_at_all += 1
    assert burned == 0, f"Water Bubble should be flatly immune to burn, but caught fire {burned} times"
    assert burn_landed_at_all > 0, "control never burned once in 20 seeds -- the test isn't exercising the immunity"


@needs_rust
def test_type_absorbing_abilities_cancel_the_move_and_agree(tmp_path: Path) -> None:
    """The first eight names in `_TYPE_ABSORBING_ABILITIES`: a move of the matching type never
    lands at all -- no damage, no secondary, no accuracy draw needed since the type check alone
    decides it -- and either heals a quarter HP (Volt Absorb, Water Absorb, Earth Eater) or bumps a
    stat by a fixed amount (Motor Drive, Lightning Rod, Storm Drain, Sap Sipper, Well Baked Body).
    Each heal-style ability is checked twice: once with prior damage taken (heals), once at full HP
    (`AbsorbBlocked`, no heal) -- the same "measure the block, don't assume it" the batch above used
    for Shield Dust.

    This is also the cluster that caught a real bug during development: `ability_before_move` fired
    for Electrify (an Electric-type move whose own effect is unmodelled — `effects` is empty in the
    exported data) before this test was even written, because nothing checked for an empty effect
    list the way Python's own `if not move.effects and not move.force_switch: MoveFailed` does one
    line above where it emits `ON_BEFORE_MOVE`. Caught by the pre-existing
    `test_the_ported_abilities_agree_too` sweep, not by a test aimed at this cluster specifically —
    see `test_effectless_moves_never_falsely_trigger_an_absorber` below for the regression test that
    followed.

    Vacuity-checked directly: each of the eight, disabled on its own by renaming its match arm in
    `hooks::ability_before_move`, turned its own case here into a plain digest mismatch (the move
    landing and dealing damage in this engine where Python absorbed it) or, for Earthquake against
    Earth Eater, a `NoEffect`/`DamageDealt` split.
    """
    heal_style = [
        ("VOLT_ABSORB", "Thunder Shock", "Machamp"),
        ("WATER_ABSORB", "Water Gun", "Machamp"),
        ("EARTH_EATER", "Earthquake", "Machamp"),
    ]
    for index, (ability, move, species) in enumerate(heal_style):
        # Tackle first, to chip HP with a Normal hit the ability has no opinion about, then the
        # real move on turn 2 -- round-robin so slot FIRST (Tackle) goes before slot SECOND (the
        # absorbed move), not a random one of the four padded copies of either.
        attacker = [_mon("Rhydon", "A0", Ability.NONE, ["Tackle", move])]
        damaged_defender = [_mon(species, "B0", Ability[ability], ["Splash"])]
        scenario, expected = record((attacker, damaged_defender), _round_robin_chooser(), seed=0, max_turns=2)
        theirs = _rust_trace(scenario, tmp_path)
        assert not isinstance(theirs, str), f"{ability} was refused: {theirs}"
        assert compare(expected, theirs) is None, f"{ability}"
        chip_turn, absorb_turn = expected[0]["events"], expected[1]["events"]
        assert any(e["type"] == "DamageDealt" for e in chip_turn), f"{ability}: Tackle never chipped it\n{chip_turn}"
        assert any(
            e["type"] == "AbsorbHealed" and e["ability"] == ability for e in absorb_turn
        ), f"{ability}: never healed off a hit it should have absorbed\n{absorb_turn}"
        assert not any(
            e["type"] == "DamageDealt" for e in absorb_turn
        ), f"{ability}: the absorbed move should never have landed\n{absorb_turn}"

        full_hp_attacker = [_mon("Rhydon", "A0", Ability.NONE, [move])]
        full_hp_defender = [_mon(species, "B0", Ability[ability], ["Splash"])]
        scenario, expected = record(
            (full_hp_attacker, full_hp_defender), _chooser(random.Random(50000 + index)), seed=0, max_turns=1
        )
        theirs = _rust_trace(scenario, tmp_path)
        assert not isinstance(theirs, str), f"{ability} (full HP) was refused: {theirs}"
        assert compare(expected, theirs) is None, f"{ability} (full HP)"
        events = expected[0]["events"]
        assert any(
            e["type"] == "AbsorbBlocked" and e["ability"] == ability for e in events
        ), f"{ability}: should report AbsorbBlocked at full HP\n{events}"

    boost_style = [
        ("MOTOR_DRIVE", "Thunder Shock", "SPEED", "motor_drive"),
        ("LIGHTNING_ROD", "Thunder Shock", "SP_ATTACK", "lightning_rod"),
        ("STORM_DRAIN", "Water Gun", "SP_ATTACK", "storm_drain"),
        ("SAP_SIPPER", "Vine Whip", "ATTACK", "sap_sipper"),
        ("WELL_BAKED_BODY", "Ember", "DEFENCE", "well_baked_body"),
    ]
    for index, (ability, move, stat, source) in enumerate(boost_style):
        attacker = [_mon("Rhydon", "A0", Ability.NONE, [move])]
        defender = [_mon("Tauros", "B0", Ability[ability], ["Splash"])]
        scenario, expected = record((attacker, defender), _chooser(random.Random(50100 + index)), seed=0, max_turns=1)
        theirs = _rust_trace(scenario, tmp_path)
        assert not isinstance(theirs, str), f"{ability} was refused: {theirs}"
        assert compare(expected, theirs) is None, f"{ability}"
        events = expected[0]["events"]
        boosts = [e for e in events if e["type"] == "StatStageChanged" and e["source"] == source]
        assert boosts and boosts[0]["stat"] == stat, f"{ability}: never boosted {stat} via {source}\n{events}"
        assert not any(e["type"] == "DamageDealt" for e in events), f"{ability}: the move should never have landed"


@needs_rust
def test_effectless_moves_never_falsely_trigger_an_absorber(tmp_path: Path) -> None:
    """The regression test for the bug `test_type_absorbing_abilities_cancel_the_move_and_agree`'s
    own docstring describes: Electrify is an Electric-type status move whose effect is entirely
    unmodelled (`effects` is empty in the exported data), so Python's own `if not move.effects and
    not move.force_switch` sends it to `MoveFailed` a line before `ON_BEFORE_MOVE` is ever emitted.
    A Motor Drive Pokemon on the receiving end must see the same `MoveFailed` this engine's own
    generic empty-effects fallback already produces, not a false Speed boost.

    Vacuity-checked directly: removing the `effects.is_empty()` guard from `ability_before_move`
    turned this into a digest mismatch -- a `StatStageChanged` in this engine where Python's
    `MoveFailed` has nothing left to trigger it.
    """
    team_a = [_mon("Rhydon", "A0", Ability.NONE, ["Electrify"])]
    team_b = [_mon("Tauros", "B0", Ability.MOTOR_DRIVE, ["Splash"])]
    scenario, expected = record((team_a, team_b), _chooser(random.Random(0)), seed=0, max_turns=1)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"was refused: {theirs}"
    assert compare(expected, theirs) is None
    events = expected[0]["events"]
    assert any(e["type"] == "MoveFailed" for e in events), events
    assert not any(e["type"] == "StatStageChanged" for e in events), events


@needs_rust
def test_flash_fire_activates_once_and_boosts_fire_moves_afterward(tmp_path: Path) -> None:
    """Flash Fire: the first Fire move absorbed sets a flag rather than healing or boosting a stat
    (`FlashFireActivated`), a second one while it's already set is a no-op beyond announcing it
    (`FlashFireAbsorbed`), and every Fire move *this Pokemon then uses itself* is boosted 1.5x for
    as long as the flag stays set -- cleared only on switch-out. The two halves live in different
    files (`hooks::ability_before_move` sets the flag, `abilities::handle` reads it), so this is the
    one absorber in the batch that needs a whole battle rather than one turn to prove both ends.

    The boost itself found a real bug during development, independently of this test: it was first
    written pushing 6144 onto `attack_mods_4096` (the same list Huge Power and Water Bubble use),
    which reads like the obvious place for a "1.5x this Pokemon's own damage" effect. The Python's
    own `boost_fire` pushes onto `pre_screen_mods_4096` instead, which folds in much later -- after
    STAB and the type multiplier, not onto the attack stat before the base-damage division -- and
    the two are not interchangeable: the 5000-battle plain-slice sweep this batch was verified
    against caught a Mind Blown landing for 36 in Python and 35 here, one point of rounding apart,
    because a Rock/Ground Golem's boosted Special Attack rounds differently depending on which
    stage of the chain the 1.5x enters at. Fixed by moving the push to `pre_screen_mods_4096`.

    Vacuity-checked directly, against the corrected code: renaming `FLASH_FIRE`'s match arm in
    `hooks::ability_before_move` (the activation guard) turned turn 1 into a tape divergence: this
    engine took Ember as a landed hit where the Python absorbed it, so the two disagreed about how
    many draws the move even needed. Turning the boost's own `pre_screen_mods_4096.push(6144)` into
    a no-op multiplier (4096, i.e. 1x) turned turn 3 into a plain digest mismatch instead -- the
    boosted `DamageDealt` this test asserts against a flat, unboosted one.
    """
    # Scripted rather than round-robin: Gengar (A0) has to sit still on Splash for the first two
    # turns to be the one *receiving* Rhydon's Ember, then switch to dishing its own out boosted --
    # a plain round-robin would have both sides attacking every turn and no way to tell "the
    # incoming hit was absorbed" apart from "the outgoing hit happened to also land" in the same
    # turn's event list.
    def scripted_both_sides(sequence: list[MoveSlot]):  # type: ignore[no-untyped-def]
        counters = {0: 0, 1: 0}

        def choose(state, side_index):  # type: ignore[no-untyped-def]
            slot = sequence[counters[side_index]]
            counters[side_index] += 1
            return Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=slot)

        return choose

    team_b = [_mon("Rhydon", "B0", Ability.NONE, ["Ember", "Splash"])]

    def play(a_ability):  # type: ignore[no-untyped-def]
        a = [_mon("Gengar", "A0", a_ability, ["Splash", "Ember"])]
        sequence = [MoveSlot.FIRST, MoveSlot.FIRST, MoveSlot.SECOND]
        scenario, expected = record((a, team_b), scripted_both_sides(sequence), seed=0, max_turns=3)
        theirs = _rust_trace(scenario, tmp_path)
        assert not isinstance(theirs, str), f"was refused: {theirs}"
        assert compare(expected, theirs) is None
        return expected

    expected = play(Ability.FLASH_FIRE)
    turn1, turn2, turn3 = expected[0]["events"], expected[1]["events"], expected[2]["events"]
    assert any(e["type"] == "FlashFireActivated" for e in turn1), turn1
    assert not any(e["type"] == "DamageDealt" and e["side"] == 0 for e in turn1), turn1
    assert any(e["type"] == "FlashFireAbsorbed" for e in turn2), turn2
    assert not any(e["type"] == "DamageDealt" and e["side"] == 0 for e in turn2), turn2
    boosted_amount = _damage_dealt(turn3)

    reference = play(Ability.NONE)
    reference_amount = _damage_dealt(reference[2]["events"])
    assert boosted_amount > reference_amount * 1.3, f"{boosted_amount} was not boosted over {reference_amount}"


@needs_rust
def test_levitate_cancels_ground_moves_and_soundproof_blocks_sound(tmp_path: Path) -> None:
    """Levitate's move-cancelling half (its grounding half, `field::is_grounded`, was already
    written and simply unreachable until "LEVITATE" joined a `PORTED` array here) and Soundproof,
    which reads the move's own `sound` flag rather than its type. Earthquake and Boomburst are both
    always-hit, so both cases are deterministic.

    Vacuity-checked directly: removing either match arm from `hooks::ability_before_move` turned
    its own case into a plain digest mismatch -- a landed `DamageDealt` here against Python's
    `AvoidedWithLevitate` or `DoesNotAffect`.
    """
    grounded_attacker = [_mon("Machamp", "A0", Ability.NONE, ["Earthquake"])]
    levitator = [_mon("Gengar", "B0", Ability.LEVITATE, ["Splash"])]
    scenario, expected = record((grounded_attacker, levitator), _chooser(random.Random(0)), seed=0, max_turns=1)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"Levitate case was refused: {theirs}"
    assert compare(expected, theirs) is None
    events = expected[0]["events"]
    assert any(e["type"] == "AvoidedWithLevitate" for e in events), events
    assert not any(e["type"] == "DamageDealt" for e in events), events

    loud_attacker = [_mon("Machamp", "A0", Ability.NONE, ["Boomburst"])]
    deaf_defender = [_mon("Rhydon", "B0", Ability.SOUNDPROOF, ["Splash"])]
    scenario, expected = record((loud_attacker, deaf_defender), _chooser(random.Random(0)), seed=0, max_turns=1)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"Soundproof case was refused: {theirs}"
    assert compare(expected, theirs) is None
    events = expected[0]["events"]
    assert any(e["type"] == "DoesNotAffect" for e in events), events
    assert not any(e["type"] == "DamageDealt" for e in events), events


@needs_rust
def test_moxie_beast_boost_and_soul_heart_boost_after_a_ko(tmp_path: Path) -> None:
    """Moxie, Beast Boost and Soul Heart all bind `ON_FAINT` -- emitted from exactly one place in
    the Python (`_apply_damage`, right after the defender's own `Fainted` line, before Destiny
    Bond's retaliation and before recoil/drain), so each fires only when *this* Pokemon's own
    ordinary damaging hit is what did the fainting. A level 1 target dies to anything, so the KO
    itself is deterministic and needs no seed.

    Beast Boost boosts whichever of its five stats is highest, not always Attack -- Golem's Defence
    (130) outranks its Attack (120), so a Golem is the only way to tell "picks the right stat" apart
    from "always boosts Attack".

    Vacuity-checked directly: renaming each ability's own arm in `hooks::ability_on_faint` turned
    its own case into a plain digest mismatch -- this engine's `MoveFailed`-shaped silence (no boost
    logged) where Python still raises the stat.
    """
    weak_target = PokemonSpec(
        species="Machamp", nickname="B0", level=1, ability=Ability.NONE, item=Item.NONE,
        nature=Nature.HARDY, moves=["Splash"],
    )
    cases = [
        (Ability.MOXIE, "Tauros", "Tackle", "ATTACK", "moxie"),
        (Ability.SOUL_HEART, "Tauros", "Tackle", "SP_ATTACK", "soul_heart"),
        (Ability.BEAST_BOOST, "Golem", "Earthquake", "DEFENCE", "beast_boost"),
    ]
    for ability, species, move, stat, source in cases:
        attacker = [_mon(species, "A0", ability, [move])]
        scenario, expected = record((attacker, [weak_target]), _chooser(random.Random(0)), seed=0, max_turns=1)
        theirs = _rust_trace(scenario, tmp_path)
        assert not isinstance(theirs, str), f"{ability} was refused: {theirs}"
        assert compare(expected, theirs) is None, f"{ability}"
        events = expected[0]["events"]
        assert any(e["type"] == "Fainted" and e["side"] == 1 for e in events), f"{ability}: never fainted\n{events}"
        boosts = [e for e in events if e["type"] == "StatStageChanged" and e["source"] == source]
        assert boosts and boosts[0]["stat"] == stat, f"{ability}: never boosted {stat} via {source}\n{events}"


@needs_rust
def test_ko_boosting_abilities_dont_fire_from_a_fixed_damage_faint(tmp_path: Path) -> None:
    """`ON_FAINT` is not "whenever a Pokemon reaches zero HP" -- it is one specific line inside
    `_apply_damage`, the ordinary variable/listed-power path. `_apply_fixed_damage` (Seismic Toss's
    own path) has no such emit, so a fixed-damage KO must not raise Moxie's Attack even though the
    target is just as dead. Same setup as the test above with only the move swapped, which is what
    makes the missing boost meaningful rather than assumed.

    Vacuity-checked directly: adding a second call to `hooks::ability_on_faint` from
    `apply_fixed_damage`'s own fainting check -- the natural mistake this hook invites, matching the
    port plan's own now-corrected claim that it should fire "at every existing `fainted()` check
    site" -- turned this into a digest mismatch: this engine's own `StatStageChanged` where Python's
    fixed-damage path has nothing to trigger it.
    """
    weak_target = PokemonSpec(
        species="Machamp", nickname="B0", level=1, ability=Ability.NONE, item=Item.NONE,
        nature=Nature.HARDY, moves=["Splash"],
    )
    attacker = [_mon("Tauros", "A0", Ability.MOXIE, ["Seismic Toss"])]
    scenario, expected = record((attacker, [weak_target]), _chooser(random.Random(0)), seed=0, max_turns=1)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"was refused: {theirs}"
    assert compare(expected, theirs) is None
    events = expected[0]["events"]
    assert any(e["type"] == "Fainted" and e["side"] == 1 for e in events), events
    assert not any(e["type"] == "StatStageChanged" for e in events), events


def _switch_on_turn(turn_to_switch: int, target_index: int):  # type: ignore[no-untyped-def]
    """Deterministic: the first available move every turn, except on `turn_to_switch`, where side 0
    switches to `target_index` -- an explicit `SWITCH_OUT` action, not a chooser hoping to land on
    one, since `ON_SWITCH_OUT` needs a voluntary switch to fire at all."""

    def choose(state, side_index):  # type: ignore[no-untyped-def]
        if side_index == 0 and state.turn == turn_to_switch:
            return Action(action=ActionType.SWITCH_OUT, switch_in=state.sides[0].team[target_index])
        options = [a for a in legal_actions(state, side_index) if a.action is ActionType.USE_MOVE]
        return (options or legal_actions(state, side_index))[0]

    return choose


@needs_rust
def test_regenerator_heals_a_third_on_switch_out(tmp_path: Path) -> None:
    """Regenerator: emitted from `ON_SWITCH_OUT`'s one site (`turn::switch_out`, before any of
    `withdraw`'s own resets), so it sees the outgoing Pokemon's HP exactly as it stood mid-battle,
    not reset to anything. Tackle is always-hit and never lethal against a bulky Golem, so the chip
    to heal back is deterministic.

    Vacuity-checked directly: renaming `"REGENERATOR"` in `hooks::ability_on_switch_out` turned this
    into a plain digest mismatch -- this engine's own un-healed HP where Python's `AbilityHealed`
    already restored a third.
    """
    team_a = [
        _mon("Golem", "A0", Ability.REGENERATOR, ["Splash"]),
        _mon("Rhydon", "A1", Ability.NONE, ["Splash"]),
    ]
    team_b = [_mon("Machamp", "B0", Ability.NONE, ["Tackle"])]
    scenario, expected = record((team_a, team_b), _switch_on_turn(1, 1), seed=0, max_turns=2)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"was refused: {theirs}"
    assert compare(expected, theirs) is None
    switch_turn_events = expected[1]["events"]
    heals = [e for e in switch_turn_events if e["type"] == "AbilityHealed" and e["ability"] == "REGENERATOR"]
    assert heals and heals[0]["amount"] > 0, switch_turn_events


@needs_rust
def test_natural_cure_clears_status_on_switch_out(tmp_path: Path) -> None:
    """Natural Cure: the same `ON_SWITCH_OUT` site, clearing whatever status the outgoing Pokemon
    is carrying. Thunder Wave is 90% accurate, so this is a seed sweep rather than a single
    deterministic case.

    Vacuity-checked directly: renaming `"NATURAL_CURE"` in `hooks::ability_on_switch_out` turned
    every seed where paralysis actually landed into a plain digest mismatch -- this engine's own
    still-paralyzed status where Python's `StatusCleared` already reset it.
    """
    team_a = [
        _mon("Golem", "A0", Ability.NATURAL_CURE, ["Splash"]),
        _mon("Rhydon", "A1", Ability.NONE, ["Splash"]),
    ]
    team_b = [_mon("Machamp", "B0", Ability.NONE, ["Thunder Wave"])]
    cured = 0
    for seed in range(20):
        scenario, expected = record((team_a, team_b), _switch_on_turn(1, 1), seed=seed, max_turns=2)
        theirs = _rust_trace(scenario, tmp_path)
        assert not isinstance(theirs, str), f"seed {seed} was refused: {theirs}"
        assert compare(expected, theirs) is None, f"seed {seed}"
        switch_turn_events = expected[1]["events"]
        if any(e["type"] == "StatusCleared" and e["clearance"] == "natural_cure" for e in switch_turn_events):
            cured += 1
    assert cured > 0, "Natural Cure never once cleared a status across 20 seeds"


@needs_rust
def test_regenerator_and_natural_cure_dont_fire_on_a_fainted_switch(tmp_path: Path) -> None:
    """`ON_SWITCH_OUT` is not emitted for a fainted switch -- the Python's own comment on the event
    says so directly ("not emitted for fainted switches"), and `turn::switch_out` guards it the same
    way. A Regenerator holder that faints and is auto-replaced must not "heal" its own corpse back to
    positive HP; a level 1 Golem takes Earthquake's overkill just as reliably as a real KO would.

    Vacuity-checked directly: dropping the `!fainted()` guard in `switch_out` before calling
    `hooks::ability_on_switch_out` turned this into a digest mismatch -- this engine's own healed,
    still-standing Golem where Python's `Fainted` line is the last thing said about it.
    """
    team_a = [
        PokemonSpec(
            species="Golem", nickname="A0", level=1, ability=Ability.REGENERATOR, item=Item.NONE,
            nature=Nature.HARDY, moves=["Splash"],
        ),
        _mon("Rhydon", "A1", Ability.NONE, ["Splash"]),
    ]
    team_b = [_mon("Machamp", "B0", Ability.NONE, ["Earthquake"])]
    # Two turns: the faint lands on the first, and the auto-replacement -- the switch this test is
    # actually about -- is the first thing the second turn resolves, not something the same turn's
    # own event list ever shows.
    scenario, expected = record((team_a, team_b), _chooser(random.Random(0)), seed=0, max_turns=2)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"was refused: {theirs}"
    assert compare(expected, theirs) is None
    faint_turn, replacement_turn = expected[0]["events"], expected[1]["events"]
    assert any(e["type"] == "Fainted" and e["side"] == 0 for e in faint_turn), faint_turn
    assert any(e["type"] == "Switched" and e["side"] == 0 for e in replacement_turn), replacement_turn
    assert not any(e["type"] == "AbilityHealed" for e in replacement_turn), replacement_turn
    a0_digest = expected[1]["state"]["sides"][0]["team"][0]
    assert a0_digest["hp"] == 0, a0_digest


@needs_rust
def test_every_ported_ability_and_item_reaches_a_battle() -> None:
    """A guard against testing nothing.

    The ability tests would pass just as happily if the generator never picked half these names, so
    this asserts the draw actually reaches all of them. It is the cheap half of the check; the
    expensive half is that turning the ability dispatch off makes 187 of 200 swept battles diverge.
    """
    rng = random.Random(0)
    generated = [spec for _ in range(400) for spec in _team(rng, size=3, abilities=True, items=True)]

    missing = set(PORTED["abilities"]) - {spec.ability.name for spec in generated}
    missing |= set(PORTED["items"]) - {spec.item.name for spec in generated}
    assert not missing, f"never generated: {sorted(missing)}"


@needs_rust
@pytest.mark.parametrize("carries", ["ability", "item"])
def test_live_abilities_and_items_are_refused_rather_than_ignored(carries: str, tmp_path: Path) -> None:
    """Reading an ability off a Pokemon and doing nothing with it is a wrong answer in silence.

    Both of these are wired to the Python's event bus, so a battle containing one is not comparable
    until the Rust engine implements it. The engine has to say so.
    """
    kind, enum = ("abilities", Ability) if carries == "ability" else ("items", Item)
    expected = _still_unported(kind, enum)
    rng = random.Random(3)
    team = [
        PokemonSpec(
            species="Rhydon",
            nickname="P0",
            level=50,
            ability=Ability[expected] if carries == "ability" else Ability.NONE,
            item=Item[expected] if carries == "item" else Item.NONE,
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
    unported = _still_unported("abilities", Ability)
    rng = random.Random(4)
    plain = PokemonSpec(
        species="Rhydon", nickname="P0", level=50, ability=Ability.NONE, item=Item.NONE,
        nature=Nature.HARDY, moves=["Earthquake", "Rock Slide"],
    )
    benched = PokemonSpec(
        species="Rhydon", nickname="P1", level=50, ability=Ability[unported], item=Item.NONE,
        nature=Nature.HARDY, moves=["Earthquake", "Rock Slide"],
    )
    scenario, _ = record(([plain, benched], [plain, benched]), _chooser(rng), seed=4, max_turns=4)

    theirs = _rust_trace(scenario, tmp_path)

    assert isinstance(theirs, str) and unported in theirs, theirs


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
