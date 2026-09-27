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
from battle_sim.models.actions import ActionType
from battle_sim.models.moves import MoveSlot
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
