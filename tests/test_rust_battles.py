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
        if move.name not in coded and not _attached(move) and all(ported(move, effect) for effect in move.effects)
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
    rules = json.loads((DATA / "rules.json").read_text())
    live = set(rules[f"live_{kind}"])
    if kind == "items":
        # A Mega Stone/Primal orb/Ultranecrozium Z is *conditionally* refused now — only if the
        # forme it reaches has an unported ability — so it cannot answer "is this always refused"
        # the way an ordinary item name can. `test_a_mega_stone_reaching_an_unported_ability_is_
        # refused_not_played_wrong` exercises that conditional case directly instead.
        forme_items = {row["item"] for row in rules["mega_formes"]} | {
            row["item"] for row in rules["ultra_burst_formes"]
        }
        live -= forme_items
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
    return (
        type(effect).__name__ == "InflictStatusEffect"
        and getattr(getattr(effect, "status", None), "name", None) == volatile
    )


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
def test_the_rarer_move_classes_agree_when_the_teams_are_built_for_them(event: str, predicate, tmp_path: Path) -> None:
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
        "Fly",
        "Bounce",
        "Dig",
        "Dive",
        "Sky Drop",
        "Phantom Force",
        "Shadow Force",
        "Solar Beam",
        "Solar Blade",
        "Meteor Beam",
        "Electro Shot",
        "Freeze Shock",
        "Geomancy",
        "Ice Burn",
        "Razor Wind",
        "Skull Bash",
        "Sky Attack",
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
        double_faints += sum(sum(1 for e in turn["events"] if e["type"] == "Fainted") >= 2 for turn in expected)

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
        landed += sum(1 for turn in expected for e in turn["events"] if e["type"] == "DamageDealt" and e["side"] == 1)

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
            species="Gyarados",
            nickname="A0",
            level=50,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Spore"],
        )
    ]
    team_b = [
        PokemonSpec(
            species="Gyarados",
            nickname="B0",
            level=50,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Sleep Talk", "Roost", "King's Shield", "Tackle"],
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
            species="Gyarados",
            nickname="A0",
            level=50,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Earthquake"],
        )
    ]
    team_b = [
        PokemonSpec(
            species="Gyarados",
            nickname="B0",
            level=50,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Roost"],
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
        "Rest",
        "Moonlight",
        "Pain Split",
        "Strength Sap",
        "Belly Drum",
        "Haze",
        "Court Change",
        "Curse",
        "Tidy Up",
        "Perish Song",
        "Heal Bell",
        "Take Heart",
        "Tackle",
        "Substitute",
        "Screech",
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
            species="Rhydon",
            nickname="A0",
            level=50,
            ability=Ability.NONE,
            item=Item.LEFTOVERS,
            nature=Nature.HARDY,
            moves=["Knock Off", "Trick", "Skill Swap", "Role Play"],
        )
    ]
    team_b = [
        PokemonSpec(
            species="Machamp",
            nickname="B0",
            level=50,
            ability=Ability.NONE,
            item=Item.SITRUS_BERRY,
            nature=Nature.HARDY,
            moves=["Knock Off", "Trick", "Entrainment", "Worry Seed"],
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
def test_a_mega_stone_reaching_an_unported_ability_is_refused_not_played_wrong(tmp_path: Path) -> None:
    """Mega Evolution itself is ported now (see the forme-swap batch), but only for a forme whose
    own ability this engine has implemented — Sableye-Mega's Magic Bounce is not one of them yet.
    A Pokemon that would reach an unplayed-out forme must still be refused, not silently left
    sitting in its base forme all battle, which is a different Pokemon from the one Python plays."""
    team = [
        PokemonSpec(
            species="Sableye",
            nickname="A0",
            level=50,
            ability=Ability.NONE,
            item=Item.SABLENITE,
            nature=Nature.HARDY,
            moves=["Tackle"],
        )
    ]
    rng = random.Random(32000)
    scenario, _ = record((team, team), _chooser(rng), seed=1, max_turns=4)
    theirs = _rust_trace(scenario, tmp_path)
    assert isinstance(theirs, str) and "not ported" in theirs, theirs


@needs_rust
def test_mega_evolution_changes_stats_ability_and_this_turns_speed_order(tmp_path: Path) -> None:
    """The same matchup `battle_sim/tests/test_mega.py::test_the_new_speed_decides_this_turn_s_order`
    and `::test_mega_rewires_the_new_forme_s_ability` use: Aerodactyl is 130 base Speed and its Mega
    is 150; Accelgor sits between at 145, so the mega has to resolve *before* `order_actions` reads
    speed for this same turn to have Aerodactyl move first, and its ability has to follow the forme
    from Pressure to Tough Claws for the differential to agree on damage at all.
    """
    team_a = [
        PokemonSpec(
            species="Aerodactyl",
            nickname="A0",
            level=50,
            ability=Ability.PRESSURE,
            item=Item.AERODACTYLITE,
            nature=Nature.HARDY,
            moves=["Rock Slide"],
        )
    ]
    team_b = [
        PokemonSpec(
            species="Accelgor",
            nickname="B0",
            level=50,
            ability=Ability.HYDRATION,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Bug Buzz"],
        )
    ]
    scenario, expected = record((team_a, team_b), _chooser(random.Random(0)), seed=0, max_turns=1)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"was refused: {theirs}"
    assert compare(expected, theirs) is None
    events = expected[0]["events"]
    forme = next(e for e in events if e["type"] == "FormeChanged")
    assert forme["pokemon"] == "A0" and forme["forme"] == "Aerodactyl-Mega", events
    assert events.index(forme) < next(i for i, e in enumerate(events) if e["type"] == "MoveUsed"), events
    a0_digest = expected[0]["state"]["sides"][0]["team"][0]
    assert a0_digest["ability"] == "TOUGH_CLAWS", a0_digest
    # Aerodactyl moved first despite Accelgor's own 145 > Aerodactyl's base 130 — only true if the
    # mega's 150 was what `order_actions` actually sorted on. (A Tough-Claws Rock Slide from a mega
    # with a stat boost this large one-shots Accelgor, which is why B0 never gets its own MoveUsed.)
    first_move = next(e for e in events if e["type"] == "MoveUsed")
    assert first_move["pokemon"] == "A0", events


@needs_rust
def test_switching_out_forfeits_mega_evolution_that_turn(tmp_path: Path) -> None:
    """A side that chose to switch does not Mega Evolve — the two are mutually exclusive in one
    turn. Aerodactyl starts on the bench, switches in on turn 1 (a switch turn: no mega yet, and it
    is not even the active Pokemon on turn 0 to be checked at all), and only Mega Evolves on turn 2,
    once it is already active and chooses to attack instead."""
    team_a = [
        PokemonSpec(
            species="Rhydon",
            nickname="A0",
            level=50,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Earthquake"],
        ),
        PokemonSpec(
            species="Aerodactyl",
            nickname="A1",
            level=50,
            ability=Ability.PRESSURE,
            item=Item.AERODACTYLITE,
            nature=Nature.HARDY,
            moves=["Rock Slide"],
        ),
    ]
    team_b = [
        PokemonSpec(
            species="Machamp",
            nickname="B0",
            level=50,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Splash"],
        )
    ]

    def choose(state, side_index):
        if side_index == 0 and state.turn == 1:
            return next(a for a in legal_actions(state, 0) if a.action is ActionType.SWITCH_OUT)
        return next(a for a in legal_actions(state, side_index) if a.action is ActionType.USE_MOVE)

    scenario, expected = record((team_a, team_b), choose, seed=0, max_turns=3)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"was refused: {theirs}"
    assert compare(expected, theirs) is None
    assert not any(e["type"] == "FormeChanged" for turn in expected[:2] for e in turn["events"]), expected[:2]
    assert any(e["type"] == "FormeChanged" for e in expected[2]["events"]), expected[2]["events"]


@needs_rust
def test_ultra_burst_changes_type_along_with_stats_and_ability(tmp_path: Path) -> None:
    """Necrozma-Dusk-Mane + Ultranecrozium Z reaches Necrozma-Ultra, the one pairing that changes
    type (Psychic/Steel to Psychic/Dragon) as well as ability (Prism Armor to Neuroforce) — the
    pairing `_ULTRA_BURST` exists for specifically, since it cannot be derived from `base_species`
    the way an ordinary Mega Stone's forme can (plain Necrozma shares the same base species and
    must not Ultra Burst)."""
    team_a = [
        PokemonSpec(
            species="Necrozma-Dusk-Mane",
            nickname="A0",
            level=50,
            ability=Ability.PRISM_ARMOR,
            item=Item.ULTRANECROZIUM_Z,
            nature=Nature.HARDY,
            moves=["Sunsteel Strike"],
        )
    ]
    team_b = [
        PokemonSpec(
            species="Machamp",
            nickname="B0",
            level=50,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Splash"],
        )
    ]
    scenario, expected = record((team_a, team_b), _chooser(random.Random(0)), seed=0, max_turns=1)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"was refused: {theirs}"
    assert compare(expected, theirs) is None
    forme = next(e for e in expected[0]["events"] if e["type"] == "FormeChanged")
    assert forme["forme"] == "Necrozma-Ultra", expected[0]["events"]
    a0_digest = expected[0]["state"]["sides"][0]["team"][0]
    assert a0_digest["ability"] == "NEUROFORCE", a0_digest


@needs_rust
def test_mega_rayquaza_is_gated_on_dragon_ascent_not_an_item(tmp_path: Path) -> None:
    """The one move-gated forme in Gen 7: no Mega Stone at all, just knowing Dragon Ascent. Also
    the only Mega whose ability sets its own weather (`Delta Stream`/`STRONG_WINDS`), exercising
    `apply_weather_from_ability`'s second call site — the switch-in binder fired long before this
    mid-turn forme swap and never saw an ability that could set anything."""
    team_a = [
        PokemonSpec(
            species="Rayquaza",
            nickname="A0",
            level=50,
            ability=Ability.AIR_LOCK,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Dragon Ascent"],
        )
    ]
    team_b = [
        PokemonSpec(
            species="Machamp",
            nickname="B0",
            level=50,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Splash"],
        )
    ]
    scenario, expected = record((team_a, team_b), _chooser(random.Random(0)), seed=0, max_turns=1)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"was refused: {theirs}"
    assert compare(expected, theirs) is None
    events = expected[0]["events"]
    forme = next(e for e in events if e["type"] == "FormeChanged")
    assert forme["forme"] == "Rayquaza-Mega", events
    a0_digest = expected[0]["state"]["sides"][0]["team"][0]
    assert a0_digest["ability"] == "DELTA_STREAM", a0_digest
    assert expected[0]["state"]["weather"] == "STRONG_WINDS", expected[0]["state"]


@needs_rust
def test_primal_reversion_sets_its_own_weather_which_an_ordinary_move_can_still_override(tmp_path: Path) -> None:
    """Groudon + Red Orb reaches Groudon-Primal and Desolate Land sets Harsh Sunlight — but this
    engine, like the Python it mirrors, does not make that weather immune to an ordinary setter
    afterward (the Python's own comments call this out as unmodelled, not a bug to fix here), so a
    later Rain Dance from the other side displaces it to plain rain rather than being refused."""
    team_a = [
        PokemonSpec(
            species="Groudon",
            nickname="A0",
            level=50,
            ability=Ability.DROUGHT,
            item=Item.RED_ORB,
            nature=Nature.HARDY,
            moves=["Splash"],
        )
    ]
    team_b = [
        PokemonSpec(
            species="Machamp",
            nickname="B0",
            level=50,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Splash", "Rain Dance"],
        )
    ]

    def choose(state, side_index):
        options = [a for a in legal_actions(state, side_index) if a.action is ActionType.USE_MOVE]
        if side_index == 1:
            wanted = "Rain Dance" if state.turn == 1 else "Splash"
            return next(a for a in options if state.sides[1].active_pokemon.moves[a.move].name == wanted)
        return options[0]

    scenario, expected = record((team_a, team_b), choose, seed=0, max_turns=2)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"was refused: {theirs}"
    assert compare(expected, theirs) is None
    turn0 = expected[0]["events"]
    forme = next(e for e in turn0 if e["type"] == "FormeChanged")
    assert forme["forme"] == "Groudon-Primal", turn0
    # Two `WeatherSetByAbility` events land this same turn: the lead's own switch-in Drought sets
    # ordinary sun first, and only once the mid-turn Primal Reversion hands over Desolate Land does
    # the second one — `apply_weather_from_ability`'s whole reason for a second call site — displace
    # it to Harsh Sunlight. Both are correct, expected Python behaviour, not a double-fire bug.
    weather_events = [e for e in turn0 if e["type"] == "WeatherSetByAbility"]
    assert [e["ability"] for e in weather_events] == ["DROUGHT", "DESOLATE_LAND"], turn0
    assert expected[0]["state"]["weather"] == "HARSH_SUN", expected[0]["state"]
    assert expected[1]["state"]["weather"] == "RAIN", expected[1]["state"]


@needs_rust
def test_delta_stream_cuts_a_flying_types_weaknesses_to_neutral(tmp_path: Path) -> None:
    """Rayquaza-Mega's whole defensive identity: an Ice-type move against its Dragon/Flying typing
    is 4x without Delta Stream (2x Dragon-half, 2x Flying-half) and 2x with it (Flying's own half
    cut to neutral, Dragon's left standing). Mega Evolution resolves before a single move is
    ordered, so even this first turn's hit already sees the post-mega Rayquaza — there is no
    "before" state to compare against within one battle.

    The log only distinguishes "super" (>=2x) from "resisted" (<=0.5x), not the exact multiplier,
    so what actually proves the negation happened — as opposed to both engines agreeing on some
    other wrong number — is `compare()` itself: Python's own recorded trace already carries the
    negated (halved) damage number, and a Rust that computed the un-negated 4x hit instead would
    diverge from it, not silently agree.
    """
    team_a = [
        PokemonSpec(
            species="Rayquaza",
            nickname="A0",
            level=50,
            ability=Ability.AIR_LOCK,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Dragon Ascent"],
        )
    ]
    team_b = [
        PokemonSpec(
            species="Dewgong",
            nickname="B0",
            level=50,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Ice Beam"],
        )
    ]
    scenario, expected = record((team_a, team_b), _chooser(random.Random(0)), seed=0, max_turns=1)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"was refused: {theirs}"
    assert compare(expected, theirs) is None
    effectiveness = next(e for e in expected[0]["events"] if e["type"] == "Effectiveness")
    assert effectiveness["level"] == "super", expected[0]["events"]


@needs_rust
def test_a_generic_z_crystal_upgrades_a_same_type_move_by_the_power_table(tmp_path: Path) -> None:
    """Groundium Z on a Ground move: the Z-move's own name and type come from the crystal's
    template, but its power is Earthquake's own 100 read through the fixed Gen 7 table (180), and
    its accuracy is unconditional — no draw for it at all, which the tape agreeing on drawn count
    without a fourth probability is what actually proves rather than merely the digest."""
    team_a = [
        PokemonSpec(
            species="Rhydon",
            nickname="A0",
            level=50,
            ability=Ability.NONE,
            item=Item.GROUNDIUM_Z,
            nature=Nature.HARDY,
            moves=["Earthquake"],
        )
    ]
    team_b = [
        PokemonSpec(
            species="Machamp",
            nickname="B0",
            level=50,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Splash"],
        )
    ]

    def choose(state, side_index):
        if side_index == 0:
            return next(a for a in legal_actions(state, 0) if a.z_move)
        return next(iter(legal_actions(state, 1)))

    scenario, expected = record((team_a, team_b), choose, seed=0, max_turns=1)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"was refused: {theirs}"
    assert compare(expected, theirs) is None
    used = next(e for e in expected[0]["events"] if e["type"] == "MoveUsed" and e["side"] == 0)
    assert used["move"] == "Earthquake" and used["unleashed_as"] == "Tectonic Rage", expected[0]["events"]
    # The crystal is spent as a *resource*, not as an item: never removed from the holder, same as
    # a Mega Stone or Primal orb (`is_fused_to` refuses to let either be taken at all).
    a0_digest = expected[0]["state"]["sides"][0]["team"][0]
    assert a0_digest["item"] == "GROUNDIUM_Z", a0_digest


@needs_rust
def test_a_signature_z_crystal_ignores_the_power_table_entirely(tmp_path: Path) -> None:
    """Aloraichium Z on Thunderbolt: Stoked Sparksurfer, at its own real 175 power (not derived from
    Thunderbolt's 90 at all), carrying its own guaranteed paralysis — proving the whole effects list
    comes from the crystal's template, not just a power number."""
    team_a = [
        PokemonSpec(
            species="Rhydon",
            nickname="A0",
            level=50,
            ability=Ability.NONE,
            item=Item.ALORAICHIUM_Z,
            nature=Nature.HARDY,
            moves=["Thunderbolt"],
        )
    ]
    team_b = [
        PokemonSpec(
            species="Machamp",
            nickname="B0",
            level=50,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Splash"],
        )
    ]

    def choose(state, side_index):
        if side_index == 0:
            return next(a for a in legal_actions(state, 0) if a.z_move)
        return next(iter(legal_actions(state, 1)))

    paralyzed = 0
    for seed in range(20):
        scenario, expected = record((team_a, team_b), choose, seed=seed, max_turns=1)
        theirs = _rust_trace(scenario, tmp_path)
        assert not isinstance(theirs, str), f"seed {seed} was refused: {theirs}"
        assert compare(expected, theirs) is None, f"seed {seed}"
        used = next(e for e in expected[0]["events"] if e["type"] == "MoveUsed" and e["side"] == 0)
        assert used["move"] == "Thunderbolt" and used["unleashed_as"] == "Stoked Sparksurfer", expected[0]["events"]
        if any(e["type"] == "StatusInflicted" and e.get("status") == "PARALYSIS" for e in expected[0]["events"]):
            paralyzed += 1

    assert paralyzed == 20, "Stoked Sparksurfer's own paralysis is supposed to be guaranteed"


@needs_rust
def test_a_z_move_only_unleashes_once_per_battle(tmp_path: Path) -> None:
    """`legal_actions` itself stops offering the Z-move variant once `has_used_z_move` is set, so
    the once-per-battle gate is exercised end to end: turn 1's own choice of "the Z-move if one is
    offered" finds none on offer any more and falls back to the ordinary move, which both engines
    have to agree resolves as plain Earthquake, `unleashed_as` and all."""
    team_a = [
        PokemonSpec(
            species="Rhydon",
            nickname="A0",
            level=50,
            ability=Ability.NONE,
            item=Item.GROUNDIUM_Z,
            nature=Nature.HARDY,
            moves=["Earthquake"],
        )
    ]
    team_b = [
        # Flying-type: naturally immune to Earthquake/Tectonic Rage, so it survives to a second
        # turn regardless of how hard a 180-power hit lands — this test cares about the once-per-
        # battle gate, not about staging a fair fight.
        PokemonSpec(
            species="Zapdos",
            nickname="B0",
            level=50,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Splash"],
        )
    ]

    def choose(state, side_index):
        if side_index != 0:
            return next(iter(legal_actions(state, 1)))
        options = legal_actions(state, 0)
        return next((a for a in options if a.z_move), options[0])

    scenario, expected = record((team_a, team_b), choose, seed=0, max_turns=2)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"was refused: {theirs}"
    assert compare(expected, theirs) is None
    first = next(e for e in expected[0]["events"] if e["type"] == "MoveUsed" and e["side"] == 0)
    second = next(e for e in expected[1]["events"] if e["type"] == "MoveUsed" and e["side"] == 0)
    assert first["unleashed_as"] == "Tectonic Rage", expected[0]["events"]
    assert second["move"] == "Earthquake" and second["unleashed_as"] is None, expected[1]["events"]
    a0_digest = expected[1]["state"]["sides"][0]["team"][0]
    assert a0_digest["item"] == "GROUNDIUM_Z", a0_digest


@needs_rust
def test_multitype_tracks_a_held_plate_and_boosts_judgment_by_it(tmp_path: Path) -> None:
    """Arceus + Iron Plate: Multitype's own type-sync (`sync_type_from_item`, checked on switch-in
    and every residual tick) makes it Steel-type, which both selects Judgment's type (`power::
    plate_type`, an independent table Multitype's own tracking has to agree with) and earns the
    plate's ordinary 1.2x same-type boost on that same hit — two separate mechanisms reading the
    same held item, which only a real battle can prove still agree with each other."""
    team_a = [
        PokemonSpec(
            species="Arceus",
            nickname="A0",
            level=50,
            ability=Ability.MULTITYPE,
            item=Item.IRON_PLATE,
            nature=Nature.HARDY,
            moves=["Judgment"],
        )
    ]
    team_b = [
        PokemonSpec(
            species="Machamp",
            nickname="B0",
            level=50,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Splash"],
        )
    ]
    scenario, expected = record((team_a, team_b), _chooser(random.Random(0)), seed=0, max_turns=1)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"was refused: {theirs}"
    assert compare(expected, theirs) is None
    type_changed = next(e for e in expected[0]["events"] if e["type"] == "TypeChanged")
    assert type_changed["pokemon"] == "A0" and type_changed["new_type"] == "STEEL", expected[0]["events"]
    a0_digest = expected[0]["state"]["sides"][0]["team"][0]
    assert a0_digest["types"] == ["STEEL"] and a0_digest["ability"] == "MULTITYPE", a0_digest


@needs_rust
def test_multitype_with_no_plate_stays_normal_and_never_logs_a_change(tmp_path: Path) -> None:
    """No plate held: Arceus is already Normal-typed at baseline, so `sync_type_from_item`'s own
    guard (only log and mutate when the wanted type actually differs) means no `TypeChanged` fires
    at all — the quiet case, and the one a sloppier port would get backwards."""
    team_a = [
        PokemonSpec(
            species="Arceus",
            nickname="A0",
            level=50,
            ability=Ability.MULTITYPE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Judgment"],
        )
    ]
    team_b = [
        PokemonSpec(
            species="Machamp",
            nickname="B0",
            level=50,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Splash"],
        )
    ]
    scenario, expected = record((team_a, team_b), _chooser(random.Random(0)), seed=0, max_turns=1)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"was refused: {theirs}"
    assert compare(expected, theirs) is None
    assert not any(e["type"] == "TypeChanged" for e in expected[0]["events"]), expected[0]["events"]
    a0_digest = expected[0]["state"]["sides"][0]["team"][0]
    assert a0_digest["types"] == ["NORMAL"], a0_digest


@needs_rust
def test_rks_system_tracks_a_held_memory_the_same_way_multitype_does(tmp_path: Path) -> None:
    """Silvally + Fire Memory: the same `sync_type_from_item` table-lookup, keyed on RKS System
    instead of Multitype and a different 17-item table — proving the shared function actually reads
    which ability it is rather than always reaching for Multitype's own plates."""
    team_a = [
        PokemonSpec(
            species="Silvally",
            nickname="A0",
            level=50,
            ability=Ability.RKS_SYSTEM,
            item=Item.FIRE_MEMORY,
            nature=Nature.HARDY,
            moves=["Tackle"],
        )
    ]
    team_b = [
        PokemonSpec(
            species="Machamp",
            nickname="B0",
            level=50,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Splash"],
        )
    ]
    scenario, expected = record((team_a, team_b), _chooser(random.Random(0)), seed=0, max_turns=1)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"was refused: {theirs}"
    assert compare(expected, theirs) is None
    type_changed = next(e for e in expected[0]["events"] if e["type"] == "TypeChanged")
    assert type_changed["pokemon"] == "A0" and type_changed["new_type"] == "FIRE", expected[0]["events"]
    a0_digest = expected[0]["state"]["sides"][0]["team"][0]
    assert a0_digest["types"] == ["FIRE"], a0_digest


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
            species="Dewgong",
            nickname="A0",
            level=50,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Trick Room", "Tackle"],
        )
    ]
    team_b = [
        PokemonSpec(
            species="Golem",
            nickname="B0",
            level=1,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Tackle", "Trick Room"],
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
                species="Rhydon",
                nickname="P0",
                level=50,
                ability=Ability.NONE,
                item=Item.NONE,
                nature=Nature.HARDY,
                moves=[first, second, "Tackle"],
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
                species=species,
                nickname=f"P{index}",
                level=50,
                ability=Ability.NONE,
                item=Item.NONE,
                nature=Nature.HARDY,
                moves=[rng.choice(pool), rng.choice(pool)],
            )
            for index, species in enumerate(["Rhydon", "Rhydon"])
        ]

    seen: Counter[str] = Counter()
    for seed in range(40):
        rng = random.Random(37000 + seed)
        team_a = team(rng, pool_a)
        team_b = [
            PokemonSpec(
                species="Machamp",
                nickname=f"P{index}",
                level=50,
                ability=Ability.NONE,
                item=Item.NONE,
                nature=Nature.HARDY,
                moves=[rng.choice(pool_b), rng.choice(pool_b)],
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
            species=species,
            nickname=nickname,
            level=50,
            ability=ability,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=moves,
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
            species="Machamp",
            nickname="A0",
            level=50,
            ability=Ability.SURGE_SURFER,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Tackle"],
        )
    ]
    team_b = [
        PokemonSpec(
            species="Kangaskhan",
            nickname="B0",
            level=50,
            ability=Ability.ELECTRIC_SURGE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Tackle"],
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
            species="Rhydon",
            nickname="A0",
            level=50,
            ability=Ability.UNBURDEN,
            item=Item.LEFTOVERS,
            nature=Nature.HARDY,
            moves=["Tackle"],
        )
    ]
    team_b = [
        PokemonSpec(
            species="Machamp",
            nickname="B0",
            level=50,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Knock Off"],
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
            species="Rhydon",
            nickname="A0",
            level=50,
            ability=Ability.QUICK_DRAW,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Tackle"],
        )
    ]
    team_b = [
        PokemonSpec(
            species="Tauros",
            nickname="B0",
            level=50,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Tackle"],
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
            species="Rhydon",
            nickname="A0",
            level=1,
            ability=Ability.STURDY,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Splash"],
        )
    ]
    team_b = [
        PokemonSpec(
            species="Machamp",
            nickname="B0",
            level=50,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Earthquake"],
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
        species=species,
        nickname=nickname,
        level=50,
        ability=ability,
        item=Item.NONE,
        nature=Nature.HARDY,
        moves=moves,
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
        assert any(e["type"] == "AbsorbHealed" and e["ability"] == ability for e in absorb_turn), (
            f"{ability}: never healed off a hit it should have absorbed\n{absorb_turn}"
        )
        assert not any(e["type"] == "DamageDealt" for e in absorb_turn), (
            f"{ability}: the absorbed move should never have landed\n{absorb_turn}"
        )

        full_hp_attacker = [_mon("Rhydon", "A0", Ability.NONE, [move])]
        full_hp_defender = [_mon(species, "B0", Ability[ability], ["Splash"])]
        scenario, expected = record(
            (full_hp_attacker, full_hp_defender), _chooser(random.Random(50000 + index)), seed=0, max_turns=1
        )
        theirs = _rust_trace(scenario, tmp_path)
        assert not isinstance(theirs, str), f"{ability} (full HP) was refused: {theirs}"
        assert compare(expected, theirs) is None, f"{ability} (full HP)"
        events = expected[0]["events"]
        assert any(e["type"] == "AbsorbBlocked" and e["ability"] == ability for e in events), (
            f"{ability}: should report AbsorbBlocked at full HP\n{events}"
        )

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
        species="Machamp",
        nickname="B0",
        level=1,
        ability=Ability.NONE,
        item=Item.NONE,
        nature=Nature.HARDY,
        moves=["Splash"],
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
        species="Machamp",
        nickname="B0",
        level=1,
        ability=Ability.NONE,
        item=Item.NONE,
        nature=Nature.HARDY,
        moves=["Splash"],
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
            species="Golem",
            nickname="A0",
            level=1,
            ability=Ability.REGENERATOR,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Splash"],
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
def test_protosynthesis_and_quark_drive_boost_the_highest_stat_and_agree(tmp_path: Path) -> None:
    """Protosynthesis and Quark Drive: the same `_paradox_evaluate` runs from `ON_SWITCH_IN`,
    `ON_TURN_START` and `ON_RESIDUAL`, boosting whichever of the holder's five relevant stats is
    highest by 1.3x in `ON_DAMAGE_CALC` (the Speed case is its own test below, since it shows up in
    turn order rather than a damage number). Both leads' switch-ins happen before `ON_TURN_START`
    fires, so a Drought/Electric Surge partner sets the field before this engine ever asks whether
    the condition is active -- deterministic, no seed needed.

    Machamp's Attack (130) is comfortably its highest stat, so `ParadoxActivated` naming anything
    else would be a real bug, not this test picking a bad example.

    Vacuity-checked directly: renaming the ability strings in `hooks::evaluate_paradox`'s match
    turned the activation itself into a plain digest mismatch (no `ParadoxActivated`, no boosted
    `DamageDealt`, where Python has both); reverting that and instead zeroing the 5325 multiplier in
    `abilities::handle` left the activation log line intact but turned the boosted `DamageDealt`
    amount into a mismatch on its own.
    """
    cases = [("PROTOSYNTHESIS", "DROUGHT"), ("QUARK_DRIVE", "ELECTRIC_SURGE")]
    for ability, setter in cases:
        boosted_a = [_mon("Machamp", "A0", Ability[ability], ["Tackle"])]
        setter_b = [_mon("Kangaskhan", "B0", Ability[setter], ["Splash"])]
        scenario, boosted = record((boosted_a, setter_b), _chooser(random.Random(0)), seed=0, max_turns=1)
        theirs = _rust_trace(scenario, tmp_path)
        assert not isinstance(theirs, str), f"{ability} was refused: {theirs}"
        assert compare(boosted, theirs) is None, f"{ability}"
        events = boosted[0]["events"]
        activation = next((e for e in events if e["type"] == "ParadoxActivated"), None)
        assert activation is not None, f"{ability}: never activated\n{events}"
        assert activation["stat"] == "ATTACK", f"{ability}: boosted {activation['stat']} instead of ATTACK"
        assert activation["from_booster"] is False, activation
        boosted_amount = _damage_dealt(events)

        plain_a = [_mon("Machamp", "A0", Ability.NONE, ["Tackle"])]
        scenario, plain = record((plain_a, setter_b), _chooser(random.Random(0)), seed=0, max_turns=1)
        theirs = _rust_trace(scenario, tmp_path)
        assert not isinstance(theirs, str), f"{ability} control was refused: {theirs}"
        assert compare(plain, theirs) is None, f"{ability} control"
        plain_amount = _damage_dealt(plain[0]["events"])
        assert boosted_amount > plain_amount * 1.2, f"{ability}: {boosted_amount} was not ~1.3x {plain_amount}"


@needs_rust
def test_quark_drive_speed_boost_reorders_turns(tmp_path: Path) -> None:
    """The Speed case of the same boost, in `turn::effective_speed` rather than `abilities::handle`:
    Tauros (110 base Speed, its own highest stat) is slower than Lycanroc (112) unboosted but faster
    once Quark Drive's 1.3x -- no, 1.5x for Speed specifically -- pushes it to 165. Both numbers are
    far enough apart that this is a strict comparison, not a tie a coin flip could still lose.

    Vacuity-checked directly: removing the `paradox_boost == SPEED` line from `effective_speed`
    turned this into a tape divergence -- Lycanroc moving first in this engine where Python's boosted
    Tauros already does.
    """
    boosted_a = [_mon("Tauros", "A0", Ability.QUARK_DRIVE, ["Splash"])]
    setter_b = [_mon("Lycanroc", "B0", Ability.ELECTRIC_SURGE, ["Splash"])]
    scenario, expected = record((boosted_a, setter_b), _chooser(random.Random(0)), seed=0, max_turns=1)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"was refused: {theirs}"
    assert compare(expected, theirs) is None
    events = expected[0]["events"]
    activation = next((e for e in events if e["type"] == "ParadoxActivated"), None)
    assert activation is not None and activation["stat"] == "SPEED", events
    movers = [e["side"] for e in events if e["type"] == "MoveUsed"]
    assert movers[0] == 0, f"Tauros should move first once boosted: {movers}"


@needs_rust
def test_paradox_boost_fades_silently_when_the_field_ends(tmp_path: Path) -> None:
    """A field-sourced boost (not from Booster Energy) is cleared the moment the condition ends --
    silently, per `_paradox_evaluate`: no log line marks the loss the way `ParadoxActivated` marked
    the gain. Drought's own sun lasts 5 turns, so by the last of 7 the boost is long gone and
    Machamp's Tackle should have quietly reverted to its unboosted amount, with `ParadoxActivated`
    never appearing a second time (nothing re-activates a name that already ended and holds no
    Booster Energy).

    Vacuity-checked directly: keeping `paradox_boost` set once activated, rather than clearing it
    when `energized` goes false, turned the final turn's `DamageDealt` into a mismatch -- this
    engine still boosting an attack Python had already let lapse.
    """
    # Rhydon, not Kangaskhan: bulky enough on both sides of this matchup that seven Tackles apiece
    # never end the battle early and shift which turn "the last one" actually is.
    team_a = [_mon("Machamp", "A0", Ability.PROTOSYNTHESIS, ["Tackle"])]
    team_b = [_mon("Rhydon", "B0", Ability.DROUGHT, ["Splash"])]
    scenario, expected = record((team_a, team_b), _chooser(random.Random(0)), seed=0, max_turns=7)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"was refused: {theirs}"
    assert compare(expected, theirs) is None
    activations = [e for turn in expected for e in turn["events"] if e["type"] == "ParadoxActivated"]
    assert len(activations) == 1, activations
    assert len(expected) == 7, f"the battle ended early: {len(expected)} turns"
    late_amount = _damage_dealt(expected[6]["events"])

    plain_a = [_mon("Machamp", "A0", Ability.NONE, ["Tackle"])]
    scenario, plain = record((plain_a, team_b), _chooser(random.Random(0)), seed=0, max_turns=7)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"control was refused: {theirs}"
    assert compare(plain, theirs) is None
    plain_late_amount = _damage_dealt(plain[6]["events"])
    assert late_amount == plain_late_amount, f"turn 7 should be unboosted: {late_amount} vs {plain_late_amount}"


@needs_rust
def test_toxic_debris_and_thermal_exchange_react_to_being_hit(tmp_path: Path) -> None:
    """Two `ON_AFTER_HIT` one-liners at the site the whole defender-reaction match block in
    `hooks::ability_after_hit` already exists for. Toxic Debris scatters a layer of Toxic Spikes on
    the attacker's side for any physical hit -- deliberately with no fainted guard, unlike every
    stat-bump reaction beside it in the Python, so a holder that faints on the triggering hit still
    scatters the layer; Thermal Exchange is the plain `_bind_hit_reaction_boost` shape Stamina and
    Justified already use, just keyed on Fire instead of Dark. Both moves are always-hit, so this is
    deterministic.

    Vacuity-checked directly: renaming each ability's own match arm turned its own case into a
    plain digest mismatch -- this engine's missing `HazardSet`/`StatStageChanged` where Python has
    one.
    """
    attacker = [_mon("Machamp", "A0", Ability.NONE, ["Tackle"])]
    debris_holder = [_mon("Rhydon", "B0", Ability.TOXIC_DEBRIS, ["Splash"])]
    scenario, expected = record((attacker, debris_holder), _chooser(random.Random(0)), seed=0, max_turns=1)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"Toxic Debris was refused: {theirs}"
    assert compare(expected, theirs) is None, "Toxic Debris"
    events = expected[0]["events"]
    assert any(e["type"] == "HazardSet" and e["side"] == 0 and e["hazard"] == "TOXIC_SPIKES" for e in events), (
        f"Toxic Debris never scattered a layer\n{events}"
    )

    fire_attacker = [_mon("Machamp", "A0", Ability.NONE, ["Ember"])]
    thermal_holder = [_mon("Rhydon", "B0", Ability.THERMAL_EXCHANGE, ["Splash"])]
    scenario, expected = record((fire_attacker, thermal_holder), _chooser(random.Random(0)), seed=0, max_turns=1)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"Thermal Exchange was refused: {theirs}"
    assert compare(expected, theirs) is None, "Thermal Exchange"
    events = expected[0]["events"]
    boosts = [e for e in events if e["type"] == "StatStageChanged" and e["source"] == "thermal_exchange"]
    assert boosts and boosts[0]["stat"] == "ATTACK", f"Thermal Exchange never boosted Attack\n{events}"


@needs_rust
def test_cursed_body_disables_the_attackers_move(tmp_path: Path) -> None:
    """Cursed Body: landing any hit gives a 30% chance to disable whatever move the attacker just
    used -- Tackle always hits, so this is a seed sweep on the 30% alone.

    Vacuity-checked directly: renaming `"CURSED_BODY"` in `hooks::ability_after_hit` turned every
    seed where the disable actually landed into a plain digest mismatch -- this engine's silence
    where Python's `DisableApplied` already fired.
    """
    attacker = [_mon("Machamp", "A0", Ability.NONE, ["Tackle"])]
    cursed_holder = [_mon("Rhydon", "B0", Ability.CURSED_BODY, ["Splash"])]
    disabled = 0
    for seed in range(20):
        rng = random.Random(42000 + seed)
        scenario, expected = record((attacker, cursed_holder), _chooser(rng), seed=seed, max_turns=1)
        theirs = _rust_trace(scenario, tmp_path)
        assert not isinstance(theirs, str), f"seed {seed} was refused: {theirs}"
        assert compare(expected, theirs) is None, f"seed {seed}"
        if any(e["type"] == "DisableApplied" and e["side"] == 0 for e in expected[0]["events"]):
            disabled += 1
    assert disabled > 0, "Cursed Body never once disabled the attacker across 20 seeds"


@needs_rust
def test_sheer_force_boosts_power_and_strips_the_secondary(tmp_path: Path) -> None:
    """Sheer Force: any move carrying a secondary effect gets 1.3x power, and the secondary itself
    is traded away entirely -- no draw, no chance for it to land regardless of its own probability.
    Rock Smash's 50% Defense-drop makes both halves deterministic in one case: a Sheer Force user's
    Rock Smash always hits harder and the Defense drop never happens, where a plain attacker's
    sometimes does and never hits as hard.

    Vacuity-checked directly: dropping the `"SHEER_FORCE"` guard from `inline::
    tune_status_secondary`/`tune_stage_secondary` turned this into a digest mismatch the moment a
    seed's own draw would have landed the drop; zeroing the 5325 power multiplier in
    `abilities::handle` turned the boosted `DamageDealt` amount into one on its own.
    """
    boosted_a = [_mon("Machamp", "A0", Ability.SHEER_FORCE, ["Rock Smash"])]
    target_b = [_mon("Rhydon", "B0", Ability.NONE, ["Splash"])]
    scenario, boosted = record((boosted_a, target_b), _chooser(random.Random(0)), seed=0, max_turns=1)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"was refused: {theirs}"
    assert compare(boosted, theirs) is None
    events = boosted[0]["events"]
    assert not any(e["type"] == "StatStageChanged" for e in events), f"the secondary should be gone\n{events}"
    boosted_amount = _damage_dealt(events)

    plain_a = [_mon("Machamp", "A0", Ability.NONE, ["Rock Smash"])]
    scenario, plain = record((plain_a, target_b), _chooser(random.Random(0)), seed=0, max_turns=1)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"control was refused: {theirs}"
    assert compare(plain, theirs) is None
    plain_amount = _damage_dealt(plain[0]["events"])
    assert boosted_amount > plain_amount * 1.2, f"{boosted_amount} was not ~1.3x {plain_amount}"


@needs_rust
def test_solar_power_boosts_special_damage_and_chips_its_holder(tmp_path: Path) -> None:
    """Solar Power: 1.5x Special damage in sun (`ON_DAMAGE_CALC`), and an eighth-HP chip every turn
    it stands in that same sun (`ON_RESIDUAL`, at `ResidualOrder.WEATHER` -- the sandstorm chip's
    own band, not the `WEATHER_ABILITY` band Ice Body and Dry Skin use). Drought's own switch-in
    sets the sun before either of this engine's own reads of it.

    Vacuity-checked directly: zeroing the 6144 attack multiplier in `abilities::handle` turned the
    boosted `DamageDealt` into a mismatch on its own; renaming `"SOLAR_POWER"` in
    `hooks::solar_power_chip` turned the missing `AbilityChipDamage` into one too.
    """
    boosted_a = [_mon("Dewgong", "A0", Ability.SOLAR_POWER, ["Ice Beam"])]
    setter_b = [_mon("Kangaskhan", "B0", Ability.DROUGHT, ["Splash"])]
    scenario, boosted = record((boosted_a, setter_b), _chooser(random.Random(0)), seed=0, max_turns=1)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"was refused: {theirs}"
    assert compare(boosted, theirs) is None
    events = boosted[0]["events"]
    chips = [e for e in events if e["type"] == "AbilityChipDamage" and e["ability"] == "SOLAR_POWER"]
    assert chips and chips[0]["amount"] > 0, f"Solar Power never chipped its holder\n{events}"
    boosted_amount = _damage_dealt(events)

    plain_a = [_mon("Dewgong", "A0", Ability.NONE, ["Ice Beam"])]
    scenario, plain = record((plain_a, setter_b), _chooser(random.Random(0)), seed=0, max_turns=1)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"control was refused: {theirs}"
    assert compare(plain, theirs) is None
    plain_amount = _damage_dealt(plain[0]["events"])
    assert boosted_amount > plain_amount * 1.2, f"{boosted_amount} was not ~1.5x {plain_amount}"


@needs_rust
def test_wonder_guard_blocks_anything_not_super_effective_including_status_moves(tmp_path: Path) -> None:
    """Wonder Guard reads the *plain* type chart against the defender's raw types -- not the
    `effective_bypass`/`battle_types` every other effectiveness site in this engine uses -- and
    blocks anything under 2x. `ON_BEFORE_MOVE` fires for a defender-facing status move too, so this
    reproduces a real quirk rather than fixing it: a status move whose type is not super effective
    against the holder gets blocked here as well, which Wonder Guard does not do in the actual
    games. Water Gun (neutral against Gengar) and Thunder Wave (also neutral, and a status move)
    are both blocked; Earthquake (super effective against Poison) still lands.

    Vacuity-checked directly: removing the `"WONDER_GUARD"` arm from `hooks::ability_before_move`
    turned every one of these three cases into a plain digest mismatch -- a landed `DamageDealt` or
    `StatusInflicted` here against Python's `DoesNotAffect`.
    """
    guard = [_mon("Gengar", "B0", Ability.WONDER_GUARD, ["Splash"])]
    cases = [("Water Gun", False), ("Earthquake", True), ("Thunder Wave", False)]
    for move, should_land in cases:
        attacker = [_mon("Machamp", "A0", Ability.NONE, [move])]
        scenario, expected = record((attacker, guard), _chooser(random.Random(0)), seed=0, max_turns=1)
        theirs = _rust_trace(scenario, tmp_path)
        assert not isinstance(theirs, str), f"{move} was refused: {theirs}"
        assert compare(expected, theirs) is None, move
        events = expected[0]["events"]
        landed = any(e["type"] in ("DamageDealt", "StatusInflicted") for e in events)
        blocked = any(e["type"] == "DoesNotAffect" for e in events)
        assert landed == should_land and blocked == (not should_land), f"{move}: {events}"


@needs_rust
def test_wind_rider_cancels_wind_moves_and_boosts_attack_as_justified(tmp_path: Path) -> None:
    """Wind Rider: a Wind-flagged move aimed at its holder is cancelled outright for a free +1
    Attack -- Gust always hits, so this is deterministic. The stat-change log line is a real bug in
    the Python, reproduced rather than fixed per invariant 3: `_bind_wind_rider` copies
    `_bind_hit_reaction_boost`'s shape by hand and keeps its `source="justified"` string, so a Wind
    Rider activation reads as Justified's in the log.

    Vacuity-checked directly: removing the `"WIND_RIDER"` arm from `hooks::ability_before_move`
    turned this into a plain digest mismatch -- a landed `DamageDealt` here against Python's
    absorbed hit and `StatStageChanged`.
    """
    attacker = [_mon("Machamp", "A0", Ability.NONE, ["Gust"])]
    rider = [_mon("Rhydon", "B0", Ability.WIND_RIDER, ["Splash"])]
    scenario, expected = record((attacker, rider), _chooser(random.Random(0)), seed=0, max_turns=1)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"was refused: {theirs}"
    assert compare(expected, theirs) is None
    events = expected[0]["events"]
    assert not any(e["type"] == "DamageDealt" for e in events), events
    boosts = [e for e in events if e["type"] == "StatStageChanged" and e["source"] == "justified"]
    assert boosts and boosts[0]["stat"] == "ATTACK", f"Wind Rider's own bug-for-bug log line: {events}"


@needs_rust
def test_liquid_voice_turns_a_sound_move_to_water(tmp_path: Path) -> None:
    """Liquid Voice: any sound move resolves as Water-type instead of its listed type, checked
    after the `-ate` abilities and before every by-name override in `power::type_override`. Round
    (Normal, sound, always-hit, no secondary) is a neutral hit against Dewgong (Water/Ice) as a
    Normal move -- no `Effectiveness` line at all, since only 2x-or-more and 0.5x-or-less get one --
    and a resisted one as Water, which is the clearest sign the type actually changed.

    Vacuity-checked directly: removing the `"LIQUID_VOICE"` clause from `power::type_override`
    turned this into a digest mismatch -- no `Effectiveness` line here where Python's resisted Water
    hit already logged one.
    """
    attacker = [_mon("Machamp", "A0", Ability.LIQUID_VOICE, ["Round"])]
    target = [_mon("Dewgong", "B0", Ability.NONE, ["Splash"])]
    scenario, expected = record((attacker, target), _chooser(random.Random(0)), seed=0, max_turns=1)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"was refused: {theirs}"
    assert compare(expected, theirs) is None
    events = expected[0]["events"]
    assert any(e["type"] == "Effectiveness" and e["level"] == "resisted" for e in events), events


@needs_rust
def test_poison_puppeteer_confuses_whatever_it_poisons(tmp_path: Path) -> None:
    """Poison Puppeteer: the *inflictor's* ability, not the target's, confuses whatever it just
    poisoned -- right after `_reflect_synchronize` in the same status-application function. Toxic
    is 90% accurate, so this is a seed sweep.

    Vacuity-checked directly: removing the `"POISON_PUPPETEER"` check after the Synchronize block
    in `apply_main_status_from` turned every seed where Toxic actually landed into a plain digest
    mismatch -- this engine's missing `VolatileInflicted` where Python's confusion already landed
    alongside the poison.
    """
    puppeteer = [_mon("Tauros", "A0", Ability.POISON_PUPPETEER, ["Toxic"])]
    target = [_mon("Rhydon", "B0", Ability.NONE, ["Splash"])]
    confused = 0
    for seed in range(20):
        rng = random.Random(42100 + seed)
        scenario, expected = record((puppeteer, target), _chooser(rng), seed=seed, max_turns=1)
        theirs = _rust_trace(scenario, tmp_path)
        assert not isinstance(theirs, str), f"seed {seed} was refused: {theirs}"
        assert compare(expected, theirs) is None, f"seed {seed}"
        events = expected[0]["events"]
        poisoned = any(e["type"] == "StatusInflicted" and e["status"] == "TOXIC" for e in events)
        confusion = any(e["type"] == "VolatileInflicted" and e["volatile"] == "CONFUSION" for e in events)
        if poisoned:
            assert confusion, f"seed {seed}: poisoned without confusion\n{events}"
            confused += 1
    assert confused > 0, "Poison Puppeteer never once poisoned (and thus confused) across 20 seeds"


@needs_rust
def test_quick_claw_occasionally_wins_the_bracket_and_consumes_itself(tmp_path: Path) -> None:
    """Quick Claw: the same `_bracket_jump` function Quick Draw already lives in, checked first --
    a 20% chance to move first within the priority bracket, drawn and consumed whether it wins or
    not. Rhydon (40 speed) is far slower than Tauros (110), so any turn it moves first is the item,
    not the stat.

    Vacuity-checked directly: renaming `"QUICK_CLAW"` in `turn::bracket_jump` turned this into a
    tape divergence -- the draw itself disappearing, not just its result, since the Python takes it
    unconditionally too.
    """
    team_a = [
        PokemonSpec(
            species="Rhydon",
            nickname="A0",
            level=50,
            ability=Ability.NONE,
            item=Item.QUICK_CLAW,
            nature=Nature.HARDY,
            moves=["Tackle"],
        )
    ]
    team_b = [
        PokemonSpec(
            species="Tauros",
            nickname="B0",
            level=50,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Tackle"],
        )
    ]
    rhydon_moved_first = 0
    for seed in range(60):
        rng = random.Random(42200 + seed)
        scenario, expected = record((team_a, team_b), _chooser(rng), seed=seed, max_turns=5)
        theirs = _rust_trace(scenario, tmp_path)
        assert not isinstance(theirs, str), f"seed {seed} was refused: {theirs}"
        assert compare(expected, theirs) is None, f"seed {seed}"
        for turn in expected:
            movers = [e["side"] for e in turn["events"] if e["type"] == "MoveUsed"]
            if movers and movers[0] == 0:
                rhydon_moved_first += 1
    assert rhydon_moved_first > 0, "Quick Claw never once won the bracket across 60 seeds x 5 turns"


@needs_rust
def test_custap_berry_guarantees_the_bracket_under_a_quarter_hp(tmp_path: Path) -> None:
    """Custap Berry: not a chance at all, unlike Quick Claw and Quick Draw beside it in the same
    function -- a guaranteed jump to the front of the bracket once its holder is at or under a
    quarter of its max HP, consuming itself the moment it does. Dragon Rage's fixed 40 damage, four
    times over, brings a 180-HP Rhydon to 20 by the start of the fifth turn -- comfortably under
    the 45-HP quarter-mark -- with no damage-roll variance to account for, since both sides carry
    exactly one move each (so any padded slot `_chooser` lands on is that same move). The whole
    scenario is deterministic despite the random pick of which identical padded slot gets used.

    Vacuity-checked directly: renaming `"CUSTAP_BERRY"` in `turn::bracket_jump` turned the final
    turn into a tape divergence -- Tauros moving first in this engine where Python's guaranteed
    jump already sent Rhydon ahead of it.
    """
    team_a = [
        PokemonSpec(
            species="Rhydon",
            nickname="A0",
            level=50,
            ability=Ability.NONE,
            item=Item.CUSTAP_BERRY,
            nature=Nature.HARDY,
            moves=["Splash"],
        )
    ]
    team_b = [
        PokemonSpec(
            species="Tauros",
            nickname="B0",
            level=50,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Dragon Rage"],
        )
    ]

    scenario, expected = record((team_a, team_b), _chooser(random.Random(0)), seed=0, max_turns=5)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"was refused: {theirs}"
    assert compare(expected, theirs) is None
    final_turn = expected[4]["events"]
    movers = [e["side"] for e in final_turn if e["type"] == "MoveUsed"]
    assert movers and movers[0] == 0, f"Rhydon should have won the bracket on the last turn: {movers}"
    a0_digest = expected[4]["state"]["sides"][0]["team"][0]
    assert a0_digest["item"] == "NONE", a0_digest
    assert a0_digest["hp"] <= 45, a0_digest


@needs_rust
def test_leppa_berry_restores_pp_when_a_move_runs_dry(tmp_path: Path) -> None:
    """Leppa Berry: restored the instant *this* spend brings a slot to zero -- a slot already empty
    before this turn takes the Struggle branch instead and never reaches this check at all. Cross
    Chop's 5 PP, spent from the same slot every turn rather than round-robining across the four
    padded copies of it Golem's one-move set carries, empties in exactly 5 turns. Gengar is immune
    to Fighting, so nothing here depends on whether Cross Chop actually lands.
    """
    team_a = [
        PokemonSpec(
            species="Golem",
            nickname="A0",
            level=50,
            ability=Ability.NONE,
            item=Item.LEPPA_BERRY,
            nature=Nature.HARDY,
            moves=["Cross Chop"],
        )
    ]
    team_b = [
        PokemonSpec(
            species="Gengar",
            nickname="B0",
            level=50,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Splash"],
        )
    ]

    def always_first(state, side_index):  # type: ignore[no-untyped-def]
        return Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.FIRST)

    scenario, expected = record((team_a, team_b), always_first, seed=0, max_turns=5)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"was refused: {theirs}"
    assert compare(expected, theirs) is None
    last_turn = expected[4]["events"]
    assert any(e["type"] == "PpRestored" and e["move"] == "Cross Chop" for e in last_turn), last_turn
    a0_digest = expected[4]["state"]["sides"][0]["team"][0]
    assert a0_digest["item"] == "NONE", a0_digest
    assert a0_digest["pp"]["FIRST"] == 5, a0_digest


@needs_rust
def test_chesto_and_lum_berry_cure_their_status_on_the_spot(tmp_path: Path) -> None:
    """Chesto Berry (sleep only) and Lum Berry (any status) cure the instant the status lands --
    same function as Synchronize's reflect and Poison Puppeteer's confusion, checked last. Spore is
    always-hit, so the Chesto case is deterministic; Thunder Wave is 90% accurate, so the Lum case
    is a seed sweep.

    Vacuity-checked directly: removing each berry's own arm from the match in
    `apply_main_status_from` turned its own case into a plain digest mismatch -- this engine's
    still-asleep or still-paralyzed status where Python's `StatusCleared` (clearance `"berry"`)
    already reset it.
    """
    attacker = [
        PokemonSpec(
            species="Machamp",
            nickname="A0",
            level=50,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Spore"],
        )
    ]
    chesto_holder = [
        PokemonSpec(
            species="Rhydon",
            nickname="B0",
            level=50,
            ability=Ability.NONE,
            item=Item.CHESTO_BERRY,
            nature=Nature.HARDY,
            moves=["Splash"],
        )
    ]
    scenario, expected = record((attacker, chesto_holder), _chooser(random.Random(0)), seed=0, max_turns=1)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"Chesto was refused: {theirs}"
    assert compare(expected, theirs) is None, "Chesto"
    events = expected[0]["events"]
    assert any(e["type"] == "StatusInflicted" and e["status"] == "SLEEP" for e in events), events
    assert any(e["type"] == "StatusCleared" and e["clearance"] == "berry" for e in events), events
    b0_digest = expected[0]["state"]["sides"][1]["team"][0]
    assert b0_digest["item"] == "NONE" and b0_digest["status"] == "NONE", b0_digest

    thunder_wave_attacker = [
        PokemonSpec(
            species="Machamp",
            nickname="A0",
            level=50,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Thunder Wave"],
        )
    ]
    lum_holder = [
        PokemonSpec(
            species="Rhydon",
            nickname="B0",
            level=50,
            ability=Ability.NONE,
            item=Item.LUM_BERRY,
            nature=Nature.HARDY,
            moves=["Splash"],
        )
    ]
    cured = 0
    for seed in range(20):
        rng = random.Random(42300 + seed)
        scenario, expected = record((thunder_wave_attacker, lum_holder), _chooser(rng), seed=seed, max_turns=1)
        theirs = _rust_trace(scenario, tmp_path)
        assert not isinstance(theirs, str), f"Lum seed {seed} was refused: {theirs}"
        assert compare(expected, theirs) is None, f"Lum seed {seed}"
        events = expected[0]["events"]
        if any(e["type"] == "StatusInflicted" and e["status"] == "PARALYSIS" for e in events):
            assert any(e["type"] == "StatusCleared" and e["clearance"] == "berry" for e in events), events
            cured += 1
    assert cured > 0, "Lum Berry never once cured a status across 20 seeds"


@needs_rust
def test_weather_rocks_extend_the_duration_to_eight_turns(tmp_path: Path) -> None:
    """The four weather rocks stretch an ordinary 5-turn weather to 8, whether the weather was set
    by a move (`Effect::WeatherEffect`) or by one of the four weather-setting abilities
    (`hooks::on_switch_in`). Heat Rock and Drought's own switch-in makes the ability half
    deterministic; Sunny Day makes the move half deterministic too. The counter itself already
    ticks once in the same turn it is set (both engines' own residual phase), so one turn in it
    reads 7, not 8.
    """
    setter = [
        PokemonSpec(
            species="Kangaskhan",
            nickname="A0",
            level=50,
            ability=Ability.DROUGHT,
            item=Item.HEAT_ROCK,
            nature=Nature.HARDY,
            moves=["Splash"],
        )
    ]
    target = [
        PokemonSpec(
            species="Rhydon",
            nickname="B0",
            level=50,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Splash"],
        )
    ]
    scenario, expected = record((setter, target), _chooser(random.Random(0)), seed=0, max_turns=1)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"ability case was refused: {theirs}"
    assert compare(expected, theirs) is None, "ability case"
    assert expected[0]["state"]["weather_turns_left"] == 7, expected[0]["state"]

    mover = [
        PokemonSpec(
            species="Machamp",
            nickname="A0",
            level=50,
            ability=Ability.NONE,
            item=Item.HEAT_ROCK,
            nature=Nature.HARDY,
            moves=["Sunny Day"],
        )
    ]
    scenario, expected = record((mover, target), _chooser(random.Random(0)), seed=0, max_turns=1)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"move case was refused: {theirs}"
    assert compare(expected, theirs) is None, "move case"
    assert expected[0]["state"]["weather_turns_left"] == 7, expected[0]["state"]


@needs_rust
def test_terrain_seeds_boost_a_stat_the_instant_their_terrain_is_up(tmp_path: Path) -> None:
    """Terrain seeds bind to `ON_SWITCH_IN` at `EventPriority.ITEM` -- after every ability above it
    including Electric Surge's own 2000 -- so a Pokemon that both sets Electric Terrain and holds
    Electric Seed sees its own terrain on the very same switch-in and consumes the seed for +1
    Defense immediately, deterministically (no draw either way).
    """
    holder = [
        PokemonSpec(
            species="Kangaskhan",
            nickname="A0",
            level=50,
            ability=Ability.ELECTRIC_SURGE,
            item=Item.ELECTRIC_SEED,
            nature=Nature.HARDY,
            moves=["Splash"],
        )
    ]
    target = [
        PokemonSpec(
            species="Rhydon",
            nickname="B0",
            level=50,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Splash"],
        )
    ]
    scenario, expected = record((holder, target), _chooser(random.Random(0)), seed=0, max_turns=1)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"was refused: {theirs}"
    assert compare(expected, theirs) is None
    events = expected[0]["events"]
    boosts = [e for e in events if e["type"] == "StatStageChanged" and e["source"] == "seed"]
    assert boosts and boosts[0]["stat"] == "DEFENCE", events
    a0_digest = expected[0]["state"]["sides"][0]["team"][0]
    assert a0_digest["item"] == "NONE", a0_digest


@needs_rust
def test_terrain_seed_fires_for_a_pokemon_that_never_switched_in(tmp_path: Path) -> None:
    """The seed's `ON_SWITCH_IN` binding is not the only path to it: `_set_terrain_from_ability` and
    `_apply_field_effect` both sweep *every* side's seed themselves the instant terrain actually
    changes, on the spot, independent of the bus. A seed holder already standing on the field --
    never switching in itself this turn -- still eats its seed the moment the other side's Pokemon
    sets the matching terrain out from under it. A large-scale differential sweep (3000 battles,
    `--slice status --abilities --switches`) is what found this: 12 seeds diverged before this
    second path existed, because Rhydon's Misty Seed sat un-eaten while an ally's fainted-and-
    replaced Misty Surge lead set the terrain on the following turn, with nobody else switching in
    to trigger the seed's own binding.

    Vacuity-checked directly: removing the sweep calls (leaving only the `ON_SWITCH_IN` path)
    reproduces exactly that: a `StatStageChanged` (source `seed`) this engine's trace has and
    Rust's does not, since the seed holder in this scenario is never the one switching in.
    """
    setter = [
        PokemonSpec(
            species="Kangaskhan",
            nickname="A0",
            level=50,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Electric Terrain"],
        )
    ]
    seed_holder = [
        PokemonSpec(
            species="Rhydon",
            nickname="B0",
            level=50,
            ability=Ability.NONE,
            item=Item.ELECTRIC_SEED,
            nature=Nature.HARDY,
            moves=["Splash"],
        )
    ]
    scenario, expected = record((setter, seed_holder), _chooser(random.Random(0)), seed=0, max_turns=1)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"was refused: {theirs}"
    assert compare(expected, theirs) is None
    events = expected[0]["events"]
    boosts = [e for e in events if e["type"] == "StatStageChanged" and e["source"] == "seed"]
    assert boosts and boosts[0]["side"] == 1 and boosts[0]["stat"] == "DEFENCE", events
    b0_digest = expected[0]["state"]["sides"][1]["team"][0]
    assert b0_digest["item"] == "NONE", b0_digest


@needs_rust
def test_choice_items_lock_the_first_move_used_and_never_re_set(tmp_path: Path) -> None:
    """Choice Band/Scarf/Specs: the same redirect Encore/rampage/charge already use at the top of
    `resolve_move`, checked first of the four -- locked onto whatever slot actually got used the
    moment `choice_locked_move` is still `None`, and never re-set after. A chooser that tries to
    switch to the second slot on turn two still gets the first move's name back, and the item itself
    is never consumed by any of this.

    Vacuity-checked directly: skipping the redirect (leaving the lock recorded but never read) let
    the second turn's `MoveUsed` say "Growl" -- a plain digest/event mismatch against the Python's
    own forced "Tackle".
    """
    team_a = [
        PokemonSpec(
            species="Machamp",
            nickname="A0",
            level=50,
            ability=Ability.NONE,
            item=Item.CHOICE_BAND,
            nature=Nature.HARDY,
            moves=["Tackle", "Growl"],
        )
    ]
    team_b = [
        PokemonSpec(
            species="Rhydon",
            nickname="B0",
            level=50,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Splash"],
        )
    ]

    def try_switch_moves(state, side_index):  # type: ignore[no-untyped-def]
        if side_index == 1:
            return Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.FIRST)
        slot = MoveSlot.FIRST if state.turn == 0 else MoveSlot.SECOND
        return Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=slot)

    scenario, expected = record((team_a, team_b), try_switch_moves, seed=0, max_turns=2)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"was refused: {theirs}"
    assert compare(expected, theirs) is None
    second_turn_moves = [e["move"] for e in expected[1]["events"] if e["type"] == "MoveUsed" and e["side"] == 0]
    assert second_turn_moves == ["Tackle"], second_turn_moves
    a0_digest = expected[1]["state"]["sides"][0]["team"][0]
    assert a0_digest["item"] == "CHOICE_BAND", a0_digest


@needs_rust
def test_focus_sash_survives_an_otherwise_lethal_hit_from_full_hp(tmp_path: Path) -> None:
    """Focus Sash: the same full-HP clamp Sturdy already has in `apply_damage`'s hit loop, checked
    first -- historically the sash is consumed in preference to the ability, which is why the Python
    binds it above the `ABILITY` band rather than beside it. A level 1 Rhydon has too little HP for
    anything else to matter, so a level 50 Machamp's Earthquake -- guaranteed to hit and super
    effective against Ground/Rock -- overkills it every single time regardless of the damage roll.

    Vacuity-checked directly: forcing the clamp's own condition to `false` turned this into a plain
    digest mismatch -- this engine's `DamageDealt` for a fainting blow where Python's own
    `SurvivedAtOneHp` (cause `"focus_sash"`) already clamped it.
    """
    team_a = [
        PokemonSpec(
            species="Rhydon",
            nickname="A0",
            level=1,
            ability=Ability.NONE,
            item=Item.FOCUS_SASH,
            nature=Nature.HARDY,
            moves=["Splash"],
        )
    ]
    team_b = [
        PokemonSpec(
            species="Machamp",
            nickname="B0",
            level=50,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Earthquake"],
        )
    ]
    scenario, expected = record((team_a, team_b), _chooser(random.Random(0)), seed=0, max_turns=1)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"was refused: {theirs}"
    assert compare(expected, theirs) is None
    events = expected[0]["events"]
    assert any(e["type"] == "SurvivedAtOneHp" and e["cause"] == "focus_sash" for e in events), events
    a0_digest = expected[0]["state"]["sides"][0]["team"][0]
    assert a0_digest["hp"] == 1 and a0_digest["item"] == "NONE", a0_digest


@needs_rust
def test_life_orb_boosts_damage_and_chips_its_holder(tmp_path: Path) -> None:
    """Life Orb: ~1.3x on `final_mods_4096` for every hit, unconditional on category -- verified by
    the differential comparison itself, since a wrong multiplier is a wrong `DamageDealt` -- plus a
    tenth of its holder's own max HP back at it, once for the whole move, logged as `ItemChipDamage`
    and never consuming the orb itself. Tackle's 100% accuracy keeps the hit itself deterministic.

    Vacuity-checked directly: removing the `ON_ACTION_RESOLVE` chip left this engine's trace one
    event short of the Python's own `ItemChipDamage` -- a plain event-count mismatch.
    """
    team_a = [
        PokemonSpec(
            species="Machamp",
            nickname="A0",
            level=50,
            ability=Ability.NONE,
            item=Item.LIFE_ORB,
            nature=Nature.HARDY,
            moves=["Tackle"],
        )
    ]
    team_b = [
        PokemonSpec(
            species="Golem",
            nickname="B0",
            level=50,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Splash"],
        )
    ]
    scenario, expected = record((team_a, team_b), _chooser(random.Random(0)), seed=0, max_turns=1)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"was refused: {theirs}"
    assert compare(expected, theirs) is None
    events = expected[0]["events"]
    chip = next((e for e in events if e["type"] == "ItemChipDamage" and e["item"] == "LIFE_ORB"), None)
    assert chip is not None and chip["pokemon"] == "A0", events
    a0_digest = expected[0]["state"]["sides"][0]["team"][0]
    assert a0_digest["item"] == "LIFE_ORB", a0_digest


@needs_rust
def test_eject_button_switches_out_only_after_the_move_finishes_resolving(tmp_path: Path) -> None:
    """Eject Button: announced and its item consumed the instant a hit lands, but the actual switch
    waits for `resolve_pending_switches` -- called once per completed action -- so a single-hit
    move's own `DamageDealt` summary still logs under the Pokemon that is, for the moment, still
    standing. Tackle's 100% accuracy makes the hit itself deterministic.

    Vacuity-checked directly: switching immediately instead of deferring reordered `Switched` ahead
    of `DamageDealt` -- a plain event-order mismatch against the Python's own later resolution.
    """
    team_a = [
        PokemonSpec(
            species="Rhydon",
            nickname="A0",
            level=50,
            ability=Ability.NONE,
            item=Item.EJECT_BUTTON,
            nature=Nature.HARDY,
            moves=["Splash"],
        ),
        PokemonSpec(
            species="Golem",
            nickname="A1",
            level=50,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Splash"],
        ),
    ]
    team_b = [
        PokemonSpec(
            species="Machamp",
            nickname="B0",
            level=50,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Tackle"],
        )
    ]
    scenario, expected = record((team_a, team_b), _chooser(random.Random(0)), seed=0, max_turns=1)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"was refused: {theirs}"
    assert compare(expected, theirs) is None
    events = expected[0]["events"]
    types = [e["type"] for e in events]
    assert "SelfSwitchPending" in types and "Switched" in types and "DamageDealt" in types, events
    assert types.index("DamageDealt") < types.index("Switched"), events
    a0_digest = expected[0]["state"]["sides"][0]["team"][0]
    assert a0_digest["item"] == "NONE", a0_digest


@needs_rust
def test_eject_pack_switches_out_after_an_opponent_inflicted_drop(tmp_path: Path) -> None:
    """Eject Pack: armed by `apply_stage_changes_from` the instant an opponent's move actually drops
    one of the holder's stats, then drained -- logged and consumed only now -- by
    `resolve_pending_switches` once the triggering action finishes. Growl's 100% accuracy and 100%
    probability keep the drop itself deterministic.

    Vacuity-checked directly: never arming the flag left this engine's trace missing the
    `SelfSwitchPending`/`Switched` pair entirely -- a plain event-count mismatch.
    """
    team_a = [
        PokemonSpec(
            species="Rhydon",
            nickname="A0",
            level=50,
            ability=Ability.NONE,
            item=Item.EJECT_PACK,
            nature=Nature.HARDY,
            moves=["Splash"],
        ),
        PokemonSpec(
            species="Golem",
            nickname="A1",
            level=50,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Splash"],
        ),
    ]
    team_b = [
        PokemonSpec(
            species="Machamp",
            nickname="B0",
            level=50,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Growl"],
        )
    ]
    scenario, expected = record((team_a, team_b), _chooser(random.Random(0)), seed=0, max_turns=1)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"was refused: {theirs}"
    assert compare(expected, theirs) is None
    events = expected[0]["events"]
    types = [e["type"] for e in events]
    assert "StatStageChanged" in types and "SelfSwitchPending" in types and "Switched" in types, events
    assert types.index("StatStageChanged") < types.index("SelfSwitchPending") < types.index("Switched"), events
    a0_digest = expected[0]["state"]["sides"][0]["team"][0]
    assert a0_digest["item"] == "NONE", a0_digest


@needs_rust
def test_red_card_forces_the_attacker_out(tmp_path: Path) -> None:
    """Red Card: the same random-bench draw Whirlwind/Roar/Dragon Tail already take
    (`turn::force_random_switch`), reused rather than reimplemented, but dragging the *attacker*
    out instead of the move's own target. Tackle's 100% accuracy makes the hit itself deterministic.
    """
    team_a = [
        PokemonSpec(
            species="Machamp",
            nickname="A0",
            level=50,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Tackle"],
        ),
        PokemonSpec(
            species="Golem",
            nickname="A1",
            level=50,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Splash"],
        ),
    ]
    team_b = [
        PokemonSpec(
            species="Rhydon",
            nickname="B0",
            level=50,
            ability=Ability.NONE,
            item=Item.RED_CARD,
            nature=Nature.HARDY,
            moves=["Splash"],
        )
    ]
    scenario, expected = record((team_a, team_b), _chooser(random.Random(0)), seed=0, max_turns=1)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"was refused: {theirs}"
    assert compare(expected, theirs) is None
    events = expected[0]["events"]
    switched = [e for e in events if e["type"] == "Switched"]
    assert switched and switched[0]["side"] == 0 and switched[0]["withdrew"] == "A0", events
    b0_digest = expected[0]["state"]["sides"][1]["team"][0]
    assert b0_digest["item"] == "NONE", b0_digest


@needs_rust
def test_red_card_forced_switch_still_lets_the_original_attacker_finish_its_own_move(tmp_path: Path) -> None:
    """The bug a 3000-battle sweep found once Red Card existed: Superpower's own Attack/Defense drop
    applies to its user *after* the hit lands, and the Python computes it against the same plain
    `attacker` object reference `_apply_damage` captured at the top of the function -- a reference
    Red Card's own forced switch (synchronous, mid-hit) does not invalidate. `turn::apply_damage` and
    `turn::resolve_move` both now pin `side`'s active back to that original index for everything
    past the switch that is still a direct procedural read of "the attacker" (this drop among them,
    plus recoil, drain and Destiny Bond), restoring the real destination only once each function is
    done needing the original. Without the pin, Rhydon's own switch left `side`'s active pointing at
    its own replacement, and Superpower's drop landed there instead of on the Machamp that was
    actually still fighting.
    """
    team_a = [
        PokemonSpec(
            species="Rhydon",
            nickname="A0",
            level=50,
            ability=Ability.NONE,
            item=Item.RED_CARD,
            nature=Nature.HARDY,
            moves=["Splash"],
        ),
        PokemonSpec(
            species="Golem",
            nickname="A1",
            level=50,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Splash"],
        ),
    ]
    team_b = [
        PokemonSpec(
            species="Machamp",
            nickname="B0",
            level=50,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Superpower"],
        ),
        PokemonSpec(
            species="Tauros",
            nickname="B1",
            level=50,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Splash"],
        ),
    ]
    scenario, expected = record((team_a, team_b), _chooser(random.Random(0)), seed=0, max_turns=1)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"was refused: {theirs}"
    assert compare(expected, theirs) is None
    events = expected[0]["events"]
    assert any(e["type"] == "Switched" and e["side"] == 1 for e in events), events
    drops = [e for e in events if e["type"] == "StatStageChanged" and e["pokemon"] == "B0"]
    assert len(drops) == 2, events  # Superpower's own Attack and Defense drop, still landing on B0


@needs_rust
def test_red_card_mid_multi_hit_stops_the_departed_attackers_own_ability(tmp_path: Path) -> None:
    """The second bug the same sweep found: Tough Claws' own damage-calc contribution is a bus
    subscription, and `_execute_switch` unregisters the departing Pokemon's handlers the instant it
    leaves -- a stale `attacker` object reference does not save a subscription that has already been
    torn down. So a second hit of the same multi-hit move, after Red Card has already forced the
    attacker's own side to switch on the first hit, gets no Tough Claws boost at all, even though the
    damage formula otherwise still reads the original (pinned) attacker's stats for that hit.
    `apply_damage_calc` takes an `attacker_registered` flag for exactly this, flipped the instant
    `apply_damage`'s own hit loop first sees `side`'s active move.

    This is the one place the fix silences a bus contribution instead of pinning it — everything
    else pins. Confirmed by the differential comparison against the Python's own trace, which is
    what would show a damage mismatch on the second hit if this ever regressed.
    """
    team_a = [
        PokemonSpec(
            species="Rhydon",
            nickname="A0",
            level=50,
            ability=Ability.NONE,
            item=Item.RED_CARD,
            nature=Nature.HARDY,
            moves=["Splash"],
        ),
        PokemonSpec(
            species="Golem",
            nickname="A1",
            level=50,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Splash"],
        ),
    ]
    team_b = [
        PokemonSpec(
            species="Kangaskhan",
            nickname="B0",
            level=50,
            ability=Ability.TOUGH_CLAWS,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Double Iron Bash"],
        ),
        PokemonSpec(
            species="Tauros",
            nickname="B1",
            level=50,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Splash"],
        ),
    ]
    scenario, expected = record((team_a, team_b), _chooser(random.Random(0)), seed=0, max_turns=1)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"was refused: {theirs}"
    assert compare(expected, theirs) is None
    events = expected[0]["events"]
    assert any(e["type"] == "Switched" and e["side"] == 1 for e in events), events
    assert any(e["type"] == "MultiHitSummary" and e["hits"] == 2 for e in events), events


@needs_rust
def test_red_card_mid_multi_hit_stops_the_departed_attackers_own_after_hit_reaction(tmp_path: Path) -> None:
    """A third bug the same territory turned up once this batch's own sweeps ran long enough: Poison
    Touch and Toxic Chain are `ON_AFTER_HIT` bus subscriptions too, just like Tough Claws' own
    `ON_DAMAGE_CALC` one -- and `ability_after_hit`'s `is_actor` arm was not gated by
    `attacker_registered` at all, only `apply_damage_calc`'s was. A second hit of the same multi-hit
    move, after Red Card forces the attacker out on the first, drew an extra probability for Poison
    Touch's 30% check that the Python's own trace never drew at all, since the departed Pokemon's
    handler no longer exists to ask -- a tape draw-count mismatch, not just a wrong value, which is
    what makes this one so disruptive: it corrupts every draw for the rest of the battle, not just
    this hit's own damage number.
    """
    team_a = [
        PokemonSpec(
            species="Rhydon",
            nickname="A0",
            level=50,
            ability=Ability.NONE,
            item=Item.RED_CARD,
            nature=Nature.HARDY,
            moves=["Splash"],
        ),
        PokemonSpec(
            species="Golem",
            nickname="A1",
            level=50,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Splash"],
        ),
    ]
    team_b = [
        PokemonSpec(
            species="Kangaskhan",
            nickname="B0",
            level=50,
            ability=Ability.POISON_TOUCH,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Arm Thrust"],
        ),
        PokemonSpec(
            species="Tauros",
            nickname="B1",
            level=50,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Splash"],
        ),
    ]
    checked = 0
    for seed in range(20):
        rng = random.Random(44500 + seed)
        scenario, expected = record((team_a, team_b), _chooser(rng), seed=seed, max_turns=1)
        theirs = _rust_trace(scenario, tmp_path)
        assert not isinstance(theirs, str), f"seed {seed} was refused: {theirs}"
        assert compare(expected, theirs) is None, f"seed {seed}"
        events = expected[0]["events"]
        if not any(e["type"] == "Switched" and e["side"] == 1 for e in events):
            continue
        checked += 1
    assert checked > 0, "Red Card never once triggered mid-multi-hit across 20 seeds"


@needs_rust
def test_eject_pack_and_a_pivots_own_switch_both_arm_before_either_executes(tmp_path: Path) -> None:
    """A fourth bug the same sweep found: an earlier version of `resolve_pending_switches` folded
    "arm this side" and "execute this side" into one call, walked side 0 then side 1 -- so a side-0
    pivot already holding `needs_switch` (armed inline, during its own move) executed its switch
    *before* a side-1 Eject Pack that same move's stat drop had just triggered even got its own
    `SelfSwitchPending` logged. The Python runs `_resolve_eject_packs` (both sides, arm and log only)
    to completion, then `_resolve_pending_switches` (both sides, execute only) to completion --
    two full passes, not one merged pass per side. Parting Shot both drops its target's Attack and
    Sp. Atk (arming Eject Pack) and self-switches its own user (arming its own pivot) in the same
    move, and both are 100% moves, making this deterministic.

    Vacuity-checked directly: re-merging the two passes back into one call per side reordered the
    events exactly as the sweep first found them -- side 0's `Switched` landing ahead of side 1's own
    `SelfSwitchPending`, a plain event-order mismatch against the Python's own arm-both-then-
    execute-both order.
    """
    team_a = [
        PokemonSpec(
            species="Machamp",
            nickname="A0",
            level=50,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Parting Shot"],
        ),
        PokemonSpec(
            species="Golem",
            nickname="A1",
            level=50,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Splash"],
        ),
    ]
    team_b = [
        PokemonSpec(
            species="Rhydon",
            nickname="B0",
            level=50,
            ability=Ability.NONE,
            item=Item.EJECT_PACK,
            nature=Nature.HARDY,
            moves=["Splash"],
        ),
        PokemonSpec(
            species="Tauros",
            nickname="B1",
            level=50,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Splash"],
        ),
    ]
    scenario, expected = record((team_a, team_b), _chooser(random.Random(0)), seed=0, max_turns=1)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"was refused: {theirs}"
    assert compare(expected, theirs) is None
    events = expected[0]["events"]
    kinds = [(e["type"], e["side"]) for e in events if e["type"] in ("SelfSwitchPending", "Switched")]
    assert kinds == [
        ("SelfSwitchPending", 0),
        ("SelfSwitchPending", 1),
        ("Switched", 0),
        ("Switched", 1),
    ], events


@needs_rust
def test_a_pivots_own_switch_is_deferred_behind_a_lower_indexed_side(tmp_path: Path) -> None:
    """A third bug the same sweep found: U-turn's own self-switch was still wired as an immediate,
    inline call in `resolve_move` -- pre-dating this whole batch, just never visible before, since
    nothing else ever competed with it for switch ordering. The Python arms it exactly like Eject
    Button (`side.needs_switch = True` plus a `SelfSwitchPending` log, in `_execute_move` itself),
    and the actual switch waits for `_resolve_pending_switches`, which walks side 0 then side 1 --
    not the order the two flags were armed in. So a defender's Eject Button, armed first
    chronologically (from `on_after_hit`, before the pivot's own check at the end of the same
    function), still resolves *after* the pivot's own switch when the pivot is on side 0: index
    order, not arming order. U-turn's 100% accuracy keeps the hit itself deterministic.

    Vacuity-checked directly: reverting the pivot's own switch to an immediate call reordered the two
    `Switched` events -- a plain event-order mismatch against the Python's own side-0-then-side-1
    resolution.
    """
    team_a = [
        PokemonSpec(
            species="Rhydon",
            nickname="A0",
            level=50,
            ability=Ability.NONE,
            item=Item.EJECT_BUTTON,
            nature=Nature.HARDY,
            moves=["Splash"],
        ),
        PokemonSpec(
            species="Golem",
            nickname="A1",
            level=50,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Splash"],
        ),
    ]
    team_b = [
        PokemonSpec(
            species="Machamp",
            nickname="B0",
            level=50,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["U-turn"],
        ),
        PokemonSpec(
            species="Tauros",
            nickname="B1",
            level=50,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Splash"],
        ),
    ]
    scenario, expected = record((team_a, team_b), _chooser(random.Random(0)), seed=0, max_turns=1)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"was refused: {theirs}"
    assert compare(expected, theirs) is None
    events = expected[0]["events"]
    switches = [e for e in events if e["type"] == "Switched"]
    assert [s["side"] for s in switches] == [0, 1], events


@needs_rust
def test_a_forced_switch_satisfies_its_own_targets_pending_eject(tmp_path: Path) -> None:
    """The bug the same sweep found in `force_random_switch` itself: Dragon Tail both damages its
    target (arming Eject Button's own `needs_switch`) and then, in the Python's own order, drags
    that same target out by force -- and `_force_random_switch` ends by setting
    `side.needs_switch = False` itself, "the drag *is* the replacement," so the Eject Button's own
    already-armed flag never gets to send out a *second* replacement behind the first. Dragon Tail's
    90% accuracy makes this a seed sweep rather than a single deterministic turn.

    Vacuity-checked directly: dropping the reset let a stale `needs_switch` survive the drag and
    send a second Pokemon out behind it -- a real extra `Switched` event, not merely a mismatch.
    """
    team_a = [
        PokemonSpec(
            species="Rhydon",
            nickname="A0",
            level=50,
            ability=Ability.NONE,
            item=Item.EJECT_BUTTON,
            nature=Nature.HARDY,
            moves=["Splash"],
        ),
        PokemonSpec(
            species="Golem",
            nickname="A1",
            level=50,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Splash"],
        ),
    ]
    team_b = [
        PokemonSpec(
            species="Machamp",
            nickname="B0",
            level=50,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Dragon Tail"],
        )
    ]
    checked_a_hit = 0
    for seed in range(20):
        rng = random.Random(44300 + seed)
        scenario, expected = record((team_a, team_b), _chooser(rng), seed=seed, max_turns=1)
        theirs = _rust_trace(scenario, tmp_path)
        assert not isinstance(theirs, str), f"seed {seed} was refused: {theirs}"
        assert compare(expected, theirs) is None, f"seed {seed}"
        events = expected[0]["events"]
        if not any(e["type"] == "DamageDealt" and e["side"] == 0 for e in events):
            continue
        switches = [e for e in events if e["type"] == "Switched" and e["side"] == 0]
        assert len(switches) == 1, (seed, events)
        checked_a_hit += 1
    assert checked_a_hit > 0, "Dragon Tail never once hit across 20 seeds"


@needs_rust
def test_a_phazing_moves_own_force_switch_sees_the_real_board_not_the_stale_attacker(tmp_path: Path) -> None:
    """A second bug the same sweep found, in the opposite direction from every other Red Card fix
    this batch: Circle Throw both damages its target (a Red Card holder, forcing *this* move's own
    attacker out mid-resolution) and then, in the Python's own order, drags that same target out by
    its own `force_switch` -- and the replacement's own `ON_SWITCH_IN` (Intimidate, here) is a fresh
    bus read in the Python, not a direct reference through `_execute_move`'s own local variables, so
    it already sees whichever Pokemon Red Card left standing on the attacker's side. Pinning the
    attacker for the rest of `resolve_move` -- correct for recoil, drain, a move's own post-damage
    self-effect -- would have this same Intimidate drop land on the *departed* attacker instead.
    Circle Throw's 90% accuracy makes this a seed sweep rather than a single deterministic turn.

    Vacuity-checked directly: leaving the attacker pinned through the `force_random_switch` call
    landed Intimidate's drop on the wrong Pokemon -- a plain event mismatch (the wrong `pokemon`
    name) against the Python's own drop on whoever is actually standing there.
    """
    team_a = [
        PokemonSpec(
            species="Machamp",
            nickname="A0",
            level=50,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Circle Throw"],
        ),
        PokemonSpec(
            species="Golem",
            nickname="A1",
            level=50,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Splash"],
        ),
    ]
    team_b = [
        PokemonSpec(
            species="Rhydon",
            nickname="B0",
            level=50,
            ability=Ability.NONE,
            item=Item.RED_CARD,
            nature=Nature.HARDY,
            moves=["Splash"],
        ),
        PokemonSpec(
            species="Kangaskhan",
            nickname="B1",
            level=50,
            ability=Ability.INTIMIDATE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Splash"],
        ),
    ]
    checked_a_hit = 0
    for seed in range(20):
        rng = random.Random(44400 + seed)
        scenario, expected = record((team_a, team_b), _chooser(rng), seed=seed, max_turns=1)
        theirs = _rust_trace(scenario, tmp_path)
        assert not isinstance(theirs, str), f"seed {seed} was refused: {theirs}"
        assert compare(expected, theirs) is None, f"seed {seed}"
        events = expected[0]["events"]
        if not any(e["type"] == "DamageDealt" and e["side"] == 1 for e in events):
            continue
        drop = next((e for e in events if e["type"] == "StatStageChanged" and e["source"] == "intimidate"), None)
        assert drop is not None and drop["pokemon"] == "A1", (seed, events)
        checked_a_hit += 1
    assert checked_a_hit > 0, "Circle Throw never once hit across 20 seeds"


@needs_rust
def test_heavy_duty_boots_blocks_an_entry_hazard(tmp_path: Path) -> None:
    """Heavy Duty Boots cancels `_apply_entry_hazards` entirely, the same way Magic Guard does as an
    ability -- not "no damage from hazards", "no hazards" -- checked at the same early-return site.
    Stealth Rock always succeeds (`accuracy_probability` is `None`), and switching a second team
    member in on turn two is what actually reaches `entry_hazards` at all: a hazard already standing
    when its holder arrives is not retroactive to whoever was already out.
    """
    team_a = [
        PokemonSpec(
            species="Golem",
            nickname="A0",
            level=50,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Stealth Rock", "Splash"],
        )
    ]
    team_b = [
        PokemonSpec(
            species="Machamp",
            nickname="B0",
            level=50,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Splash"],
        ),
        PokemonSpec(
            species="Rhydon",
            nickname="B1",
            level=50,
            ability=Ability.NONE,
            item=Item.HEAVY_DUTY_BOOTS,
            nature=Nature.HARDY,
            moves=["Splash"],
        ),
    ]

    def switch_b_on_turn_one(state, side_index):  # type: ignore[no-untyped-def]
        if side_index == 1 and state.turn == 1:
            return Action(action=ActionType.SWITCH_OUT, switch_in=state.sides[1].team[1])
        options = [a for a in legal_actions(state, side_index) if a.action is ActionType.USE_MOVE]
        return (options or legal_actions(state, side_index))[0]

    scenario, expected = record((team_a, team_b), switch_b_on_turn_one, seed=0, max_turns=2)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"was refused: {theirs}"
    assert compare(expected, theirs) is None
    assert any(e["type"] == "HazardSet" for e in expected[0]["events"]), expected[0]["events"]
    second_turn = expected[1]["events"]
    assert any(e["type"] == "Switched" and e["sent_out"] == "B1" for e in second_turn), second_turn
    assert not any(e["type"] == "HazardDamage" for e in second_turn), second_turn
    b1_digest = expected[1]["state"]["sides"][1]["team"][1]
    assert b1_digest["item"] == "HEAVY_DUTY_BOOTS" and b1_digest["hp"] == b1_digest["max_hp"], b1_digest


@needs_rust
def test_the_drives_change_techno_blasts_type(tmp_path: Path) -> None:
    """The four drives: `power::type_override`'s own already-written match arm, unblocked the
    moment each joins a `PORTED` array. Douse Drive turns Techno Blast to Water, super effective
    against Golem's Rock/Ground -- unreachable if the move had stayed Normal, which has no
    super-effective matchups at all. Techno Blast's 100% accuracy keeps the hit deterministic.
    """
    team_a = [
        PokemonSpec(
            species="Machamp",
            nickname="A0",
            level=50,
            ability=Ability.NONE,
            item=Item.DOUSE_DRIVE,
            nature=Nature.HARDY,
            moves=["Techno Blast"],
        )
    ]
    team_b = [
        PokemonSpec(
            species="Golem",
            nickname="B0",
            level=50,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Splash"],
        )
    ]
    scenario, expected = record((team_a, team_b), _chooser(random.Random(0)), seed=0, max_turns=1)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"was refused: {theirs}"
    assert compare(expected, theirs) is None
    events = expected[0]["events"]
    assert any(e["type"] == "Effectiveness" and e["level"] == "super" for e in events), events


@needs_rust
def test_booster_energy_activates_paradox_without_the_field_condition(tmp_path: Path) -> None:
    """Booster Energy: `evaluate_paradox`'s own already-written fallback, unblocked the moment the
    item joins a `PORTED` array -- no weather, no terrain, just the held item consumed on switch-in.
    """
    team_a = [
        PokemonSpec(
            species="Rhydon",
            nickname="A0",
            level=50,
            ability=Ability.PROTOSYNTHESIS,
            item=Item.BOOSTER_ENERGY,
            nature=Nature.HARDY,
            moves=["Splash"],
        )
    ]
    team_b = [
        PokemonSpec(
            species="Machamp",
            nickname="B0",
            level=50,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Splash"],
        )
    ]
    scenario, expected = record((team_a, team_b), _chooser(random.Random(0)), seed=0, max_turns=1)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"was refused: {theirs}"
    assert compare(expected, theirs) is None
    events = expected[0]["events"]
    activated = next((e for e in events if e["type"] == "ParadoxActivated"), None)
    assert activated is not None and activated["from_booster"], events
    a0_digest = expected[0]["state"]["sides"][0]["team"][0]
    assert a0_digest["item"] == "NONE", a0_digest


@needs_rust
def test_punching_glove_boosts_punching_moves_without_making_them_contact(tmp_path: Path) -> None:
    """Punching Glove's other half: 1.1x on `power_mods_4096` for any punching move, unconditional
    of category -- verified by the differential comparison, since a wrong multiplier is a wrong
    `DamageDealt`. Its contact-negation half already lived at `makes_contact`'s own inline site
    before this batch; Mach Punch's 100% accuracy keeps the hit itself deterministic.
    """
    team_a = [
        PokemonSpec(
            species="Machamp",
            nickname="A0",
            level=50,
            ability=Ability.NONE,
            item=Item.PUNCHING_GLOVE,
            nature=Nature.HARDY,
            moves=["Mach Punch"],
        )
    ]
    team_b = [
        PokemonSpec(
            species="Golem",
            nickname="B0",
            level=50,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Splash"],
        )
    ]
    scenario, expected = record((team_a, team_b), _chooser(random.Random(0)), seed=0, max_turns=1)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"was refused: {theirs}"
    assert compare(expected, theirs) is None
    events = expected[0]["events"]
    assert any(e["type"] == "DamageDealt" for e in events), events


@needs_rust
def test_terrain_extender_stretches_the_terrain_to_eight_turns(tmp_path: Path) -> None:
    """Terrain Extender: the same duration bump the weather rocks already have, at both places
    terrain gets set (an ability's own switch-in, and a terrain move) -- checked here via the
    ability path, which is deterministic on its own switch-in.
    """
    team_a = [
        PokemonSpec(
            species="Kangaskhan",
            nickname="A0",
            level=50,
            ability=Ability.ELECTRIC_SURGE,
            item=Item.TERRAIN_EXTENDER,
            nature=Nature.HARDY,
            moves=["Splash"],
        )
    ]
    team_b = [
        PokemonSpec(
            species="Rhydon",
            nickname="B0",
            level=50,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Splash"],
        )
    ]
    scenario, expected = record((team_a, team_b), _chooser(random.Random(0)), seed=0, max_turns=1)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"was refused: {theirs}"
    assert compare(expected, theirs) is None
    # One turn already ticked the counter down from its nominal 8, the same as the weather rocks.
    assert expected[0]["state"]["terrain_turns_left"] == 7, expected[0]["state"]


@needs_rust
def test_light_clay_stretches_a_screen_to_eight_turns(tmp_path: Path) -> None:
    """Light Clay: the caster's own held item extending Reflect/Light Screen/Aurora Veil from 5
    turns to 8, checked at the same site the screen itself gets set. The counter itself already
    ticks once in the same turn it's set (the field's own residual tick), so one turn in it reads
    7, not 8 -- the same convention the weather rocks and Terrain Extender both follow.
    """
    team_a = [
        PokemonSpec(
            species="Machamp",
            nickname="A0",
            level=50,
            ability=Ability.NONE,
            item=Item.LIGHT_CLAY,
            nature=Nature.HARDY,
            moves=["Reflect"],
        )
    ]
    team_b = [
        PokemonSpec(
            species="Rhydon",
            nickname="B0",
            level=50,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Splash"],
        )
    ]
    scenario, expected = record((team_a, team_b), _chooser(random.Random(0)), seed=0, max_turns=1)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"was refused: {theirs}"
    assert compare(expected, theirs) is None
    a0_digest = expected[0]["state"]["sides"][0]
    assert a0_digest["screens"]["REFLECT"] == 7, a0_digest


@needs_rust
def test_clear_amulet_blocks_an_opponent_inflicted_drop(tmp_path: Path) -> None:
    """Clear Amulet: the same whole-request block Clear Body/Full Metal Body/White Smoke already
    have in `intercept_drops`, checked as an item rather than an ability. Growl's 100% accuracy and
    100% probability keep the drop attempt itself deterministic.
    """
    team_a = [
        PokemonSpec(
            species="Rhydon",
            nickname="A0",
            level=50,
            ability=Ability.NONE,
            item=Item.CLEAR_AMULET,
            nature=Nature.HARDY,
            moves=["Splash"],
        )
    ]
    team_b = [
        PokemonSpec(
            species="Machamp",
            nickname="B0",
            level=50,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Growl"],
        )
    ]
    scenario, expected = record((team_a, team_b), _chooser(random.Random(0)), seed=0, max_turns=1)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"was refused: {theirs}"
    assert compare(expected, theirs) is None
    events = expected[0]["events"]
    assert any(e["type"] == "StatDropBlockedByItem" and e["item"] == "CLEAR_AMULET" for e in events), events
    assert not any(e["type"] == "StatStageChanged" for e in events), events
    a0_digest = expected[0]["state"]["sides"][0]["team"][0]
    assert a0_digest["item"] == "CLEAR_AMULET", a0_digest


@needs_rust
def test_white_herb_restores_a_negative_stage_the_instant_it_appears(tmp_path: Path) -> None:
    """White Herb: unconditional of whether the call that just ran caused a drop at all -- any stat
    still sitting negative, from any earlier call, resets to zero the moment any stage change
    resolves. Growl's own drop is what puts a stat negative in the first place, and the tail of the
    very same `apply_stage_changes_from` call restores it, all on one turn.
    """
    team_a = [
        PokemonSpec(
            species="Rhydon",
            nickname="A0",
            level=50,
            ability=Ability.NONE,
            item=Item.WHITE_HERB,
            nature=Nature.HARDY,
            moves=["Splash"],
        )
    ]
    team_b = [
        PokemonSpec(
            species="Machamp",
            nickname="B0",
            level=50,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Growl"],
        )
    ]
    scenario, expected = record((team_a, team_b), _chooser(random.Random(0)), seed=0, max_turns=1)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"was refused: {theirs}"
    assert compare(expected, theirs) is None
    events = expected[0]["events"]
    assert any(e["type"] == "StatStageChanged" and e["stat"] == "ATTACK" and e["delta"] == -1 for e in events), events
    assert any(e["type"] == "WhiteHerbRestored" and e["pokemon"] == "A0" for e in events), events
    a0_digest = expected[0]["state"]["sides"][0]["team"][0]
    assert a0_digest["item"] == "NONE" and a0_digest["stages"]["ATTACK"] == 0, a0_digest


@needs_rust
def test_adrenaline_orb_boosts_speed_off_an_intimidate_drop(tmp_path: Path) -> None:
    """Adrenaline Orb: Intimidate specifically, not any other opponent-inflicted drop -- checked
    only once the drop has actually gone through, on the lead switch-in both sides get for free.
    """
    team_a = [
        PokemonSpec(
            species="Rhydon",
            nickname="A0",
            level=50,
            ability=Ability.NONE,
            item=Item.ADRENALINE_ORB,
            nature=Nature.HARDY,
            moves=["Splash"],
        )
    ]
    team_b = [
        PokemonSpec(
            species="Machamp",
            nickname="B0",
            level=50,
            ability=Ability.INTIMIDATE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Splash"],
        )
    ]
    scenario, expected = record((team_a, team_b), _chooser(random.Random(0)), seed=0, max_turns=1)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"was refused: {theirs}"
    assert compare(expected, theirs) is None
    events = expected[0]["events"]
    assert any(
        e["type"] == "StatStageChanged" and e["stat"] == "ATTACK" and e["source"] == "intimidate" for e in events
    ), events
    assert any(
        e["type"] == "StatStageChanged" and e["stat"] == "SPEED" and e["source"] == "seed" and e["pokemon"] == "A0"
        for e in events
    ), events
    a0_digest = expected[0]["state"]["sides"][0]["team"][0]
    assert a0_digest["item"] == "NONE", a0_digest


@needs_rust
def test_covert_cloak_blocks_a_secondary_effect_like_shield_dust(tmp_path: Path) -> None:
    """Covert Cloak: `_blocks_secondaries`, Shield Dust or the item, either one refusing a secondary
    the same way -- no draw at all, not even a suppressed one, since the block happens ahead of the
    probability roll. Rock Smash's 50% Defense-drop secondary would make this a seed sweep without
    the block; with it, the outcome is deterministic regardless of what the roll would have been.
    """
    team_a = [
        PokemonSpec(
            species="Machamp",
            nickname="A0",
            level=50,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Rock Smash"],
        )
    ]
    team_b = [
        PokemonSpec(
            species="Rhydon",
            nickname="B0",
            level=50,
            ability=Ability.NONE,
            item=Item.COVERT_CLOAK,
            nature=Nature.HARDY,
            moves=["Splash"],
        )
    ]
    cleared = 0
    for seed in range(20):
        rng = random.Random(44100 + seed)
        scenario, expected = record((team_a, team_b), _chooser(rng), seed=seed, max_turns=1)
        theirs = _rust_trace(scenario, tmp_path)
        assert not isinstance(theirs, str), f"seed {seed} was refused: {theirs}"
        assert compare(expected, theirs) is None, f"seed {seed}"
        events = expected[0]["events"]
        assert not any(e["type"] == "StatStageChanged" for e in events), (seed, events)
        cleared += 1
    assert cleared == 20


@needs_rust
def test_loaded_dice_never_rolls_the_bottom_of_a_wide_multi_hit_range(tmp_path: Path) -> None:
    """Loaded Dice: only on a spread of two or more between the low and high hit counts -- folds
    the roll up to `high - 1` or `high`, so a 2-5-hit move (Bullet Seed) never lands at 2 or 3 while
    the item is held. Machamp is the target rather than the (4x-weak) Rhydon specifically so it
    survives all five hits on every seed -- `MultiHitSummary` reports hits *landed*, and a target
    that faints partway through would report a short count for a reason that has nothing to do with
    the item.

    Not vacuity-checked by disabling the code and watching the differential fail: the tape replays
    whatever integer the Python's own recording drew, and a 4-or-5 draw from the item's own narrow
    range is *also* a valid answer to the plain 2-through-5 draw the disabled code would ask for
    instead, so the comparator has nothing to catch. This statistical check against the Python's own
    independently-computed distribution, across seeds the item's own narrower range actually
    constrains, is what stands in for it.
    """
    team_a = [
        PokemonSpec(
            species="Golem",
            nickname="A0",
            level=50,
            ability=Ability.NONE,
            item=Item.LOADED_DICE,
            nature=Nature.HARDY,
            moves=["Bullet Seed"],
        )
    ]
    team_b = [
        PokemonSpec(
            species="Machamp",
            nickname="B0",
            level=50,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Splash"],
        )
    ]
    seen_hits: set[int] = set()
    for seed in range(20):
        rng = random.Random(44200 + seed)
        scenario, expected = record((team_a, team_b), _chooser(rng), seed=seed, max_turns=1)
        theirs = _rust_trace(scenario, tmp_path)
        assert not isinstance(theirs, str), f"seed {seed} was refused: {theirs}"
        assert compare(expected, theirs) is None, f"seed {seed}"
        summary = next(e for e in expected[0]["events"] if e["type"] == "MultiHitSummary")
        seen_hits.add(summary["hits"])
    assert seen_hits and seen_hits.issubset({4, 5}), seen_hits


@needs_rust
def test_power_herb_skips_the_charging_turn(tmp_path: Path) -> None:
    """Power Herb: consumed to fire a charge move the same turn it is chosen, checked last in
    `_skips_charge_turn`'s own order (after the sun-skip check) -- so a Solar Beam not helped by any
    weather still lands on turn one instead of announcing `ChargingUp`.
    """
    team_a = [
        PokemonSpec(
            species="Machamp",
            nickname="A0",
            level=50,
            ability=Ability.NONE,
            item=Item.POWER_HERB,
            nature=Nature.HARDY,
            moves=["Solar Beam"],
        )
    ]
    team_b = [
        PokemonSpec(
            species="Rhydon",
            nickname="B0",
            level=50,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Splash"],
        )
    ]
    scenario, expected = record((team_a, team_b), _chooser(random.Random(0)), seed=0, max_turns=1)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"was refused: {theirs}"
    assert compare(expected, theirs) is None
    events = expected[0]["events"]
    assert not any(e["type"] == "ChargingUp" for e in events), events
    assert any(e["type"] == "DamageDealt" for e in events), events
    a0_digest = expected[0]["state"]["sides"][0]["team"][0]
    assert a0_digest["item"] == "NONE", a0_digest


@needs_rust
def test_mental_herb_cures_taunt_the_instant_it_lands(tmp_path: Path) -> None:
    """Mental Herb: Taunt, Encore or Disable, cured the instant any of them lands -- Taunt's own
    100% accuracy keeps this deterministic, and it reaches the cure through the same generic
    `apply_volatile` path every other status-inflicting volatile does.
    """
    team_a = [
        PokemonSpec(
            species="Rhydon",
            nickname="A0",
            level=50,
            ability=Ability.NONE,
            item=Item.MENTAL_HERB,
            nature=Nature.HARDY,
            moves=["Splash"],
        )
    ]
    team_b = [
        PokemonSpec(
            species="Machamp",
            nickname="B0",
            level=50,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Taunt"],
        )
    ]
    scenario, expected = record((team_a, team_b), _chooser(random.Random(0)), seed=0, max_turns=1)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"was refused: {theirs}"
    assert compare(expected, theirs) is None
    events = expected[0]["events"]
    assert any(e["type"] == "VolatileInflicted" and e["volatile"] == "TAUNT" for e in events), events
    assert any(e["type"] == "StatusCleared" and e["clearance"] == "berry" for e in events), events
    a0_digest = expected[0]["state"]["sides"][0]["team"][0]
    assert a0_digest["item"] == "NONE" and "TAUNT" not in a0_digest["volatiles"], a0_digest


@needs_rust
def test_mirror_herb_copies_the_opponents_own_self_raise(tmp_path: Path) -> None:
    """Mirror Herb: the opponent's own self-targeted raise, copied onto the holder the instant it
    lands -- checked against the move's own requested stages, unconditional of whether any of them
    actually moved a clamped stat. Swords Dance's own effect always succeeds, so the copy is
    deterministic.
    """
    team_a = [
        PokemonSpec(
            species="Machamp",
            nickname="A0",
            level=50,
            ability=Ability.NONE,
            item=Item.NONE,
            nature=Nature.HARDY,
            moves=["Swords Dance"],
        )
    ]
    team_b = [
        PokemonSpec(
            species="Rhydon",
            nickname="B0",
            level=50,
            ability=Ability.NONE,
            item=Item.MIRROR_HERB,
            nature=Nature.HARDY,
            moves=["Splash"],
        )
    ]
    scenario, expected = record((team_a, team_b), _chooser(random.Random(0)), seed=0, max_turns=1)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"was refused: {theirs}"
    assert compare(expected, theirs) is None
    events = expected[0]["events"]
    a0_boost = next(e for e in events if e["type"] == "StatStageChanged" and e["pokemon"] == "A0")
    assert a0_boost["stat"] == "ATTACK" and a0_boost["delta"] == 2, events
    b0_boost = next(e for e in events if e["type"] == "StatStageChanged" and e["pokemon"] == "B0")
    assert b0_boost["stat"] == "ATTACK" and b0_boost["delta"] == 2 and b0_boost["source"] == "seed", events
    b0_digest = expected[0]["state"]["sides"][1]["team"][0]
    assert b0_digest["item"] == "NONE", b0_digest


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
        species="Rhydon",
        nickname="P0",
        level=50,
        ability=Ability.NONE,
        item=Item.NONE,
        nature=Nature.HARDY,
        moves=["Earthquake", "Rock Slide"],
    )
    benched = PokemonSpec(
        species="Rhydon",
        nickname="P1",
        level=50,
        ability=Ability[unported],
        item=Item.NONE,
        nature=Nature.HARDY,
        moves=["Earthquake", "Rock Slide"],
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
