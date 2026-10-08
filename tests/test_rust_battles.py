"""Whole battles, played by both engines, compared turn for turn."""

import json
import random
import subprocess
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Any, Final

import pytest

from battle_sim.database.loader import get_all_moves
from battle_sim.differential import Chooser, Scenario, compare, record
from battle_sim.engine import legal_actions
from battle_sim.models.actions import Action, ActionType
from battle_sim.models.moves import (
    CodedEffect,
    DamageEffect,
    FixedDamageEffect,
    HealEffect,
    InflictStatusEffect,
    Move,
    MoveEffect,
    MoveSlot,
    PseudoWeatherEffect,
    RemoveHazardsEffect,
    SideConditionEffect,
    StatStageChangeEffect,
    TerrainEffect,
    WeatherEffect,
)
from battle_sim.models.spec import PokemonSpec
from battle_sim.models.stats import EVs, IVs
from battle_sim.utils import Ability, Item, Nature, Target

RUST = Path(__file__).resolve().parent.parent / "rust"
BINARY = RUST / "target" / "release" / "replay"
DATA = RUST / "data"

needs_rust = pytest.mark.skipif(
    not BINARY.exists(), reason="the Rust crate is not built; run `cargo build --release` in rust/"
)

_ALWAYS_PORTED: Final = (
    StatStageChangeEffect,
    FixedDamageEffect,
    HealEffect,
    WeatherEffect,
    TerrainEffect,
    PseudoWeatherEffect,
    SideConditionEffect,
    RemoveHazardsEffect,
)
# Species with no ability the engine binds behaviour for, so nothing modifies damage on either side.
PLAIN_SPECIES = ("Rhydon", "Machamp", "Kangaskhan", "Tauros", "Dewgong", "Golem")


def _plain_damage(effect: MoveEffect) -> bool:
    """A damage effect with a listed power."""
    return isinstance(effect, DamageEffect) and bool(effect.power)


def _plain_move_names() -> list[str]:
    """Damaging moves with a fixed power, no secondary, and nothing clever attached."""
    coded = set(json.loads((DATA / "rules.json").read_text())["coded_moves"])
    wanted = []
    for move in get_all_moves().values():
        effects = move.effects
        if len(effects) != 1 or not _plain_damage(effects[0]):
            continue
        if move.name in coded:
            continue
        wanted.append(move.name)
    return sorted(wanted)


def _status_and_stage_move_names() -> list[str]:
    """The wider slice: moves that also inflict a status, move stat stages or land a volatile."""
    landable = {"BURN", "FREEZE", "PARALYSIS", "POISON", "TOXIC", "SLEEP", *PORTED["volatiles"]}

    def ported(move: Move, effect: MoveEffect) -> bool:
        if isinstance(effect, InflictStatusEffect):
            return effect.status.name in landable
        if isinstance(effect, CodedEffect):
            return effect.kind.name in PORTED["coded_kinds"]
        if isinstance(effect, DamageEffect):
            # A coded move with a power formula is exempt from the plain-damage test.
            return _plain_damage(effect) or move.name in PORTED["coded_moves"]
        return isinstance(effect, _ALWAYS_PORTED)

    coded = set(json.loads((DATA / "rules.json").read_text())["coded_moves"]) - set(PORTED["coded_moves"])
    return sorted(
        move.name
        for move in get_all_moves().values()
        if move.name not in coded and all(ported(move, effect) for effect in move.effects)
    )


def _ported() -> dict[str, list[str]]:
    """What the Rust engine says it has implemented."""
    if not BINARY.exists():
        return {"abilities": [], "items": [], "coded_moves": [], "coded_kinds": [], "volatiles": []}
    out = subprocess.run([str(BINARY), "--ported"], capture_output=True, text=True, check=True).stdout
    listed: dict[str, list[str]] = json.loads(out)
    return listed


PORTED = _ported()


def _still_unported(kind: str, enum: type[Enum]) -> str:
    """Some live ability or item the engine has *not* implemented, whichever one that happens to be."""
    rules = json.loads((DATA / "rules.json").read_text())
    live: set[str] = set(rules[f"live_{kind}"])
    if kind == "items":
        # Mega Stones are refused only conditionally, so they cannot serve as the unported example.
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
    # NONE stays in each draw so a bare Pokemon and a mixed board are tested too.
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


def _chooser(rng: random.Random):
    """Only moves, so the ported slice is what gets exercised."""

    def choose(state, side_index):
        options = [a for a in legal_actions(state, side_index) if a.action is ActionType.USE_MOVE]
        return rng.choice(options or legal_actions(state, side_index))

    return choose


UNPORTED, DIVERGED = 2, 3


def _switching_chooser(rng: random.Random):
    """Like `_chooser`, but switches sometimes — which some mechanics need in order to happen."""

    def choose(state, side_index):
        options = legal_actions(state, side_index)
        moves = [a for a in options if a.action is ActionType.USE_MOVE]
        if moves and rng.random() < 0.3 and len(options) > len(moves):
            return rng.choice([a for a in options if a.action is not ActionType.USE_MOVE])
        return rng.choice(moves or options)

    return choose


def _rust_trace(scenario: Scenario, tmp_path: Path) -> list[dict[str, Any]] | str:
    """The Rust engine's answer, or the reason it declined to give one."""
    path = tmp_path / "scenario.json"
    path.write_text(scenario.to_json())
    result = subprocess.run([str(BINARY), str(path), str(DATA)], capture_output=True, text=True)
    if result.returncode == UNPORTED:
        return result.stderr.strip()
    assert result.returncode != DIVERGED, f"the engines took different paths: {result.stderr.strip()}"
    assert result.returncode == 0, f"replay failed ({result.returncode}): {result.stderr}"
    parsed: list[dict[str, Any]] = json.loads(result.stdout)
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
    """The wider slice: burns, poisons, sleeps and stat drops."""
    rng = random.Random(1000 + seed)
    teams = (_team(rng, pool=STATUS_MOVES), _team(rng, pool=STATUS_MOVES))
    scenario, expected = record(teams, _chooser(rng), seed=seed, max_turns=40)

    theirs = _rust_trace(scenario, tmp_path)

    if isinstance(theirs, str):
        pytest.skip(f"outside the ported slice: {theirs}")
    divergence = compare(expected, theirs)
    assert divergence is None, f"seed {seed}\n{divergence}"


# No unported coded move remains, so the refusal test that needed one is gone.


@needs_rust
@pytest.mark.parametrize("seed", range(25))
def test_the_ported_abilities_agree_too(seed: int, tmp_path: Path) -> None:
    """The same battles, with an ability on nearly every Pokemon."""
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
    """Flinches and confusions have to happen, not merely be permitted."""
    # A hundred and twenty battles, so a flinch reliably lands on someone still waiting.
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


def _inflicts(effect: MoveEffect, volatile: str) -> bool:
    """A move effect that lands this particular volatile."""
    return isinstance(effect, InflictStatusEffect) and effect.status.name == volatile


def _moves_where(predicate) -> list[str]:
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
    """Multi-hit, drain and recoil, drawn deliberately rather than hoped for."""
    pool = _moves_where(predicate)
    assert pool, "no move in the slice carries this effect"

    def team(rng: random.Random) -> list[PokemonSpec]:
        # One move from the class and one ordinary attack each.
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
        # Switching on purpose, so entry hazards are paid for by a Pokemon arriving.
        scenario, expected = record(teams, _switching_chooser(rng), seed=seed, max_turns=60)
        theirs = _rust_trace(scenario, tmp_path)
        assert not isinstance(theirs, str), f"seed {seed} was refused: {theirs}"
        divergence = compare(expected, theirs)
        assert divergence is None, f"seed {seed}\n{divergence}"
        seen += sum(entry["type"] == event for turn in expected for entry in turn["events"])

    assert seen > 0, f"{event} never happened even with a team built for it"


@needs_rust
def test_charge_moves_actually_charge_and_agree(tmp_path: Path) -> None:
    """All 17 two-turn moves, forced to actually charge rather than merely being legal."""
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
    """Outrage and its three relatives: the lock, the free turns and the confusion on the way out."""
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
    """Rollout and Ice Ball: a run that doubles its power and stops clean."""
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
    """Taunt, Encore and Disable: three ways of taking a move off the table."""
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
    """A turn where the Destiny Bond user faints has to take its attacker with it."""
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
    """Foresight/Odor Sleuth and Miracle Eye actually let their moves land, not just resolve."""

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
    """The whole Substitute lifecycle: standing up, soaking, breaking and blocking."""
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
    """Sleep Talk: acting through sleep."""
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
    """A roosted Flying type loses its immunity to Ground moves for the turn it roosted."""
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
    """Rest, Haze, Court Change, Perish Song and Curse's Ghost half each log their own entry and agree."""
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
    """Knock Off's item removal, Trick, Skill Swap, Entrainment and Worry Seed."""
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


def _mega_reaching_an_unported_ability() -> tuple[str, Item]:
    """A base species and the stone that takes it to a forme whose ability is still unported."""
    rules = json.loads((DATA / "rules.json").read_text())
    species = {entry["name"]: entry for entry in json.loads((DATA / "species.json").read_text())["species"].values()}
    live = set(rules["live_abilities"]) - set(PORTED["abilities"])
    for row in rules["mega_formes"]:
        forme = species.get(row["forme"])
        if forme is not None and forme.get("regular_ability") in live and row["item"] in Item.__members__:
            return str(forme["base_species"]), Item[row["item"]]
    raise AssertionError("every Mega forme's ability is ported; this test needs rewriting")


@needs_rust
def test_a_mega_stone_reaching_an_unported_ability_is_refused_not_played_wrong(tmp_path: Path) -> None:
    """A Pokemon that would reach a forme with an unported ability is refused."""
    base, stone = _mega_reaching_an_unported_ability()
    team = [
        PokemonSpec(
            species=base,
            nickname="A0",
            level=50,
            ability=Ability.NONE,
            item=stone,
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
    """Mega Evolution resolves before the turn is ordered, so the Mega's speed decides it."""
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
    # Aerodactyl moved first, which needs the Mega's 150 Speed.
    first_move = next(e for e in events if e["type"] == "MoveUsed")
    assert first_move["pokemon"] == "A0", events


@needs_rust
def test_switching_out_forfeits_mega_evolution_that_turn(tmp_path: Path) -> None:
    """A side that chose to switch does not Mega Evolve — the two are mutually exclusive in one turn."""
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
    """Necrozma-Dusk-Mane with Ultranecrozium Z Ultra Bursts into Necrozma-Ultra."""
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
def test_primal_reversion_does_not_use_the_mega_slot(tmp_path: Path) -> None:
    """Primal Groudon and Mega Rayquaza transform on the same side."""
    team_a = [
        PokemonSpec(
            species="Groudon",
            nickname="A0",
            level=50,
            ability=Ability.DROUGHT,
            item=Item.RED_ORB,
            moves=["Precipice Blades"],
        ),
        PokemonSpec(
            species="Rayquaza",
            nickname="A1",
            level=50,
            ability=Ability.AIR_LOCK,
            item=Item.LIFE_ORB,
            moves=["Dragon Ascent"],
        ),
    ]
    team_b = [PokemonSpec(species="Blissey", nickname="B0", level=100, moves=["Soft-Boiled"])]
    turn = {0: 0}

    def choose(state, side_index):
        options = legal_actions(state, side_index)
        if side_index == 0:
            turn[0] += 1
            if turn[0] == 2:
                return next(a for a in options if a.action is ActionType.SWITCH_OUT)
        return next(a for a in options if a.action is ActionType.USE_MOVE)

    scenario, expected = record((team_a, team_b), choose, seed=0, max_turns=3)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"was refused: {theirs}"
    assert compare(expected, theirs) is None
    formes = [e["forme"] for turn_ in expected for e in turn_["events"] if e["type"] == "FormeChanged"]
    assert formes == ["Groudon-Primal", "Rayquaza-Mega"], formes


@needs_rust
def test_mega_rayquaza_is_gated_on_dragon_ascent_not_an_item(tmp_path: Path) -> None:
    """Mega Rayquaza, reached by knowing Dragon Ascent, sets Delta Stream."""
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
    """Groudon with the Red Orb reaches Groudon-Primal and Desolate Land sets harsh sunlight."""
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
    # Two weather events land this turn: Drought's sun, then Desolate Land after Primal Reversion.
    weather_events = [e for e in turn0 if e["type"] == "WeatherSetByAbility"]
    assert [e["ability"] for e in weather_events] == ["DROUGHT", "DESOLATE_LAND"], turn0
    assert expected[0]["state"]["weather"] == "HARSH_SUN", expected[0]["state"]
    assert expected[1]["state"]["weather"] == "RAIN", expected[1]["state"]


@needs_rust
def test_delta_stream_cuts_a_flying_types_weaknesses_to_neutral(tmp_path: Path) -> None:
    """Delta Stream halves Ice's 4x against Mega Rayquaza."""
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
    """Groundium Z on a Ground move: the crystal's name and type, Earthquake's power through the table."""
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
    # The crystal is spent as a resource, never removed from its holder.
    a0_digest = expected[0]["state"]["sides"][0]["team"][0]
    assert a0_digest["item"] == "GROUNDIUM_Z", a0_digest


@needs_rust
def test_a_signature_z_crystal_ignores_the_power_table_entirely(tmp_path: Path) -> None:
    """Aloraichium Z on Thunderbolt becomes Stoked Sparksurfer with its own power and paralysis."""
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
    """`legal_actions` stops offering a Z-move once one has been used."""
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
        # Flying-type, so it survives the Z-move to a second turn.
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
    """Arceus with an Iron Plate is Steel-type and its Judgment is Steel."""
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
    """Arceus with no plate stays Normal and logs no type change."""
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
    """Silvally with a Fire Memory is Fire-type."""
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
@pytest.mark.parametrize(
    ("species", "ability", "item"),
    [
        ("Arceus", Ability.MULTITYPE, Item.IRON_PLATE),
        ("Giratina-Origin", Ability.LEVITATE, Item.GRISEOUS_ORB),
    ],
)
def test_knock_off_cannot_take_an_item_that_defines_its_holder(
    species: str, ability: Ability, item: Item, tmp_path: Path
) -> None:
    """`formes.is_fused_to`'s two special cases."""
    team_a = [PokemonSpec(species=species, nickname="A0", level=50, ability=ability, item=item, moves=["Splash"])]
    team_b = [PokemonSpec(species="Machamp", nickname="B0", level=50, moves=["Knock Off"])]
    scenario, expected = record((team_a, team_b), _chooser(random.Random(0)), seed=0, max_turns=2)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"was refused: {theirs}"
    assert compare(expected, theirs) is None
    assert any(e["type"] == "MoveUsed" and e["move"] == "Knock Off" for e in expected[0]["events"])
    assert expected[-1]["state"]["sides"][0]["team"][0]["item"] == item.name


@needs_rust
def test_the_griseous_orb_boosts_giratinas_dragon_and_ghost_moves(tmp_path: Path) -> None:
    """The Griseous Orb boosts Dragon and Ghost moves for the Giratina line."""

    def damage(item: Item) -> int:
        team_a = [
            PokemonSpec(
                species="Giratina-Origin",
                nickname="A0",
                level=50,
                ability=Ability.LEVITATE,
                item=item,
                moves=["Dragon Claw"],
            )
        ]
        team_b = [PokemonSpec(species="Rhydon", nickname="B0", level=50, moves=["Splash"])]
        scenario, expected = record((team_a, team_b), _chooser(random.Random(0)), seed=3, max_turns=1)
        theirs = _rust_trace(scenario, tmp_path)
        assert not isinstance(theirs, str), f"was refused: {theirs}"
        assert compare(expected, theirs) is None
        b0 = expected[0]["state"]["sides"][1]["team"][0]
        return int(b0["max_hp"] - b0["hp"])

    assert damage(Item.GRISEOUS_ORB) > damage(Item.NONE)


@needs_rust
def test_an_air_balloon_floats_over_ground_until_a_hit_pops_it(tmp_path: Path) -> None:
    """Air Balloon announces itself, floats over Earthquake and bursts on the next hit."""
    team_a = [PokemonSpec(species="Rhydon", nickname="A0", level=50, item=Item.AIR_BALLOON, moves=["Splash"])]
    team_b = [PokemonSpec(species="Machamp", nickname="B0", level=50, moves=["Earthquake", "Tackle"])]
    scenario, expected = record((team_a, team_b), _round_robin_chooser(), seed=0, max_turns=3)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"was refused: {theirs}"
    assert compare(expected, theirs) is None
    kinds = [[e["type"] for e in turn["events"]] for turn in expected]
    assert "AirBalloonRevealed" in kinds[0] and "FloatedOnAirBalloon" in kinds[0], kinds[0]
    assert kinds[1].index("AirBalloonPopped") < kinds[1].index("DamageDealt"), kinds[1]
    assert "FloatedOnAirBalloon" not in kinds[2] and "DamageDealt" in kinds[2], kinds[2]


@needs_rust
def test_a_balloon_popped_by_residual_chip_is_logged_against_side_zero(tmp_path: Path) -> None:
    """A balloon popped by poison chip is logged against side 0, a quirk pinned in both engines."""
    team_a = [PokemonSpec(species="Machamp", nickname="A0", level=50, moves=["Toxic"])]
    team_b = [PokemonSpec(species="Rhydon", nickname="B0", level=50, item=Item.AIR_BALLOON, moves=["Splash"])]
    scenario, expected = record((team_a, team_b), _round_robin_chooser(), seed=1, max_turns=2)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"was refused: {theirs}"
    assert compare(expected, theirs) is None
    popped = [e for turn in expected for e in turn["events"] if e["type"] == "AirBalloonPopped"]
    assert popped == [{"type": "AirBalloonPopped", "side": 0, "pokemon": "B0"}], popped


def _ditto(nickname: str) -> PokemonSpec:
    return PokemonSpec(
        species="Ditto",
        nickname=nickname,
        level=50,
        ability=Ability.IMPOSTER,
        item=Item.CHOICE_SCARF,
        moves=["Transform"],
    )


@needs_rust
def test_imposter_transforms_on_arrival_and_fights_with_the_copied_moves(tmp_path: Path) -> None:
    """Imposter transforms on switch-in, so the Ditto's first action is one of its opponent's moves."""
    team_a = [_ditto("A0")]
    team_b = [PokemonSpec(species="Rhydon", nickname="B0", level=50, moves=["Rock Slide"])]
    scenario, expected = record((team_a, team_b), _chooser(random.Random(0)), seed=0, max_turns=2)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"was refused: {theirs}"
    assert compare(expected, theirs) is None
    first = expected[0]["events"]
    assert {"type": "Transformed", "side": 0, "pokemon": "A0", "into": "B0"} in first, first
    used = [e for e in first if e["type"] == "MoveUsed" and e["pokemon"] == "A0"]
    assert used and used[0]["move"] == "Rock Slide", used


def _mimikyu() -> PokemonSpec:
    return PokemonSpec(species="Mimikyu", nickname="B0", level=50, ability=Ability.DISGUISE, moves=["Splash"])


@needs_rust
def test_a_disguise_takes_one_hit_whole_then_busts(tmp_path: Path) -> None:
    """A disguise takes the first damaging move whole, and the second lands normally."""
    team_a = [PokemonSpec(species="Rhydon", nickname="A0", level=50, moves=["Rock Slide"])]
    scenario, expected = record((team_a, [_mimikyu()]), _chooser(random.Random(0)), seed=0, max_turns=2)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"was refused: {theirs}"
    assert compare(expected, theirs) is None
    first, second = (turn["events"] for turn in expected)
    assert {"type": "FormeChanged", "side": 1, "pokemon": "B0", "forme": "Mimikyu-Busted"} in first, first
    assert not any(e["type"] == "DamageDealt" and e["side"] == 1 for e in first), first
    assert any(e["type"] == "DamageDealt" and e["side"] == 1 for e in second), second


@needs_rust
def test_a_disguise_blocks_the_secondary_along_with_the_hit(tmp_path: Path) -> None:
    """A disguise that takes the hit also blocks the move's secondary effects."""
    team_a = [PokemonSpec(species="Rhydon", nickname="A0", level=50, moves=["Nuzzle"])]
    scenario, expected = record((team_a, [_mimikyu()]), _chooser(random.Random(0)), seed=0, max_turns=1)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"was refused: {theirs}"
    assert compare(expected, theirs) is None
    mimikyu = expected[0]["state"]["sides"][1]["team"][0]
    assert mimikyu["species"] == "Mimikyu-Busted" and mimikyu["status"] == "NONE", mimikyu


@needs_rust
def test_a_fixed_damage_move_is_blocked_by_a_disguise_without_busting_it(tmp_path: Path) -> None:
    """A quirk pinned: a disguise blocks fixed-damage moves outright."""
    team_a = [PokemonSpec(species="Rhydon", nickname="A0", level=50, moves=["Night Shade"])]
    scenario, expected = record((team_a, [_mimikyu()]), _chooser(random.Random(0)), seed=0, max_turns=1)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"was refused: {theirs}"
    assert compare(expected, theirs) is None
    kinds = [e["type"] for e in expected[0]["events"]]
    assert "MoveFailed" in kinds and "FormeChanged" not in kinds, kinds


@needs_rust
def test_power_construct_completes_zygarde_at_half_hp_and_adds_the_new_hp(tmp_path: Path) -> None:
    """Zygarde becomes Complete after the action that takes it to half HP, and its extra HP is real."""
    team_a = [PokemonSpec(species="Machamp", nickname="A0", level=50, moves=["Super Fang"])]
    zygarde = PokemonSpec(species="Zygarde", nickname="B0", level=50, ability=Ability.POWER_CONSTRUCT, moves=["Splash"])
    team_b = [zygarde]
    scenario, expected = record((team_a, team_b), _chooser(random.Random(0)), seed=0, max_turns=2)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"was refused: {theirs}"
    assert compare(expected, theirs) is None
    before, after = (turn["state"]["sides"][1]["team"][0] for turn in expected)
    assert before["species"] == "Zygarde" and after["species"] == "Zygarde-Complete", (before, after)
    assert after["ability"] == "POWER_CONSTRUCT" and 2 * after["hp"] > after["max_hp"], after


@needs_rust
@pytest.mark.parametrize("seed", range(6))
def test_sucker_punch_reads_the_chosen_slot_as_it_is_now(seed: int, tmp_path: Path) -> None:
    """Sucker Punch reads the target's chosen slot when asked, not when chosen."""
    # No Choice Scarf, so the transformed Ditto ties Kangaskhan's speed.
    ditto = _ditto("A0").model_copy(update={"item": Item.NONE})
    team_b = [PokemonSpec(species="Kangaskhan", nickname="B0", level=50, moves=["Sucker Punch"])]
    scenario, expected = record(([ditto], team_b), _chooser(random.Random(seed)), seed=seed, max_turns=1)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"was refused: {theirs}"
    assert compare(expected, theirs) is None


@needs_rust
def test_damage_above_a_transformed_maximum_also_takes_the_overflow(tmp_path: Path) -> None:
    """A transformed Ditto can stand above its new maximum HP, and the next chip clamps it."""
    ditto = _ditto("A0").model_copy(update={"effort_values": EVs(HP=252)})
    team_b = [PokemonSpec(species="Machamp", nickname="B0", level=50, moves=["Toxic"])]
    scenario, expected = record(([ditto], team_b), _chooser(random.Random(0)), seed=0, max_turns=1)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"was refused: {theirs}"
    assert compare(expected, theirs) is None
    chip = next(e for e in expected[0]["events"] if e["type"] == "ResidualDamage" and e["pokemon"] == "A0")
    after = expected[0]["state"]["sides"][0]["team"][0]
    assert chip["amount"] > after["max_hp"] // 16 and after["hp"] <= after["max_hp"], (chip, after)


@needs_rust
@pytest.mark.parametrize("move", ["Toxic", "Encore", "Taunt"])
def test_prankster_status_moves_do_not_affect_a_dark_type(move: str, tmp_path: Path) -> None:
    """Gen 7 Prankster status moves fail against Dark types."""
    team_a = [PokemonSpec(species="Klefki", nickname="A0", level=50, ability=Ability.PRANKSTER, moves=[move])]
    team_b = [PokemonSpec(species="Yveltal", nickname="B0", level=50, moves=["Roost"])]
    scenario, expected = record((team_a, team_b), _chooser(random.Random(0)), seed=0, max_turns=1)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"was refused: {theirs}"
    assert compare(expected, theirs) is None
    assert {"type": "DoesNotAffect", "side": 1, "pokemon": "B0"} in expected[0]["events"], expected[0]["events"]


@needs_rust
def test_eviolite_reads_the_species_a_pokemon_was_built_as(tmp_path: Path) -> None:
    """`fully_evolved` survives a forme change, so Eviolite still counts."""
    team_a = [PokemonSpec(species="Machamp", nickname="A0", level=50, moves=["Rock Slide"])]
    team_b = [
        PokemonSpec(
            species="Rhydon", nickname="B0", level=50, ability=Ability.DISGUISE, item=Item.EVIOLITE, moves=["Splash"]
        )
    ]
    scenario, expected = record((team_a, team_b), _chooser(random.Random(0)), seed=0, max_turns=2)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"was refused: {theirs}"
    assert compare(expected, theirs) is None
    assert expected[1]["state"]["sides"][1]["team"][0]["species"] == "Mimikyu-Busted"


@needs_rust
def test_a_heal_that_lowers_hp_is_not_announced(tmp_path: Path) -> None:
    """Above a Transform-shrunk maximum, Leftovers heals down to the cap silently."""
    ditto = _ditto("A0").model_copy(update={"effort_values": EVs(HP=252), "item": Item.LEFTOVERS})
    team_b = [PokemonSpec(species="Machamp", nickname="B0", level=50, moves=["Splash"])]
    scenario, expected = record(([ditto], team_b), _chooser(random.Random(0)), seed=0, max_turns=1)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"was refused: {theirs}"
    assert compare(expected, theirs) is None
    assert not any(e["type"] == "ItemHealed" for e in expected[0]["events"]), expected[0]["events"]
    after = expected[0]["state"]["sides"][0]["team"][0]
    assert after["hp"] == after["max_hp"], after


@needs_rust
def test_a_copied_multitype_resyncs_in_its_own_sides_residual(tmp_path: Path) -> None:
    """Residual ordering of Paradox abilities and Multitype re-sync matches across a Transform."""
    team_a = [
        PokemonSpec(
            species="Arceus-Water",
            nickname="A0",
            level=50,
            ability=Ability.MULTITYPE,
            item=Item.SPLASH_PLATE,
            moves=["Toxic"],
        )
    ]
    scenario, expected = record((team_a, [_ditto("B0")]), _chooser(random.Random(0)), seed=0, max_turns=1)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"was refused: {theirs}"
    assert compare(expected, theirs) is None
    kinds = [(e["type"], e.get("side")) for e in expected[0]["events"]]
    assert kinds.index(("ResidualDamage", 0)) < kinds.index(("TypeChanged", 1)), kinds


@needs_rust
def test_two_imposters_in_a_mirror_only_one_transforms(tmp_path: Path) -> None:
    """Mirror battles hand both sides the same Ditto."""
    scenario, expected = record(([_ditto("A0")], [_ditto("B0")]), _chooser(random.Random(0)), seed=0, max_turns=1)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"was refused: {theirs}"
    assert compare(expected, theirs) is None
    kinds = [e["type"] for e in expected[0]["events"]]
    assert kinds.count("Transformed") == 1 and "MoveFailed" in kinds, kinds


@needs_rust
def test_a_transformed_beat_up_counts_the_copied_attack(tmp_path: Path) -> None:
    """Beat Up sums each teammate's current base Attack, which Transform overwrites."""
    team_a = [_ditto("A0"), PokemonSpec(species="Rhydon", nickname="A1", level=50, moves=["Tackle"])]
    team_b = [PokemonSpec(species="Machamp", nickname="B0", level=50, moves=["Beat Up"])]
    scenario, expected = record((team_a, team_b), _chooser(random.Random(0)), seed=0, max_turns=1)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"was refused: {theirs}"
    assert compare(expected, theirs) is None
    used = [e for e in expected[0]["events"] if e["type"] == "MoveUsed" and e["pokemon"] == "A0"]
    assert used and used[0]["move"] == "Beat Up", used


@needs_rust
@pytest.mark.parametrize(
    "ability", [Ability.MOLD_BREAKER, Ability.TERAVOLT, Ability.TURBOBLAZE, Ability.NONE], ids=lambda a: a.name
)
def test_a_mold_breaker_earthquake_lands_on_levitate(ability: Ability, tmp_path: Path) -> None:
    """Mold Breaker and its variants ground a Levitate target for the whole move."""
    team_a = [PokemonSpec(species="Excadrill", nickname="A0", level=50, ability=ability, moves=["Earthquake"])]
    team_b = [PokemonSpec(species="Bronzong", nickname="B0", level=50, ability=Ability.LEVITATE, moves=["Splash"])]
    scenario, expected = record((team_a, team_b), _chooser(random.Random(0)), seed=0, max_turns=1)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"was refused: {theirs}"
    assert compare(expected, theirs) is None
    kinds = [e["type"] for e in expected[0]["events"]]
    if ability is Ability.NONE:
        assert "AvoidedWithLevitate" in kinds, kinds
    else:
        assert "AvoidedWithLevitate" not in kinds and "DamageDealt" in kinds, kinds


@needs_rust
@pytest.mark.parametrize(
    ("move", "user_ability"),
    [("Toxic", Ability.NONE), ("Stealth Rock", Ability.NONE), ("Toxic", Ability.MOLD_BREAKER)],
    ids=["toxic", "stealth-rock", "toxic-from-a-mold-breaker"],
)
def test_magic_bounce_sends_a_status_move_back_at_its_user(move: str, user_ability: Ability, tmp_path: Path) -> None:
    """Magic Bounce swaps attacker and defender for the rest of the move."""
    team_a = [PokemonSpec(species="Machamp", nickname="A0", level=50, ability=user_ability, moves=[move])]
    team_b = [PokemonSpec(species="Espeon", nickname="B0", level=50, ability=Ability.MAGIC_BOUNCE, moves=["Splash"])]
    scenario, expected = record((team_a, team_b), _chooser(random.Random(0)), seed=0, max_turns=1)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"was refused: {theirs}"
    assert compare(expected, theirs) is None
    events = expected[0]["events"]
    assert {"type": "MoveBounced", "side": 1, "pokemon": "B0"} in events, events
    sides = expected[0]["state"]["sides"]
    if move == "Toxic":
        assert sides[0]["team"][0]["status"] == "TOXIC" and sides[1]["team"][0]["status"] == "NONE", sides
    else:
        assert "STEALTH_ROCK" in sides[0]["hazards"] and not sides[1]["hazards"], sides


@needs_rust
@pytest.mark.parametrize("ability", [Ability.MOLD_BREAKER, Ability.NONE], ids=lambda a: a.name)
def test_a_mold_breaker_knocks_out_through_sturdy(ability: Ability, tmp_path: Path) -> None:
    """Mold Breaker gets through Sturdy's clamp."""
    team_a = [PokemonSpec(species="Machamp", nickname="A0", level=100, ability=ability, moves=["Surf"])]
    team_b = [PokemonSpec(species="Rhydon", nickname="B0", level=5, ability=Ability.STURDY, moves=["Splash"])]
    scenario, expected = record((team_a, team_b), _chooser(random.Random(0)), seed=0, max_turns=1)
    theirs = _rust_trace(scenario, tmp_path)
    assert not isinstance(theirs, str), f"was refused: {theirs}"
    assert compare(expected, theirs) is None
    kinds = [e["type"] for e in expected[0]["events"]]
    assert ("SurvivedAtOneHp" in kinds) is (ability is Ability.NONE), kinds


@needs_rust
def test_trick_room_inverts_the_speed_sort(tmp_path: Path) -> None:
    """The four pseudo-weather rooms."""
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
    """Rooms ending on the same pass are announced in the order they were cast."""

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

    # Scripted, so both room timers start together and run out together.
    def choose(state, side_index):
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
    """Wish, Healing Wish/Lunar Dance, Revival Blessing and Future Sight/Doom Desire agree."""
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
            # Future Sight landing after its user switched away is refused, which is correct here.
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
    """Shed Tail's replacement arrives the same turn, right after the substitute."""
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
    """Transform copies everything but HP and restores it on switch-out."""
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
    """Prankster, Gale Wings, Triage and Mycelium Might reorder a turn without touching a stat."""

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
        # Prankster: Growl outruns a much faster Tackle.
        ([mon("Rhydon", "A0", Ability.PRANKSTER, ["Growl"])], [mon("Tauros", "B0", Ability.NONE, ["Tackle"])], 0),
        # Gale Wings: Peck at full HP gets +1 priority.
        ([mon("Rhydon", "A0", Ability.GALE_WINGS, ["Peck"])], [mon("Tauros", "B0", Ability.NONE, ["Tackle"])], 0),
        # Triage: Drain Punch's +3 is the largest of the three, and unconditional.
        ([mon("Rhydon", "A0", Ability.TRIAGE, ["Drain Punch"])], [mon("Tauros", "B0", Ability.NONE, ["Tackle"])], 0),
        # Mycelium Might: a status move sorts last in its bracket.
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
    """Surge Surfer and Unburden change speed off terrain and a lost item."""
    # Surge Surfer: Electric Surge doubles Machamp's speed from the first turn.
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

    # Unburden: losing Leftovers doubles Rhydon's speed from turn two.
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
    """Quick Draw: a 30% chance to move first in the bracket."""
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
    """Sturdy: an OHKO from full HP leaves exactly one hit point."""
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
    """Serene Grace doubles a secondary's chance and Shield Dust blocks it."""
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
    """Skill Link: a multi-hit move always lands its maximum hits."""
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
    """Scrappy and Mind's Eye: Normal and Fighting moves hit Ghost types."""
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


def _round_robin_chooser() -> Chooser:
    """Cycles a Pokemon through its four move slots in order, each side independently."""
    slots = [MoveSlot.FIRST, MoveSlot.SECOND, MoveSlot.THIRD, MoveSlot.FOURTH]
    counters = {0: 0, 1: 0}

    def choose(state, side_index):
        slot = slots[counters[side_index] % 4]
        counters[side_index] += 1
        return Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=slot)

    return choose


@needs_rust
def test_pressure_doubles_the_pp_cost_of_a_move_that_faces_it(tmp_path: Path) -> None:
    """Pressure: a move aimed at its holder spends 2 PP."""
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
    """Steadfast: a flinch raises its holder's Speed by one stage."""
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
    """Synchronize reflects a burn, paralysis or poison back onto the inflictor."""
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
    """Corrosion lets Toxic poison a Steel type."""
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


def _damage_dealt(events: list[dict[str, Any]]) -> int:
    amount: int = next(e["amount"] for e in events if e["type"] == "DamageDealt")
    return amount


@needs_rust
def test_water_bubble_doubles_its_own_water_and_halves_fire_taken(tmp_path: Path) -> None:
    """Water Bubble halves Fire damage taken, doubles Water damage dealt and blocks burns."""
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
    """The type-absorbing abilities cancel the move and heal or boost."""
    heal_style = [
        ("VOLT_ABSORB", "Thunder Shock", "Machamp"),
        ("WATER_ABSORB", "Water Gun", "Machamp"),
        ("EARTH_EATER", "Earthquake", "Machamp"),
    ]
    for index, (ability, move, species) in enumerate(heal_style):
        # Tackle first to chip HP, then the absorbed move on turn two.
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
    """An absorbing ability cancels even a move with no modelled effects, such as Electrify."""
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
    """Flash Fire: absorbs the first Fire move, then boosts its own Fire moves."""

    # Scripted, so Gengar receives Ember first and then uses its own boosted.
    def scripted_both_sides(sequence: list[MoveSlot]):
        counters = {0: 0, 1: 0}

        def choose(state, side_index):
            slot = sequence[counters[side_index]]
            counters[side_index] += 1
            return Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=slot)

        return choose

    team_b = [_mon("Rhydon", "B0", Ability.NONE, ["Ember", "Splash"])]

    def play(a_ability: Ability) -> list[dict[str, Any]]:
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
    """Levitate cancels Ground moves and Soundproof cancels sound moves."""
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
    """Moxie, Beast Boost and Soul Heart trigger when the holder's own move knocks out a foe."""
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
    """A fixed-damage KO does not trigger ON_FAINT abilities."""
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


def _switch_on_turn(turn_to_switch: int, target_index: int):
    """Deterministic: the first available move every turn, with one voluntary switch."""

    def choose(state, side_index):
        if side_index == 0 and state.turn == turn_to_switch:
            return Action(action=ActionType.SWITCH_OUT, switch_in=state.sides[0].team[target_index])
        options = [a for a in legal_actions(state, side_index) if a.action is ActionType.USE_MOVE]
        return (options or legal_actions(state, side_index))[0]

    return choose


@needs_rust
def test_regenerator_heals_a_third_on_switch_out(tmp_path: Path) -> None:
    """Regenerator heals on switch-out, from the HP it left with."""
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
    """Natural Cure clears status on switch-out."""
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
    """A fainted Pokemon's replacement does not trigger ON_SWITCH_OUT."""
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
    # Two turns: the faint, then the auto-replacement this test is about.
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
    """Protosynthesis and Quark Drive boost the holder's highest stat by 1.3x."""
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
    """Quark Drive's Speed case boosts by 1.5x in `effective_speed`."""
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
    """A field-sourced Paradox boost ends silently when its condition does."""
    # Rhydon, bulky enough that seven Tackles apiece never end the battle early.
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
    """Toxic Debris and the other after-hit one-liners agree."""
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
    """Cursed Body: a 30% chance to disable the move that hit."""
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
    """Sheer Force: 1.3x power on moves with secondaries, which never land."""
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
    """Solar Power: 1.5x special damage in sun and an eighth of max HP chip each turn."""
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
    """Wonder Guard blocks anything under 2x on the plain type chart."""
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
    """Wind Rider: a wind move aimed at its holder is cancelled for +1 Attack."""
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
    """Liquid Voice makes sound moves Water-type."""
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
    """Poison Puppeteer confuses whatever its holder poisons."""
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
    """Quick Claw: a 20% chance to move first in the bracket."""
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
    """Custap Berry: a guaranteed jump to the front of the bracket at a quarter HP."""
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
    """Leppa Berry restores PP the instant a slot hits zero."""
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

    def always_first(state, side_index):
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
    """Chesto and Lum Berry cure the instant their status lands."""
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
    """Weather rocks stretch a five-turn weather to eight, whoever set it."""
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
    """A terrain seed is consumed on the same switch-in that sets its terrain."""
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
    """Terrain changes trigger every side's seed on the spot."""
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
    """Choice Band, Scarf and Specs lock the first move used."""
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

    def try_switch_moves(state, side_index):
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
    """Focus Sash leaves one HP from full and is consumed before Sturdy."""
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
    """Life Orb: 1.3x damage and a tenth of max HP recoil, once per move."""
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
    """Eject Button switches its holder out once the hitting action finishes."""
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
    """Eject Pack switches its holder out after an opponent drops its stats."""
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
    """Red Card drags the attacker out at random."""
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
    """Superpower's self-drop applies to the user even after Red Card drags it out."""
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
    """Tough Claws stops applying once Red Card has dragged its holder out."""
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
    """Poison Touch and Toxic Chain stop applying once their holder is dragged out."""
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
    """Pending switches are armed and executed in the right order across both sides."""
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
    """U-turn's self-switch is armed and resolved with every other pending switch."""
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
    """Dragon Tail into an Eject Button holder resolves both switches in Python's order."""
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
    """Circle Throw into a Red Card holder resolves both switches in Python's order."""
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
    """Heavy-Duty Boots cancels entry hazards entirely."""
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

    def switch_b_on_turn_one(state, side_index):
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
    """The four drives change Techno Blast's type."""
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
    """Booster Energy activates a Paradox ability on switch-in."""
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
    """Punching Glove boosts punching moves by 1.1x."""
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
    """Terrain Extender stretches terrain to eight turns."""
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
    """Light Clay stretches screens to eight turns."""
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
    """Clear Amulet blocks stat drops from opponents."""
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
    """White Herb resets negative stats whenever any stage change resolves."""
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
    """Adrenaline Orb answers Intimidate."""
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
    """Covert Cloak blocks secondary effects."""
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
    """Loaded Dice folds a multi-hit roll up to its top two counts."""
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
    """Power Herb fires a charge move on the turn it is chosen."""
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
    """Mental Herb cures Taunt, Encore and Disable."""
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
    """Mirror Herb copies the opponent's stat raises."""
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
    """A guard against testing nothing."""
    rng = random.Random(0)
    generated = [spec for _ in range(1000) for spec in _team(rng, size=3, abilities=True, items=True)]

    missing = set(PORTED["abilities"]) - {spec.ability.name for spec in generated}
    missing |= set(PORTED["items"]) - {spec.item.name for spec in generated}
    assert not missing, f"never generated: {sorted(missing)}"


@needs_rust
def test_every_live_item_but_the_forme_items_is_ported() -> None:
    """Shed Shell — whose whole effect is on legality, ported with `legal_actions` — was the last."""
    rules = json.loads((DATA / "rules.json").read_text())
    forme_items = {row["item"] for row in rules["mega_formes"]} | {row["item"] for row in rules["ultra_burst_formes"]}
    live = set(rules["live_items"]) - forme_items
    missing = sorted(name for name in live - set(PORTED["items"]) if name in Item.__members__)
    assert not missing, missing


@needs_rust
@pytest.mark.parametrize("carries", ["ability"])
def test_live_abilities_and_items_are_refused_rather_than_ignored(carries: str, tmp_path: Path) -> None:
    """Reading an ability off a Pokemon and doing nothing with it is a wrong answer in silence."""
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
    """The lead is comparable; the one behind it is not."""
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
    """Asking for more randomness than Python used is a divergence, never a fresh number."""
    rng = random.Random(7)
    scenario, _ = record((_team(rng), _team(rng)), _chooser(rng), seed=7, max_turns=30)
    starved = Scenario(teams=scenario.teams, actions=scenario.actions, tape=scenario.tape[:1], seed=scenario.seed)

    path = tmp_path / "starved.json"
    path.write_text(starved.to_json())
    result = subprocess.run([str(BINARY), str(path), str(DATA)], capture_output=True, text=True)

    assert result.returncode == DIVERGED, f"exit {result.returncode}: {result.stderr}"
    assert "asked for" in result.stderr, result.stderr
