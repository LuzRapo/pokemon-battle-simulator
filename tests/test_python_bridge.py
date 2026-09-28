"""The PyO3 bridge: this engine, called from Python in-process rather than through the `replay`
binary's stdin/stdout.

Two things are worth proving separately, because they are different claims:

  * **Replay mode agrees with the subprocess harness.** The bridge is a second way to run the exact
    same tape/actions/database `bin/replay.rs` already runs — if it produced a different trace, the
    bridge itself would be the bug, not anything about the engine. `compare()` is the same check
    `test_rust_battles.py` already trusts for that.
  * **Live mode plays a whole battle to completion with no Python RNG involved at all.** This is the
    capability replay mode structurally cannot exercise (a replay tape has to come from a prior
    Python run) and the one self-play actually needs: this engine drawing its own randomness,
    resolving turns nothing else is watching.

Neither test drives the bridge through `battle_sim.engine.legal_actions` — that needs a
`BattleState`, which nothing here builds. Action choice for both tests reads the bridge's own
digest (`pp` per move name) rather than reimplementing legality, which is the same boundary
`docs/rust-port-plan.md` draws for what self-play still needs on the Python side.
"""

import importlib.machinery
import importlib.util
import json
import random
import types
from pathlib import Path
from typing import Any

import pytest

from battle_sim.differential import compare, encode_spec, record
from battle_sim.models.spec import PokemonSpec
from battle_sim.models.stats import EVs, IVs
from battle_sim.utils import Ability, Nature
from tests.test_rust_battles import PLAIN_MOVES, PLAIN_SPECIES, _chooser, _still_unported

RUST = Path(__file__).resolve().parent.parent / "rust"
DATA = RUST / "data"
LIBRARY = RUST / "target" / "release" / "libpokemon_engine.so"

needs_bridge = pytest.mark.skipif(
    not LIBRARY.exists(),
    reason="the Rust cdylib is not built; run `cargo build --release` in rust/ (the `python` feature is on by default)",
)


def _load_bridge() -> types.ModuleType:
    """The compiled extension, loaded straight from its `.so` rather than via `sys.path` — nothing
    needs it importable as a normal package, and a stale copy on `sys.path` cannot shadow it."""
    loader = importlib.machinery.ExtensionFileLoader("pokemon_engine_rs", str(LIBRARY))
    spec = importlib.util.spec_from_file_location("pokemon_engine_rs", str(LIBRARY), loader=loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def _team(rng: random.Random, size: int = 2) -> list[PokemonSpec]:
    return [
        PokemonSpec(
            species=rng.choice(PLAIN_SPECIES),
            nickname=f"P{index}",
            level=50,
            nature=Nature.HARDY,
            effort_values=EVs(),
            individual_values=IVs(),
            moves=rng.sample(PLAIN_MOVES, 4),
        )
        for index in range(size)
    ]


@needs_bridge
@pytest.mark.parametrize("seed", range(10))
def test_replay_mode_agrees_with_the_subprocess_harness(seed: int) -> None:
    rng = random.Random(f"bridge-replay:{seed}")
    scenario, expected = record((_team(rng), _team(rng)), _chooser(rng), seed=seed, max_turns=20)

    rs = _load_bridge()
    db = rs.Database(str(DATA))
    battle = rs.Battle.new(
        db,
        [json.dumps(scenario.teams[0]), json.dumps(scenario.teams[1])],
        json.dumps(scenario.tape),
    )

    theirs = []
    for actions in scenario.actions:
        events = json.loads(battle.step(list(actions)))
        theirs.append(
            {"actions": list(actions), "events": events, "state": json.loads(battle.digest()), "drawn": battle.drawn()}
        )

    divergence = compare(expected, theirs)
    assert divergence is None, f"seed {seed}\n{divergence}"


@needs_bridge
def test_an_unported_ability_raises_unported_not_a_generic_error() -> None:
    """`Refusal::Unported` has to cross the FFI boundary as its own exception type, not collapse
    into a bare `RuntimeError` a caller cannot distinguish from a real divergence -- the same
    distinction `bin/replay.rs`'s exit codes 2 vs 3 exist to preserve."""
    rng = random.Random("bridge-unported")
    unported = _still_unported("abilities", Ability)
    team = _team(rng)
    team[0] = team[0].model_copy(update={"ability": Ability[unported]})

    rs = _load_bridge()
    db = rs.Database(str(DATA))
    teams_json = [json.dumps([encode_spec(s) for s in team]), json.dumps([encode_spec(s) for s in _team(rng)])]
    with pytest.raises(rs.Unported):
        rs.Battle.new(db, teams_json, "[]")


@needs_bridge
def test_a_corrupted_tape_raises_diverged_not_a_generic_error() -> None:
    """`Refusal::Diverged` has to cross the same boundary as its own exception too -- a caller that
    cannot tell a divergence apart from an ordinary error could go on treating a battle as valid
    after the two engines have already parted."""
    rng = random.Random("bridge-diverged")
    scenario, _ = record((_team(rng), _team(rng)), _chooser(rng), seed=1, max_turns=10)

    rs = _load_bridge()
    db = rs.Database(str(DATA))
    # A probability recorded as a wildly out-of-range integer instead: whatever the first draw
    # actually was, this cannot be it, so the very first turn must diverge.
    corrupted = [999999999] + list(scenario.tape[1:])
    battle = rs.Battle.new(db, [json.dumps(scenario.teams[0]), json.dumps(scenario.teams[1])], json.dumps(corrupted))
    with pytest.raises(rs.Diverged):
        battle.step(list(scenario.actions[0]))


def _pick_move(digest: dict[str, Any], spec: PokemonSpec, slot_names: tuple[str, ...]) -> str:
    """A random still-usable move for whoever is active, read off the bridge's own digest -- no
    `BattleState`, no `legal_actions`, exactly the boundary self-play still has to build past this
    bridge (see the module docstring). `pp` is keyed by slot name ("FIRST"/"SECOND"/...), the same
    keys `battle::SLOT_NAMES` uses everywhere else, not by move name."""
    pp = digest["pp"]
    usable = [(slot_names[i], name) for i, name in enumerate(spec.moves) if pp.get(slot_names[i], 0) > 0]
    slot, name = random.choice(usable)
    return f"move:{slot}:{name}"


@needs_bridge
def test_live_mode_plays_a_whole_battle_with_no_python_rng() -> None:
    """No `Scenario`, no `record()`, no recorded tape anywhere in this test -- every draw the
    battle takes comes from the Rust engine's own seeded RNG (`Tape::live`), which is what makes
    this the one capability replay mode cannot stand in for."""
    rng = random.Random("bridge-live")
    specs = (_team(rng), _team(rng))
    slot_names = ("FIRST", "SECOND", "THIRD", "FOURTH")

    rs = _load_bridge()
    db = rs.Database(str(DATA))
    teams_json = [json.dumps([encode_spec(s) for s in team]) for team in specs]
    battle = rs.Battle.live(db, teams_json, seed=12345)

    # Neither side ever switches, so this stops the moment either lead faints rather than trying to
    # act for a Pokemon that needs replacing -- that replacement decision is `legal_actions`'s job,
    # not this smoke test's. Reaching a handful of turns first is what it is actually checking.
    turns = 0
    while battle.outcome() is None and turns < 50:
        state = json.loads(battle.digest())
        leads = [state["sides"][side]["team"][0] for side in range(2)]
        if any(lead["fainted"] for lead in leads):
            break
        actions = [_pick_move(leads[side], specs[side][0], slot_names) for side in range(2)]
        battle.step(actions)
        turns += 1

    assert turns > 0, "the battle never took a single turn"
    assert battle.drawn() > 0, "a live battle that never drew a random number is not exercising Tape::live"
