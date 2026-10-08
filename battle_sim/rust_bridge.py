"""The Rust engine's Python extension, and a driver that replays a `PlayedScenario` through it."""

import importlib.machinery
import importlib.util
import json
from collections.abc import Callable, Iterator
from functools import cache
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from battle_sim.differential import PlayedScenario

RUST = Path(__file__).resolve().parent.parent / "rust"
DATA = RUST / "data"
LIBRARY = RUST / "target" / "release" / "libpokemon_engine.so"

# `(side, index, move, foe index, low, high)`.
type Estimate = tuple[int, int, str, int, int, int]
# `(outcome, turns, survivors_a, survivors_b, error)`.
type TournamentResult = tuple[str | None, int, int, int, str | None]
# `(env, side, decision, ids, pokemon, field, mask)`, one row per open request.
type Observed = tuple[
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.bool_],
]
# One side's view as `(ids, pokemon, field)`, flat, in `battle_sim.rl.encode`'s order.
type RustObservation = tuple[list[int], list[float], list[float]]
# `("done", log_json)` once the turn has finished, or `("switch", side)` while a pivot waits.
type TurnStatus = tuple[str, str | int]


class RustDatabase(Protocol):
    def move_playable(self, name: str) -> bool: ...
    def ability_playable(self, name: str) -> bool: ...


class RustBattle(Protocol):
    def step(self, actions: list[str]) -> str: ...
    def begin_turn(self, actions: list[str]) -> TurnStatus: ...
    def resume(self, action: str) -> TurnStatus: ...
    def forced_switch(self, side: int, action: str) -> str: ...
    def waiting_on(self) -> int | None: ...
    def needs_replacement(self, side: int) -> bool: ...
    def legal_actions(self, side: int) -> list[str]: ...
    def matchup_features(self, side: int) -> list[tuple[str, list[float] | None]]: ...
    def observe(self, viewer: int, decision: str) -> RustObservation: ...
    def digest(self) -> str: ...
    def outcome(self) -> str | None: ...
    def drawn(self) -> int: ...


class RustBattleFactory(Protocol):
    def new(
        self, db: RustDatabase, teams_json: list[str], tape_json: str, orders: list[list[int]] | None = None
    ) -> RustBattle: ...
    def live(
        self, db: RustDatabase, teams_json: list[str], seed: int, orders: list[list[int]] | None = None
    ) -> RustBattle: ...


class RustVecEnv(Protocol):
    def __len__(self) -> int: ...
    def reset(self, index: int, teams_json: list[str], seed: int) -> None: ...
    def replay(self, index: int, teams_json: list[str], tape_json: str) -> None: ...
    def observe(self) -> Observed: ...
    def act(self, env: NDArray[np.int64], side: NDArray[np.int64], action: NDArray[np.int64]) -> None: ...
    def collect(self) -> list[tuple[int, str, int | None, int]]: ...
    def empty(self) -> list[int]: ...
    def legal(self, index: int, side: int) -> list[tuple[int, str]]: ...
    def digest(self, index: int) -> str: ...
    def drawn(self, index: int) -> int: ...


class RustEngine(Protocol):
    """The parts of `pokemon_engine_rs` Python calls, as `rust/src/python.rs` defines them."""

    Battle: RustBattleFactory
    Unported: type[RuntimeError]
    Diverged: type[RuntimeError]

    def Database(self, data_dir: str) -> RustDatabase: ...
    def VecEnv(self, db: RustDatabase, size: int, max_turns: int = 300, threads: int = 0) -> RustVecEnv: ...
    def matchup_order(self, db: RustDatabase, own_json: str, opponent_json: str) -> list[int]: ...
    def matchup_damage(self, db: RustDatabase, own_json: str, opponent_json: str) -> list[Estimate]: ...
    def play_matchup_battles(
        self,
        db: RustDatabase,
        jobs: list[tuple[str, str, int]],
        weights_json: str | None = None,
        max_turns: int = 1000,
        threads: int = 0,
    ) -> list[TournamentResult]: ...


@cache
def load() -> RustEngine:
    """The compiled `pokemon_engine_rs` module."""
    if not LIBRARY.exists():
        raise FileNotFoundError(f"{LIBRARY} is missing; run `cargo build --release` in {RUST}")
    loader = importlib.machinery.ExtensionFileLoader("pokemon_engine_rs", str(LIBRARY))
    spec = importlib.util.spec_from_file_location("pokemon_engine_rs", str(LIBRARY), loader=loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    # A module loaded from a path carries no static types; `RustEngine` is its declared interface.
    return cast(RustEngine, module)


@cache
def database() -> RustDatabase:
    return load().Database(str(DATA))


Observer = Callable[[RustBattle, int, str], None]


def _next_pivot(decisions: Iterator[dict[str, Any]], side: int) -> dict[str, Any]:
    """The recorded answer to the mid-turn replacement the Rust engine is waiting on."""
    pivot = next(decisions)
    if pivot["kind"] != "pivot" or pivot["side"] != side:
        raise AssertionError(f"rust waits on side {side}'s pivot; the python recorded {pivot}")
    return pivot


def replay_played(
    scenario: PlayedScenario, observe: Observer | None = None
) -> tuple[list[dict[str, Any]], list[str | None]]:
    """Play a `PlayedScenario`'s decisions through the Rust engine, in order."""
    rs = load()
    battle = rs.Battle.new(
        database(),
        [json.dumps(scenario.teams[0]), json.dumps(scenario.teams[1])],
        json.dumps(scenario.tape),
        [list(scenario.orders[0]), list(scenario.orders[1])],
    )
    segments: list[dict[str, Any]] = []
    legality: list[str | None] = []

    def seen(side: int, kind: str) -> None:
        if observe is not None:
            observe(battle, side, kind)

    seen(0, "lead")
    seen(1, "lead")

    def check(side: int, expected: list[str]) -> None:
        offered = list(battle.legal_actions(side))
        legality.append(None if offered == expected else f"side {side}: rust {offered} != python {expected}")

    def segment(labels: list[str], log_json: str) -> None:
        state = json.loads(battle.digest())
        segments.append({"actions": labels, "events": json.loads(log_json), "state": state, "drawn": battle.drawn()})

    decisions = iter(scenario.decisions)
    for decision in decisions:
        kind = decision["kind"]
        if kind == "turn":
            seen(0, "turn")
            seen(1, "turn")
            check(0, decision["legal"][0])
            check(1, decision["legal"][1])
            labels = list(decision["actions"])
            _, value = battle.begin_turn(labels)
            while isinstance(value, int):
                pivot = _next_pivot(decisions, value)
                seen(value, "switch")
                check(value, pivot["legal"])
                labels.append(pivot["action"])
                _, value = battle.resume(pivot["action"])
            segment(labels, value)
        elif kind == "replace":
            side = decision["side"]
            if not battle.needs_replacement(side):
                raise AssertionError(f"the python replaced side {side}; rust does not need a replacement there")
            seen(side, "switch")
            check(side, decision["legal"])
            segment([f"replace:{side}", decision["action"]], battle.forced_switch(side, decision["action"]))
        else:
            raise AssertionError(f"the python asked for a {kind} the rust engine never paused for: {decision}")
    return segments, legality
