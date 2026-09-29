"""The Rust engine's Python extension, and a driver that replays a `PlayedScenario` through it.

The extension is built by `cargo build --release` in `rust/` and loaded straight from its `.so`,
so nothing has to install it as a package — a stale copy on `sys.path` cannot shadow it.
"""

import importlib.machinery
import importlib.util
import json
import types
from collections.abc import Callable, Iterator
from functools import cache
from pathlib import Path
from typing import Any

from battle_sim.differential import PlayedScenario

RUST = Path(__file__).resolve().parent.parent / "rust"
DATA = RUST / "data"
LIBRARY = RUST / "target" / "release" / "libpokemon_engine.so"


@cache
def load() -> types.ModuleType:
    """The compiled `pokemon_engine_rs` module."""
    if not LIBRARY.exists():
        raise FileNotFoundError(f"{LIBRARY} is missing; run `cargo build --release` in {RUST}")
    loader = importlib.machinery.ExtensionFileLoader("pokemon_engine_rs", str(LIBRARY))
    spec = importlib.util.spec_from_file_location("pokemon_engine_rs", str(LIBRARY), loader=loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


@cache
def database() -> Any:
    return load().Database(str(DATA))


Observer = Callable[[Any, int, str], None]


def _next_pivot(decisions: Iterator[dict[str, Any]], side: int) -> dict[str, Any]:
    """The recorded answer to the mid-turn replacement the Rust engine is waiting on."""
    pivot = next(decisions)
    if pivot["kind"] != "pivot" or pivot["side"] != side:
        raise AssertionError(f"rust waits on side {side}'s pivot; the python recorded {pivot}")
    return pivot


def replay_played(
    scenario: PlayedScenario, observe: Observer | None = None
) -> tuple[list[dict[str, Any]], list[str | None]]:
    """Play a `PlayedScenario`'s decisions through the Rust engine, in order.

    Returns the Rust trace, segment for segment as `record_played` writes the Python's, and — per
    decision — `None` if the Rust engine offered exactly the legal actions the Python did, or a
    message naming the difference. A decision the Rust engine is not waiting for (a pivot the Python
    asked about and Rust did not, say) raises: the two have already parted.

    `observe(battle, side, kind)` is called at every decision, in `record_played`'s order: both
    sides' "lead" first, then "turn" for each side, and "switch" for a pivot or a replacement.
    """
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
            status, value = battle.begin_turn(labels)
            while status == "switch":
                pivot = _next_pivot(decisions, value)
                seen(value, "switch")
                check(value, pivot["legal"])
                labels.append(pivot["action"])
                status, value = battle.resume(pivot["action"])
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
