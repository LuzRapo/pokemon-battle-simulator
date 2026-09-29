"""The self-play environment (`rust/src/env.rs`) asks for decisions exactly when the Python does.

`VecEnv` answers single-option decisions itself and sequences leads, turns, mid-turn pivots and
faint replacements on its own, so its sequencing is checked against `record_played` (which plays
like `runner.run_battle`): a Python battle's decisions are fed to a `VecEnv` replaying the same
draws, and it must ask for every multi-option decision, in order, and end in the same state.
"""

import json
import random
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from battle_sim.ag_sets import mirror_team
from battle_sim.differential import PlayedScenario, encode_spec, name_action, record_played
from battle_sim.engine import legal_actions
from battle_sim.mechanics.battle import BattleState
from battle_sim.models.actions import Action, ActionType
from battle_sim.models.spec import PokemonSpec
from battle_sim.rl import encode as py
from battle_sim.rl.encode import Decision
from battle_sim.rl.net_player import action_index, decision_for
from battle_sim.rust_bridge import LIBRARY, database, load

pytestmark = pytest.mark.skipif(not LIBRARY.exists(), reason="run `cargo build --release` in rust/")


def _teams_json(team: Sequence[PokemonSpec]) -> list[str]:
    encoded = json.dumps([encode_spec(spec) for spec in team])
    return [encoded, encoded]


type Reads = list[tuple[Decision, dict[str, int]]]


def _played(seed: int) -> tuple[Sequence[PokemonSpec], tuple[int, int], PlayedScenario, list[dict[str, Any]], Reads]:
    """A random mirror battle played the `run_battle` way, and what `NetPlayer` would have made of
    each of its decisions: the kind, and each legal action's number."""
    rng = random.Random(f"env:{seed}")
    team = mirror_team(rng)
    leads = (rng.randrange(6), rng.randrange(6))
    orders = tuple([lead, *[i for i in range(6) if i != lead]] for lead in leads)

    def pick(state: BattleState, side: int, offered: list[Action]) -> Action:
        moves = [a for a in offered if a.action is ActionType.USE_MOVE]
        return rng.choice(moves) if moves and rng.random() < 0.8 else rng.choice(offered)

    reads: Reads = []

    def read(state: BattleState, side: int, kind: str) -> None:
        if kind != "lead":
            numbered = {name_action(a, state, side): action_index(a, state, side) for a in legal_actions(state, side)}
            reads.append((decision_for(state, side), numbered))
            assert reads[-1][0] is {"turn": Decision.TURN, "switch": Decision.SWITCH}[kind]

    scenario, expected = record_played(
        (team, team),
        orders,  # type: ignore[arg-type]
        pick,
        seed=seed,
        max_turns=150,
        observe=read,
    )
    return team, leads, scenario, expected, reads


@pytest.mark.parametrize("seed", range(40))
def test_the_environment_asks_what_run_battle_asks_and_ends_where_it_ends(seed: int) -> None:
    team, leads, scenario, expected, python_reads = _played(seed)
    reads = iter(python_reads)
    env = load().VecEnv(database(), 1, max_turns=150, threads=1)
    env.replay(0, _teams_json(team), json.dumps(scenario.tape))

    def answer(side: int, name: str, python_legal: list[str], kind: int) -> None:
        rows, sides, decisions = env.observe()[:3]
        asked = {(int(s), int(d)) for r, s, d in zip(rows, sides, decisions, strict=True) if r == 0}
        numbered = next(reads)[1] if kind else None
        if len(python_legal) == 1:
            return  # answered by the environment itself; the final state checks it answered right
        assert (side, kind) in asked, f"side {side} had a choice of {python_legal} and was not asked"
        named = {label: action for action, label in env.legal(0, side)}
        assert sorted(named) == sorted(python_legal)
        assert numbered is None or named == numbered, "NetPlayer numbers these actions differently"
        env.act(np.array([0]), np.array([side]), np.array([named[name]]))

    for side, lead in enumerate(leads):
        answer(side, f"lead:{lead}", [f"lead:{i}" for i in range(6)], 0)
    for decision in scenario.decisions:
        if decision["kind"] == "turn":
            for side in (0, 1):
                answer(side, decision["actions"][side], decision["legal"][side], 1)
        else:
            answer(decision["side"], decision["action"], decision["legal"], 2)

    final = expected[-1]
    if final["state"]["outcome"] is None:
        assert env.collect() == [(0, "timeout", None, 150)]
    else:
        assert json.loads(env.digest(0)) == final["state"]
        assert env.drawn(0) == final["drawn"]
        [(_, ending, winner, _)] = env.collect()
        outcome = {"P1_WIN": ("won", 0), "P2_WIN": ("won", 1), "DRAW": ("draw", None)}[final["state"]["outcome"]]
        assert (ending, winner) == outcome


def test_a_random_policy_finishes_every_battle_through_the_masks() -> None:
    rng = np.random.default_rng(0)
    teams = random.Random(0)
    env = load().VecEnv(database(), 32, max_turns=300)
    results: list[tuple[int, str, int | None, int]] = []
    started = 0
    while len(results) < 200:
        for index in env.empty():
            if started < 200:
                env.reset(index, _teams_json(mirror_team(teams)), started)
                started += 1
        rows, sides, decisions, ids, pokemon, field, mask = env.observe()
        assert mask.any(axis=1).all(), "every request has a legal answer"
        assert (mask.sum(axis=1) > 1).all(), "single-answer decisions are answered, not asked"
        assert ids.shape == (len(rows), py.SLOTS, py.POKEMON_IDS)
        assert pokemon.shape == (len(rows), py.SLOTS, py.POKEMON_FLOATS)
        assert field.shape == (len(rows), py.FIELD_FLOATS)
        choice = np.array([rng.choice(np.flatnonzero(legal)) for legal in mask])
        env.act(rows, sides, choice)
        results += env.collect()
    endings = [ending for _, ending, _, _ in results]
    assert not [e for e in endings if e.startswith("failed")], endings
    assert endings.count("won") > 150, endings


def test_an_illegal_answer_is_refused() -> None:
    env = load().VecEnv(database(), 1)
    env.reset(0, _teams_json(mirror_team(random.Random(0))), 0)
    with pytest.raises(ValueError, match="not legal"):
        env.act(np.array([0]), np.array([0]), np.array([0]))  # a move is no lead
