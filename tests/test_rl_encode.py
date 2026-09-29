"""The two observation encoders — `rust/src/obs.rs` and `battle_sim/rl/encode.py` — agree exactly.

A network trains on the Rust encoder and plays through the Python one, so any difference between
them is a position the network has never seen. Both engines already agree on every field the digest
covers, so each array is compared at every decision of played mirror battles: a difference here is
an encoder bug, not an engine one.
"""

import json
import random
from typing import Any

import numpy as np
import pytest

from battle_sim.differential import compare, record_played
from battle_sim.mechanics.battle import BattleState
from battle_sim.models.actions import Action, ActionType
from battle_sim.rl import encode as py
from battle_sim.rl.vocab import vocabulary
from battle_sim.rust_bridge import DATA, LIBRARY, replay_played

needs_bridge = pytest.mark.skipif(not LIBRARY.exists(), reason="run `cargo build --release` in rust/")

DECISIONS = {"lead": py.Decision.LEAD, "turn": py.Decision.TURN, "switch": py.Decision.SWITCH}


def test_the_exported_vocabulary_is_current() -> None:
    """`rust/data/vocab.json` is what the Rust encoder reads; stale, it would disagree on ids."""
    exported = json.loads((DATA / "vocab.json").read_text())
    exported.pop("schema")
    assert exported == vocabulary(), "re-run `uv run python -m battle_sim.export_data --out rust/data`"


def _played_observations(seed: int) -> tuple[list[Any], list[Any]]:
    from battle_sim.ag_sets import mirror_team

    rng = random.Random(f"encode:{seed}")
    team = mirror_team(rng)
    orders = tuple([lead, *[i for i in range(6) if i != lead]] for lead in (rng.randrange(6), rng.randrange(6)))

    def pick(state: BattleState, side: int, offered: list[Action]) -> Action:
        moves = [a for a in offered if a.action is ActionType.USE_MOVE]
        return rng.choice(moves) if moves and rng.random() < 0.8 else rng.choice(offered)

    ours: list[Any] = []
    theirs: list[Any] = []

    def observe_python(state: BattleState, side: int, kind: str) -> None:
        ours.append((side, kind, py.encode(state, side, DECISIONS[kind])))

    def observe_rust(battle: Any, side: int, kind: str) -> None:
        theirs.append((side, kind, battle.observe(side, kind)))

    scenario, expected = record_played(
        (team, team),
        orders,  # type: ignore[arg-type]
        pick,
        seed=seed,
        max_turns=150,
        observe=observe_python,
    )
    trace, _ = replay_played(scenario, observe=observe_rust)
    assert compare(expected, trace) is None, compare(expected, trace)
    return ours, theirs


@needs_bridge
@pytest.mark.parametrize("seed", range(8))
def test_both_encoders_agree_at_every_decision(seed: int) -> None:
    ours, theirs = _played_observations(seed)
    assert [(s, k) for s, k, _ in ours] == [(s, k) for s, k, _ in theirs]
    for at, ((side, kind, (ids, pokemon, field)), (_, _, (rs_ids, rs_pokemon, rs_field))) in enumerate(
        zip(ours, theirs, strict=True)
    ):
        where = f"decision {at} ({kind}, side {side})"
        np.testing.assert_array_equal(ids, np.asarray(rs_ids).reshape(ids.shape), err_msg=f"{where}: ids")
        np.testing.assert_array_equal(
            pokemon, np.asarray(rs_pokemon, dtype=np.float32).reshape(pokemon.shape), err_msg=f"{where}: pokemon"
        )
        np.testing.assert_array_equal(field, np.asarray(rs_field, dtype=np.float32), err_msg=f"{where}: field")


def test_the_arrays_have_their_documented_shapes() -> None:
    from battle_sim.ag_sets import mirror_team
    from battle_sim.runner import build_side

    team = mirror_team(random.Random(0))
    state = BattleState(sides=(build_side(team, range(6)), build_side(team, range(6))))
    ids, pokemon, field = py.encode(state, 0, py.Decision.LEAD)
    assert ids.shape == (py.SLOTS, py.POKEMON_IDS) and (ids[:, 0] > 0).all()
    assert pokemon.shape == (py.SLOTS, py.POKEMON_FLOATS)
    assert field.shape == (py.FIELD_FLOATS,)
    assert pokemon[:, 0].all(), "every slot of a full mirror is present"
    active = 1 + 1 + 1 + 5 + len(py.STAGES) + len(py.STATUSES) + 1 + 1
    assert not pokemon[:, active].any(), "nothing is active before the leads are out"
    _, during, _ = py.encode(state, 0, py.Decision.TURN)
    assert during[:, active].tolist() == [1.0] + [0.0] * 5 + [1.0] + [0.0] * 5


@needs_bridge
def test_both_encoders_mark_a_transformed_pokemon() -> None:
    """No mirror set transforms, so the random battles never set the flag; Imposter does."""
    from battle_sim.models.spec import PokemonSpec
    from battle_sim.utils import Ability

    def spec(species: str, nickname: str, moves: list[str], **extra: Any) -> PokemonSpec:
        return PokemonSpec(species=species, nickname=nickname, level=50, moves=moves, **extra)

    team_a = [spec("Ditto", "A0", ["Transform"], ability=Ability.IMPOSTER), spec("Machamp", "A1", ["Tackle"])]
    team_b = [spec("Tauros", "B0", ["Tackle"]), spec("Rhydon", "B1", ["Tackle"])]
    ours: list[Any] = []
    theirs: list[Any] = []
    scenario, expected = record_played(
        (team_a, team_b),
        ([0, 1], [0, 1]),
        lambda state, side, offered: offered[0],
        max_turns=2,
        observe=lambda state, side, kind: ours.append(py.encode(state, side, DECISIONS[kind])),
    )
    trace, _ = replay_played(scenario, observe=lambda battle, side, kind: theirs.append(battle.observe(side, kind)))
    assert compare(expected, trace) is None, compare(expected, trace)
    transformed = 1 + 1 + 1 + 5 + len(py.STAGES) + len(py.STATUSES) + 1 + 1 + 1
    assert any(pokemon[0, transformed] for _, pokemon, _ in ours), "Ditto never transformed"
    for (_, pokemon, _), (_, rs_pokemon, _) in zip(ours, theirs, strict=True):
        np.testing.assert_array_equal(pokemon, np.asarray(rs_pokemon, dtype=np.float32).reshape(pokemon.shape))
