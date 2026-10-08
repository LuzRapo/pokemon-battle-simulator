"""A trained self-play network as a `runner.Player`, playing in the Python engine through ONNX."""

import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import onnxruntime
from numpy.typing import NDArray

from battle_sim.mechanics.battle import BattleState
from battle_sim.models.actions import Action, ActionType
from battle_sim.models.spec import PokemonSpec
from battle_sim.rl.encode import Decision, encode
from battle_sim.runner import build_side

ACTIONS = 14
Z_MOVE = 4
SWITCH = 8


def action_index(action: Action, state: BattleState, side: int) -> int:
    """`action` in the fixed 14-action space: a move slot, that slot plus 4 as a Z-move, or 8 plus a team index."""
    if action.action is ActionType.SWITCH_OUT:
        team = state.sides[side].team
        for index, member in enumerate(team):
            if member is action.switch_in:
                return SWITCH + index
        # A belief view holds copies of the Pokemon, not the engine's own objects.
        assert action.switch_in is not None
        return SWITCH + [member.nickname for member in team].index(action.switch_in.nickname)
    assert action.move is not None
    return action.move.index + (Z_MOVE if action.z_move else 0)


def decision_for(state: BattleState, side: int) -> Decision:
    """A turn, or a replacement: after a knockout, or mid-turn after a pivot (`needs_switch`)."""
    own = state.sides[side]
    return Decision.SWITCH if own.active_pokemon.is_fainted() or own.needs_switch else Decision.TURN


class NetPlayer:
    def __init__(self, model: Path, greedy: bool = True, seed: int = 0) -> None:
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self._session = onnxruntime.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])
        self._greedy = greedy
        self._rng = random.Random(seed)

    def _pick(self, state: BattleState, side: int, decision: Decision, legal: Sequence[int]) -> int:
        ids, pokemon, field = encode(state, side, decision)
        mask = np.zeros((1, ACTIONS), dtype=bool)
        mask[0, list(legal)] = True
        feeds = {"ids": ids[None], "pokemon": pokemon[None], "field": field[None], "mask": mask}
        logits: NDArray[np.float32] = self._session.run(["logits"], feeds)[0][0]
        if self._greedy:
            return int(max(legal, key=lambda a: logits[a]))
        weights = np.exp(logits[list(legal)] - logits[list(legal)].max())
        return int(self._rng.choices(list(legal), weights=weights.tolist())[0])

    def choose_order(self, own: Sequence[PokemonSpec], opponent: Sequence[PokemonSpec]) -> Sequence[int]:
        if sorted(spec.species for spec in own) != sorted(spec.species for spec in opponent):
            raise ValueError("NetPlayer plays mirror battles only: both sides must field the same six")
        listed = range(len(own))
        state = BattleState(sides=(build_side(own, listed), build_side(own, listed)))
        lead = self._pick(state, 0, Decision.LEAD, [SWITCH + i for i in listed]) - SWITCH
        return [lead, *(i for i in listed if i != lead)]

    def choose_action(self, state: BattleState, side_index: int, actions: Sequence[Action]) -> Action:
        by_index = {action_index(action, state, side_index): action for action in actions}
        if len(by_index) == 1:
            return actions[0]
        chosen = self._pick(state, side_index, decision_for(state, side_index), sorted(by_index))
        return by_index[chosen]
