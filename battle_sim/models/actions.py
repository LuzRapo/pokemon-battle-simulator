from dataclasses import dataclass
from enum import Enum, auto

from battle_sim.models.moves import MoveSlot
from battle_sim.models.pokemon import Pokemon
from battle_sim.utils import Target


class ActionType(Enum):
    USE_MOVE = auto()
    SWITCH_OUT = auto()
    USE_ITEM = auto()
    RUN = auto()


@dataclass(frozen=True, slots=True)
class Action:
    action: ActionType
    target: Target | None = None
    move: MoveSlot | None = None
    switch_in: Pokemon | None = None
    z_move: bool = False  # unleash this slot's move through the held Z-Crystal, once per battle

    def __post_init__(self) -> None:
        if self.action is ActionType.USE_MOVE and (self.target is None or self.move is None):
            raise ValueError("USE_MOVE requires a target and a move.")
        if self.action is ActionType.SWITCH_OUT and self.switch_in is None:
            raise ValueError("SWITCH_OUT requires a pokemon to switch_in.")
        if self.z_move and self.action is not ActionType.USE_MOVE:
            raise ValueError("Only a USE_MOVE action can be a Z-move.")
