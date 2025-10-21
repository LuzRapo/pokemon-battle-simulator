from enum import Enum, auto

from pydantic import BaseModel, model_validator

from battle_sim.models.moves import MoveSlot
from battle_sim.models.pokemon import Pokemon
from battle_sim.utils import Target


class ActionType(Enum):
    USE_MOVE = auto()
    SWITCH_OUT = auto()
    USE_ITEM = auto()
    RUN = auto()


class Action(BaseModel):
    action: ActionType
    target: Target | None = None
    move: MoveSlot | None = None
    switch_in: Pokemon | None = None

    @model_validator(mode="after")
    def _check_constraints(self):
        if self.action == ActionType.USE_MOVE and (self.target is None or self.move is None):
            raise ValueError("USE_MOVE requires a target and a move.")
        elif self.action == ActionType.SWITCH_OUT and self.switch_in is None:
            raise ValueError("SWITCH_OUT requires a pokemon to switch_in.")
        return self
