from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Literal, Sequence, Union

from battle_sim.utils import Category, ExtraStatus, PriorityLevel, Stats, Status, Target, Terrain, Type, Weather


@dataclass(frozen=True)
class DamageEffect:
    power: int | None
    category: Category
    crit_stage: int
    contact: bool
    multi_hit: tuple[int, int] | None = None  # e.g., (2, 5) for Bullet Seed
    recoil_percent: float | None = None
    drain_percent: float | None = None


@dataclass(frozen=True)
class FixedDamageEffect:
    amount_formula: Literal["LEVEL", "SET"]
    set_amount: int | None


@dataclass(frozen=True)
class InflictStatusEffect:
    status: Status | ExtraStatus
    probability: float
    only_if_contact: bool = False


@dataclass(frozen=True)
class StatStageChangeEffect:
    target: Literal["SELF", "TARGET"]
    stages: dict[Stats, int]  # e.g., {"ATTACK": +2} for Swords Dance
    probability: float


@dataclass(frozen=True)
class WeatherEffect:
    kind: Weather
    duration_turns: int | None = None


@dataclass(frozen=True)
class TerrainEffect:
    kind: Terrain
    duration_turns: int | None = None


MoveEffect = Union[
    DamageEffect,
    FixedDamageEffect,
    InflictStatusEffect,
    StatStageChangeEffect,
    WeatherEffect,
    TerrainEffect,
    # TODO: add more effects as needed
]


@dataclass(frozen=True)
class Move:
    name: str
    type: Type
    category: Category
    accuracy_probability: float | None
    priority: PriorityLevel
    pp: int
    target: Target
    effects: Sequence[MoveEffect] = ()
    # TODO: flags can be added later (e.g., "protectable", "makes contact", "triggers switch", etc.)


class MoveSlot(Enum):
    FIRST = 1
    SECOND = 2
    THIRD = 3
    FOURTH = 4

    @property
    def index(self) -> int:
        return self.value - 1


@dataclass
class MoveSet:
    move_one: Move
    move_two: Move | None
    move_three: Move | None
    move_four: Move | None
    slots: ClassVar[tuple[str, ...]] = ("move_one", "move_two", "move_three", "move_four")

    def __getitem__(self, slot: MoveSlot) -> Move | None:
        return [self.move_one, self.move_two, self.move_three, self.move_four][slot.index]

    def __iter__(self):
        yield from (self.move_one, self.move_two, self.move_three, self.move_four)

    def __len__(self) -> int:
        return sum(1 for move in self if move is not None)

    def to_list(self) -> list[Move]:
        return [m for m in (self.move_one, self.move_two, self.move_three, self.move_four) if m is not None]

    def contains(self, move: Move) -> bool:
        return any(m == move for m in self.to_list())

    def learn_move(self, new_move: Move, move_slot: MoveSlot) -> None:
        if len(self) < 4:
            for slot in self.slots:
                if getattr(self, slot) is None:
                    setattr(self, slot, new_move)
                    return
        else:
            setattr(self, self.slots[move_slot.index], new_move)

    def forget_move(self, move_slot: MoveSlot) -> None:
        ordered_moves = [getattr(self, slot) for slot in self.slots]
        ordered_moves.pop(move_slot.index)
        ordered_moves.append(None)
        for slot, move in zip(self.slots, ordered_moves, strict=True):
            setattr(self, slot, move)
