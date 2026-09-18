from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import ClassVar, Literal

from battle_sim.utils import (
    Category,
    ExtraStatus,
    Hazards,
    PriorityLevel,
    PseudoWeather,
    Stats,
    Status,
    Target,
    Terrain,
    Type,
    Weather,
)


@dataclass(frozen=True)
class DamageEffect:
    power: int | None
    category: Category
    crit_stage: int
    contact: bool
    multi_hit: tuple[int, int] | None = None  # e.g., (2, 5) for Bullet Seed
    recoil_percent: float | None = None
    drain_percent: float | None = None
    struggle_recoil: bool = False  # PS struggleRecoil: 1/4 of the user's max HP, unconditionally


@dataclass(frozen=True)
class FixedDamageEffect:
    amount_formula: Literal[
        "LEVEL", "SET", "HALF_TARGET_HP", "ENDEAVOR", "COUNTER", "MIRROR_COAT", "USER_HP", "TARGET_HP", "PSYWAVE"
    ]
    set_amount: int | None


@dataclass(frozen=True)
class InflictStatusEffect:
    status: Status | ExtraStatus
    probability: float
    to_self: bool = False  # PS `self.volatileStatus`: applies to the user of a defender-facing move
    is_secondary: bool = False  # secondaries are blocked by Shield Dust / Covert Cloak, doubled by Serene Grace


@dataclass(frozen=True)
class HealEffect:
    fraction: float  # of the user's max HP


@dataclass(frozen=True)
class RemoveHazardsEffect:
    style: Literal[
        "RAPID_SPIN", "DEFOG"
    ]  # spin: own hazards + own Leech Seed; defog: both sides' hazards, target screens, terrain


@dataclass(frozen=True)
class StatStageChangeEffect:
    target: Literal["SELF", "TARGET"]
    stages: dict[Stats, int]  # e.g., {"ATTACK": +2} for Swords Dance
    probability: float
    is_secondary: bool = False  # secondaries are blocked by Shield Dust / Covert Cloak, doubled by Serene Grace


@dataclass(frozen=True)
class WeatherEffect:
    kind: Weather
    duration_turns: int | None = None


@dataclass(frozen=True)
class TerrainEffect:
    kind: Terrain
    duration_turns: int | None = None


@dataclass(frozen=True)
class PseudoWeatherEffect:
    kind: PseudoWeather
    duration_turns: int | None = None


@dataclass(frozen=True)
class SideConditionEffect:
    kind: Hazards
    duration_turns: int | None = None


class CodedMoveKind(Enum):
    """Move behaviours that live in Showdown code; the engine implements each in engine/coded.py."""

    REST = auto()
    WEATHER_HEAL = auto()  # Morning Sun / Moonlight / Synthesis
    PAIN_SPLIT = auto()
    STRENGTH_SAP = auto()
    BELLY_DRUM = auto()
    HAZE = auto()
    COURT_CHANGE = auto()
    WISH = auto()
    HEALING_WISH = auto()
    CURE_SELF = auto()  # Take Heart
    CURE_PARTY = auto()  # Heal Bell / Aromatherapy
    TIDY_UP = auto()
    CURSE = auto()
    PERISH_SONG = auto()
    REVIVAL_BLESSING = auto()
    TRANSFORM = auto()
    SHED_TAIL = auto()
    KNOCK_OFF_ITEM = auto()
    TRICK = auto()
    SKILL_SWAP = auto()
    ROLE_PLAY = auto()  # the user takes a copy of the target's ability
    ENTRAINMENT = auto()  # the target is given the user's
    WORRY_SEED = auto()  # the target is given Insomnia
    SIMPLE_BEAM = auto()  # the target is given Simple
    FUTURE_SIGHT = auto()  # Future Sight / Doom Desire: queued now, lands two turns later


@dataclass(frozen=True)
class CodedEffect:
    kind: CodedMoveKind


type MoveEffect = (
    DamageEffect
    | FixedDamageEffect
    | HealEffect
    | InflictStatusEffect
    | RemoveHazardsEffect
    | StatStageChangeEffect
    | WeatherEffect
    | TerrainEffect
    | PseudoWeatherEffect
    | SideConditionEffect
    | CodedEffect
)


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
    self_switch: bool = False
    protectable: bool = False  # PS flag `protect`: blocked by Protect/Detect
    bypass_substitute: bool = False  # PS flag `bypasssub`: sound moves, Chatter, etc.
    stalling: bool = False  # PS `stallingMove`: Protect-likes sharing the consecutive-use failure counter
    typeless: bool = False  # Struggle: ignores type effectiveness (PS onEffectiveness -> 0)
    slicing: bool = False  # PS flag `slicing`: boosted by Sharpness
    sound: bool = False  # PS flag `sound`: Liquid Voice / Punk Rock
    punching: bool = False  # PS flag `punch`: boosted by Iron Fist
    biting: bool = False  # PS flag `bite`: boosted by Strong Jaw
    pulse: bool = False  # PS flag `pulse`: boosted by Mega Launcher
    wind: bool = False  # PS flag `wind`: absorbed by Wind Rider
    bullet: bool = False  # PS flag `bullet`: blocked by Bulletproof
    healing: bool = False  # PS flag `heal`: Triage priority
    reflectable: bool = False  # PS flag `reflectable`: bounced by Magic Bounce
    charge: bool = False  # PS flag `charge`: two-turn moves (Solar Beam, Meteor Beam)
    self_destructs: bool = False  # PS `selfdestruct`: the user faints on use (Explosion, Memento)
    recharges: bool = False  # PS `mustrecharge` self-volatile: the next turn is lost
    force_switch: bool = False  # PS `forceSwitch`: phazing (Whirlwind, Roar, Dragon Tail)
    has_crash_damage: bool = False  # PS `hasCrashDamage`: (High) Jump Kick — half the user's max HP if it fails
    defrosts_user: bool = False  # PS flag `defrost`: a frozen user melts itself free and attacks anyway
    thaws_target: bool = False  # PS `thawsTarget`: Scald and friends unfreeze whatever they hit


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

    def __setitem__(self, slot: MoveSlot, move: Move | None) -> None:
        setattr(self, self.slots[slot.index], move)

    def __iter__(self) -> Iterator[Move | None]:
        yield from (self.move_one, self.move_two, self.move_three, self.move_four)

    def __len__(self) -> int:
        return sum(1 for move in self if move is not None)

    def to_list(self) -> list[Move]:
        return [m for m in (self.move_one, self.move_two, self.move_three, self.move_four) if m is not None]

    def contains(self, move: Move) -> bool:
        return any(m == move for m in self.to_list())

    def learn_move(self, new_move: Move, move_slot: MoveSlot) -> None:
        if len(self) < 4:
            for slot in MoveSlot:
                if self[slot] is None:
                    self[slot] = new_move
                    return
        else:
            self[move_slot] = new_move

    def forget_move(self, move_slot: MoveSlot) -> None:
        remaining = [move for slot, move in zip(MoveSlot, self, strict=True) if slot is not move_slot]
        remaining.append(None)
        for slot, move in zip(MoveSlot, remaining, strict=True):
            self[slot] = move
