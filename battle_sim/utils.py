from dataclasses import dataclass
from enum import Enum, IntEnum, auto
from typing import Literal

Stats = Literal["HP", "ATTACK", "DEFENCE", "SP_ATTACK", "SP_DEFENCE", "SPEED"]


class Type(Enum):
    NORMAL = auto()
    FIRE = auto()
    WATER = auto()
    GRASS = auto()
    ELECTRIC = auto()
    ICE = auto()
    FIGHTING = auto()
    POISON = auto()
    GROUND = auto()
    FLYING = auto()
    PSYCHIC = auto()
    BUG = auto()
    ROCK = auto()
    GHOST = auto()
    DRAGON = auto()
    DARK = auto()
    STEEL = auto()
    FAIRY = auto()


class Category(Enum):
    PHYSICAL = auto()
    SPECIAL = auto()
    STATUS = auto()


class PriorityLevel(IntEnum):
    """
    These are codenames for various priority levels based on the most common moves that have them.
    Note: Higher number means higher priority; this is based on Gen 9 mechanics.
    """

    HELPING_HAND = 5
    PROTECT = 4  # or Protect-like moves
    FAKE_OUT = 3  # or Quick Guard, Upper Hand, Wide Guard
    E_SPEED = 2  # or First Impression, Follow Me, Rage Powder, etc.
    QUICK_ATTACK = 1  # or Sucker Punch, Bullet Punch, etc.
    NORMAL = 0
    VITAL_THROW = -1
    # NOTHING = -2
    FOCUS_PUNCH = -3
    AVALANCHE = -4
    COUNTER = -5
    ROAR = -6
    TRICK_ROOM = -7


class Status(Enum):
    NONE = auto()
    BURN = auto()
    POISON = auto()
    PARALYSIS = auto()
    SLEEP = auto()
    FREEZE = auto()


class ExtraStatus(Enum):
    NONE = auto()
    CONFUSION = auto()
    FLINCH = auto()
    INFATUATION = auto()
    TRAPPED = auto()
    CURSE = auto()
    EMBARGO = auto()
    IDENTIFIED = auto()
    LEECH_SEED = auto()
    NIGHTMARE = auto()
    PERISH_SONG = auto()
    TAUNT = auto()


class Weather(Enum):
    NONE = auto()
    SUN = auto()
    RAIN = auto()
    HARSH_SUN = auto()
    HEAVY_RAIN = auto()
    SANDSTORM = auto()
    SNOW = auto()
    STRONG_WINDS = auto()


class Terrain(Enum):
    NONE = auto()
    ELECTRIC = auto()
    GRASSY = auto()
    MISTY = auto()
    PSYCHIC = auto()


class Hazards(Enum):
    NONE = auto()
    REFLECT = auto()
    LIGHT_SCREEN = auto()
    AURORA_VEIL = auto()
    SPIKES = auto()
    TOXIC_SPIKES = auto()
    STEALTH_ROCK = auto()
    STICKY_WEB = auto()


@dataclass(frozen=True)
class NatureEffect:
    UP: Stats
    DOWN: Stats


class Nature(Enum):
    HARDY = NatureEffect("ATTACK", "ATTACK")
    DOCILE = NatureEffect("DEFENCE", "DEFENCE")
    BASHFUL = NatureEffect("SP_ATTACK", "SP_ATTACK")
    QUIRKY = NatureEffect("SP_DEFENCE", "SP_DEFENCE")
    SERIOUS = NatureEffect("SPEED", "SPEED")

    LONELY = NatureEffect("ATTACK", "DEFENCE")
    BRAVE = NatureEffect("ATTACK", "SPEED")
    ADAMANT = NatureEffect("ATTACK", "SP_ATTACK")
    NAUGHTY = NatureEffect("ATTACK", "SP_DEFENCE")

    BOLD = NatureEffect("DEFENCE", "ATTACK")
    RELAXED = NatureEffect("DEFENCE", "SPEED")
    IMPISH = NatureEffect("DEFENCE", "SP_ATTACK")
    LAX = NatureEffect("DEFENCE", "SP_DEFENCE")

    TIMID = NatureEffect("SPEED", "ATTACK")
    HASTY = NatureEffect("SPEED", "DEFENCE")
    JOLLY = NatureEffect("SPEED", "SP_ATTACK")
    NAIVE = NatureEffect("SPEED", "SP_DEFENCE")

    MODEST = NatureEffect("SP_ATTACK", "ATTACK")
    MILD = NatureEffect("SP_ATTACK", "DEFENCE")
    QUIET = NatureEffect("SP_ATTACK", "SPEED")
    RASH = NatureEffect("SP_ATTACK", "SP_DEFENCE")

    CALM = NatureEffect("SP_DEFENCE", "ATTACK")
    GENTLE = NatureEffect("SP_DEFENCE", "DEFENCE")
    SASSY = NatureEffect("SP_DEFENCE", "SPEED")
    CAREFUL = NatureEffect("SP_DEFENCE", "SP_ATTACK")
