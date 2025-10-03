from dataclasses import dataclass
from enum import Enum, IntEnum, StrEnum, auto


class Stats(StrEnum):
    HP = "HP"
    ATTACK = "ATTACK"
    DEFENCE = "DEFENCE"
    SP_ATTACK = "SP_ATTACK"
    SP_DEFENCE = "SP_DEFENCE"
    SPEED = "SPEED"
    ACCURACY = "ACCURACY"
    EVASION = "EVASION"


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


class Target(Enum):
    SINGLE_OPPONENT = auto()
    SELF = auto()
    USER_SIDE = auto()
    OPPONENT_SIDE = auto()
    FIELD = auto()
    # TODO: Add support for these in the engine:
    ALL_ADJACENT_ENEMIES = auto()
    ALL_ADJACENT = auto()


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
    HARDY = NatureEffect(Stats.ATTACK, Stats.ATTACK)
    DOCILE = NatureEffect(Stats.DEFENCE, Stats.DEFENCE)
    BASHFUL = NatureEffect(Stats.SP_ATTACK, Stats.SP_ATTACK)
    QUIRKY = NatureEffect(Stats.SP_DEFENCE, Stats.SP_DEFENCE)
    SERIOUS = NatureEffect(Stats.SPEED, Stats.SPEED)

    LONELY = NatureEffect(Stats.ATTACK, Stats.DEFENCE)
    BRAVE = NatureEffect(Stats.ATTACK, Stats.SPEED)
    ADAMANT = NatureEffect(Stats.ATTACK, Stats.SP_ATTACK)
    NAUGHTY = NatureEffect(Stats.ATTACK, Stats.SP_DEFENCE)

    BOLD = NatureEffect(Stats.DEFENCE, Stats.ATTACK)
    RELAXED = NatureEffect(Stats.DEFENCE, Stats.SPEED)
    IMPISH = NatureEffect(Stats.DEFENCE, Stats.SP_ATTACK)
    LAX = NatureEffect(Stats.DEFENCE, Stats.SP_DEFENCE)

    TIMID = NatureEffect(Stats.SPEED, Stats.ATTACK)
    HASTY = NatureEffect(Stats.SPEED, Stats.DEFENCE)
    JOLLY = NatureEffect(Stats.SPEED, Stats.SP_ATTACK)
    NAIVE = NatureEffect(Stats.SPEED, Stats.SP_DEFENCE)

    MODEST = NatureEffect(Stats.SP_ATTACK, Stats.ATTACK)
    MILD = NatureEffect(Stats.SP_ATTACK, Stats.DEFENCE)
    QUIET = NatureEffect(Stats.SP_ATTACK, Stats.SPEED)
    RASH = NatureEffect(Stats.SP_ATTACK, Stats.SP_DEFENCE)

    CALM = NatureEffect(Stats.SP_DEFENCE, Stats.ATTACK)
    GENTLE = NatureEffect(Stats.SP_DEFENCE, Stats.DEFENCE)
    SASSY = NatureEffect(Stats.SP_DEFENCE, Stats.SPEED)
    CAREFUL = NatureEffect(Stats.SP_DEFENCE, Stats.SP_ATTACK)
