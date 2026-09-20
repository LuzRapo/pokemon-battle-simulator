from dataclasses import dataclass
from enum import Enum, IntEnum, StrEnum, auto


class Stats(StrEnum):
    HP = "HP"
    ATTACK = "ATTACK"
    DEFENCE = "DEFENCE"
    SP_ATTACK = "SP_ATTACK"
    SP_DEFENCE = "SP_DEFENCE"
    SPEED = "SPEED"
    # Battle-only: exist in StatStages, not in StatTotals/EVs/IVs (indexing those with these crashes)
    ACCURACY = "ACCURACY"
    EVASION = "EVASION"


class StageBases(IntEnum):
    STATS = 2
    EVASION = 3
    ACCURACY = 3


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
    TOXIC = auto()
    PARALYSIS = auto()
    SLEEP = auto()
    FREEZE = auto()


class ExtraStatus(Enum):
    NONE = auto()
    NINE_LIVES = auto()  # lives still in hand; see abilities._bind_nine_lives
    NINE_LIVES_SPENT = auto()  # set the instant one is used, cleared once the heal lands
    CONFUSION = auto()
    DISABLE = auto()
    ENCORE = auto()
    FLINCH = auto()
    FOCUS_ENERGY = auto()
    IDENTIFIED = auto()
    MIRACLE_EYE = auto()
    LEECH_SEED = auto()
    LOCKED_MOVE = auto()
    NIGHTMARE = auto()
    PROTECT = auto()
    YAWN = auto()
    SUBSTITUTE = auto()  # volatile value holds the substitute's remaining HP
    TAUNT = auto()
    SALT_CURE = auto()
    CURSE = auto()  # the ghost variant's residual
    PERISH = auto()  # value counts down; faints at zero
    DESTINY_BOND = auto()  # cleared when the user next acts
    ENDURE = auto()
    CHARGING = auto()  # two-turn moves: committed to release next turn
    MUST_RECHARGE = auto()  # Hyper Beam-likes: the next turn is lost
    SLOW_START = auto()  # Regigigas: value counts down from 5, halves Attack/Speed while present
    LOAFING = auto()  # Truant: present means the next move attempt is skipped, then it is removed
    ROOSTED = auto()  # Roost: the user's Flying type is ignored for the rest of the turn
    PARTIALLY_TRAPPED = auto()  # Wrap/Magma Storm/Whirlpool: value counts down; chips and blocks switching


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
    TAILWIND = auto()


class Item(Enum):
    NONE = auto()
    LEFTOVERS = auto()
    LIFE_ORB = auto()
    CHOICE_BAND = auto()
    CHOICE_SPECS = auto()
    CHOICE_SCARF = auto()
    FOCUS_SASH = auto()
    BLACK_SLUDGE = auto()
    HEAVY_DUTY_BOOTS = auto()
    AIR_BALLOON = auto()
    ROCKY_HELMET = auto()
    EXPERT_BELT = auto()
    ASSAULT_VEST = auto()
    IRON_PLATE = auto()
    CHOPLE_BERRY = auto()
    SHUCA_BERRY = auto()
    COLBUR_BERRY = auto()
    SITRUS_BERRY = auto()
    LUM_BERRY = auto()
    CHESTO_BERRY = auto()
    BOOSTER_ENERGY = auto()
    LOADED_DICE = auto()
    LIGHT_CLAY = auto()
    COVERT_CLOAK = auto()
    DAMP_ROCK = auto()
    HEAT_ROCK = auto()
    ICY_ROCK = auto()
    SMOOTH_ROCK = auto()
    TERRAIN_EXTENDER = auto()
    TOXIC_ORB = auto()
    FLAME_ORB = auto()
    BLACK_GLASSES = auto()
    MYSTIC_WATER = auto()
    METAL_COAT = auto()
    NEVER_MELT_ICE = auto()
    SILK_SCARF = auto()
    EARTH_PLATE = auto()
    SPOOKY_PLATE = auto()
    PIXIE_PLATE = auto()
    SOUL_DEW = auto()
    GRISEOUS_CORE = auto()
    GRISEOUS_ORB = auto()  # the Gen 4-7 name for the same item; Gen 9 renamed it Griseous Core
    SHED_SHELL = auto()
    WIKI_BERRY = auto()
    WELLSPRING_MASK = auto()
    CORNERSTONE_MASK = auto()
    WEAKNESS_POLICY = auto()
    EVIOLITE = auto()
    WIDE_LENS = auto()
    SCOPE_LENS = auto()
    CLEAR_AMULET = auto()
    WHITE_HERB = auto()
    GRASSY_SEED = auto()
    ELECTRIC_SEED = auto()
    PSYCHIC_SEED = auto()
    MISTY_SEED = auto()
    RUSTED_SWORD = auto()  # Zacian-Crowned's forme item: the forme (already in species data) is the effect
    RUSTED_SHIELD = auto()  # Zamazenta-Crowned's forme item, likewise
    POWER_HERB = auto()
    EJECT_BUTTON = auto()
    EJECT_PACK = auto()
    RED_CARD = auto()
    MENTAL_HERB = auto()
    CUSTAP_BERRY = auto()
    QUICK_CLAW = auto()
    ADRENALINE_ORB = auto()
    MIRROR_HERB = auto()
    STARF_BERRY = auto()
    LEPPA_BERRY = auto()
    WISE_GLASSES = auto()
    MUSCLE_BAND = auto()
    MIRACLE_SEED = auto()
    SILVER_POWDER = auto()
    SPLASH_PLATE = auto()
    STONE_PLATE = auto()
    PROTECTIVE_PADS = auto()
    ADAMANT_CRYSTAL = auto()
    LUSTROUS_GLOBE = auto()
    PUNCHING_GLOVE = auto()
    # Arceus plates (Multitype) — IRON/EARTH/SPOOKY/PIXIE/SPLASH/STONE_PLATE already exist above
    FIST_PLATE = auto()
    SKY_PLATE = auto()
    TOXIC_PLATE = auto()
    INSECT_PLATE = auto()
    FLAME_PLATE = auto()
    MEADOW_PLATE = auto()
    ZAP_PLATE = auto()
    MIND_PLATE = auto()
    ICICLE_PLATE = auto()
    DRACO_PLATE = auto()
    DREAD_PLATE = auto()
    # Silvally memory discs (RKS System) — one per type except Normal
    BUG_MEMORY = auto()
    DARK_MEMORY = auto()
    DRAGON_MEMORY = auto()
    ELECTRIC_MEMORY = auto()
    FAIRY_MEMORY = auto()
    FIGHTING_MEMORY = auto()
    FIRE_MEMORY = auto()
    FLYING_MEMORY = auto()
    GHOST_MEMORY = auto()
    GRASS_MEMORY = auto()
    GROUND_MEMORY = auto()
    ICE_MEMORY = auto()
    POISON_MEMORY = auto()
    PSYCHIC_MEMORY = auto()
    ROCK_MEMORY = auto()
    STEEL_MEMORY = auto()
    WATER_MEMORY = auto()

    # Genesect's Drives: retype its signature move Techno Blast (see engine.moves._move_type_override)
    DOUSE_DRIVE = auto()
    SHOCK_DRIVE = auto()
    BURN_DRIVE = auto()
    CHILL_DRIVE = auto()

    # Mega Stones: held by the matching species to Mega Evolve; inert on anything else
    ABOMASITE = auto()
    ABSOLITE = auto()
    AERODACTYLITE = auto()
    AGGRONITE = auto()
    ALAKAZITE = auto()
    ALTARIANITE = auto()
    AMPHAROSITE = auto()
    AUDINITE = auto()
    BANETTITE = auto()
    BEEDRILLITE = auto()
    BLASTOISINITE = auto()
    BLAZIKENITE = auto()
    CAMERUPTITE = auto()
    CHARIZARDITE_X = auto()
    CHARIZARDITE_Y = auto()
    DIANCITE = auto()
    GALLADITE = auto()
    GARCHOMPITE = auto()
    GARDEVOIRITE = auto()
    GENGARITE = auto()
    GLALITITE = auto()
    GYARADOSITE = auto()
    HERACRONITE = auto()
    HOUNDOOMINITE = auto()
    KANGASKHANITE = auto()
    LATIASITE = auto()
    LATIOSITE = auto()
    LOPUNNITE = auto()
    LUCARIONITE = auto()
    MANECTITE = auto()
    MAWILITE = auto()
    MEDICHAMITE = auto()
    METAGROSSITE = auto()
    MEWTWONITE_X = auto()
    MEWTWONITE_Y = auto()
    PIDGEOTITE = auto()
    PINSIRITE = auto()
    SABLENITE = auto()
    SALAMENCITE = auto()
    SCEPTILITE = auto()
    SCIZORITE = auto()
    SHARPEDONITE = auto()
    SLOWBRONITE = auto()
    STEELIXITE = auto()
    SWAMPERTITE = auto()
    TYRANITARITE = auto()
    VENUSAURITE = auto()

    # The Legends: Z-A stones. Every one of these formes was already in the vendored species
    # data with real stats and a `requiredItem`; the only thing missing was the stone itself, so
    # `item_from_showdown` returned None and `_forme_by_base_and_item` skipped the pairing. Forty
    # nine Mega Evolutions were unreachable for want of forty nine enum members.
    ABSOLITE_Z = auto()
    BARBARACITE = auto()
    BAXCALIBRITE = auto()
    CHANDELURITE = auto()
    CHESNAUGHTITE = auto()
    CHIMECHITE = auto()
    CLEFABLITE = auto()
    CRABOMINITE = auto()
    DARKRANITE = auto()
    DELPHOXITE = auto()
    DRAGALGITE = auto()
    DRAGONINITE = auto()
    DRAMPANITE = auto()
    EELEKTROSSITE = auto()
    EMBOARITE = auto()
    EXCADRITE = auto()
    FALINKSITE = auto()
    FERALIGITE = auto()
    FLOETTITE = auto()
    FROSLASSITE = auto()
    GARCHOMPITE_Z = auto()
    GLIMMORANITE = auto()
    GOLISOPITE = auto()
    GOLURKITE = auto()
    GRENINJITE = auto()
    HAWLUCHANITE = auto()
    HEATRANITE = auto()
    LUCARIONITE_Z = auto()
    MAGEARNITE = auto()
    MALAMARITE = auto()
    MEGANIUMITE = auto()
    MEOWSTICITE = auto()
    PYROARITE = auto()
    RAICHUNITE_X = auto()
    RAICHUNITE_Y = auto()
    SCOLIPITE = auto()
    SCOVILLAINITE = auto()
    SCRAFTINITE = auto()
    SKARMORITE = auto()
    STARAPTITE = auto()
    STARMINITE = auto()
    TATSUGIRINITE = auto()
    VICTREEBELITE = auto()
    ZERAORITE = auto()
    ZYGARDITE = auto()

    # The butler's own, and the only item here that is not from the games. A flat 1.25x on
    # everything he throws, with none of the Life Orb recoil that usually pays for it.
    MEOWFREDS_MONOCLE = auto()

    # Primal Reversion orbs
    BLUE_ORB = auto()
    RED_ORB = auto()

    # Z-Crystals: one Z-move per battle
    ALORAICHIUM_Z = auto()
    BUGINIUM_Z = auto()
    DARKINIUM_Z = auto()
    DECIDIUM_Z = auto()
    DRAGONIUM_Z = auto()
    ELECTRIUM_Z = auto()
    FAIRIUM_Z = auto()
    FIGHTINIUM_Z = auto()
    FIRIUM_Z = auto()
    FLYINIUM_Z = auto()
    GHOSTIUM_Z = auto()
    GRASSIUM_Z = auto()
    GROUNDIUM_Z = auto()
    ICIUM_Z = auto()
    INCINIUM_Z = auto()
    KOMMONIUM_Z = auto()
    LUNALIUM_Z = auto()
    LYCANIUM_Z = auto()
    MARSHADIUM_Z = auto()
    MEWNIUM_Z = auto()
    MIMIKIUM_Z = auto()
    NORMALIUM_Z = auto()
    POISONIUM_Z = auto()
    PSYCHIUM_Z = auto()
    ROCKIUM_Z = auto()
    SOLGANIUM_Z = auto()
    STEELIUM_Z = auto()
    ULTRANECROZIUM_Z = auto()
    EEVIUM_Z = auto()
    PIKANIUM_Z = auto()
    PIKASHUNIUM_Z = auto()
    PRIMARIUM_Z = auto()
    SNORLIUM_Z = auto()
    TAPUNIUM_Z = auto()
    WATERIUM_Z = auto()


CHOICE_ITEMS = frozenset({Item.CHOICE_BAND, Item.CHOICE_SPECS, Item.CHOICE_SCARF})


class Ability(Enum):
    NONE = auto()
    SPEED_BOOST = auto()
    INTIMIDATE = auto()
    DROUGHT = auto()
    DRIZZLE = auto()
    SAND_STREAM = auto()
    SNOW_WARNING = auto()
    LEVITATE = auto()
    WONDER_GUARD = auto()
    NINE_LIVES = auto()
    STURDY = auto()
    FLASH_FIRE = auto()
    VOLT_ABSORB = auto()
    WATER_ABSORB = auto()
    MAGIC_GUARD = auto()
    PRANKSTER = auto()
    QUICK_FEET = auto()
    ADAPTABILITY = auto()
    HUGE_POWER = auto()
    PURE_POWER = auto()
    GUTS = auto()
    ROCK_HEAD = auto()
    NO_GUARD = auto()
    MOTOR_DRIVE = auto()
    PRESSURE = auto()
    SOLAR_POWER = auto()
    MOXIE = auto()
    CLEAR_BODY = auto()
    SAND_VEIL = auto()
    ROUGH_SKIN = auto()
    PROTOSYNTHESIS = auto()
    QUARK_DRIVE = auto()
    REGENERATOR = auto()
    NATURAL_CURE = auto()
    SUPREME_OVERLORD = auto()
    GOOD_AS_GOLD = auto()
    MULTISCALE = auto()
    VESSEL_OF_RUIN = auto()
    BEADS_OF_RUIN = auto()
    SWORD_OF_RUIN = auto()
    TABLETS_OF_RUIN = auto()
    SHARPNESS = auto()
    TECHNICIAN = auto()
    TOXIC_DEBRIS = auto()
    SWIFT_SWIM = auto()
    CHLOROPHYLL = auto()
    SAND_RUSH = auto()
    SLUSH_RUSH = auto()
    UNBURDEN = auto()
    FLAME_BODY = auto()
    STATIC = auto()
    POISON_TOUCH = auto()
    TOXIC_CHAIN = auto()
    DAUNTLESS_SHIELD = auto()
    INTREPID_SWORD = auto()
    DOWNLOAD = auto()
    MULTITYPE = auto()  # Arceus formes already carry their per-forme typing in species data
    HADRON_ENGINE = auto()
    ORICHALCUM_PULSE = auto()
    GRASSY_SURGE = auto()
    ELECTRIC_SURGE = auto()
    PSYCHIC_SURGE = auto()
    MISTY_SURGE = auto()
    BAD_DREAMS = auto()
    WEAK_ARMOR = auto()
    STAMINA = auto()
    BERSERK = auto()
    JUSTIFIED = auto()
    THERMAL_EXCHANGE = auto()
    CURSED_BODY = auto()
    PURIFYING_SALT = auto()
    WATER_BUBBLE = auto()
    CONTRARY = auto()
    DEFIANT = auto()
    COMPETITIVE = auto()
    POISON_HEAL = auto()
    SHIELD_DUST = auto()
    SERENE_GRACE = auto()
    BLAZE = auto()
    TORRENT = auto()
    OVERGROW = auto()
    SWARM = auto()
    PRISM_ARMOR = auto()
    FILTER = auto()
    TINTED_LENS = auto()
    HEATPROOF = auto()
    THICK_FAT = auto()
    ICE_SCALES = auto()
    DRAGONS_MAW = auto()
    HYDRATION = auto()
    ICE_BODY = auto()
    SAP_SIPPER = auto()
    EARTH_EATER = auto()
    WELL_BAKED_BODY = auto()
    DRY_SKIN = auto()
    SOUL_HEART = auto()
    CHILLING_NEIGH = auto()
    AS_ONE_GLASTRIER = auto()
    OVERCOAT = auto()
    SKILL_LINK = auto()
    INNER_FOCUS = auto()
    UNAWARE = auto()
    INFILTRATOR = auto()
    LIBERO = auto()
    PROTEAN = auto()
    POISON_PUPPETEER = auto()
    MAGIC_BOUNCE = auto()
    POWER_CONSTRUCT = auto()
    DELTA_STREAM = auto()
    LIQUID_VOICE = auto()
    NEUTRALIZING_GAS = auto()
    MIRROR_ARMOR = auto()
    PICKPOCKET = auto()
    MAGICIAN = auto()
    MOLD_BREAKER = auto()
    TERAVOLT = auto()
    SHEER_FORCE = auto()
    ZERO_TO_HERO = auto()
    IMPOSTER = auto()
    AIR_LOCK = auto()
    ELECTROMORPHOSIS = auto()
    IRON_FIST = auto()
    SCRAPPY = auto()
    MINDS_EYE = auto()
    TRIAGE = auto()
    MAGNET_PULL = auto()
    ARENA_TRAP = auto()
    SHADOW_TAG = auto()
    HARVEST = auto()
    DAZZLING = auto()
    GALE_WINGS = auto()
    LIQUID_OOZE = auto()
    MYCELIUM_MIGHT = auto()
    FULL_METAL_BODY = auto()
    QUICK_DRAW = auto()
    CORROSION = auto()
    LEAF_GUARD = auto()
    PUNK_ROCK = auto()
    WIND_RIDER = auto()
    # Added 2026-09-08 after a coverage audit found each of these unwired while its species was being
    # rated as though it had no ability at all (see the species-rating notes): Ferrothorn's Iron
    # Barbs, Archeops' Defeatist, Toxapex's Merciless, and so on.
    IRON_BARBS = auto()
    GALVANIZE = auto()
    DEFEATIST = auto()
    MARVEL_SCALE = auto()
    FUR_COAT = auto()
    STEELWORKER = auto()
    LONG_REACH = auto()
    QUEENLY_MAJESTY = auto()
    STAKEOUT = auto()
    SURGE_SURFER = auto()
    MERCILESS = auto()
    EMERGENCY_EXIT = auto()
    WIMP_OUT = auto()
    FLUFFY = auto()
    STORM_DRAIN = auto()
    COMATOSE = auto()
    MUMMY = auto()
    FLOWER_GIFT = auto()
    SHED_SKIN = auto()
    ANALYTIC = auto()
    SUPER_LUCK = auto()
    AFTERMATH = auto()
    POISON_POINT = auto()
    EFFECT_SPORE = auto()
    SIMPLE = auto()
    # The forme-changing family: each swaps its holder onto another species entry mid-battle, which
    # is why they live in `formes.py` rather than as event handlers like every other ability.
    STANCE_CHANGE = auto()
    DISGUISE = auto()
    ZEN_MODE = auto()
    SCHOOLING = auto()
    SHIELDS_DOWN = auto()
    ILLUSION = auto()  # disguise only: our battle state is fully observable, so there is nothing to hide
    TRACE = auto()
    TERA_SHIFT = auto()
    GUARD_DOG = auto()
    BULLETPROOF = auto()
    ROCKY_PAYLOAD = auto()
    TRANSISTOR = auto()
    BATTLE_BOND = auto()
    LIMBER = auto()
    INSOMNIA = auto()
    VITAL_SPIRIT = auto()
    WATER_VEIL = auto()
    MAGMA_ARMOR = auto()
    IMMUNITY = auto()
    OWN_TEMPO = auto()
    WHITE_SMOKE = auto()
    SNOW_CLOAK = auto()
    COMPOUND_EYES = auto()
    TANGLED_FEET = auto()
    BATTLE_ARMOR = auto()
    SHELL_ARMOR = auto()
    SNIPER = auto()
    SOUNDPROOF = auto()
    RECKLESS = auto()
    RAIN_DISH = auto()
    STEADFAST = auto()
    HUSTLE = auto()
    KEEN_EYE = auto()
    HYPER_CUTTER = auto()
    BIG_PECKS = auto()
    RKS_SYSTEM = auto()
    BEAST_BOOST = auto()
    TURBOBLAZE = auto()
    SHADOW_SHIELD = auto()
    NEUROFORCE = auto()
    VICTORY_STAR = auto()
    FAIRY_AURA = auto()
    DARK_AURA = auto()
    AURA_BREAK = auto()
    SYNCHRONIZE = auto()
    SLOW_START = auto()
    # Abilities introduced by Mega Evolution / Primal Reversion formes
    TOUGH_CLAWS = auto()
    STRONG_JAW = auto()
    MEGA_LAUNCHER = auto()
    PARENTAL_BOND = auto()
    SAND_FORCE = auto()
    AERILATE = auto()
    PIXILATE = auto()
    REFRIGERATE = auto()
    PRIMORDIAL_SEA = auto()
    DESOLATE_LAND = auto()
    LIGHTNING_ROD = auto()
    HEALER = auto()
    TRUANT = auto()


class PseudoWeather(Enum):
    TRICK_ROOM = auto()
    GRAVITY = auto()
    MAGIC_ROOM = auto()
    WONDER_ROOM = auto()


class BattleFormat(Enum):
    SINGLES = auto()


class Outcome(Enum):
    P1_WIN = auto()
    P2_WIN = auto()
    DRAW = auto()


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


# Abilities that cannot be swapped away, copied, replaced or taken by anything — the games keep such
# a list and this is ours. An ability here stays on whoever was born with it and never appears on
# anybody else, so Skill Swap, Trace, Role Play, Entrainment, Worry Seed, Simple Beam and Mummy all
# read it before they touch a thing.
#
# Two kinds of member. Nine Lives is ours and needs both halves: losing it would undo the one thing
# the butler's final form *is*, and gaining it would be far worse — a Trace on the switch-in and the
# challenger has nine lives of their own, which no amount of his nine can answer.
#
# The rest are the canonical Gen 7 list, and they are here for a different reason: each one *defines*
# a Pokemon rather than decorating it. Multitype is what makes an Arceus its type; Disguise is
# Mimikyu; Schooling is Wishiwashi. Moving one produces a Pokemon the game has no rules for.
#
# The real games keep four slightly different lists — what may be copied, swapped, replaced and
# suppressed each differ at the edges (Illusion may be swapped but not copied, and so on). One list
# is a deliberate simplification: it errs toward leaving forme-defining abilities alone, which is the
# right side to err on, and it is one rule rather than four nearly-identical ones to keep true.
UNTOUCHABLE_ABILITIES: frozenset["Ability"] = frozenset(
    {
        Ability.NINE_LIVES,
        Ability.MULTITYPE,
        Ability.RKS_SYSTEM,
        Ability.STANCE_CHANGE,
        Ability.SCHOOLING,
        Ability.SHIELDS_DOWN,
        Ability.DISGUISE,
        Ability.COMATOSE,
        Ability.BATTLE_BOND,
        Ability.POWER_CONSTRUCT,
        Ability.ZEN_MODE,
        Ability.ILLUSION,
        Ability.IMPOSTER,
    }
)


def is_untouchable(ability: "Ability") -> bool:
    """Whether this ability refuses to be moved, copied, replaced or taken away."""
    return ability in UNTOUCHABLE_ABILITIES
