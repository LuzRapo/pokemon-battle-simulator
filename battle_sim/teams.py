import re
from dataclasses import replace

from battle_sim.database.loader import get_move, get_species, normalize_id
from battle_sim.models.moves import MoveSet
from battle_sim.models.pokemon import Pokemon
from battle_sim.models.spec import ParseResult, ParseWarning, ParseWarningKind, PokemonSpec
from battle_sim.models.stats import EVs, IVs
from battle_sim.utils import Ability, Item, Nature

_ABILITY_SHOWDOWN_NAMES: dict[Ability, str] = {
    Ability.SPEED_BOOST: "Speed Boost",
    Ability.INTIMIDATE: "Intimidate",
    Ability.DROUGHT: "Drought",
    Ability.DRIZZLE: "Drizzle",
    Ability.SAND_STREAM: "Sand Stream",
    Ability.SNOW_WARNING: "Snow Warning",
    Ability.LEVITATE: "Levitate",
    Ability.WONDER_GUARD: "Wonder Guard",
    Ability.NINE_LIVES: "9 Lives",
    Ability.STURDY: "Sturdy",
    Ability.FLASH_FIRE: "Flash Fire",
    Ability.VOLT_ABSORB: "Volt Absorb",
    Ability.WATER_ABSORB: "Water Absorb",
    Ability.MAGIC_GUARD: "Magic Guard",
    Ability.PRANKSTER: "Prankster",
    Ability.QUICK_FEET: "Quick Feet",
    Ability.ADAPTABILITY: "Adaptability",
    Ability.HUGE_POWER: "Huge Power",
    Ability.PURE_POWER: "Pure Power",
    Ability.GUTS: "Guts",
    Ability.ROCK_HEAD: "Rock Head",
    Ability.NO_GUARD: "No Guard",
    Ability.MOTOR_DRIVE: "Motor Drive",
    Ability.PRESSURE: "Pressure",
    Ability.SOLAR_POWER: "Solar Power",
    Ability.MOXIE: "Moxie",
    Ability.CLEAR_BODY: "Clear Body",
    Ability.SAND_VEIL: "Sand Veil",
    Ability.ROUGH_SKIN: "Rough Skin",
    Ability.IRON_BARBS: "Iron Barbs",
    Ability.GALVANIZE: "Galvanize",
    Ability.DEFEATIST: "Defeatist",
    Ability.MARVEL_SCALE: "Marvel Scale",
    Ability.FUR_COAT: "Fur Coat",
    Ability.STEELWORKER: "Steelworker",
    Ability.LONG_REACH: "Long Reach",
    Ability.QUEENLY_MAJESTY: "Queenly Majesty",
    Ability.STAKEOUT: "Stakeout",
    Ability.SURGE_SURFER: "Surge Surfer",
    Ability.MERCILESS: "Merciless",
    Ability.ARENA_TRAP: "Arena Trap",
    Ability.EMERGENCY_EXIT: "Emergency Exit",
    Ability.WIMP_OUT: "Wimp Out",
    Ability.FLUFFY: "Fluffy",
    Ability.STORM_DRAIN: "Storm Drain",
    Ability.COMATOSE: "Comatose",
    Ability.MUMMY: "Mummy",
    Ability.FLOWER_GIFT: "Flower Gift",
    Ability.SHED_SKIN: "Shed Skin",
    Ability.ANALYTIC: "Analytic",
    Ability.SUPER_LUCK: "Super Luck",
    Ability.AFTERMATH: "Aftermath",
    Ability.POISON_POINT: "Poison Point",
    Ability.EFFECT_SPORE: "Effect Spore",
    Ability.SIMPLE: "Simple",
    Ability.STANCE_CHANGE: "Stance Change",
    Ability.DISGUISE: "Disguise",
    Ability.ZEN_MODE: "Zen Mode",
    Ability.SCHOOLING: "Schooling",
    Ability.SHIELDS_DOWN: "Shields Down",
    Ability.PROTOSYNTHESIS: "Protosynthesis",
    Ability.QUARK_DRIVE: "Quark Drive",
    Ability.REGENERATOR: "Regenerator",
    Ability.NATURAL_CURE: "Natural Cure",
    Ability.SUPREME_OVERLORD: "Supreme Overlord",
    Ability.GOOD_AS_GOLD: "Good as Gold",
    Ability.MULTISCALE: "Multiscale",
    Ability.VESSEL_OF_RUIN: "Vessel of Ruin",
    Ability.BEADS_OF_RUIN: "Beads of Ruin",
    Ability.SWORD_OF_RUIN: "Sword of Ruin",
    Ability.TABLETS_OF_RUIN: "Tablets of Ruin",
    Ability.SHARPNESS: "Sharpness",
    Ability.TECHNICIAN: "Technician",
    Ability.TOXIC_DEBRIS: "Toxic Debris",
    Ability.SWIFT_SWIM: "Swift Swim",
    Ability.CHLOROPHYLL: "Chlorophyll",
    Ability.SAND_RUSH: "Sand Rush",
    Ability.SLUSH_RUSH: "Slush Rush",
    Ability.UNBURDEN: "Unburden",
    Ability.FLAME_BODY: "Flame Body",
    Ability.STATIC: "Static",
    Ability.POISON_TOUCH: "Poison Touch",
    Ability.TOXIC_CHAIN: "Toxic Chain",
    Ability.DAUNTLESS_SHIELD: "Dauntless Shield",
    Ability.INTREPID_SWORD: "Intrepid Sword",
    Ability.DOWNLOAD: "Download",
    Ability.MULTITYPE: "Multitype",
    Ability.HADRON_ENGINE: "Hadron Engine",
    Ability.ORICHALCUM_PULSE: "Orichalcum Pulse",
    Ability.GRASSY_SURGE: "Grassy Surge",
    Ability.ELECTRIC_SURGE: "Electric Surge",
    Ability.PSYCHIC_SURGE: "Psychic Surge",
    Ability.MISTY_SURGE: "Misty Surge",
    Ability.BAD_DREAMS: "Bad Dreams",
    Ability.WEAK_ARMOR: "Weak Armor",
    Ability.STAMINA: "Stamina",
    Ability.BERSERK: "Berserk",
    Ability.JUSTIFIED: "Justified",
    Ability.THERMAL_EXCHANGE: "Thermal Exchange",
    Ability.CURSED_BODY: "Cursed Body",
    Ability.PURIFYING_SALT: "Purifying Salt",
    Ability.WATER_BUBBLE: "Water Bubble",
    Ability.CONTRARY: "Contrary",
    Ability.DEFIANT: "Defiant",
    Ability.COMPETITIVE: "Competitive",
    Ability.POISON_HEAL: "Poison Heal",
    Ability.SHIELD_DUST: "Shield Dust",
    Ability.SERENE_GRACE: "Serene Grace",
    Ability.BLAZE: "Blaze",
    Ability.TORRENT: "Torrent",
    Ability.OVERGROW: "Overgrow",
    Ability.SWARM: "Swarm",
    Ability.PRISM_ARMOR: "Prism Armor",
    Ability.FILTER: "Filter",
    Ability.TINTED_LENS: "Tinted Lens",
    Ability.HEATPROOF: "Heatproof",
    Ability.THICK_FAT: "Thick Fat",
    Ability.ICE_SCALES: "Ice Scales",
    Ability.DRAGONS_MAW: "Dragon's Maw",
    Ability.HYDRATION: "Hydration",
    Ability.ICE_BODY: "Ice Body",
    Ability.SAP_SIPPER: "Sap Sipper",
    Ability.EARTH_EATER: "Earth Eater",
    Ability.WELL_BAKED_BODY: "Well-Baked Body",
    Ability.DRY_SKIN: "Dry Skin",
    Ability.SOUL_HEART: "Soul-Heart",
    Ability.CHILLING_NEIGH: "Chilling Neigh",
    Ability.AS_ONE_GLASTRIER: "As One (Glastrier)",
    Ability.OVERCOAT: "Overcoat",
    Ability.SKILL_LINK: "Skill Link",
    Ability.INNER_FOCUS: "Inner Focus",
    Ability.UNAWARE: "Unaware",
    Ability.INFILTRATOR: "Infiltrator",
    Ability.LIBERO: "Libero",
    Ability.PROTEAN: "Protean",
    Ability.POISON_PUPPETEER: "Poison Puppeteer",
    Ability.MAGIC_BOUNCE: "Magic Bounce",
    Ability.POWER_CONSTRUCT: "Power Construct",
    Ability.DELTA_STREAM: "Delta Stream",
    Ability.LIQUID_VOICE: "Liquid Voice",
    Ability.NEUTRALIZING_GAS: "Neutralizing Gas",
    Ability.MIRROR_ARMOR: "Mirror Armor",
    Ability.PICKPOCKET: "Pickpocket",
    Ability.MAGICIAN: "Magician",
    Ability.MOLD_BREAKER: "Mold Breaker",
    Ability.TERAVOLT: "Teravolt",
    Ability.SHEER_FORCE: "Sheer Force",
    Ability.ZERO_TO_HERO: "Zero to Hero",
    Ability.IMPOSTER: "Imposter",
    Ability.AIR_LOCK: "Air Lock",
    Ability.ELECTROMORPHOSIS: "Electromorphosis",
    Ability.IRON_FIST: "Iron Fist",
    Ability.SCRAPPY: "Scrappy",
    Ability.MINDS_EYE: "Mind's Eye",
    Ability.TRIAGE: "Triage",
    Ability.MAGNET_PULL: "Magnet Pull",
    Ability.SHADOW_TAG: "Shadow Tag",
    Ability.HARVEST: "Harvest",
    Ability.DAZZLING: "Dazzling",
    Ability.GALE_WINGS: "Gale Wings",
    Ability.LIQUID_OOZE: "Liquid Ooze",
    Ability.MYCELIUM_MIGHT: "Mycelium Might",
    Ability.FULL_METAL_BODY: "Full Metal Body",
    Ability.QUICK_DRAW: "Quick Draw",
    Ability.CORROSION: "Corrosion",
    Ability.LEAF_GUARD: "Leaf Guard",
    Ability.PUNK_ROCK: "Punk Rock",
    Ability.WIND_RIDER: "Wind Rider",
    Ability.ILLUSION: "Illusion",
    Ability.TRACE: "Trace",
    Ability.TERA_SHIFT: "Tera Shift",
    Ability.GUARD_DOG: "Guard Dog",
    Ability.BULLETPROOF: "Bulletproof",
    Ability.ROCKY_PAYLOAD: "Rocky Payload",
    Ability.TRANSISTOR: "Transistor",
    Ability.BATTLE_BOND: "Battle Bond",
    Ability.LIMBER: "Limber",
    Ability.INSOMNIA: "Insomnia",
    Ability.VITAL_SPIRIT: "Vital Spirit",
    Ability.WATER_VEIL: "Water Veil",
    Ability.MAGMA_ARMOR: "Magma Armor",
    Ability.IMMUNITY: "Immunity",
    Ability.OWN_TEMPO: "Own Tempo",
    Ability.WHITE_SMOKE: "White Smoke",
    Ability.SNOW_CLOAK: "Snow Cloak",
    Ability.COMPOUND_EYES: "Compound Eyes",
    Ability.TANGLED_FEET: "Tangled Feet",
    Ability.BATTLE_ARMOR: "Battle Armor",
    Ability.SHELL_ARMOR: "Shell Armor",
    Ability.SNIPER: "Sniper",
    Ability.SOUNDPROOF: "Soundproof",
    Ability.RECKLESS: "Reckless",
    Ability.RAIN_DISH: "Rain Dish",
    Ability.STEADFAST: "Steadfast",
    Ability.HUSTLE: "Hustle",
    Ability.KEEN_EYE: "Keen Eye",
    Ability.HYPER_CUTTER: "Hyper Cutter",
    Ability.BIG_PECKS: "Big Pecks",
    Ability.RKS_SYSTEM: "RKS System",
    Ability.BEAST_BOOST: "Beast Boost",
    Ability.TURBOBLAZE: "Turboblaze",
    Ability.SHADOW_SHIELD: "Shadow Shield",
    Ability.NEUROFORCE: "Neuroforce",
    Ability.VICTORY_STAR: "Victory Star",
    Ability.FAIRY_AURA: "Fairy Aura",
    Ability.DARK_AURA: "Dark Aura",
    Ability.AURA_BREAK: "Aura Break",
    Ability.SYNCHRONIZE: "Synchronize",
    Ability.SLOW_START: "Slow Start",
    # Abilities introduced by Mega Evolution / Primal Reversion formes
    Ability.TOUGH_CLAWS: "Tough Claws",
    Ability.STRONG_JAW: "Strong Jaw",
    Ability.MEGA_LAUNCHER: "Mega Launcher",
    Ability.PARENTAL_BOND: "Parental Bond",
    Ability.SAND_FORCE: "Sand Force",
    Ability.AERILATE: "Aerilate",
    Ability.PIXILATE: "Pixilate",
    Ability.REFRIGERATE: "Refrigerate",
    Ability.PRIMORDIAL_SEA: "Primordial Sea",
    Ability.DESOLATE_LAND: "Desolate Land",
    Ability.LIGHTNING_ROD: "Lightning Rod",
    Ability.HEALER: "Healer",
    Ability.TRUANT: "Truant",
}
_ABILITY_BY_SHOWDOWN_NAME: dict[str, Ability] = {
    normalize_id(name): ability for ability, name in _ABILITY_SHOWDOWN_NAMES.items()
}

_ITEM_SHOWDOWN_NAMES: dict[Item, str] = {
    Item.LEFTOVERS: "Leftovers",
    Item.LIFE_ORB: "Life Orb",
    Item.MEOWFREDS_MONOCLE: "Meowfred's Monocle",
    Item.CHOICE_BAND: "Choice Band",
    Item.CHOICE_SPECS: "Choice Specs",
    Item.CHOICE_SCARF: "Choice Scarf",
    Item.FOCUS_SASH: "Focus Sash",
    Item.BLACK_SLUDGE: "Black Sludge",
    Item.HEAVY_DUTY_BOOTS: "Heavy-Duty Boots",
    Item.AIR_BALLOON: "Air Balloon",
    Item.ROCKY_HELMET: "Rocky Helmet",
    Item.EXPERT_BELT: "Expert Belt",
    Item.ASSAULT_VEST: "Assault Vest",
    Item.IRON_PLATE: "Iron Plate",
    Item.CHOPLE_BERRY: "Chople Berry",
    Item.SHUCA_BERRY: "Shuca Berry",
    Item.COLBUR_BERRY: "Colbur Berry",
    Item.SITRUS_BERRY: "Sitrus Berry",
    Item.LUM_BERRY: "Lum Berry",
    Item.CHESTO_BERRY: "Chesto Berry",
    Item.BOOSTER_ENERGY: "Booster Energy",
    Item.LOADED_DICE: "Loaded Dice",
    Item.LIGHT_CLAY: "Light Clay",
    Item.COVERT_CLOAK: "Covert Cloak",
    Item.DAMP_ROCK: "Damp Rock",
    Item.HEAT_ROCK: "Heat Rock",
    Item.ICY_ROCK: "Icy Rock",
    Item.SMOOTH_ROCK: "Smooth Rock",
    Item.TERRAIN_EXTENDER: "Terrain Extender",
    Item.TOXIC_ORB: "Toxic Orb",
    Item.FLAME_ORB: "Flame Orb",
    Item.BLACK_GLASSES: "Black Glasses",
    Item.MYSTIC_WATER: "Mystic Water",
    Item.METAL_COAT: "Metal Coat",
    Item.NEVER_MELT_ICE: "Never-Melt Ice",
    Item.SILK_SCARF: "Silk Scarf",
    Item.EARTH_PLATE: "Earth Plate",
    Item.SPOOKY_PLATE: "Spooky Plate",
    Item.PIXIE_PLATE: "Pixie Plate",
    Item.SOUL_DEW: "Soul Dew",
    Item.GRISEOUS_CORE: "Griseous Core",
    Item.GRISEOUS_ORB: "Griseous Orb",
    Item.SHED_SHELL: "Shed Shell",
    Item.WIKI_BERRY: "Wiki Berry",
    Item.WELLSPRING_MASK: "Wellspring Mask",
    Item.CORNERSTONE_MASK: "Cornerstone Mask",
    Item.WEAKNESS_POLICY: "Weakness Policy",
    Item.EVIOLITE: "Eviolite",
    Item.WIDE_LENS: "Wide Lens",
    Item.SCOPE_LENS: "Scope Lens",
    Item.CLEAR_AMULET: "Clear Amulet",
    Item.WHITE_HERB: "White Herb",
    Item.GRASSY_SEED: "Grassy Seed",
    Item.ELECTRIC_SEED: "Electric Seed",
    Item.PSYCHIC_SEED: "Psychic Seed",
    Item.MISTY_SEED: "Misty Seed",
    Item.RUSTED_SWORD: "Rusted Sword",
    Item.RUSTED_SHIELD: "Rusted Shield",
    Item.POWER_HERB: "Power Herb",
    Item.EJECT_BUTTON: "Eject Button",
    Item.EJECT_PACK: "Eject Pack",
    Item.RED_CARD: "Red Card",
    Item.MENTAL_HERB: "Mental Herb",
    Item.CUSTAP_BERRY: "Custap Berry",
    Item.QUICK_CLAW: "Quick Claw",
    Item.ADRENALINE_ORB: "Adrenaline Orb",
    Item.MIRROR_HERB: "Mirror Herb",
    Item.STARF_BERRY: "Starf Berry",
    Item.LEPPA_BERRY: "Leppa Berry",
    Item.WISE_GLASSES: "Wise Glasses",
    Item.MUSCLE_BAND: "Muscle Band",
    Item.MIRACLE_SEED: "Miracle Seed",
    Item.SILVER_POWDER: "Silver Powder",
    Item.SPLASH_PLATE: "Splash Plate",
    Item.STONE_PLATE: "Stone Plate",
    Item.PROTECTIVE_PADS: "Protective Pads",
    Item.ADAMANT_CRYSTAL: "Adamant Crystal",
    Item.LUSTROUS_GLOBE: "Lustrous Globe",
    Item.PUNCHING_GLOVE: "Punching Glove",
    Item.FIST_PLATE: "Fist Plate",
    Item.SKY_PLATE: "Sky Plate",
    Item.TOXIC_PLATE: "Toxic Plate",
    Item.INSECT_PLATE: "Insect Plate",
    Item.FLAME_PLATE: "Flame Plate",
    Item.MEADOW_PLATE: "Meadow Plate",
    Item.ZAP_PLATE: "Zap Plate",
    Item.MIND_PLATE: "Mind Plate",
    Item.ICICLE_PLATE: "Icicle Plate",
    Item.DRACO_PLATE: "Draco Plate",
    Item.DREAD_PLATE: "Dread Plate",
    Item.BUG_MEMORY: "Bug Memory",
    Item.DARK_MEMORY: "Dark Memory",
    Item.DRAGON_MEMORY: "Dragon Memory",
    Item.ELECTRIC_MEMORY: "Electric Memory",
    Item.FAIRY_MEMORY: "Fairy Memory",
    Item.FIGHTING_MEMORY: "Fighting Memory",
    Item.FIRE_MEMORY: "Fire Memory",
    Item.FLYING_MEMORY: "Flying Memory",
    Item.GHOST_MEMORY: "Ghost Memory",
    Item.GRASS_MEMORY: "Grass Memory",
    Item.GROUND_MEMORY: "Ground Memory",
    Item.ICE_MEMORY: "Ice Memory",
    Item.POISON_MEMORY: "Poison Memory",
    Item.PSYCHIC_MEMORY: "Psychic Memory",
    Item.ROCK_MEMORY: "Rock Memory",
    Item.STEEL_MEMORY: "Steel Memory",
    Item.WATER_MEMORY: "Water Memory",
    Item.DOUSE_DRIVE: "Douse Drive",
    Item.SHOCK_DRIVE: "Shock Drive",
    Item.BURN_DRIVE: "Burn Drive",
    Item.CHILL_DRIVE: "Chill Drive",
    # Mega Stones: held by the matching species to Mega Evolve; inert on anything else
    Item.ABOMASITE: "Abomasite",
    Item.ABSOLITE: "Absolite",
    Item.AERODACTYLITE: "Aerodactylite",
    Item.AGGRONITE: "Aggronite",
    Item.ALAKAZITE: "Alakazite",
    Item.ALTARIANITE: "Altarianite",
    Item.AMPHAROSITE: "Ampharosite",
    Item.AUDINITE: "Audinite",
    Item.BANETTITE: "Banettite",
    Item.BEEDRILLITE: "Beedrillite",
    Item.BLASTOISINITE: "Blastoisinite",
    Item.BLAZIKENITE: "Blazikenite",
    Item.CAMERUPTITE: "Cameruptite",
    Item.CHARIZARDITE_X: "Charizardite X",
    Item.CHARIZARDITE_Y: "Charizardite Y",
    Item.DIANCITE: "Diancite",
    Item.GALLADITE: "Galladite",
    Item.GARCHOMPITE: "Garchompite",
    Item.GARDEVOIRITE: "Gardevoirite",
    Item.GENGARITE: "Gengarite",
    Item.GLALITITE: "Glalitite",
    Item.GYARADOSITE: "Gyaradosite",
    Item.HERACRONITE: "Heracronite",
    Item.HOUNDOOMINITE: "Houndoominite",
    Item.KANGASKHANITE: "Kangaskhanite",
    Item.LATIASITE: "Latiasite",
    Item.LATIOSITE: "Latiosite",
    Item.LOPUNNITE: "Lopunnite",
    Item.LUCARIONITE: "Lucarionite",
    Item.MANECTITE: "Manectite",
    Item.MAWILITE: "Mawilite",
    Item.MEDICHAMITE: "Medichamite",
    Item.METAGROSSITE: "Metagrossite",
    Item.MEWTWONITE_X: "Mewtwonite X",
    Item.MEWTWONITE_Y: "Mewtwonite Y",
    Item.PIDGEOTITE: "Pidgeotite",
    Item.PINSIRITE: "Pinsirite",
    Item.SABLENITE: "Sablenite",
    Item.SALAMENCITE: "Salamencite",
    Item.SCEPTILITE: "Sceptilite",
    Item.SCIZORITE: "Scizorite",
    Item.SHARPEDONITE: "Sharpedonite",
    Item.SLOWBRONITE: "Slowbronite",
    Item.STEELIXITE: "Steelixite",
    Item.SWAMPERTITE: "Swampertite",
    Item.TYRANITARITE: "Tyranitarite",
    Item.VENUSAURITE: "Venusaurite",
    Item.ABSOLITE_Z: "Absolite Z",
    Item.BARBARACITE: "Barbaracite",
    Item.BAXCALIBRITE: "Baxcalibrite",
    Item.CHANDELURITE: "Chandelurite",
    Item.CHESNAUGHTITE: "Chesnaughtite",
    Item.CHIMECHITE: "Chimechite",
    Item.CLEFABLITE: "Clefablite",
    Item.CRABOMINITE: "Crabominite",
    Item.DARKRANITE: "Darkranite",
    Item.DELPHOXITE: "Delphoxite",
    Item.DRAGALGITE: "Dragalgite",
    Item.DRAGONINITE: "Dragoninite",
    Item.DRAMPANITE: "Drampanite",
    Item.EELEKTROSSITE: "Eelektrossite",
    Item.EMBOARITE: "Emboarite",
    Item.EXCADRITE: "Excadrite",
    Item.FALINKSITE: "Falinksite",
    Item.FERALIGITE: "Feraligite",
    Item.FLOETTITE: "Floettite",
    Item.FROSLASSITE: "Froslassite",
    Item.GARCHOMPITE_Z: "Garchompite Z",
    Item.GLIMMORANITE: "Glimmoranite",
    Item.GOLISOPITE: "Golisopite",
    Item.GOLURKITE: "Golurkite",
    Item.GRENINJITE: "Greninjite",
    Item.HAWLUCHANITE: "Hawluchanite",
    Item.HEATRANITE: "Heatranite",
    Item.LUCARIONITE_Z: "Lucarionite Z",
    Item.MAGEARNITE: "Magearnite",
    Item.MALAMARITE: "Malamarite",
    Item.MEGANIUMITE: "Meganiumite",
    Item.MEOWSTICITE: "Meowsticite",
    Item.PYROARITE: "Pyroarite",
    Item.RAICHUNITE_X: "Raichunite X",
    Item.RAICHUNITE_Y: "Raichunite Y",
    Item.SCOLIPITE: "Scolipite",
    Item.SCOVILLAINITE: "Scovillainite",
    Item.SCRAFTINITE: "Scraftinite",
    Item.SKARMORITE: "Skarmorite",
    Item.STARAPTITE: "Staraptite",
    Item.STARMINITE: "Starminite",
    Item.TATSUGIRINITE: "Tatsugirinite",
    Item.VICTREEBELITE: "Victreebelite",
    Item.ZERAORITE: "Zeraorite",
    Item.ZYGARDITE: "Zygardite",
    # Primal Reversion orbs
    Item.BLUE_ORB: "Blue Orb",
    Item.RED_ORB: "Red Orb",
    # Z-Crystals: one Z-move per battle
    Item.ALORAICHIUM_Z: "Aloraichium Z",
    Item.BUGINIUM_Z: "Buginium Z",
    Item.DARKINIUM_Z: "Darkinium Z",
    Item.DECIDIUM_Z: "Decidium Z",
    Item.DRAGONIUM_Z: "Dragonium Z",
    Item.ELECTRIUM_Z: "Electrium Z",
    Item.FAIRIUM_Z: "Fairium Z",
    Item.FIGHTINIUM_Z: "Fightinium Z",
    Item.FIRIUM_Z: "Firium Z",
    Item.FLYINIUM_Z: "Flyinium Z",
    Item.GHOSTIUM_Z: "Ghostium Z",
    Item.GRASSIUM_Z: "Grassium Z",
    Item.GROUNDIUM_Z: "Groundium Z",
    Item.ICIUM_Z: "Icium Z",
    Item.INCINIUM_Z: "Incinium Z",
    Item.KOMMONIUM_Z: "Kommonium Z",
    Item.LUNALIUM_Z: "Lunalium Z",
    Item.LYCANIUM_Z: "Lycanium Z",
    Item.MARSHADIUM_Z: "Marshadium Z",
    Item.MEWNIUM_Z: "Mewnium Z",
    Item.MIMIKIUM_Z: "Mimikium Z",
    Item.NORMALIUM_Z: "Normalium Z",
    Item.POISONIUM_Z: "Poisonium Z",
    Item.PSYCHIUM_Z: "Psychium Z",
    Item.ROCKIUM_Z: "Rockium Z",
    Item.SOLGANIUM_Z: "Solganium Z",
    Item.STEELIUM_Z: "Steelium Z",
    Item.ULTRANECROZIUM_Z: "Ultranecrozium Z",
    Item.EEVIUM_Z: "Eevium Z",
    Item.PIKANIUM_Z: "Pikanium Z",
    Item.PIKASHUNIUM_Z: "Pikashunium Z",
    Item.PRIMARIUM_Z: "Primarium Z",
    Item.SNORLIUM_Z: "Snorlium Z",
    Item.TAPUNIUM_Z: "Tapunium Z",
    Item.WATERIUM_Z: "Waterium Z",
}
_ITEM_BY_SHOWDOWN_NAME: dict[str, Item] = {normalize_id(name): item for item, name in _ITEM_SHOWDOWN_NAMES.items()}


def ability_from_showdown(name: str) -> Ability:
    """The `Ability` a Showdown ability name maps to; unmapped names are ones we don't model yet."""
    return _ABILITY_BY_SHOWDOWN_NAME.get(normalize_id(name), Ability.NONE)


def item_from_showdown(name: str) -> Item | None:
    return _ITEM_BY_SHOWDOWN_NAME.get(normalize_id(name))


def item_showdown_name(item: Item) -> str:
    """Item.NONE and anything we hold but never name maps to the empty string."""
    return _ITEM_SHOWDOWN_NAMES.get(item, "")


_STAT_KEYS: dict[str, str] = {
    "HP": "HP",
    "Atk": "ATTACK",
    "Def": "DEFENCE",
    "SpA": "SP_ATTACK",
    "SpD": "SP_DEFENCE",
    "Spe": "SPEED",
}
_STAT_LABELS: list[tuple[str, str]] = [
    ("HP", "HP"),
    ("ATTACK", "Atk"),
    ("DEFENCE", "Def"),
    ("SP_ATTACK", "SpA"),
    ("SP_DEFENCE", "SpD"),
    ("SPEED", "Spe"),
]
_FIRST_LINE_RE = re.compile(r"^(?P<head>.+?)(?:\s*\((?P<paren>[^)]+)\))?(?:\s*\([MFN]\))?(?:\s*@\s*(?P<item>.+))?$")
_HIDDEN_POWER_TYPE_RE = re.compile(r"\s*\[[^\]]+\]\s*$")


def parse_showdown_team(text: str) -> ParseResult:
    """Lenient: unknown values are dropped, but every drop surfaces as a ParseWarning."""
    blocks = [b for b in re.split(r"\n\s*\n", text.strip()) if b.strip()]
    specs: list[PokemonSpec] = []
    warnings: list[ParseWarning] = []
    for index, block in enumerate(blocks):
        spec, block_warnings = _parse_pokemon_block(block)
        specs.append(spec)
        warnings.extend(replace(w, block_index=index) for w in block_warnings)
    return ParseResult(specs=tuple(specs), warnings=tuple(warnings))


def _warn(warnings: list[ParseWarning], kind: ParseWarningKind, detail: str) -> None:
    warnings.append(ParseWarning(kind=kind, block_index=-1, detail=detail))  # caller stamps block_index


def _parse_pokemon_block(block: str) -> tuple[PokemonSpec, list[ParseWarning]]:
    warnings: list[ParseWarning] = []
    lines = [line.rstrip() for line in block.strip().splitlines() if line.strip()]
    species, nickname, item = _parse_first_line(lines[0], warnings)
    moves: list[str] = []
    level = 100
    ability = Ability.NONE
    nature = Nature.HARDY
    evs = EVs()
    ivs = IVs()
    pp_ups = 0
    for line in lines[1:]:
        stripped = line.strip()
        if stripped.startswith("- "):
            moves.append(_HIDDEN_POWER_TYPE_RE.sub("", stripped[2:].strip()))
        elif stripped.startswith("Ability:"):
            ability = _parse_ability(stripped.split(":", 1)[1].strip(), warnings)
        elif stripped.startswith("PP Ups:"):
            pp_ups = _parse_pp_ups(stripped.split(":", 1)[1].strip())
        elif stripped.startswith("Level:"):
            level = _parse_level(stripped.split(":", 1)[1].strip(), warnings)
        elif stripped.startswith("EVs:"):
            evs = EVs(**_parse_stats_line(stripped.split(":", 1)[1], warnings))
        elif stripped.startswith("IVs:"):
            ivs = IVs(**_parse_stats_line(stripped.split(":", 1)[1], warnings))
        elif stripped.endswith(" Nature"):
            nature = _parse_nature(stripped[: -len(" Nature")].strip(), warnings)
    if len(moves) > 4:
        _warn(warnings, ParseWarningKind.EXTRA_MOVES_TRUNCATED, f"{len(moves)} moves listed; keeping the first 4")
        moves = moves[:4]
    spec = PokemonSpec(
        species=species,
        nickname=nickname,
        level=level,
        ability=ability,
        item=item,
        nature=nature,
        effort_values=evs,
        individual_values=ivs,
        moves=moves,
        pp_ups=pp_ups,
    )
    return spec, warnings


def _parse_pp_ups(value: str) -> int:
    """`PP Ups: 3`, or `max` for the same thing. Out-of-range or unreadable means none."""
    text = value.strip().casefold()
    if text in {"max", "maxed", "all"}:
        return 3
    try:
        return max(0, min(3, int(text)))
    except ValueError:
        return 0


def _parse_first_line(line: str, warnings: list[ParseWarning]) -> tuple[str, str | None, Item]:
    """`Nickname (Species) @ Item` and simpler variants -> (species, nickname, item)."""
    match = _FIRST_LINE_RE.match(line.strip())
    if not match:
        return line.strip(), None, Item.NONE
    head = match.group("head").strip()
    paren = (match.group("paren") or "").strip()
    item_raw = (match.group("item") or "").strip()
    if paren and paren not in {"M", "F", "N"}:
        species, nickname = paren, head
    else:
        species, nickname = head, None
    item = Item.NONE
    if item_raw:
        mapped = _ITEM_BY_SHOWDOWN_NAME.get(normalize_id(item_raw))
        if mapped is None:
            _warn(warnings, ParseWarningKind.UNKNOWN_ITEM, item_raw)
        else:
            item = mapped
    return species, nickname, item


def _parse_ability(name: str, warnings: list[ParseWarning]) -> Ability:
    mapped = _ABILITY_BY_SHOWDOWN_NAME.get(normalize_id(name))
    if mapped is None:
        _warn(warnings, ParseWarningKind.UNKNOWN_ABILITY, name)
        return Ability.NONE
    return mapped


def _parse_level(text: str, warnings: list[ParseWarning]) -> int:
    try:
        level = int(text)
    except ValueError:
        _warn(warnings, ParseWarningKind.MALFORMED_LEVEL, text)
        return 100
    if not 1 <= level <= 100:
        _warn(warnings, ParseWarningKind.MALFORMED_LEVEL, f"{level} out of range 1-100")
        return 100
    return level


def _parse_nature(name: str, warnings: list[ParseWarning]) -> Nature:
    try:
        return Nature[name.strip().upper()]
    except KeyError:
        _warn(warnings, ParseWarningKind.UNKNOWN_NATURE, name)
        return Nature.HARDY


def _parse_stats_line(text: str, warnings: list[ParseWarning]) -> dict[str, int]:
    result: dict[str, int] = {}
    for part in text.split("/"):
        words = part.strip().split()
        if len(words) < 2:
            continue
        try:
            value = int(words[0])
        except ValueError:
            _warn(warnings, ParseWarningKind.MALFORMED_STAT_VALUE, part.strip())
            continue
        label = " ".join(words[1:])
        if label in _STAT_KEYS:
            result[_STAT_KEYS[label]] = value
        else:
            _warn(warnings, ParseWarningKind.UNKNOWN_STAT_KEY, label)
    return result


def build_pokemon(spec: PokemonSpec) -> Pokemon:
    species = get_species(spec.species)
    moves = [get_move(name) for name in spec.moves]
    while len(moves) < 4:
        moves.append(moves[0] if moves else get_move("Tackle"))
    return Pokemon(
        name=species.name,
        nickname=spec.nickname or species.name,
        level=spec.level,
        base_stats=species.base_stats,
        effort_values=spec.effort_values,
        individual_values=spec.individual_values,
        types=species.types,
        moves=MoveSet(moves[0], moves[1], moves[2], moves[3]),
        nature=spec.nature,
        item=spec.item,
        ability=spec.ability,
        pp_ups=spec.pp_ups,
        fully_evolved=species.fully_evolved,
        weight_kg=species.weight_kg,
    )


def export_to_showdown(team: list[Pokemon]) -> str:
    return "\n\n".join(_export_pokemon(p) for p in team)


def _export_pokemon(pokemon: Pokemon) -> str:
    lines = []
    header = (
        pokemon.name
        if not pokemon.nickname or pokemon.nickname == pokemon.name
        else f"{pokemon.nickname} ({pokemon.name})"
    )
    if pokemon.item is not Item.NONE and pokemon.item in _ITEM_SHOWDOWN_NAMES:
        header = f"{header} @ {_ITEM_SHOWDOWN_NAMES[pokemon.item]}"
    lines.append(header)
    if pokemon.ability is not Ability.NONE and pokemon.ability in _ABILITY_SHOWDOWN_NAMES:
        lines.append(f"Ability: {_ABILITY_SHOWDOWN_NAMES[pokemon.ability]}")
    if pokemon.level != 100:
        lines.append(f"Level: {pokemon.level}")
    evs = pokemon.effort_values.model_dump()
    ev_parts = [f"{evs[stat]} {label}" for stat, label in _STAT_LABELS if evs[stat] > 0]
    if ev_parts:
        lines.append("EVs: " + " / ".join(ev_parts))
    ivs = pokemon.individual_values.model_dump()
    iv_parts = [f"{ivs[stat]} {label}" for stat, label in _STAT_LABELS if ivs[stat] < 31]
    if iv_parts:
        lines.append("IVs: " + " / ".join(iv_parts))
    if pokemon.nature is not Nature.HARDY:
        lines.append(f"{pokemon.nature.name.title()} Nature")
    for move in pokemon.known_moves():
        lines.append(f"- {move.name}")
    return "\n".join(lines)
