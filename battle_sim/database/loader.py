import json
from collections import Counter
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Literal

from loguru import logger

from battle_sim.database.raw import RawMoveData, RawSecondary, RawSpeciesData
from battle_sim.models.moves import (
    CodedEffect,
    CodedMoveKind,
    DamageEffect,
    FixedDamageEffect,
    HealEffect,
    InflictStatusEffect,
    Move,
    MoveEffect,
    PseudoWeatherEffect,
    RemoveHazardsEffect,
    SideConditionEffect,
    StatStageChangeEffect,
    TerrainEffect,
    WeatherEffect,
)
from battle_sim.models.species import BaseSpecies
from battle_sim.models.stats import BaseStats
from battle_sim.models.type_matchups import TypePair
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

_VENDOR_DIR = Path(__file__).parent / "_vendor"

_TYPE_MAP: dict[str, Type] = {t.name.capitalize(): t for t in Type}
_CATEGORY_MAP: dict[str, Category] = {c.name.capitalize(): c for c in Category}
_STAT_MAP: dict[str, Stats] = {
    "hp": Stats.HP,
    "atk": Stats.ATTACK,
    "def": Stats.DEFENCE,
    "spa": Stats.SP_ATTACK,
    "spd": Stats.SP_DEFENCE,
    "spe": Stats.SPEED,
}
_BOOST_MAP: dict[str, Stats] = {**_STAT_MAP, "accuracy": Stats.ACCURACY, "evasion": Stats.EVASION}
_TARGET_MAP: dict[str, Target] = {
    "normal": Target.SINGLE_OPPONENT,
    "self": Target.SELF,
    "allySide": Target.USER_SIDE,
    "foeSide": Target.OPPONENT_SIDE,
    "all": Target.FIELD,
    "allAdjacent": Target.ALL_ADJACENT,
    "allAdjacentFoes": Target.ALL_ADJACENT_ENEMIES,
    "adjacentFoe": Target.SINGLE_OPPONENT,
    "randomNormal": Target.SINGLE_OPPONENT,
    "any": Target.SINGLE_OPPONENT,
    "scripted": Target.SINGLE_OPPONENT,
    "allyTeam": Target.USER_SIDE,
}
_STATUS_MAP: dict[str, Status] = {
    "brn": Status.BURN,
    "psn": Status.POISON,
    "tox": Status.TOXIC,
    "par": Status.PARALYSIS,
    "slp": Status.SLEEP,
    "frz": Status.FREEZE,
}
_VOLATILE_MAP: dict[str, ExtraStatus] = {
    "confusion": ExtraStatus.CONFUSION,
    "disable": ExtraStatus.DISABLE,
    "encore": ExtraStatus.ENCORE,
    "flinch": ExtraStatus.FLINCH,
    "focusenergy": ExtraStatus.FOCUS_ENERGY,
    "yawn": ExtraStatus.YAWN,
    "foresight": ExtraStatus.IDENTIFIED,
    "miracleeye": ExtraStatus.MIRACLE_EYE,
    "leechseed": ExtraStatus.LEECH_SEED,
    "lockedmove": ExtraStatus.LOCKED_MOVE,
    "nightmare": ExtraStatus.NIGHTMARE,
    "protect": ExtraStatus.PROTECT,
    "substitute": ExtraStatus.SUBSTITUTE,
    "taunt": ExtraStatus.TAUNT,
    "burningbulwark": ExtraStatus.PROTECT,  # the contact-burn rider is not modelled
    "saltcure": ExtraStatus.SALT_CURE,
    "endure": ExtraStatus.ENDURE,
    "destinybond": ExtraStatus.DESTINY_BOND,
}
_WEATHER_MAP: dict[str, Weather] = {
    "raindance": Weather.RAIN,
    "sunnyday": Weather.SUN,
    "sandstorm": Weather.SANDSTORM,
    "snowscape": Weather.SNOW,
    "hail": Weather.SNOW,
    "primordialsea": Weather.HEAVY_RAIN,
    "desolateland": Weather.HARSH_SUN,
    "deltastream": Weather.STRONG_WINDS,
}
_TERRAIN_MAP: dict[str, Terrain] = {
    "electricterrain": Terrain.ELECTRIC,
    "grassyterrain": Terrain.GRASSY,
    "mistyterrain": Terrain.MISTY,
    "psychicterrain": Terrain.PSYCHIC,
}
_PSEUDO_WEATHER_MAP: dict[str, PseudoWeather] = {
    "trickroom": PseudoWeather.TRICK_ROOM,
    "gravity": PseudoWeather.GRAVITY,
    "magicroom": PseudoWeather.MAGIC_ROOM,
    "wonderroom": PseudoWeather.WONDER_ROOM,
}
_SIDE_CONDITION_MAP: dict[str, Hazards] = {
    "stealthrock": Hazards.STEALTH_ROCK,
    "spikes": Hazards.SPIKES,
    "toxicspikes": Hazards.TOXIC_SPIKES,
    "stickyweb": Hazards.STICKY_WEB,
    "reflect": Hazards.REFLECT,
    "lightscreen": Hazards.LIGHT_SCREEN,
    "auroraveil": Hazards.AURORA_VEIL,
    "tailwind": Hazards.TAILWIND,
}
_SCREEN_HAZARDS = frozenset({Hazards.REFLECT, Hazards.LIGHT_SCREEN, Hazards.AURORA_VEIL})
_SIDE_CONDITION_DURATIONS: dict[Hazards, int] = dict.fromkeys(_SCREEN_HAZARDS, 5) | {Hazards.TAILWIND: 4}

# Effects whose behaviour lives in PS *code* (onHit callbacks) rather than move data.
_CODED_EFFECTS: dict[str, tuple[MoveEffect, ...]] = {
    "rapidspin": (RemoveHazardsEffect(style="RAPID_SPIN"),),
    "mortalspin": (RemoveHazardsEffect(style="RAPID_SPIN"),),
    "defog": (
        StatStageChangeEffect(target="TARGET", stages={Stats.EVASION: -1}, probability=1.0),
        RemoveHazardsEffect(style="DEFOG"),
    ),
    "rest": (CodedEffect(kind=CodedMoveKind.REST),),
    "morningsun": (CodedEffect(kind=CodedMoveKind.WEATHER_HEAL),),
    "moonlight": (CodedEffect(kind=CodedMoveKind.WEATHER_HEAL),),
    "synthesis": (CodedEffect(kind=CodedMoveKind.WEATHER_HEAL),),
    "painsplit": (CodedEffect(kind=CodedMoveKind.PAIN_SPLIT),),
    "strengthsap": (CodedEffect(kind=CodedMoveKind.STRENGTH_SAP),),
    "bellydrum": (CodedEffect(kind=CodedMoveKind.BELLY_DRUM),),
    "haze": (CodedEffect(kind=CodedMoveKind.HAZE),),
    "courtchange": (CodedEffect(kind=CodedMoveKind.COURT_CHANGE),),
    "wish": (CodedEffect(kind=CodedMoveKind.WISH),),
    "healingwish": (CodedEffect(kind=CodedMoveKind.HEALING_WISH),),
    "lunardance": (CodedEffect(kind=CodedMoveKind.HEALING_WISH),),
    "takeheart": (
        StatStageChangeEffect(target="SELF", stages={Stats.SP_ATTACK: 1, Stats.SP_DEFENCE: 1}, probability=1.0),
        CodedEffect(kind=CodedMoveKind.CURE_SELF),
    ),
    "tidyup": (
        StatStageChangeEffect(target="SELF", stages={Stats.ATTACK: 1, Stats.SPEED: 1}, probability=1.0),
        CodedEffect(kind=CodedMoveKind.TIDY_UP),
    ),
    "partingshot": (
        StatStageChangeEffect(target="TARGET", stages={Stats.ATTACK: -1, Stats.SP_ATTACK: -1}, probability=1.0),
    ),
    "curse": (CodedEffect(kind=CodedMoveKind.CURSE),),
    "perishsong": (CodedEffect(kind=CodedMoveKind.PERISH_SONG),),
    "revivalblessing": (CodedEffect(kind=CodedMoveKind.REVIVAL_BLESSING),),
    "transform": (CodedEffect(kind=CodedMoveKind.TRANSFORM),),
    "shedtail": (CodedEffect(kind=CodedMoveKind.SHED_TAIL),),
    "knockoff": (CodedEffect(kind=CodedMoveKind.KNOCK_OFF_ITEM),),
    "trick": (CodedEffect(kind=CodedMoveKind.TRICK),),
    "switcheroo": (CodedEffect(kind=CodedMoveKind.TRICK),),
    "skillswap": (CodedEffect(kind=CodedMoveKind.SKILL_SWAP),),
    "futuresight": (CodedEffect(kind=CodedMoveKind.FUTURE_SIGHT),),
    "doomdesire": (CodedEffect(kind=CodedMoveKind.FUTURE_SIGHT),),
    "ruination": (FixedDamageEffect(amount_formula="HALF_TARGET_HP", set_amount=None),),
    "superfang": (FixedDamageEffect(amount_formula="HALF_TARGET_HP", set_amount=None),),
    "endeavor": (FixedDamageEffect(amount_formula="ENDEAVOR", set_amount=None),),
    "counter": (FixedDamageEffect(amount_formula="COUNTER", set_amount=None),),
    "mirrorcoat": (FixedDamageEffect(amount_formula="MIRROR_COAT", set_amount=None),),
    "finalgambit": (FixedDamageEffect(amount_formula="USER_HP", set_amount=None),),
}
# Revival Blessing's selfSwitch in data only drives PS's revive prompt; the user stays in.
# Shed Tail's switch and substitute are both handled by its coded effect.
_SUPPRESS_SELF_SWITCH = frozenset({"revivalblessing", "shedtail"})
# Generation 9 halved the PP on the recovery moves and the vendor dex carries those values, but this
# format is gen 7: a Blissey gets sixteen Soft-Boileds, not eight. Only moves whose vendor PP
# actually disagrees with gen 7 are listed, so an unlisted move keeps whatever the dex says.
_GEN7_PP: dict[str, int] = {
    "recover": 10,
    "roost": 10,
    "softboiled": 10,
    "milkdrink": 10,
    "slackoff": 10,
    "rest": 10,
    "shoreup": 10,
}
_SUPPRESS_VOLATILE = frozenset({"shedtail"})

_EXCLUDE_NONSTANDARD_SPECIES = frozenset({"CAP", "Custom"})
_EXCLUDE_NONSTANDARD_MOVES = frozenset({"CAP", "Custom", "LGPE", "Gigantamax", "Unobtainable"})
_LEGENDARY_OR_MYTHICAL_TAGS = frozenset({"Sub-Legendary", "Restricted Legendary", "Mythical"})


@dataclass
class LoaderDiagnostics:
    """Counts unmappable vendor values so drops are visible instead of silent."""

    skipped: Counter[str] = field(default_factory=Counter)

    def skip(self, kind: str, move: str, value: object) -> None:
        self.skipped[f"{kind}:{value}"] += 1
        logger.debug(f"loader: skipped {kind}={value!r} on move {move!r}")

    def log_summary(self, source: str) -> None:
        if not self.skipped:
            return
        total = sum(self.skipped.values())
        logger.info(f"loader: skipped {total} unmappable values across {len(self.skipped)} kinds in {source}")


def normalize_id(name: str) -> str:
    return "".join(c for c in name.lower() if c.isalnum())


def _to_type_pair(types: list[str]) -> TypePair:
    primary = _TYPE_MAP[types[0]]
    secondary = _TYPE_MAP[types[1]] if len(types) > 1 else None
    return (primary, secondary)


def _to_base_stats(raw: dict[str, int]) -> BaseStats:
    return BaseStats(
        HP=raw["hp"],
        ATTACK=raw["atk"],
        DEFENCE=raw["def"],
        SP_ATTACK=raw["spa"],
        SP_DEFENCE=raw["spd"],
        SPEED=raw["spe"],
    )


def _adapt_species(raw: RawSpeciesData) -> BaseSpecies:
    assert raw.num is not None and raw.base_stats is not None  # guaranteed by get_all_species filter
    assert raw.height_m is not None and raw.weight_kg is not None  # holds for every non-filtered vendor entry
    regular = tuple(raw.abilities[slot] for slot in ("0", "1") if slot in raw.abilities)
    return BaseSpecies(
        name=raw.name,
        dex_number=raw.num,
        types=_to_type_pair(raw.types),
        base_stats=_to_base_stats(raw.base_stats),
        regular_abilities=regular,
        hidden_ability=raw.abilities.get("H"),
        height_m=raw.height_m,
        weight_kg=raw.weight_kg,
        fully_evolved=not raw.evos,
        is_legendary_or_mythical=bool(_LEGENDARY_OR_MYTHICAL_TAGS.intersection(raw.tags)),
        base_species=raw.base_species,
        required_item=raw.required_item,
    )


def _accuracy(raw_value: int | bool) -> float | None:
    if raw_value is True:
        return None
    return float(raw_value) / 100.0


def _build_boost_effect(
    boosts: dict[str, int],
    target: Literal["SELF", "TARGET"],
    probability: float = 1.0,
    is_secondary: bool = False,
) -> StatStageChangeEffect | None:
    stages = {_BOOST_MAP[k]: v for k, v in boosts.items() if k in _BOOST_MAP}
    if not stages:
        return None
    return StatStageChangeEffect(target=target, stages=stages, probability=probability, is_secondary=is_secondary)


# Damaging moves whose base power is 0 in data because it lives in a PS callback (engine/power.py).
_FORMULA_POWER_MOVES = frozenset(
    {"lowkick", "grassknot", "heavyslam", "heatcrash", "electroball", "beatup", "gyroball"}
)


def _damage_effects(raw: RawMoveData, category: Category) -> list[MoveEffect]:
    effects: list[MoveEffect] = []
    has_formula_power = normalize_id(raw.name) in _FORMULA_POWER_MOVES
    if category is not Category.STATUS and (raw.base_power or has_formula_power):
        multi_hit = (raw.multihit, raw.multihit) if isinstance(raw.multihit, int) else raw.multihit
        effects.append(
            DamageEffect(
                power=raw.base_power or None,
                category=category,
                crit_stage=(raw.crit_ratio or 1) - 1,
                contact=bool(raw.flags.get("contact")),
                multi_hit=multi_hit,
                recoil_percent=raw.recoil[0] / raw.recoil[1] if raw.recoil else None,
                drain_percent=raw.drain[0] / raw.drain[1] if raw.drain else None,
                struggle_recoil=raw.struggle_recoil,
            )
        )
    if raw.damage == "level":
        effects.append(FixedDamageEffect(amount_formula="LEVEL", set_amount=None))
    elif isinstance(raw.damage, int):
        effects.append(FixedDamageEffect(amount_formula="SET", set_amount=raw.damage))
    if raw.heal:
        effects.append(HealEffect(fraction=raw.heal[0] / raw.heal[1]))
    return effects


def _field_effects(raw: RawMoveData, diag: LoaderDiagnostics) -> list[MoveEffect]:
    effects: list[MoveEffect] = []
    if raw.weather is not None:
        weather_kind = _WEATHER_MAP.get(raw.weather.lower())
        if weather_kind is not None:
            effects.append(WeatherEffect(kind=weather_kind, duration_turns=5))
        else:
            diag.skip("weather", raw.name, raw.weather)
    if raw.terrain is not None:
        terrain_kind = _TERRAIN_MAP.get(raw.terrain.lower())
        if terrain_kind is not None:
            effects.append(TerrainEffect(kind=terrain_kind, duration_turns=5))
        else:
            diag.skip("terrain", raw.name, raw.terrain)
    if raw.pseudo_weather is not None:
        pseudo_kind = _PSEUDO_WEATHER_MAP.get(raw.pseudo_weather.lower())
        if pseudo_kind is not None:
            effects.append(PseudoWeatherEffect(kind=pseudo_kind, duration_turns=5))
        else:
            diag.skip("pseudo_weather", raw.name, raw.pseudo_weather)
    if raw.side_condition is not None:
        mapped_side = _SIDE_CONDITION_MAP.get(raw.side_condition.lower())
        if mapped_side is not None:
            duration = _SIDE_CONDITION_DURATIONS.get(mapped_side)
            effects.append(SideConditionEffect(kind=mapped_side, duration_turns=duration))
        else:
            diag.skip("side_condition", raw.name, raw.side_condition)
    return effects


def _primary_status_effects(raw: RawMoveData, target: Target, diag: LoaderDiagnostics) -> list[MoveEffect]:
    effects: list[MoveEffect] = []
    if raw.volatile_status is not None and normalize_id(raw.name) not in _SUPPRESS_VOLATILE:
        mapped_volatile = _VOLATILE_MAP.get(raw.volatile_status)
        if mapped_volatile is not None:
            effects.append(InflictStatusEffect(status=mapped_volatile, probability=1.0))
        else:
            diag.skip("volatile_status", raw.name, raw.volatile_status)
    if raw.status is not None:
        mapped_status = _STATUS_MAP.get(raw.status)
        if mapped_status is not None:
            effects.append(InflictStatusEffect(status=mapped_status, probability=1.0))
        else:
            diag.skip("status", raw.name, raw.status)
    if raw.boosts is not None:
        boost_target: Literal["SELF", "TARGET"] = "SELF" if target is Target.SELF else "TARGET"
        effect = _build_boost_effect(raw.boosts, boost_target)
        if effect is not None:
            effects.append(effect)
    effects.extend(_self_effects(raw, diag))
    return effects


def _self_effects(raw: RawMoveData, diag: LoaderDiagnostics) -> list[MoveEffect]:
    if raw.self_effects is None:
        return []
    effects: list[MoveEffect] = []
    if raw.self_effects.boosts is not None:
        effect = _build_boost_effect(raw.self_effects.boosts, "SELF")
        if effect is not None:
            effects.append(effect)
    if raw.self_effects.volatile_status is not None and raw.self_effects.volatile_status != "mustrecharge":
        mapped = _VOLATILE_MAP.get(raw.self_effects.volatile_status)
        if mapped is not None:
            effects.append(InflictStatusEffect(status=mapped, probability=1.0, to_self=True))
        else:
            diag.skip("self_volatile_status", raw.name, raw.self_effects.volatile_status)
    return effects


def _secondary_effects(raw: RawMoveData, diag: LoaderDiagnostics) -> list[MoveEffect]:
    effects: list[MoveEffect] = []
    for secondary in raw.all_secondaries():
        effects.extend(_one_secondary(secondary, raw.name, diag))
    return effects


def _one_secondary(secondary: RawSecondary, move_name: str, diag: LoaderDiagnostics) -> list[MoveEffect]:
    effects: list[MoveEffect] = []
    prob = secondary.chance / 100.0
    if secondary.status is not None:
        mapped_status = _STATUS_MAP.get(secondary.status)
        if mapped_status is not None:
            effects.append(InflictStatusEffect(status=mapped_status, probability=prob, is_secondary=True))
        else:
            diag.skip("secondary_status", move_name, secondary.status)
    if secondary.volatile_status is not None:
        mapped_volatile = _VOLATILE_MAP.get(secondary.volatile_status)
        if mapped_volatile is not None:
            effects.append(InflictStatusEffect(status=mapped_volatile, probability=prob, is_secondary=True))
        else:
            diag.skip("secondary_volatile_status", move_name, secondary.volatile_status)
    if secondary.boosts is not None:
        effect = _build_boost_effect(secondary.boosts, "TARGET", probability=prob, is_secondary=True)
        if effect is not None:
            effects.append(effect)
    if secondary.self_effects is not None and secondary.self_effects.boosts is not None:
        effect = _build_boost_effect(secondary.self_effects.boosts, "SELF", probability=prob, is_secondary=True)
        if effect is not None:
            effects.append(effect)
    return effects


def _adapt_move(raw: RawMoveData, diag: LoaderDiagnostics) -> Move | None:
    target = _TARGET_MAP.get(raw.target)
    if target is None:
        diag.skip("target", raw.name, raw.target)
        return None
    category = _CATEGORY_MAP[raw.category]
    effects: list[MoveEffect] = [
        *_damage_effects(raw, category),
        *_field_effects(raw, diag),
        *_primary_status_effects(raw, target, diag),
        *_secondary_effects(raw, diag),
        *_CODED_EFFECTS.get(normalize_id(raw.name), ()),
    ]
    return Move(
        name=raw.name,
        type=_TYPE_MAP[raw.type],
        category=category,
        accuracy_probability=_accuracy(raw.accuracy),
        priority=PriorityLevel(raw.priority),
        pp=_GEN7_PP.get(normalize_id(raw.name), raw.pp),
        target=target,
        effects=tuple(effects),
        self_switch=bool(raw.self_switch) and normalize_id(raw.name) not in _SUPPRESS_SELF_SWITCH,
        protectable=bool(raw.flags.get("protect")),
        bypass_substitute=bool(raw.flags.get("bypasssub")),
        stalling=raw.stalling_move,
        slicing=bool(raw.flags.get("slicing")),
        sound=bool(raw.flags.get("sound")),
        punching=bool(raw.flags.get("punch")),
        biting=bool(raw.flags.get("bite")),
        pulse=bool(raw.flags.get("pulse")),
        wind=bool(raw.flags.get("wind")),
        bullet=bool(raw.flags.get("bullet")),
        healing=bool(raw.flags.get("heal")),
        reflectable=bool(raw.flags.get("reflectable")),
        charge=bool(raw.flags.get("charge")),
        self_destructs=raw.self_destruct is not None,
        recharges=raw.self_effects is not None and raw.self_effects.volatile_status == "mustrecharge",
        force_switch=raw.force_switch,
        typeless=raw.struggle_recoil,  # only Struggle; its typelessness lives in PS code (onEffectiveness -> 0)
        has_crash_damage=raw.has_crash_damage,
    )


@cache
def get_all_species() -> dict[str, BaseSpecies]:
    raw_data = json.loads((_VENDOR_DIR / "pokedex.json").read_text())
    result: dict[str, BaseSpecies] = {}
    for key, entry in raw_data.items():
        raw = RawSpeciesData.model_validate(entry)
        if raw.is_nonstandard in _EXCLUDE_NONSTANDARD_SPECIES:
            continue
        if raw.is_cosmetic_forme:
            continue
        if raw.num is None or raw.base_stats is None:
            continue
        result[key] = _adapt_species(raw)
    return result


@cache
def get_all_z_moves() -> dict[str, Move]:
    """Z-Crystal id -> the Z-move it unleashes.

    Deliberately a separate table from `get_all_moves`: a Z-move is never an ordinary move slot, so
    letting it into the main pool would make it selectable in team building. The 18 generic type
    Z-moves arrive with placeholder data (`basePower: 1`, and a `category` that ignores whichever
    base move triggered them) — `zmoves.z_move_for` supplies both from the base move instead.
    """
    raw_data = json.loads((_VENDOR_DIR / "moves.json").read_text())
    diag = LoaderDiagnostics()
    result: dict[str, Move] = {}
    for entry in raw_data.values():
        raw = RawMoveData.model_validate(entry)
        if not isinstance(raw.is_z, str) or raw.is_max is not None:
            continue
        move = _adapt_move(raw, diag)
        if move is not None:
            result[raw.is_z] = move
    return result


@cache
def get_all_moves() -> dict[str, Move]:
    raw_data = json.loads((_VENDOR_DIR / "moves.json").read_text())
    diag = LoaderDiagnostics()
    result: dict[str, Move] = {}
    for key, entry in raw_data.items():
        raw = RawMoveData.model_validate(entry)
        if raw.is_nonstandard in _EXCLUDE_NONSTANDARD_MOVES:
            continue
        if raw.is_z is not None or raw.is_max is not None:
            continue
        move = _adapt_move(raw, diag)
        if move is not None:
            result[key] = move
    diag.log_summary("moves.json")
    return result


# Pre-rename forme names still common in older pastes.
_SPECIES_ALIASES: dict[str, str] = {
    "taurospaldeafire": "taurospaldeablaze",
    "taurospaldeawater": "taurospaldeaaqua",
}


def get_species(name: str) -> BaseSpecies:
    species = get_all_species()
    key = normalize_id(name)
    key = _SPECIES_ALIASES.get(key, key)
    if key not in species:
        raise KeyError(f"Unknown species: {name!r}")
    return species[key]


def get_move(name: str) -> Move:
    moves = get_all_moves()
    key = normalize_id(name)
    if key not in moves:
        raise KeyError(f"Unknown move: {name!r}")
    return moves[key]


@cache
def _gen7_singles_tiers() -> dict[str, str]:
    """`{species_id: "OU"|"UU"|"Uber"|...}`, Gen 7's own competitive placement — not `pokedex.json`'s
    `tier` field, which always reflects the *current* generation's metagame instead."""
    raw: dict[str, str] = json.loads((_VENDOR_DIR / "gen7_tiers.json").read_text())
    return raw


def gen7_singles_tier(name: str) -> str | None:
    """A species' Gen 7 singles tier (`"OU"`, `"UU"`, `"Uber"`, ...), or None if it was never placed —
    e.g. it postdates Gen 7, or only ever had a doubles tier."""
    return _gen7_singles_tiers().get(normalize_id(name))
