import json
from functools import cache
from pathlib import Path
from typing import Literal

from battle_sim.models.moves import (
    DamageEffect,
    InflictStatusEffect,
    Move,
    MoveEffect,
    StatStageChangeEffect,
)
from battle_sim.models.species import BaseSpecies
from battle_sim.models.stats import BaseStats
from battle_sim.models.type_matchups import TypePair
from battle_sim.utils import Category, ExtraStatus, PriorityLevel, Stats, Status, Target, Type

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
    "tox": Status.POISON,
    "par": Status.PARALYSIS,
    "slp": Status.SLEEP,
    "frz": Status.FREEZE,
}
_VOLATILE_MAP: dict[str, ExtraStatus] = {
    "confusion": ExtraStatus.CONFUSION,
    "flinch": ExtraStatus.FLINCH,
    "attract": ExtraStatus.INFATUATION,
    "partiallytrapped": ExtraStatus.TRAPPED,
    "curse": ExtraStatus.CURSE,
    "embargo": ExtraStatus.EMBARGO,
    "leechseed": ExtraStatus.LEECH_SEED,
    "nightmare": ExtraStatus.NIGHTMARE,
    "perishsong": ExtraStatus.PERISH_SONG,
    "taunt": ExtraStatus.TAUNT,
}

_EXCLUDE_NONSTANDARD_SPECIES = frozenset({"CAP", "Custom"})
_EXCLUDE_NONSTANDARD_MOVES = frozenset({"CAP", "Custom", "LGPE", "Gigantamax", "Unobtainable"})


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


def _adapt_species(raw: dict) -> BaseSpecies:
    abilities = raw["abilities"]
    regular = tuple(abilities[slot] for slot in ("0", "1") if slot in abilities)
    return BaseSpecies(
        name=raw["name"],
        dex_number=raw["num"],
        types=_to_type_pair(raw["types"]),
        base_stats=_to_base_stats(raw["baseStats"]),
        regular_abilities=regular,
        hidden_ability=abilities.get("H"),
        height_m=float(raw["heightm"]),
        weight_kg=float(raw["weightkg"]),
    )


def _accuracy(raw_value: int | bool) -> float | None:
    if raw_value is True:
        return None
    return float(raw_value) / 100.0


def _build_boost_effect(
    boosts: dict[str, int],
    target: Literal["SELF", "TARGET"],
    probability: float = 1.0,
) -> StatStageChangeEffect | None:
    stages = {_STAT_MAP[k]: v for k, v in boosts.items() if k in _STAT_MAP}
    if not stages:
        return None
    return StatStageChangeEffect(target=target, stages=stages, probability=probability)


def _adapt_move(raw: dict) -> Move | None:
    target = _TARGET_MAP.get(raw["target"])
    if target is None:
        return None

    category = _CATEGORY_MAP[raw["category"]]
    flags = raw.get("flags", {})
    effects: list[MoveEffect] = []

    if category != Category.STATUS and raw.get("basePower"):
        multi_hit: tuple[int, int] | None = None
        mh = raw.get("multihit")
        if isinstance(mh, list) and len(mh) == 2:
            multi_hit = (mh[0], mh[1])
        elif isinstance(mh, int):
            multi_hit = (mh, mh)

        recoil_percent: float | None = None
        if "recoil" in raw:
            num, den = raw["recoil"]
            recoil_percent = num / den

        drain_percent: float | None = None
        if "drain" in raw:
            num, den = raw["drain"]
            drain_percent = num / den

        effects.append(
            DamageEffect(
                power=raw["basePower"],
                category=category,
                crit_stage=raw.get("critRatio", 1) - 1,
                contact=bool(flags.get("contact")),
                multi_hit=multi_hit,
                recoil_percent=recoil_percent,
                drain_percent=drain_percent,
            )
        )

    primary_status = raw.get("status")
    if isinstance(primary_status, str):
        mapped = _STATUS_MAP.get(primary_status)
        if mapped is not None:
            effects.append(InflictStatusEffect(status=mapped, probability=1.0))

    if isinstance(raw.get("boosts"), dict):
        boost_target: Literal["SELF", "TARGET"] = "SELF" if target == Target.SELF else "TARGET"
        effect = _build_boost_effect(raw["boosts"], boost_target)
        if effect is not None:
            effects.append(effect)

    self_section = raw.get("self")
    if isinstance(self_section, dict) and isinstance(self_section.get("boosts"), dict):
        effect = _build_boost_effect(self_section["boosts"], "SELF")
        if effect is not None:
            effects.append(effect)

    secondary = raw.get("secondary")
    if isinstance(secondary, dict):
        prob = secondary.get("chance", 0) / 100.0
        sec_status = secondary.get("status")
        if isinstance(sec_status, str):
            mapped_status = _STATUS_MAP.get(sec_status)
            if mapped_status is not None:
                effects.append(InflictStatusEffect(status=mapped_status, probability=prob))
        sec_volatile = secondary.get("volatileStatus")
        if isinstance(sec_volatile, str):
            mapped_volatile = _VOLATILE_MAP.get(sec_volatile)
            if mapped_volatile is not None:
                effects.append(InflictStatusEffect(status=mapped_volatile, probability=prob))
        if isinstance(secondary.get("boosts"), dict):
            effect = _build_boost_effect(secondary["boosts"], "TARGET", probability=prob)
            if effect is not None:
                effects.append(effect)
        sec_self = secondary.get("self")
        if isinstance(sec_self, dict) and isinstance(sec_self.get("boosts"), dict):
            effect = _build_boost_effect(sec_self["boosts"], "SELF", probability=prob)
            if effect is not None:
                effects.append(effect)

    return Move(
        name=raw["name"],
        type=_TYPE_MAP[raw["type"]],
        category=category,
        accuracy_probability=_accuracy(raw["accuracy"]),
        priority=PriorityLevel(raw["priority"]),
        pp=raw["pp"],
        target=target,
        effects=tuple(effects),
    )


@cache
def get_all_species() -> dict[str, BaseSpecies]:
    raw_data = json.loads((_VENDOR_DIR / "pokedex.json").read_text())
    result: dict[str, BaseSpecies] = {}
    for key, raw in raw_data.items():
        if raw.get("isNonstandard") in _EXCLUDE_NONSTANDARD_SPECIES:
            continue
        if raw.get("isCosmeticForme"):
            continue
        if "num" not in raw or "baseStats" not in raw:
            continue
        result[key] = _adapt_species(raw)
    return result


@cache
def get_all_moves() -> dict[str, Move]:
    raw_data = json.loads((_VENDOR_DIR / "moves.json").read_text())
    result: dict[str, Move] = {}
    for key, raw in raw_data.items():
        if raw.get("isNonstandard") in _EXCLUDE_NONSTANDARD_MOVES:
            continue
        if "isZ" in raw or "isMax" in raw:
            continue
        move = _adapt_move(raw)
        if move is not None:
            result[key] = move
    return result


def get_species(name: str) -> BaseSpecies:
    species = get_all_species()
    key = normalize_id(name)
    if key not in species:
        raise KeyError(f"Unknown species: {name!r}")
    return species[key]


def get_move(name: str) -> Move:
    moves = get_all_moves()
    key = normalize_id(name)
    if key not in moves:
        raise KeyError(f"Unknown move: {name!r}")
    return moves[key]
