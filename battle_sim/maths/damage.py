from battle_sim.maths.rng import RNG
from battle_sim.maths.stats import apply_stage_multiplier
from battle_sim.mechanics.battle import FieldState, SideState
from battle_sim.mechanics.events import Payload
from battle_sim.models.moves import DamageEffect, Move
from battle_sim.models.pokemon import Pokemon
from battle_sim.models.type_matchups import type_effectiveness
from battle_sim.utils import Ability, Category, ExtraStatus, Hazards, Item, Stats, Status, Terrain, Type, Weather

_CRIT_PROBABILITIES = (1 / 24, 1 / 8, 1 / 2, 1.0)


def calculate_damage(  # noqa: C901 — the Gen-9 modifier chain; its sequence IS the contract, splitting would hide it
    attacker: Pokemon,
    defender: Pokemon,
    move: Move,
    field: FieldState,
    defender_side: SideState,
    rng: RNG,
    is_crit: bool | None = None,
    random_roll: int | None = None,
    target_count: int = 1,
    modifiers: Payload | None = None,
) -> int:
    """Pure Gen-9 damage formula. Ability/item contributions arrive pre-collected in `modifiers`
    (see Payload in mechanics.events); their fold positions are fixed so rounding is exact
    regardless of handler registration order."""
    payload: Payload = {} if modifiers is None else modifiers
    damage_effect = next((e for e in move.effects if isinstance(e, DamageEffect)), None)
    if damage_effect is None:
        return 0
    power: int | None = payload.get("power_override", damage_effect.power)
    if not power:
        return 0

    if defender.item is Item.AIR_BALLOON and move.type is Type.GROUND:
        return 0

    type_multiplier = move_effectiveness(move, attacker, defender)
    if type_multiplier == 0:
        return 0

    if is_crit is None:
        crit_stage = damage_effect.crit_stage + (2 if ExtraStatus.FOCUS_ENERGY in attacker.volatiles else 0)
        if attacker.item is Item.SCOPE_LENS:
            crit_stage += 1
        rolled_crit = rng.roll_chance(_crit_chance(crit_stage))
        is_crit = rolled_crit and defender.ability not in (Ability.BATTLE_ARMOR, Ability.SHELL_ARMOR)

    if damage_effect.category is Category.PHYSICAL:
        attack_stat, defense_stat = Stats.ATTACK, Stats.DEFENCE
    else:
        attack_stat, defense_stat = Stats.SP_ATTACK, Stats.SP_DEFENCE
    attack_stat = payload.get("attack_stat_override", attack_stat)  # Body Press attacks with Defence
    attack_owner = defender if payload.get("use_target_attack", False) else attacker  # Foul Play

    attack = _crit_aware_offensive_stat(attack_owner, attack_stat, is_crit, payload.get("ignore_attack_stages", False))
    for modifier in payload.get("attack_mods_4096", []):
        attack = _chain(attack, modifier)
    defense = _crit_aware_defensive_stat(defender, defense_stat, is_crit, payload.get("ignore_defense_stages", False))
    for modifier in payload.get("defense_mods_4096", []):
        defense = _chain(defense, modifier)

    for modifier in payload.get("power_mods_4096", []):
        power = _chain(power, modifier)

    damage = ((2 * attacker.level) // 5 + 2) * power * attack // defense // 50 + 2

    if target_count > 1:
        damage = _chain(damage, 3072)
    active_weather = Weather.NONE if payload.get("weather_suppressed", False) else field.weather
    weather_mod = _weather_modifier(move.type, active_weather)
    if weather_mod < 4096 and payload.get("ignore_weather_drop", False):
        weather_mod = 4096  # Hydro Steam thrives in the sun
    damage = _chain(damage, weather_mod)
    if is_crit:
        damage = _chain(damage, 9216 if attacker.ability is Ability.SNIPER else 6144)

    if random_roll is None:
        random_roll = rng.random_integer(85, 101)
    damage = damage * random_roll // 100

    if move.type in attacker.types:
        damage = _chain(damage, payload.get("stab_4096", 6144))

    damage = int(damage * type_multiplier)

    if (
        attacker.status is Status.BURN
        and damage_effect.category is Category.PHYSICAL
        and not payload.get("ignore_burn", False)
    ):
        damage = _chain(damage, 2048)

    for modifier in payload.get("pre_screen_mods_4096", []):
        damage = _chain(damage, modifier)

    if not is_crit and not payload.get("bypass_screens", False):
        damage = _chain(damage, _screen_modifier(damage_effect.category, defender_side.screens))

    damage = _chain(damage, _terrain_modifier(move.type, attacker, defender, field.terrain))
    for modifier in payload.get("final_mods_4096", []):
        damage = _chain(damage, modifier)

    return max(1, damage)


_EFFECTIVENESS_OVERRIDES: dict[str, dict[Type, float]] = {
    "Freeze-Dry": {Type.WATER: 2.0},  # PS onEffectiveness: super effective against Water
}


def move_effectiveness(move: Move, attacker: Pokemon, defender: Pokemon) -> float:
    """Type effectiveness with per-move (Freeze-Dry) and per-attacker (Scrappy) overrides folded in."""
    if move.typeless:
        return 1.0
    bypass = immunity_bypass(defender)
    scrappy = attacker.ability in (Ability.SCRAPPY, Ability.MINDS_EYE) and move.type in (Type.NORMAL, Type.FIGHTING)
    if scrappy and Type.GHOST in defender.types:
        bypass = bypass | {Type.GHOST}
    multiplier = type_effectiveness(move.type, defender.types, immunity_bypass=bypass)
    overrides = _EFFECTIVENESS_OVERRIDES.get(move.name)
    if overrides is None:
        return multiplier
    for defending_type, forced in overrides.items():
        if defending_type in defender.types:
            natural = type_effectiveness(move.type, (defending_type, None))
            if natural > 0:
                multiplier = multiplier / natural * forced
    return multiplier


def immunity_bypass(defender: Pokemon) -> set[Type]:
    """Defending types whose 0× immunity is bypassed by the defender's current volatiles."""
    bypass: set[Type] = set()
    if Type.GHOST in defender.types and ExtraStatus.IDENTIFIED in defender.volatiles:
        bypass.add(Type.GHOST)
    if Type.DARK in defender.types and ExtraStatus.MIRACLE_EYE in defender.volatiles:
        bypass.add(Type.DARK)
    return bypass


def _chain(value: int, modifier_4096: int) -> int:
    return ((value * modifier_4096) + 2047) // 4096


def _crit_chance(crit_stage: int) -> float:
    return _CRIT_PROBABILITIES[min(max(crit_stage, 0), len(_CRIT_PROBABILITIES) - 1)]


def _crit_aware_offensive_stat(attacker: Pokemon, stat: Stats, is_crit: bool, ignore_stages: bool) -> int:
    stage = 0 if ignore_stages else attacker.stat_stages[stat]  # Unaware defenders ignore the boosts
    if is_crit:
        stage = max(0, stage)
    return apply_stage_multiplier(attacker.stat_totals[stat], stage)


def _crit_aware_defensive_stat(defender: Pokemon, stat: Stats, is_crit: bool, ignore_stages: bool) -> int:
    stage = 0 if ignore_stages else defender.stat_stages[stat]  # Unaware attackers ignore the bulk-ups
    if is_crit:
        stage = min(0, stage)
    return apply_stage_multiplier(defender.stat_totals[stat], stage)


def _weather_modifier(move_type: Type, weather: Weather) -> int:
    if weather in (Weather.SUN, Weather.HARSH_SUN):
        if move_type is Type.FIRE:
            return 6144
        if move_type is Type.WATER:
            return 2048
    if weather in (Weather.RAIN, Weather.HEAVY_RAIN):
        if move_type is Type.WATER:
            return 6144
        if move_type is Type.FIRE:
            return 2048
    return 4096


def _terrain_modifier(move_type: Type, attacker: Pokemon, defender: Pokemon, terrain: Terrain) -> int:
    if terrain is Terrain.ELECTRIC and move_type is Type.ELECTRIC and attacker.is_grounded():
        return 5324
    if terrain is Terrain.GRASSY and move_type is Type.GRASS and attacker.is_grounded():
        return 5324
    if terrain is Terrain.PSYCHIC and move_type is Type.PSYCHIC and attacker.is_grounded():
        return 5324
    if terrain is Terrain.MISTY and move_type is Type.DRAGON and defender.is_grounded():
        return 2048
    return 4096


def _screen_modifier(category: Category, screens: dict[Hazards, int]) -> int:
    if Hazards.AURORA_VEIL in screens:
        return 2048
    if category is Category.PHYSICAL and Hazards.REFLECT in screens:
        return 2048
    if category is Category.SPECIAL and Hazards.LIGHT_SCREEN in screens:
        return 2048
    return 4096
