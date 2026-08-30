"""Move power formulas and conditions that live in Showdown *code* rather than move data.

Tables are keyed by display move name. `effective_power` resolves the final base power
(weight/speed/counter formulas, then conditional multipliers); `coded_move_fails` covers
moves with a code-driven failure clause; `payload_overrides` seeds per-move damage-calc
payload keys (stat overrides, burn/weather exemptions).
"""

from collections.abc import Callable

from battle_sim.mechanics.battle import BattleState, SideState, effective_weather
from battle_sim.mechanics.events import Payload
from battle_sim.mechanics.priority import effective_speed
from battle_sim.models.actions import ActionType
from battle_sim.models.moves import DamageEffect, FixedDamageEffect, Move
from battle_sim.models.pokemon import Pokemon
from battle_sim.models.type_matchups import type_effectiveness
from battle_sim.utils import Item, Stats, Status, Terrain, Weather

type PowerCondition = Callable[[Pokemon, Pokemon, BattleState], bool]
type PowerFormula = Callable[[Pokemon, Pokemon, BattleState], int]


ROLLING_MOVES: frozenset[str] = frozenset({"Rollout", "Ice Ball"})
_ROLLING_POWERS: tuple[int, ...] = (30, 60, 120, 240, 480)  # doubles per connected hit, then holds
ROLLING_LOCK_TURNS = len(_ROLLING_POWERS)  # committed for the whole run; a miss ends it early


def _rolling_power(attacker: Pokemon, defender: Pokemon, state: BattleState) -> int:
    """Rollout and Ice Ball double for every consecutive hit that has already connected."""
    return _ROLLING_POWERS[min(attacker.rolling_hits, len(_ROLLING_POWERS) - 1)]


def _weight_class_power(attacker: Pokemon, defender: Pokemon, state: BattleState) -> int:
    for threshold, power in ((200.0, 120), (100.0, 100), (50.0, 80), (25.0, 60), (10.0, 40)):
        if defender.weight_kg >= threshold:
            return power
    return 20


def _weight_ratio_power(attacker: Pokemon, defender: Pokemon, state: BattleState) -> int:
    ratio = attacker.weight_kg / defender.weight_kg
    for threshold, power in ((5.0, 120), (4.0, 100), (3.0, 80), (2.0, 60)):
        if ratio >= threshold:
            return power
    return 40


def _speed_ratio_power(attacker: Pokemon, defender: Pokemon, state: BattleState) -> int:
    attacker_side, defender_side = _sides_of(attacker, defender, state)
    own = effective_speed(attacker, attacker_side, state.field)
    other = max(1, effective_speed(defender, defender_side, state.field))
    ratio = own / other
    for threshold, power in ((4.0, 150), (3.0, 120), (2.0, 80), (1.0, 60)):
        if ratio >= threshold:
            return power
    return 40


def _gyro_ball_power(attacker: Pokemon, defender: Pokemon, state: BattleState) -> int:
    """The slower the user next to its target, the harder Gyro Ball hits."""
    attacker_side, defender_side = _sides_of(attacker, defender, state)
    own = max(1, effective_speed(attacker, attacker_side, state.field))
    other = effective_speed(defender, defender_side, state.field)
    return min(150, 25 * other // own + 1)


def _boost_count_power(attacker: Pokemon, defender: Pokemon, state: BattleState) -> int:
    positive = sum(max(0, attacker.stat_stages[stat]) for stat in _STAGED_STATS)
    return 20 + 20 * positive


def _last_respects_power(attacker: Pokemon, defender: Pokemon, state: BattleState) -> int:
    side, _ = _sides_of(attacker, defender, state)
    fallen = sum(1 for member in side.team if member.is_fainted())
    return 50 * (1 + fallen)


def _rage_fist_power(attacker: Pokemon, defender: Pokemon, state: BattleState) -> int:
    return min(350, 50 * (1 + attacker.times_hit))


def _beat_up_power(attacker: Pokemon, defender: Pokemon, state: BattleState) -> int:
    """PS hits once per healthy ally; approximated as one hit carrying the summed power."""
    side, _ = _sides_of(attacker, defender, state)
    contributors = [m for m in side.team if not m.is_fainted() and m.status is Status.NONE]
    return sum(5 + m.base_stats.ATTACK // 10 for m in contributors) or 5


def _hp_scaled_power(attacker: Pokemon, defender: Pokemon, state: BattleState) -> int:
    """Water Spout / Eruption / Dragon Energy: power falls off with the user's current HP%."""
    return max(1, 150 * attacker.live_stats.HP // attacker.stat_totals.HP)


_STAGED_STATS = tuple(s for s in Stats if s is not Stats.HP)

_POWER_FORMULAS: dict[str, PowerFormula] = {
    "Low Kick": _weight_class_power,
    "Grass Knot": _weight_class_power,
    "Heavy Slam": _weight_ratio_power,
    "Heat Crash": _weight_ratio_power,
    "Electro Ball": _speed_ratio_power,
    "Gyro Ball": _gyro_ball_power,
    "Stored Power": _boost_count_power,
    "Power Trip": _boost_count_power,
    "Last Respects": _last_respects_power,
    "Rage Fist": _rage_fist_power,
    "Beat Up": _beat_up_power,
    "Rollout": _rolling_power,
    "Ice Ball": _rolling_power,
    "Water Spout": _hp_scaled_power,
    "Eruption": _hp_scaled_power,
    "Dragon Energy": _hp_scaled_power,
}


def _defender_statused(attacker: Pokemon, defender: Pokemon, state: BattleState) -> bool:
    return defender.status is not Status.NONE


def _defender_poisoned(attacker: Pokemon, defender: Pokemon, state: BattleState) -> bool:
    return defender.status in (Status.POISON, Status.TOXIC)


def _attacker_statused(attacker: Pokemon, defender: Pokemon, state: BattleState) -> bool:
    return attacker.status is not Status.NONE


def _attacker_itemless(attacker: Pokemon, defender: Pokemon, state: BattleState) -> bool:
    return attacker.item is Item.NONE


def _defender_has_item(attacker: Pokemon, defender: Pokemon, state: BattleState) -> bool:
    return defender.item is not Item.NONE


def _hit_by_target_this_turn(attacker: Pokemon, defender: Pokemon, state: BattleState) -> bool:
    return attacker.last_hit_taken > 0


def _weather_active(attacker: Pokemon, defender: Pokemon, state: BattleState) -> bool:
    return effective_weather(state) is not Weather.NONE


def _electric_terrain_grounded_target(attacker: Pokemon, defender: Pokemon, state: BattleState) -> bool:
    return state.field.terrain is Terrain.ELECTRIC and defender.is_grounded()


def _psychic_terrain_grounded_user(attacker: Pokemon, defender: Pokemon, state: BattleState) -> bool:
    return state.field.terrain is Terrain.PSYCHIC and attacker.is_grounded()


def _electric_terrain_grounded_user(attacker: Pokemon, defender: Pokemon, state: BattleState) -> bool:
    return state.field.terrain is Terrain.ELECTRIC and attacker.is_grounded()


def _in_sun(attacker: Pokemon, defender: Pokemon, state: BattleState) -> bool:
    return effective_weather(state) in (Weather.SUN, Weather.HARSH_SUN)


# (condition, numerator, denominator) applied to the resolved base power.
_POWER_CONDITIONS: dict[str, tuple[PowerCondition, int, int]] = {
    "Facade": (_attacker_statused, 2, 1),
    "Hex": (_defender_statused, 2, 1),
    "Infernal Parade": (_defender_statused, 2, 1),
    "Barb Barrage": (_defender_poisoned, 2, 1),
    "Acrobatics": (_attacker_itemless, 2, 1),
    "Avalanche": (_hit_by_target_this_turn, 2, 1),
    "Weather Ball": (_weather_active, 2, 1),
    "Rising Voltage": (_electric_terrain_grounded_target, 2, 1),
    "Knock Off": (_defender_has_item, 3, 2),
    "Expanding Force": (_psychic_terrain_grounded_user, 3, 2),
    "Psyblade": (_electric_terrain_grounded_user, 3, 2),
    "Hydro Steam": (_in_sun, 3, 2),
}
_SE_BONUS_MOVES = frozenset({"Electro Drift", "Collision Course"})  # 5461/4096 when super effective


def effective_power(move: Move, effect: DamageEffect, attacker: Pokemon, defender: Pokemon, state: BattleState) -> int:
    formula = _POWER_FORMULAS.get(move.name)
    power = formula(attacker, defender, state) if formula is not None else effect.power
    assert power is not None  # every damaging move has data power or a formula above
    condition_entry = _POWER_CONDITIONS.get(move.name)
    if condition_entry is not None:
        condition, numerator, denominator = condition_entry
        if condition(attacker, defender, state):
            power = power * numerator // denominator
    if move.name in _SE_BONUS_MOVES and type_effectiveness(move.type, defender.types) >= 2:
        power = power * 5461 // 4096
    return power


def coded_move_fails(move: Move, attacker: Pokemon, defender: Pokemon, state: BattleState) -> bool:
    if move.name == "Poltergeist":
        return defender.item is Item.NONE
    if move.name in ("Sucker Punch", "Thunderclap"):
        return not _target_is_about_to_attack(defender, state)
    if move.name == "Dream Eater":
        return defender.status is not Status.SLEEP
    if move.name in ("Fake Out", "First Impression"):
        return attacker.turns_active > 0
    return False


def _target_is_about_to_attack(defender: Pokemon, state: BattleState) -> bool:
    defender_side, _ = _sides_of(defender, defender, state)
    if defender_side.acted_this_turn:
        return False
    action = defender_side.chosen_action
    if action is None or action.action is not ActionType.USE_MOVE or action.move is None:
        return False
    chosen = defender.moves[action.move]
    return chosen is not None and any(isinstance(e, (DamageEffect, FixedDamageEffect)) for e in chosen.effects)


def payload_overrides(move: Move, attacker: Pokemon, defender: Pokemon) -> Payload:
    if move.name == "Body Press":
        return {"attack_stat_override": Stats.DEFENCE}
    if move.name == "Foul Play":
        return {"use_target_attack": True}
    if move.name == "Facade":
        return {"ignore_burn": True}
    if move.name == "Hydro Steam":
        return {"ignore_weather_drop": True}
    return {}


def _sides_of(attacker: Pokemon, defender: Pokemon, state: BattleState) -> tuple[SideState, SideState]:
    """(attacker's side, defender's side); the attacker must be one of the actives."""
    if state.sides[0].active_pokemon is attacker:
        return state.sides[0], state.sides[1]
    return state.sides[1], state.sides[0]
