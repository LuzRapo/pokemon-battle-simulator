"""Move power formulas and conditions that live in Showdown *code* rather than move data.

Tables are keyed by display move name. `effective_power` resolves the final base power
(weight/speed/counter formulas, then conditional multipliers); `coded_move_fails` covers
moves with a code-driven failure clause; `payload_overrides` seeds per-move damage-calc
payload keys (stat overrides, burn/weather exemptions).
"""

from collections.abc import Callable

from battle_sim.database.loader import get_move
from battle_sim.mechanics.battle import BattleState, SideState, effective_weather
from battle_sim.mechanics.events import Payload
from battle_sim.mechanics.priority import effective_speed
from battle_sim.models.actions import ActionType
from battle_sim.models.moves import DamageEffect, FixedDamageEffect, Move, MoveSlot
from battle_sim.models.pokemon import Pokemon
from battle_sim.models.type_matchups import type_effectiveness
from battle_sim.utils import Ability, Category, Item, Stats, Status, Terrain, Type, Weather

# Belch only works once its user has actually eaten one.
_BERRIES = frozenset(item for item in Item if item.name.endswith("_BERRY"))

type PowerCondition = Callable[[Pokemon, Pokemon, BattleState], bool]
type PowerFormula = Callable[[Pokemon, Pokemon, BattleState], int]


ROLLING_MOVES: frozenset[str] = frozenset({"Rollout", "Ice Ball"})
# Fury Cutter escalates the same way but commits its user to nothing, so it counts hits without
# taking the lock — which is why it is a separate set rather than another rolling move.
ESCALATING_MOVES: frozenset[str] = ROLLING_MOVES | frozenset({"Fury Cutter"})
_FURY_CUTTER_POWERS: tuple[int, ...] = (40, 80, 160)
_ROLLING_POWERS: tuple[int, ...] = (30, 60, 120, 240, 480)  # doubles per connected hit, then holds
ROLLING_LOCK_TURNS = len(_ROLLING_POWERS)  # committed for the whole run; a miss ends it early


def _rolling_power(attacker: Pokemon, defender: Pokemon, state: BattleState) -> int:
    """Rollout and Ice Ball double for every consecutive hit that has already connected."""
    return _ROLLING_POWERS[min(attacker.rolling_hits, len(_ROLLING_POWERS) - 1)]


def _fury_cutter_power(attacker: Pokemon, defender: Pokemon, state: BattleState) -> int:
    """Doubles per consecutive connection and then holds, with no lock-in to pay for it."""
    return _FURY_CUTTER_POWERS[min(attacker.rolling_hits, len(_FURY_CUTTER_POWERS) - 1)]


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


def _friendship_power(attacker: Pokemon, defender: Pokemon, state: BattleState) -> int:
    """Return and Frustration both come out at 102 in practice.

    Their power is friendship/2.5 and (255-friendship)/2.5 respectively, and a competitive set is
    always built at whichever end its move wants — max friendship for Return, zero for Frustration.
    Friendship is not modelled here (nothing else in the format reads it), so rather than load both
    moves as 0 power and have them do nothing at all, both take the value every real set gives them.
    """
    return 102


def _low_hp_power(attacker: Pokemon, defender: Pokemon, state: BattleState) -> int:
    """Flail / Reversal: the closer to fainting, the harder it hits."""
    scaled = 48 * attacker.live_stats.HP // max(1, attacker.stat_totals.HP)
    for threshold, power in ((1, 200), (4, 150), (9, 100), (16, 80), (32, 40)):
        if scaled < threshold:
            return power
    return 20


def _target_hp_power(attacker: Pokemon, defender: Pokemon, state: BattleState) -> int:
    """Crush Grip / Wring Out: strongest against a healthy target."""
    return max(1, 120 * defender.live_stats.HP // max(1, defender.stat_totals.HP))


def _punishment_power(attacker: Pokemon, defender: Pokemon, state: BattleState) -> int:
    """Punishment: 60, plus 20 for each stage the target has raised, capped at 200."""
    positive = sum(max(0, defender.stat_stages[stat]) for stat in _STAGED_STATS)
    return min(200, 60 + 20 * positive)


def _magnitude_power(attacker: Pokemon, defender: Pokemon, state: BattleState) -> int:
    """Magnitude 4 through 10, on the games' own weighting."""
    roll = state.rng.random_integer(1, 20)
    for threshold, power in ((1, 10), (3, 30), (7, 50), (13, 70), (17, 90), (19, 110)):
        if roll <= threshold:
            return power
    return 150


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
    "Fury Cutter": _fury_cutter_power,
    "Water Spout": _hp_scaled_power,
    "Eruption": _hp_scaled_power,
    "Return": _friendship_power,
    "Frustration": _friendship_power,
    "Flail": _low_hp_power,
    "Reversal": _low_hp_power,
    "Crush Grip": _target_hp_power,
    "Wring Out": _target_hp_power,
    "Punishment": _punishment_power,
    "Magnitude": _magnitude_power,
    "Dragon Energy": _hp_scaled_power,
}


def _defender_statused(attacker: Pokemon, defender: Pokemon, state: BattleState) -> bool:
    return defender.status is not Status.NONE


def _defender_poisoned(attacker: Pokemon, defender: Pokemon, state: BattleState) -> bool:
    return defender.status in (Status.POISON, Status.TOXIC)


def _defender_asleep(attacker: Pokemon, defender: Pokemon, state: BattleState) -> bool:
    return defender.status is Status.SLEEP


def _defender_paralysed(attacker: Pokemon, defender: Pokemon, state: BattleState) -> bool:
    return defender.status is Status.PARALYSIS


def _defender_below_half(attacker: Pokemon, defender: Pokemon, state: BattleState) -> bool:
    return defender.live_stats.HP * 2 <= defender.stat_totals.HP


def _defender_already_acted(attacker: Pokemon, defender: Pokemon, state: BattleState) -> bool:
    """Payback: doubled for going second, which the target having already moved is the record of."""
    _, defender_side = _sides_of(attacker, defender, state)
    return defender_side.acted_this_turn


def _defender_already_damaged(attacker: Pokemon, defender: Pokemon, state: BattleState) -> bool:
    """Assurance: doubled if something has already hurt the target this turn."""
    return defender.last_hit_taken > 0


def _defender_is_fleeing(attacker: Pokemon, defender: Pokemon, state: BattleState) -> bool:
    """Pursuit: doubled against a target caught on its way out.

    `mechanics.priority` is what makes the catch happen at all, by sorting Pursuit ahead of the
    switch it would otherwise arrive too late for; this is the other half, the reward for reading it.
    """
    _, defender_side = _sides_of(attacker, defender, state)
    chosen = defender_side.chosen_action
    if defender_side.acted_this_turn or chosen is None:
        return False
    return chosen.action is ActionType.SWITCH_OUT


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
    # Added 2026-09-08: all of these fired at base power whatever their condition said, which on a
    # move that is only worth a slot when it doubles is most of the move missing.
    "Revenge": (_hit_by_target_this_turn, 2, 1),
    "Payback": (_defender_already_acted, 2, 1),
    "Assurance": (_defender_already_damaged, 2, 1),
    "Pursuit": (_defender_is_fleeing, 2, 1),
    "Brine": (_defender_below_half, 2, 1),
    "Venoshock": (_defender_poisoned, 2, 1),
    "Wake-Up Slap": (_defender_asleep, 2, 1),
    "Smelling Salts": (_defender_paralysed, 2, 1),
    "Weather Ball": (_weather_active, 2, 1),
    "Rising Voltage": (_electric_terrain_grounded_target, 2, 1),
    "Knock Off": (_defender_has_item, 3, 2),
    "Expanding Force": (_psychic_terrain_grounded_user, 3, 2),
    "Psyblade": (_electric_terrain_grounded_user, 3, 2),
    "Hydro Steam": (_in_sun, 3, 2),
}
_SE_BONUS_MOVES = frozenset({"Electro Drift", "Collision Course"})  # 5461/4096 when super effective

_WEATHER_BALL_TYPES: dict[Weather, Type] = {
    Weather.SUN: Type.FIRE,
    Weather.HARSH_SUN: Type.FIRE,
    Weather.RAIN: Type.WATER,
    Weather.HEAVY_RAIN: Type.WATER,
    Weather.SANDSTORM: Type.ROCK,
    Weather.SNOW: Type.ICE,
}
_OGERPON_CUDGEL_TYPES: dict[str, Type] = {
    "Ogerpon-Wellspring": Type.WATER,
    "Ogerpon-Hearthflame": Type.FIRE,
    "Ogerpon-Cornerstone": Type.ROCK,
}
_TAUROS_BULL_TYPES: dict[str, Type] = {
    "Tauros-Paldea-Combat": Type.FIGHTING,
    "Tauros-Paldea-Blaze": Type.FIRE,
    "Tauros-Paldea-Aqua": Type.WATER,
}
_JUDGMENT_PLATE_TYPES: dict[Item, Type] = {
    Item.FIST_PLATE: Type.FIGHTING,
    Item.SKY_PLATE: Type.FLYING,
    Item.TOXIC_PLATE: Type.POISON,
    Item.EARTH_PLATE: Type.GROUND,
    Item.STONE_PLATE: Type.ROCK,
    Item.INSECT_PLATE: Type.BUG,
    Item.SPOOKY_PLATE: Type.GHOST,
    Item.IRON_PLATE: Type.STEEL,
    Item.FLAME_PLATE: Type.FIRE,
    Item.SPLASH_PLATE: Type.WATER,
    Item.MEADOW_PLATE: Type.GRASS,
    Item.ZAP_PLATE: Type.ELECTRIC,
    Item.MIND_PLATE: Type.PSYCHIC,
    Item.ICICLE_PLATE: Type.ICE,
    Item.DRACO_PLATE: Type.DRAGON,
    Item.DREAD_PLATE: Type.DARK,
    Item.PIXIE_PLATE: Type.FAIRY,
}
_MULTI_ATTACK_MEMORY_TYPES: dict[Item, Type] = {
    Item.BUG_MEMORY: Type.BUG,
    Item.DARK_MEMORY: Type.DARK,
    Item.DRAGON_MEMORY: Type.DRAGON,
    Item.ELECTRIC_MEMORY: Type.ELECTRIC,
    Item.FAIRY_MEMORY: Type.FAIRY,
    Item.FIGHTING_MEMORY: Type.FIGHTING,
    Item.FIRE_MEMORY: Type.FIRE,
    Item.FLYING_MEMORY: Type.FLYING,
    Item.GHOST_MEMORY: Type.GHOST,
    Item.GRASS_MEMORY: Type.GRASS,
    Item.GROUND_MEMORY: Type.GROUND,
    Item.ICE_MEMORY: Type.ICE,
    Item.POISON_MEMORY: Type.POISON,
    Item.PSYCHIC_MEMORY: Type.PSYCHIC,
    Item.ROCK_MEMORY: Type.ROCK,
    Item.STEEL_MEMORY: Type.STEEL,
    Item.WATER_MEMORY: Type.WATER,
}
_TECHNO_BLAST_DRIVE_TYPES: dict[Item, Type] = {
    Item.DOUSE_DRIVE: Type.WATER,
    Item.SHOCK_DRIVE: Type.ELECTRIC,
    Item.BURN_DRIVE: Type.FIRE,
    Item.CHILL_DRIVE: Type.ICE,
}
# The "-ate" family: a Normal-type move becomes this type and gains a 4915/4096 (~1.2x, Gen 7+)
# power boost. Liquid Voice also changes a move's type (sound moves -> Water) but grants no boost,
# so it is not in this table; `move_type_override` still handles it separately below.
_NORMAL_TYPE_ABILITIES: dict[Ability, Type] = {
    Ability.AERILATE: Type.FLYING,
    Ability.PIXILATE: Type.FAIRY,
    Ability.REFRIGERATE: Type.ICE,
    Ability.GALVANIZE: Type.ELECTRIC,
}
_ATE_POWER_MOD_4096 = 4915


def move_type_override(move: Move, attacker: Pokemon, state: BattleState) -> Type | None:
    """The type this move actually resolves as, if something changes it from its listed type."""
    ate_type = _NORMAL_TYPE_ABILITIES.get(attacker.ability)
    if ate_type is not None and move.type is Type.NORMAL:
        return ate_type
    if attacker.ability is Ability.LIQUID_VOICE and move.sound:
        return Type.WATER
    if move.name == "Weather Ball":
        return _WEATHER_BALL_TYPES.get(effective_weather(state))
    if move.name == "Ivy Cudgel":
        return _OGERPON_CUDGEL_TYPES.get(attacker.name, Type.GRASS)
    if move.name == "Raging Bull":
        return _TAUROS_BULL_TYPES.get(attacker.name)
    if move.name == "Judgment":
        return _JUDGMENT_PLATE_TYPES.get(attacker.item)
    if move.name == "Multi-Attack":
        return _MULTI_ATTACK_MEMORY_TYPES.get(attacker.item)
    if move.name == "Techno Blast":
        return _TECHNO_BLAST_DRIVE_TYPES.get(attacker.item)
    return None


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
    # The turn-start commitment moves. Each is priority -3 so the opponent almost always acts first,
    # and what happened in between is the whole move: Focus Punch is lost if anything hit its user,
    # and Shell Trap only goes off if something physical did. Left unconditional, Focus Punch could
    # never be broken and Shell Trap — a 150 BP move that in practice usually does nothing — always
    # landed. `last_hit_taken` is reset at the top of every turn, so it reads this turn only.
    if move.name == "Focus Punch":
        return attacker.last_hit_taken > 0
    if move.name == "Shell Trap":
        return not (attacker.last_hit_taken > 0 and attacker.last_hit_category is Category.PHYSICAL)
    if move.name == "Last Resort":
        return not _every_other_move_used(attacker)
    if move.name == "Belch":
        return attacker.last_consumed_item not in _BERRIES
    return False


def _every_other_move_used(attacker: Pokemon) -> bool:
    """Last Resort's condition, read off the PP its other moves have spent.

    Not quite the real rule — that one tracks which moves have actually been used — but a slot with
    full PP has certainly not been used, so this refuses exactly the cases the real one refuses at
    the start of a battle, which is when Last Resort would otherwise be a free 140 BP move.
    """
    others = [(slot, move) for slot, move in zip(MoveSlot, attacker.moves, strict=True) if move is not None]
    return all(attacker.pp[slot] < move.pp for slot, move in others if move.name != "Last Resort")


def _target_is_about_to_attack(defender: Pokemon, state: BattleState) -> bool:
    defender_side, _ = _sides_of(defender, defender, state)
    if defender_side.acted_this_turn:
        return False
    action = defender_side.chosen_action
    if action is None or action.action is not ActionType.USE_MOVE or action.move is None:
        return False
    chosen = defender.moves[action.move]
    return chosen is not None and any(isinstance(e, (DamageEffect, FixedDamageEffect)) for e in chosen.effects)


# Special moves that are checked against the target's physical Defence. Their listed category is
# right about which of *our* stats attacks; it is wrong about which of theirs defends, and a
# category cannot say one without the other.
_HITS_PHYSICAL_DEFENCE = {"Psyshock", "Psystrike", "Secret Sword"}


def payload_overrides(move: Move, attacker: Pokemon, defender: Pokemon) -> Payload:
    if move.name == "Body Press":
        return {"attack_stat_override": Stats.DEFENCE}
    if move.name in _HITS_PHYSICAL_DEFENCE:
        return {"defense_stat_override": Stats.DEFENCE}
    if move.name in {"Photon Geyser", "Light That Burns the Sky"}:
        # Listed Special, but it uses whichever attacking stat is higher *after* boosts, and swaps
        # the defending stat to match. On a Necrozma-Dusk-Mane -- base 157 Attack against 113 Special
        # Attack, and EV'd physical -- running it as written meant the move did so little that the
        # search declined it in 31 out of 31 opportunities, which also left its Ultranecrozium Z
        # with nothing to fire.
        if attacker.effective_stat(Stats.ATTACK) > attacker.effective_stat(Stats.SP_ATTACK):
            return {"attack_stat_override": Stats.ATTACK, "defense_stat_override": Stats.DEFENCE}
        return {}
    if move.name == "Foul Play":
        return {"use_target_attack": True}
    if move.name == "Facade":
        return {"ignore_burn": True}
    if move.name == "Hydro Steam":
        return {"ignore_weather_drop": True}
    if _ate_boost_applies(move, attacker):
        return {"power_mods_4096": [_ATE_POWER_MOD_4096]}
    return {}


def _ate_boost_applies(move: Move, attacker: Pokemon) -> bool:
    """True once an "-ate" ability has actually converted this move, not merely alongside one.

    `move.type` may already be the converted type by the time this runs (the live path replaces
    it before building the damage payload), so the only reliable "was this Normal-type" check left
    is the move's own listed type in the database — a naturally Flying/Fairy/Ice move used by an
    Aerilate/Pixilate/Refrigerate holder must not get this boost.
    """
    ate_type = _NORMAL_TYPE_ABILITIES.get(attacker.ability)
    return ate_type is not None and move.type is ate_type and get_move(move.name).type is Type.NORMAL


def _sides_of(attacker: Pokemon, defender: Pokemon, state: BattleState) -> tuple[SideState, SideState]:
    """(attacker's side, defender's side); the attacker must be one of the actives."""
    if state.sides[0].active_pokemon is attacker:
        return state.sides[0], state.sides[1]
    return state.sides[1], state.sides[0]
