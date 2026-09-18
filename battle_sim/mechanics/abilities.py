"""One binder per ability, registered via @ability; wired/unwired on switch-in/out by mechanics.effects."""

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from battle_sim.mechanics.battle import effective_weather
from battle_sim.mechanics.effects import EffectOwner
from battle_sim.mechanics.events import (
    Event,
    EventBus,
    EventContext,
    EventPriority,
    HandlerResult,
    Payload,
    ResidualOrder,
)
from battle_sim.mechanics.items import WEATHER_ROCKS, consume_terrain_seed
from battle_sim.mechanics.stages import apply_stage_changes
from battle_sim.models.log_events import (
    AbilityChanged,
    AbilityChipDamage,
    AbilityCopied,
    AbilityHealed,
    AbilityUnchanged,
    AbsorbBlocked,
    AbsorbHealed,
    AvoidedWithLevitate,
    DoesNotAffect,
    Fainted,
    FlashFireAbsorbed,
    FlashFireActivated,
    FormeChanged,
    HazardSet,
    ItemStolen,
    ParadoxActivated,
    StatChangeSource,
    StatStageChanged,
    StatusCleared,
    StatusMoveBlocked,
    SurvivedAtOneHp,
    TerrainSetByAbility,
    TypeChanged,
    WeatherSetByAbility,
)
from battle_sim.models.moves import DamageEffect, Move
from battle_sim.models.pokemon import Pokemon
from battle_sim.models.type_matchups import type_effectiveness
from battle_sim.utils import (
    Ability,
    Category,
    ExtraStatus,
    Hazards,
    Item,
    Stats,
    Status,
    Terrain,
    Type,
    Weather,
    is_untouchable,
)

if TYPE_CHECKING:
    from battle_sim.mechanics.battle import BattleState
    from battle_sim.mechanics.log import BattleLog

type AbilityBinder = Callable[[EventBus, Pokemon, EffectOwner], None]

ABILITY_BINDERS: dict[Ability, AbilityBinder] = {}


def ability(kind: Ability) -> Callable[[AbilityBinder], AbilityBinder]:
    def register(binder: AbilityBinder) -> AbilityBinder:
        ABILITY_BINDERS[kind] = binder
        return binder

    return register


@ability(Ability.ADAPTABILITY)
def _bind_adaptability(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    def boost_stab(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.actor is not pokemon:
            return None
        return HandlerResult(updated_payload={"stab_4096": 8192})

    bus.on(Event.ON_DAMAGE_CALC, boost_stab, priority=EventPriority.ABILITY, owner=owner)


def _bind_double_attack(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    """Huge Power / Pure Power: doubles the Attack stat input on physical hits."""

    def double_attack(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.actor is pokemon and payload["category"] is Category.PHYSICAL:
            payload.setdefault("attack_mods_4096", []).append(8192)
        return None

    bus.on(Event.ON_DAMAGE_CALC, double_attack, priority=EventPriority.ABILITY, owner=owner)


ABILITY_BINDERS[Ability.HUGE_POWER] = _bind_double_attack
ABILITY_BINDERS[Ability.PURE_POWER] = _bind_double_attack


@ability(Ability.GUTS)
def _bind_guts(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    def guts(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.actor is not pokemon:
            return None
        if payload["category"] is Category.PHYSICAL and pokemon.status is not Status.NONE:
            payload.setdefault("attack_mods_4096", []).append(6144)
        return HandlerResult(updated_payload={"ignore_burn": True})

    bus.on(Event.ON_DAMAGE_CALC, guts, priority=EventPriority.ABILITY, owner=owner)


@ability(Ability.HUSTLE)
def _bind_hustle(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    """The accuracy penalty on physical moves lives in engine.moves._accuracy_check."""

    def boost_physical(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.actor is pokemon and payload["category"] is Category.PHYSICAL:
            payload.setdefault("attack_mods_4096", []).append(6144)
        return None

    bus.on(Event.ON_DAMAGE_CALC, boost_physical, priority=EventPriority.ABILITY, owner=owner)


@ability(Ability.FLASH_FIRE)
def _bind_flash_fire(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    def boost_fire(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.actor is pokemon and payload["move_type"] is Type.FIRE and pokemon.flash_fire_active:
            payload.setdefault("pre_screen_mods_4096", []).append(6144)
        return None

    def absorb_fire(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.defender is not pokemon or payload["move_type"] is not Type.FIRE:
            return None
        if not pokemon.flash_fire_active:
            pokemon.flash_fire_active = True
            context.log.add(FlashFireActivated(side=payload["defender_index"], pokemon=pokemon.nickname))
        else:
            context.log.add(FlashFireAbsorbed(side=payload["defender_index"], pokemon=pokemon.nickname))
        return HandlerResult(cancel=True, updated_payload={"absorbed": True, "immune_reason": "flash_fire"})

    bus.on(Event.ON_DAMAGE_CALC, boost_fire, priority=EventPriority.ABILITY, owner=owner)
    bus.on(Event.ON_BEFORE_MOVE, absorb_fire, priority=EventPriority.ABILITY, owner=owner)


@ability(Ability.LEVITATE)
def _bind_levitate(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    def avoid_ground(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.defender is not pokemon or payload["move_type"] is not Type.GROUND:
            return None
        context.log.add(AvoidedWithLevitate(side=payload["defender_index"], pokemon=pokemon.nickname))
        return HandlerResult(cancel=True, updated_payload={"absorbed": True, "immune_reason": "levitate"})

    bus.on(Event.ON_BEFORE_MOVE, avoid_ground, priority=EventPriority.ABILITY, owner=owner)


@ability(Ability.WONDER_GUARD)
def _bind_wonder_guard(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    def block_non_super_effective(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.defender is not pokemon or type_effectiveness(payload["move_type"], pokemon.types) >= 2:
            return None
        context.log.add(DoesNotAffect(side=payload["defender_index"], pokemon=pokemon.nickname))
        return HandlerResult(cancel=True, updated_payload={"absorbed": True, "immune_reason": "wonder_guard"})

    bus.on(Event.ON_BEFORE_MOVE, block_non_super_effective, priority=EventPriority.ABILITY, owner=owner)


@ability(Ability.SOUNDPROOF)
def _bind_soundproof(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    def block_sound(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.defender is not pokemon or context.move is None or not context.move.sound:
            return None
        context.log.add(DoesNotAffect(side=payload["defender_index"], pokemon=pokemon.nickname))
        return HandlerResult(cancel=True, updated_payload={"absorbed": True, "immune_reason": "soundproof"})

    bus.on(Event.ON_BEFORE_MOVE, block_sound, priority=EventPriority.ABILITY, owner=owner)


def _bind_type_absorb(kind: Ability, absorbed_type: Type) -> AbilityBinder:
    def binder(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
        def absorb(context: EventContext, payload: Payload) -> HandlerResult | None:
            if context.defender is not pokemon or payload["move_type"] is not absorbed_type:
                return None
            healed = pokemon.apply_healing(max(1, pokemon.stat_totals.HP // 4))
            if healed > 0:
                context.log.add(
                    AbsorbHealed(side=payload["defender_index"], pokemon=pokemon.nickname, ability=kind, amount=healed)
                )
            else:
                context.log.add(AbsorbBlocked(side=payload["defender_index"], pokemon=pokemon.nickname, ability=kind))
            return HandlerResult(cancel=True, updated_payload={"absorbed": True, "immune_reason": kind.name.lower()})

        bus.on(Event.ON_BEFORE_MOVE, absorb, priority=EventPriority.ABILITY, owner=owner)

    return binder


ABILITY_BINDERS[Ability.VOLT_ABSORB] = _bind_type_absorb(Ability.VOLT_ABSORB, Type.ELECTRIC)
ABILITY_BINDERS[Ability.WATER_ABSORB] = _bind_type_absorb(Ability.WATER_ABSORB, Type.WATER)


@ability(Ability.MOTOR_DRIVE)
def _bind_motor_drive(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    def absorb_electric(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.defender is not pokemon or payload["move_type"] is not Type.ELECTRIC:
            return None
        delta = pokemon.change_stat_stage(Stats.SPEED, 1)
        if delta > 0:
            context.log.add(
                StatStageChanged(
                    side=payload["defender_index"],
                    pokemon=pokemon.nickname,
                    stat=Stats.SPEED,
                    delta=delta,
                    requested=1,
                    source="motor_drive",
                )
            )
        return HandlerResult(cancel=True, updated_payload={"absorbed": True, "immune_reason": "motor_drive"})

    bus.on(Event.ON_BEFORE_MOVE, absorb_electric, priority=EventPriority.ABILITY, owner=owner)


@ability(Ability.SOLAR_POWER)
def _bind_solar_power(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    def boost_special(context: EventContext, payload: Payload) -> HandlerResult | None:
        in_sun = effective_weather(context.battle) in (Weather.SUN, Weather.HARSH_SUN)
        if context.actor is pokemon and in_sun and payload["category"] is Category.SPECIAL:
            payload.setdefault("attack_mods_4096", []).append(6144)
        return None

    def sun_chip(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.actor is not pokemon or payload.get("magic_guard", False):
            return None
        if effective_weather(context.battle) not in (Weather.SUN, Weather.HARSH_SUN):
            return None
        dealt = pokemon.apply_damage(max(1, pokemon.stat_totals.HP // 8))
        context.log.add(
            AbilityChipDamage(
                side=payload["side_index"], pokemon=pokemon.nickname, ability=Ability.SOLAR_POWER, amount=dealt
            )
        )
        return None

    bus.on(Event.ON_DAMAGE_CALC, boost_special, priority=EventPriority.ABILITY, owner=owner)
    bus.on(Event.ON_RESIDUAL, sun_chip, priority=ResidualOrder.WEATHER, owner=owner)


def _bind_ko_boost(stat: Stats, source: StatChangeSource) -> AbilityBinder:
    """Moxie-likes: +1 to a stat after KOing with an attack."""

    def binder(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
        def on_ko(context: EventContext, payload: Payload) -> HandlerResult | None:
            if context.actor is not pokemon or pokemon.is_fainted():
                return None
            delta = pokemon.change_stat_stage(stat, 1)
            if delta > 0:
                context.log.add(
                    StatStageChanged(
                        side=payload["attacker_index"],
                        pokemon=pokemon.nickname,
                        stat=stat,
                        delta=delta,
                        requested=1,
                        source=source,
                    )
                )
            return None

        bus.on(Event.ON_FAINT, on_ko, priority=EventPriority.ABILITY, owner=owner)

    return binder


ABILITY_BINDERS[Ability.MOXIE] = _bind_ko_boost(Stats.ATTACK, "moxie")
ABILITY_BINDERS[Ability.CHILLING_NEIGH] = _bind_ko_boost(Stats.ATTACK, "chilling_neigh")
ABILITY_BINDERS[Ability.AS_ONE_GLASTRIER] = _bind_ko_boost(Stats.ATTACK, "as_one_glastrier")
ABILITY_BINDERS[Ability.SOUL_HEART] = _bind_ko_boost(Stats.SP_ATTACK, "soul_heart")

_BEAST_BOOST_STATS = (Stats.ATTACK, Stats.DEFENCE, Stats.SP_ATTACK, Stats.SP_DEFENCE, Stats.SPEED)


@ability(Ability.BEAST_BOOST)
def _bind_beast_boost(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    """+1 to whichever of its stats is highest after KOing with an attack (not always Attack)."""

    def on_ko(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.actor is not pokemon or pokemon.is_fainted():
            return None
        best = max(_BEAST_BOOST_STATS, key=lambda stat: pokemon.stat_totals[stat])
        delta = pokemon.change_stat_stage(best, 1)
        if delta > 0:
            context.log.add(
                StatStageChanged(
                    side=payload["attacker_index"],
                    pokemon=pokemon.nickname,
                    stat=best,
                    delta=delta,
                    requested=1,
                    source="beast_boost",
                )
            )
        return None

    bus.on(Event.ON_FAINT, on_ko, priority=EventPriority.ABILITY, owner=owner)


@ability(Ability.ROUGH_SKIN)
def _bind_rough_skin(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    def spike_back(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.defender is not pokemon or not payload["contact"]:
            return None
        attacker = context.actor
        assert attacker is not None  # hit events always carry the attacker
        if attacker.is_fainted() or attacker.ability is Ability.MAGIC_GUARD:
            return None
        chip = attacker.apply_damage(max(1, attacker.stat_totals.HP // 8))
        context.log.add(
            AbilityChipDamage(
                side=payload["attacker_index"], pokemon=attacker.nickname, ability=Ability.ROUGH_SKIN, amount=chip
            )
        )
        return None

    bus.on(Event.ON_AFTER_HIT, spike_back, priority=EventPriority.ABILITY, owner=owner)


@ability(Ability.MAGIC_GUARD)
def _bind_magic_guard(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    """Magic Guard prevents all indirect damage: residual chips, hazards, Life Orb.

    (Move recoil suppression is part of the core recoil rule in damage_apply.)
    """

    def flag(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.actor is not pokemon:
            return None
        return HandlerResult(updated_payload={"magic_guard": True})

    def block_hazards(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.actor is not pokemon:
            return None
        return HandlerResult(cancel=True, updated_payload={"hazards_blocked": True})

    bus.on(Event.ON_RESIDUAL, flag, priority=ResidualOrder.MAGIC_GUARD, owner=owner)
    bus.on(Event.ON_ACTION_RESOLVE, flag, priority=EventPriority.SYSTEM, owner=owner)
    bus.on(Event.ON_ENTRY_HAZARD, block_hazards, priority=EventPriority.ABILITY, owner=owner)


@ability(Ability.STURDY)
def _bind_sturdy(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    def endure(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.defender is not pokemon:
            return None
        max_hp = pokemon.stat_totals.HP
        if max_hp == pokemon.live_stats.HP and payload["damage"] >= max_hp:
            context.log.add(SurvivedAtOneHp(side=payload["defender_index"], pokemon=pokemon.nickname, cause="sturdy"))
            return HandlerResult(updated_payload={"damage": max_hp - 1})
        return None

    bus.on(Event.ON_BEFORE_HIT, endure, priority=EventPriority.ABILITY, owner=owner)


def _set_weather_from_ability(kind: Ability, weather: Weather, pokemon: Pokemon, context: EventContext) -> None:
    field = context.battle.field
    if field.weather is weather:
        return
    field.weather = weather
    field.weather_turns_left = 8 if pokemon.item is WEATHER_ROCKS.get(weather) else 5
    side_index = next(i for i, s in enumerate(context.battle.sides) if s.active_pokemon is pokemon)
    context.log.add(WeatherSetByAbility(side=side_index, pokemon=pokemon.nickname, ability=kind))


def _set_terrain_from_ability(kind: Ability, terrain: Terrain, pokemon: Pokemon, context: EventContext) -> None:
    field = context.battle.field
    if field.terrain is terrain:
        return
    field.terrain = terrain
    field.terrain_turns_left = 8 if pokemon.item is Item.TERRAIN_EXTENDER else 5
    side_index = next(i for i, s in enumerate(context.battle.sides) if s.active_pokemon is pokemon)
    context.log.add(TerrainSetByAbility(side=side_index, pokemon=pokemon.nickname, ability=kind))
    for seed_side_index, side in enumerate(context.battle.sides):
        consume_terrain_seed(side.active_pokemon, seed_side_index, terrain, context.log)


def _bind_weather_setter(kind: Ability, weather: Weather) -> AbilityBinder:
    def binder(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
        def set_weather(context: EventContext, payload: Payload) -> HandlerResult | None:
            if context.actor is pokemon:
                _set_weather_from_ability(kind, weather, pokemon, context)
            return None

        bus.on(Event.ON_SWITCH_IN, set_weather, priority=EventPriority.ABILITY, owner=owner)

    return binder


# The weather each ability brings with it. Kept as a table rather than only as binder registrations
# because a Pokemon can *gain* one of these mid-turn: Primal Reversion and Mega Evolution swap the
# ability in after the switch-in event has already gone by, so the engine has to be able to ask
# "what weather does this ability want?" outside of a switch. Until it could, Primal Groudon set
# ordinary sun rather than Desolate Land's, and Mega Rayquaza set nothing at all.
ABILITY_WEATHER: dict[Ability, Weather] = {
    Ability.DROUGHT: Weather.SUN,
    Ability.DRIZZLE: Weather.RAIN,
    Ability.SAND_STREAM: Weather.SANDSTORM,
    Ability.SNOW_WARNING: Weather.SNOW,
    Ability.PRIMORDIAL_SEA: Weather.HEAVY_RAIN,
    Ability.DESOLATE_LAND: Weather.HARSH_SUN,
    Ability.DELTA_STREAM: Weather.STRONG_WINDS,
}
for _ability, _weather in ABILITY_WEATHER.items():
    ABILITY_BINDERS[_ability] = _bind_weather_setter(_ability, _weather)
# Primal Reversion weather. The real pair also cannot be overridden by ordinary weather moves;
# that lock is not modelled, so a later Rain Dance still displaces harsh sun.
# Mega Rayquaza's, and the third of the "primal" weathers: it cannot be overwritten by an ordinary
# setter, exactly as the other two cannot.


def _bind_terrain_setter(kind: Ability, terrain: Terrain) -> AbilityBinder:
    def binder(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
        def set_terrain(context: EventContext, payload: Payload) -> HandlerResult | None:
            if context.actor is pokemon:
                _set_terrain_from_ability(kind, terrain, pokemon, context)
            return None

        bus.on(Event.ON_SWITCH_IN, set_terrain, priority=EventPriority.ABILITY, owner=owner)

    return binder


# Paired with ABILITY_WEATHER so a scorer can ask what a Pokemon puts on the field without having
# to run the switch-in that does it.
ABILITY_TERRAIN: dict[Ability, Terrain] = {
    Ability.GRASSY_SURGE: Terrain.GRASSY,
    Ability.ELECTRIC_SURGE: Terrain.ELECTRIC,
    Ability.PSYCHIC_SURGE: Terrain.PSYCHIC,
    Ability.MISTY_SURGE: Terrain.MISTY,
}
for _ability, _terrain in ABILITY_TERRAIN.items():
    ABILITY_BINDERS[_ability] = _bind_terrain_setter(_ability, _terrain)


@ability(Ability.INTIMIDATE)
def _bind_intimidate(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    def intimidate(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.actor is not pokemon:
            return None
        side_index = payload["side_index"]
        opponent = context.battle.sides[1 - side_index].active_pokemon
        if opponent.is_fainted() or ExtraStatus.SUBSTITUTE in opponent.volatiles:  # a sub blocks Intimidate (gen 8+)
            return None
        apply_stage_changes(
            opponent,
            1 - side_index,
            {Stats.ATTACK: -1},
            context.log,
            inflicted_by_opponent=True,
            source="intimidate",
            inflictor=pokemon,
        )
        return None

    bus.on(Event.ON_SWITCH_IN, intimidate, priority=EventPriority.ABILITY, owner=owner)


@ability(Ability.SPEED_BOOST)
def _bind_speed_boost(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    def raise_speed(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.actor is not pokemon or pokemon.is_fainted():
            return None
        if pokemon.just_switched_in:  # not this turn — takes a full turn out to kick in
            return None
        delta = pokemon.change_stat_stage(Stats.SPEED, 1)
        if delta > 0:
            context.log.add(
                StatStageChanged(
                    side=payload["side_index"],
                    pokemon=pokemon.nickname,
                    stat=Stats.SPEED,
                    delta=delta,
                    requested=1,
                    source="speed_boost",
                )
            )
        return None

    bus.on(Event.ON_RESIDUAL, raise_speed, priority=ResidualOrder.SPEED_BOOST, owner=owner)


@ability(Ability.SLOW_START)
def _bind_slow_start(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    """5 turns of halved Attack (here) and halved Speed (mechanics.priority.effective_speed),

    both gated on the ExtraStatus.SLOW_START volatile ticked down in engine.residuals.
    """

    def start(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.actor is pokemon:
            pokemon.volatiles[ExtraStatus.SLOW_START] = 5
        return None

    def halve_attack(context: EventContext, payload: Payload) -> HandlerResult | None:
        if (
            context.actor is pokemon
            and payload["category"] is Category.PHYSICAL
            and ExtraStatus.SLOW_START in pokemon.volatiles
        ):
            payload.setdefault("attack_mods_4096", []).append(2048)
        return None

    bus.on(Event.ON_SWITCH_IN, start, priority=EventPriority.ABILITY, owner=owner)
    bus.on(Event.ON_DAMAGE_CALC, halve_attack, priority=EventPriority.ABILITY, owner=owner)


# -- Damage-calc modifiers -----------------------------------------------------


@ability(Ability.MULTISCALE)
def _bind_multiscale(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    def halve_at_full_hp(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.defender is pokemon and pokemon.live_stats.HP == pokemon.stat_totals.HP:
            payload.setdefault("final_mods_4096", []).append(2048)
        return None

    bus.on(Event.ON_DAMAGE_CALC, halve_at_full_hp, priority=EventPriority.ABILITY, owner=owner)


ABILITY_BINDERS[Ability.SHADOW_SHIELD] = _bind_multiscale  # Lunala's name for the identical mechanic


def _bind_ruin_offense(category: Category) -> AbilityBinder:
    """Tablets/Vessel of Ruin: every other active attacks with 0.75x Atk/SpA."""

    def binder(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
        def weaken(context: EventContext, payload: Payload) -> HandlerResult | None:
            if context.actor is not pokemon and payload["category"] is category:
                payload.setdefault("attack_mods_4096", []).append(3072)
            return None

        bus.on(Event.ON_DAMAGE_CALC, weaken, priority=EventPriority.ABILITY, owner=owner)

    return binder


def _bind_ruin_defense(category: Category) -> AbilityBinder:
    """Sword/Beads of Ruin: every other active defends with 0.75x Def/SpD."""

    def binder(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
        def weaken(context: EventContext, payload: Payload) -> HandlerResult | None:
            if context.defender is not pokemon and payload["category"] is category:
                payload.setdefault("defense_mods_4096", []).append(3072)
            return None

        bus.on(Event.ON_DAMAGE_CALC, weaken, priority=EventPriority.ABILITY, owner=owner)

    return binder


ABILITY_BINDERS[Ability.TABLETS_OF_RUIN] = _bind_ruin_offense(Category.PHYSICAL)
ABILITY_BINDERS[Ability.VESSEL_OF_RUIN] = _bind_ruin_offense(Category.SPECIAL)
ABILITY_BINDERS[Ability.SWORD_OF_RUIN] = _bind_ruin_defense(Category.PHYSICAL)
ABILITY_BINDERS[Ability.BEADS_OF_RUIN] = _bind_ruin_defense(Category.SPECIAL)


def _bind_aura(aura_type: Type) -> AbilityBinder:
    """Fairy/Dark Aura: every use of the matching type, by either side, hits 1.33x harder —

    or 0.75x softer field-wide if anyone active has Aura Break. (Aura Break has no binder of
    its own; it's a passive flag only ever read from here.)
    """

    def binder(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
        def modify(context: EventContext, payload: Payload) -> HandlerResult | None:
            if payload["move_type"] is not aura_type:
                return None
            broken = any(side.active_pokemon.ability is Ability.AURA_BREAK for side in context.battle.sides)
            payload.setdefault("power_mods_4096", []).append(3072 if broken else 5461)
            return None

        bus.on(Event.ON_DAMAGE_CALC, modify, priority=EventPriority.ABILITY, owner=owner)

    return binder


ABILITY_BINDERS[Ability.FAIRY_AURA] = _bind_aura(Type.FAIRY)
ABILITY_BINDERS[Ability.DARK_AURA] = _bind_aura(Type.DARK)


@ability(Ability.SHARPNESS)
def _bind_sharpness(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    def boost_slicing(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.actor is pokemon and context.move is not None and context.move.slicing:
            payload.setdefault("power_mods_4096", []).append(6144)
        return None

    bus.on(Event.ON_DAMAGE_CALC, boost_slicing, priority=EventPriority.ABILITY, owner=owner)


@ability(Ability.TECHNICIAN)
def _bind_technician(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    def boost_weak_moves(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.actor is not pokemon or context.move is None:
            return None
        damage_effect = next((e for e in context.move.effects if isinstance(e, DamageEffect)), None)
        if damage_effect is not None and damage_effect.power is not None and damage_effect.power <= 60:
            payload.setdefault("power_mods_4096", []).append(6144)
        return None

    bus.on(Event.ON_DAMAGE_CALC, boost_weak_moves, priority=EventPriority.ABILITY, owner=owner)


@ability(Ability.RECKLESS)
def _bind_reckless(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    def boost_recoil_moves(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.actor is not pokemon or context.move is None:
            return None
        damage_effect = next((e for e in context.move.effects if isinstance(e, DamageEffect)), None)
        if damage_effect is not None and damage_effect.recoil_percent is not None:
            payload.setdefault("power_mods_4096", []).append(4915)
        return None

    bus.on(Event.ON_DAMAGE_CALC, boost_recoil_moves, priority=EventPriority.ABILITY, owner=owner)


def _bind_pinch_boost(move_type: Type) -> AbilityBinder:
    """Blaze/Torrent/Overgrow/Swarm: 1.5x power on the matching type at or below 1/3 HP."""

    def binder(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
        def boost(context: EventContext, payload: Payload) -> HandlerResult | None:
            in_pinch = 3 * pokemon.live_stats.HP <= pokemon.stat_totals.HP
            if context.actor is pokemon and payload["move_type"] is move_type and in_pinch:
                payload.setdefault("power_mods_4096", []).append(6144)
            return None

        bus.on(Event.ON_DAMAGE_CALC, boost, priority=EventPriority.ABILITY, owner=owner)

    return binder


ABILITY_BINDERS[Ability.BLAZE] = _bind_pinch_boost(Type.FIRE)
ABILITY_BINDERS[Ability.TORRENT] = _bind_pinch_boost(Type.WATER)
ABILITY_BINDERS[Ability.OVERGROW] = _bind_pinch_boost(Type.GRASS)
ABILITY_BINDERS[Ability.SWARM] = _bind_pinch_boost(Type.BUG)

_OVERLORD_POWER_4096 = (4096, 4506, 4915, 5325, 5734, 6144)  # PS powMod: +10% per fallen teammate


@ability(Ability.SUPREME_OVERLORD)
def _bind_supreme_overlord(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    def boost_per_fallen(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.actor is not pokemon:
            return None
        side = context.battle.sides[payload["attacker_index"]]
        fallen = min(5, sum(1 for member in side.team if member.is_fainted()))
        if fallen:
            payload.setdefault("power_mods_4096", []).append(_OVERLORD_POWER_4096[fallen])
        return None

    bus.on(Event.ON_DAMAGE_CALC, boost_per_fallen, priority=EventPriority.ABILITY, owner=owner)


def _bind_super_effective_reduction(kind: Ability) -> AbilityBinder:
    """Prism Armor / Filter: super-effective hits deal 0.75x."""

    def binder(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
        def reduce(context: EventContext, payload: Payload) -> HandlerResult | None:
            if context.defender is pokemon and type_effectiveness(payload["move_type"], pokemon.types) >= 2:
                payload.setdefault("final_mods_4096", []).append(3072)
            return None

        bus.on(Event.ON_DAMAGE_CALC, reduce, priority=EventPriority.ABILITY, owner=owner)

    return binder


ABILITY_BINDERS[Ability.PRISM_ARMOR] = _bind_super_effective_reduction(Ability.PRISM_ARMOR)
ABILITY_BINDERS[Ability.FILTER] = _bind_super_effective_reduction(Ability.FILTER)


@ability(Ability.TINTED_LENS)
def _bind_tinted_lens(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    def double_resisted(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.actor is not pokemon or context.defender is None:
            return None
        effectiveness = type_effectiveness(payload["move_type"], context.defender.types)
        if 0 < effectiveness < 1:
            payload.setdefault("final_mods_4096", []).append(8192)
        return None

    bus.on(Event.ON_DAMAGE_CALC, double_resisted, priority=EventPriority.ABILITY, owner=owner)


@ability(Ability.NEUROFORCE)
def _bind_neuroforce(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    def boost_super_effective(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.actor is not pokemon or context.defender is None:
            return None
        effectiveness = type_effectiveness(payload["move_type"], context.defender.types)
        if effectiveness > 1:
            payload.setdefault("final_mods_4096", []).append(5120)  # 1.25x
        return None

    bus.on(Event.ON_DAMAGE_CALC, boost_super_effective, priority=EventPriority.ABILITY, owner=owner)


def _bind_incoming_type_weakening(types: frozenset[Type]) -> AbilityBinder:
    """Heatproof / Thick Fat: the listed attacking types hit with halved offense."""

    def binder(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
        def weaken(context: EventContext, payload: Payload) -> HandlerResult | None:
            if context.defender is pokemon and payload["move_type"] in types:
                payload.setdefault("attack_mods_4096", []).append(2048)
            return None

        bus.on(Event.ON_DAMAGE_CALC, weaken, priority=EventPriority.ABILITY, owner=owner)

    return binder


ABILITY_BINDERS[Ability.HEATPROOF] = _bind_incoming_type_weakening(frozenset({Type.FIRE}))
ABILITY_BINDERS[Ability.THICK_FAT] = _bind_incoming_type_weakening(frozenset({Type.FIRE, Type.ICE}))
# Iron Barbs is Rough Skin under another name — same trigger, same fraction, different flavour text.
ABILITY_BINDERS[Ability.IRON_BARBS] = _bind_rough_skin


@ability(Ability.DEFEATIST)
def _bind_defeatist(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    """Halved offense at or below half HP. It is the whole reason Archeops is not a top-tier
    attacker, so leaving it out rated the drawback as though it were free."""

    def weaken(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.actor is pokemon and pokemon.live_stats.HP * 2 <= pokemon.stat_totals.HP:
            payload.setdefault("attack_mods_4096", []).append(2048)
        return None

    bus.on(Event.ON_DAMAGE_CALC, weaken, priority=EventPriority.ABILITY, owner=owner)


@ability(Ability.MARVEL_SCALE)
def _bind_marvel_scale(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    """1.5x Defence while statused — Milotic's reason to accept a burn."""

    def toughen(context: EventContext, payload: Payload) -> HandlerResult | None:
        if (
            context.defender is pokemon
            and pokemon.status is not Status.NONE
            and payload["category"] is Category.PHYSICAL
        ):
            payload.setdefault("defense_mods_4096", []).append(6144)
        return None

    bus.on(Event.ON_DAMAGE_CALC, toughen, priority=EventPriority.ABILITY, owner=owner)


@ability(Ability.FUR_COAT)
def _bind_fur_coat(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    """Physical hits land against doubled Defence."""

    def toughen(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.defender is pokemon and payload["category"] is Category.PHYSICAL:
            payload.setdefault("defense_mods_4096", []).append(8192)
        return None

    bus.on(Event.ON_DAMAGE_CALC, toughen, priority=EventPriority.ABILITY, owner=owner)


@ability(Ability.STEELWORKER)
def _bind_steelworker(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    """1.5x on Steel moves, the way STAB would if Dhelmise's Steel were one of its own types."""

    def boost(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.actor is pokemon and payload["move_type"] is Type.STEEL:
            payload.setdefault("attack_mods_4096", []).append(6144)
        return None

    bus.on(Event.ON_DAMAGE_CALC, boost, priority=EventPriority.ABILITY, owner=owner)


@ability(Ability.STAKEOUT)
def _bind_stakeout(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    """Double damage against something that came in this turn — punishing the switch is the point,
    so it reads the same "did you just arrive" flag Speed Boost sits out for."""

    def punish_the_switch(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.actor is pokemon and context.defender is not None and context.defender.just_switched_in:
            payload.setdefault("attack_mods_4096", []).append(8192)
        return None

    bus.on(Event.ON_DAMAGE_CALC, punish_the_switch, priority=EventPriority.ABILITY, owner=owner)


@ability(Ability.WATER_BUBBLE)
def _bind_water_bubble(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    def modify(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.actor is pokemon and payload["move_type"] is Type.WATER:
            payload.setdefault("attack_mods_4096", []).append(8192)
        if context.defender is pokemon and payload["move_type"] is Type.FIRE:
            payload.setdefault("attack_mods_4096", []).append(2048)
        return None

    bus.on(Event.ON_DAMAGE_CALC, modify, priority=EventPriority.ABILITY, owner=owner)


@ability(Ability.ICE_SCALES)
def _bind_ice_scales(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    def halve_special(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.defender is pokemon and payload["category"] is Category.SPECIAL:
            payload.setdefault("final_mods_4096", []).append(2048)
        return None

    bus.on(Event.ON_DAMAGE_CALC, halve_special, priority=EventPriority.ABILITY, owner=owner)


def _bind_type_attack_boost(move_type: Type) -> AbilityBinder:
    """Dragon's Maw-likes: 1.5x offense on the matching type."""

    def binder(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
        def boost(context: EventContext, payload: Payload) -> HandlerResult | None:
            if context.actor is pokemon and payload["move_type"] is move_type:
                payload.setdefault("attack_mods_4096", []).append(6144)
            return None

        bus.on(Event.ON_DAMAGE_CALC, boost, priority=EventPriority.ABILITY, owner=owner)

    return binder


ABILITY_BINDERS[Ability.DRAGONS_MAW] = _bind_type_attack_boost(Type.DRAGON)


# -- Switch-in abilities -------------------------------------------------------


@ability(Ability.ORICHALCUM_PULSE)
def _bind_orichalcum_pulse(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    def set_sun(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.actor is pokemon:
            _set_weather_from_ability(Ability.ORICHALCUM_PULSE, Weather.SUN, pokemon, context)
        return None

    def boost_in_sun(context: EventContext, payload: Payload) -> HandlerResult | None:
        in_sun = effective_weather(context.battle) in (Weather.SUN, Weather.HARSH_SUN)
        if context.actor is pokemon and payload["category"] is Category.PHYSICAL and in_sun:
            payload.setdefault("attack_mods_4096", []).append(5461)
        return None

    bus.on(Event.ON_SWITCH_IN, set_sun, priority=EventPriority.ABILITY, owner=owner)
    bus.on(Event.ON_DAMAGE_CALC, boost_in_sun, priority=EventPriority.ABILITY, owner=owner)


@ability(Ability.HADRON_ENGINE)
def _bind_hadron_engine(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    def set_terrain(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.actor is pokemon:
            _set_terrain_from_ability(Ability.HADRON_ENGINE, Terrain.ELECTRIC, pokemon, context)
        return None

    def boost_on_terrain(context: EventContext, payload: Payload) -> HandlerResult | None:
        on_terrain = context.battle.field.terrain is Terrain.ELECTRIC
        if context.actor is pokemon and payload["category"] is Category.SPECIAL and on_terrain:
            payload.setdefault("attack_mods_4096", []).append(5461)
        return None

    bus.on(Event.ON_SWITCH_IN, set_terrain, priority=EventPriority.ABILITY, owner=owner)
    bus.on(Event.ON_DAMAGE_CALC, boost_on_terrain, priority=EventPriority.ABILITY, owner=owner)


def _bind_once_switch_in_boost(stat: Stats, source: StatChangeSource) -> AbilityBinder:
    """Dauntless Shield / Intrepid Sword: +1 on switch-in, once per battle."""

    def binder(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
        def boost_once(context: EventContext, payload: Payload) -> HandlerResult | None:
            if context.actor is not pokemon or pokemon.switch_in_boost_used:
                return None
            pokemon.switch_in_boost_used = True
            apply_stage_changes(
                pokemon, payload["side_index"], {stat: 1}, context.log, inflicted_by_opponent=False, source=source
            )
            return None

        bus.on(Event.ON_SWITCH_IN, boost_once, priority=EventPriority.ABILITY, owner=owner)

    return binder


ABILITY_BINDERS[Ability.DAUNTLESS_SHIELD] = _bind_once_switch_in_boost(Stats.DEFENCE, "dauntless_shield")
ABILITY_BINDERS[Ability.INTREPID_SWORD] = _bind_once_switch_in_boost(Stats.ATTACK, "intrepid_sword")


@ability(Ability.DOWNLOAD)
def _bind_download(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    def analyse(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.actor is not pokemon:
            return None
        side_index = payload["side_index"]
        opponent = context.battle.sides[1 - side_index].active_pokemon
        if opponent.is_fainted():
            return None
        totals = opponent.stat_totals
        stat = Stats.SP_ATTACK if totals.DEFENCE >= totals.SP_DEFENCE else Stats.ATTACK
        apply_stage_changes(pokemon, side_index, {stat: 1}, context.log, inflicted_by_opponent=False, source="download")
        return None

    bus.on(Event.ON_SWITCH_IN, analyse, priority=EventPriority.ABILITY, owner=owner)


# -- Paradox abilities ----------------------------------------------------------

_PARADOX_STATS = (Stats.ATTACK, Stats.DEFENCE, Stats.SP_ATTACK, Stats.SP_DEFENCE, Stats.SPEED)
_SAND_FORCE_TYPES = (Type.ROCK, Type.GROUND, Type.STEEL)


def _has_multiple_hits(move: Move) -> bool:
    return any(isinstance(effect, DamageEffect) and effect.multi_hit is not None for effect in move.effects)


def _best_stat(pokemon: Pokemon) -> Stats:
    return max(_PARADOX_STATS, key=pokemon.effective_stat)


def _sun_active(battle: "BattleState") -> bool:
    return effective_weather(battle) in (Weather.SUN, Weather.HARSH_SUN)


def _electric_terrain_active(battle: "BattleState") -> bool:
    return battle.field.terrain is Terrain.ELECTRIC


def _paradox_activate(kind: Ability, pokemon: Pokemon, context: EventContext, from_booster: bool) -> None:
    pokemon.paradox_boost = _best_stat(pokemon)
    pokemon.paradox_from_booster = from_booster
    side_index = next(i for i, s in enumerate(context.battle.sides) if s.active_pokemon is pokemon)
    context.log.add(
        ParadoxActivated(
            side=side_index,
            pokemon=pokemon.nickname,
            ability=kind,
            stat=pokemon.paradox_boost,
            from_booster=from_booster,
        )
    )


def _paradox_evaluate(
    kind: Ability, energized: Callable[["BattleState"], bool], pokemon: Pokemon, context: EventContext
) -> None:
    if pokemon.is_fainted():
        return
    field_active = energized(context.battle)
    if pokemon.paradox_boost is not None:
        if field_active or pokemon.paradox_from_booster:
            return
        pokemon.paradox_boost = None  # the condition ended; Booster Energy may take over below
    if field_active:
        _paradox_activate(kind, pokemon, context, from_booster=False)
    elif pokemon.item is Item.BOOSTER_ENERGY:
        pokemon.consume_item()
        _paradox_activate(kind, pokemon, context, from_booster=True)


def _paradox_damage_mods(pokemon: Pokemon, context: EventContext, payload: Payload) -> None:
    boost = pokemon.paradox_boost
    if boost is None:
        return
    category = payload["category"]
    offensive = (boost is Stats.ATTACK and category is Category.PHYSICAL) or (
        boost is Stats.SP_ATTACK and category is Category.SPECIAL
    )
    defensive = (boost is Stats.DEFENCE and category is Category.PHYSICAL) or (
        boost is Stats.SP_DEFENCE and category is Category.SPECIAL
    )
    if context.actor is pokemon and offensive:
        payload.setdefault("attack_mods_4096", []).append(5325)
    if context.defender is pokemon and defensive:
        payload.setdefault("defense_mods_4096", []).append(5325)


def _bind_paradox(kind: Ability, energized: Callable[["BattleState"], bool]) -> AbilityBinder:
    """Protosynthesis / Quark Drive: the field condition (or a consumed Booster Energy)
    boosts the holder's strongest stat by 1.3x (1.5x speed lives in priority.effective_speed)."""

    def binder(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
        def evaluate(context: EventContext, payload: Payload) -> HandlerResult | None:
            _paradox_evaluate(kind, energized, pokemon, context)
            return None

        def damage_mods(context: EventContext, payload: Payload) -> HandlerResult | None:
            _paradox_damage_mods(pokemon, context, payload)
            return None

        bus.on(Event.ON_SWITCH_IN, evaluate, priority=EventPriority.ABILITY, owner=owner)
        bus.on(Event.ON_TURN_START, evaluate, priority=EventPriority.ABILITY, owner=owner)
        bus.on(Event.ON_RESIDUAL, evaluate, priority=ResidualOrder.PARADOX, owner=owner)
        bus.on(Event.ON_DAMAGE_CALC, damage_mods, priority=EventPriority.ABILITY, owner=owner)

    return binder


ABILITY_BINDERS[Ability.PROTOSYNTHESIS] = _bind_paradox(Ability.PROTOSYNTHESIS, _sun_active)
ABILITY_BINDERS[Ability.QUARK_DRIVE] = _bind_paradox(Ability.QUARK_DRIVE, _electric_terrain_active)


# -- Reacting to hits ------------------------------------------------------------


@ability(Ability.TOXIC_DEBRIS)
def _bind_toxic_debris(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    def scatter(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.defender is not pokemon or payload["category"] is not Category.PHYSICAL:
            return None
        attacker_index = payload["attacker_index"]
        hazards = context.battle.sides[attacker_index].hazards
        layers = hazards.get(Hazards.TOXIC_SPIKES, 0)
        if layers >= 2:
            return None
        hazards[Hazards.TOXIC_SPIKES] = layers + 1
        context.log.add(HazardSet(side=attacker_index, hazard=Hazards.TOXIC_SPIKES))
        return None

    bus.on(Event.ON_AFTER_HIT, scatter, priority=EventPriority.ABILITY, owner=owner)


def _bind_contact_status(status: Status, chance: float = 0.3) -> AbilityBinder:
    """Flame Body / Static: a contact hit risks afflicting the attacker."""

    def binder(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
        def afflict(context: EventContext, payload: Payload) -> HandlerResult | None:
            if context.defender is not pokemon or not payload["contact"]:
                return None
            attacker = context.actor
            assert attacker is not None  # hit events always carry the attacker
            if attacker.is_fainted() or not context.rng.roll_chance(chance):
                return None
            from battle_sim.engine.status_apply import _apply_main_status  # lazy: avoids a mechanics->engine cycle

            teams = (context.battle.sides[0].team, context.battle.sides[1].team)
            _apply_main_status(status, attacker, payload["attacker_index"], context.rng, context.log, teams)
            return None

        bus.on(Event.ON_AFTER_HIT, afflict, priority=EventPriority.ABILITY, owner=owner)

    return binder


ABILITY_BINDERS[Ability.FLAME_BODY] = _bind_contact_status(Status.BURN)
ABILITY_BINDERS[Ability.STATIC] = _bind_contact_status(Status.PARALYSIS)


def _bind_on_hit_status(status: Status, requires_contact: bool, chance: float = 0.3) -> AbilityBinder:
    """Poison Touch / Toxic Chain: the attacker's hits risk afflicting the defender."""

    def binder(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
        def afflict(context: EventContext, payload: Payload) -> HandlerResult | None:
            if context.actor is not pokemon or (requires_contact and not payload["contact"]):
                return None
            defender = context.defender
            assert defender is not None  # hit events always carry the defender
            if defender.is_fainted() or not context.rng.roll_chance(chance):
                return None
            from battle_sim.engine.status_apply import _apply_main_status  # lazy: avoids a mechanics->engine cycle

            teams = (context.battle.sides[0].team, context.battle.sides[1].team)
            _apply_main_status(status, defender, payload["defender_index"], context.rng, context.log, teams)
            return None

        bus.on(Event.ON_AFTER_HIT, afflict, priority=EventPriority.ABILITY, owner=owner)

    return binder


ABILITY_BINDERS[Ability.POISON_TOUCH] = _bind_on_hit_status(Status.POISON, requires_contact=True)
ABILITY_BINDERS[Ability.TOXIC_CHAIN] = _bind_on_hit_status(Status.TOXIC, requires_contact=False)


@ability(Ability.WEAK_ARMOR)
def _bind_weak_armor(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    def crack(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.defender is not pokemon or payload["category"] is not Category.PHYSICAL or pokemon.is_fainted():
            return None
        apply_stage_changes(
            pokemon,
            payload["defender_index"],
            {Stats.DEFENCE: -1, Stats.SPEED: 2},
            context.log,
            inflicted_by_opponent=False,
            source="weak_armor",
        )
        return None

    bus.on(Event.ON_AFTER_HIT, crack, priority=EventPriority.ABILITY, owner=owner)


def _bind_hit_reaction_boost(stat: Stats, source: StatChangeSource, move_type: Type | None = None) -> AbilityBinder:
    """Stamina / Justified / Thermal Exchange: being hit (optionally by a given type) grants +1."""

    def binder(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
        def react(context: EventContext, payload: Payload) -> HandlerResult | None:
            if context.defender is not pokemon or pokemon.is_fainted():
                return None
            if move_type is not None and payload["move_type"] is not move_type:
                return None
            apply_stage_changes(
                pokemon,
                payload["defender_index"],
                {stat: 1},
                context.log,
                inflicted_by_opponent=False,
                source=source,
            )
            return None

        bus.on(Event.ON_AFTER_HIT, react, priority=EventPriority.ABILITY, owner=owner)

    return binder


ABILITY_BINDERS[Ability.STAMINA] = _bind_hit_reaction_boost(Stats.DEFENCE, "stamina")
ABILITY_BINDERS[Ability.JUSTIFIED] = _bind_hit_reaction_boost(Stats.ATTACK, "justified", Type.DARK)
ABILITY_BINDERS[Ability.THERMAL_EXCHANGE] = _bind_hit_reaction_boost(Stats.ATTACK, "thermal_exchange", Type.FIRE)


@ability(Ability.BERSERK)
def _bind_berserk(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    def enrage(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.defender is not pokemon or pokemon.is_fainted():
            return None
        dealt = payload["dealt"]
        max_hp, live = pokemon.stat_totals.HP, pokemon.live_stats.HP
        if dealt > 0 and 2 * live <= max_hp < 2 * (live + dealt):  # this hit crossed the half mark
            apply_stage_changes(
                pokemon,
                payload["defender_index"],
                {Stats.SP_ATTACK: 1},
                context.log,
                inflicted_by_opponent=False,
                source="berserk",
            )
        return None

    bus.on(Event.ON_AFTER_HIT, enrage, priority=EventPriority.ABILITY, owner=owner)


@ability(Ability.CURSED_BODY)
def _bind_cursed_body(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    def curse(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.defender is not pokemon or payload["dealt"] <= 0:
            return None
        attacker = context.actor
        assert attacker is not None  # hit events always carry the attacker
        if attacker.is_fainted() or attacker.disabled_slot is not None or attacker.last_move_slot is None:
            return None
        if not context.rng.roll_chance(0.3):
            return None
        from battle_sim.engine.status_apply import _start_disable  # lazy: avoids a mechanics->engine cycle

        _start_disable(attacker, payload["attacker_index"], context.log)
        return None

    bus.on(Event.ON_AFTER_HIT, curse, priority=EventPriority.ABILITY, owner=owner)


def _bind_absorb_boost(absorbed_type: Type, stat: Stats, amount: int, source: StatChangeSource) -> AbilityBinder:
    """Sap Sipper / Well-Baked Body: absorb a type for a stat boost (Motor Drive pattern)."""

    def binder(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
        def absorb(context: EventContext, payload: Payload) -> HandlerResult | None:
            if context.defender is not pokemon or payload["move_type"] is not absorbed_type:
                return None
            delta = pokemon.change_stat_stage(stat, amount)
            if delta > 0:
                context.log.add(
                    StatStageChanged(
                        side=payload["defender_index"],
                        pokemon=pokemon.nickname,
                        stat=stat,
                        delta=delta,
                        requested=amount,
                        source=source,
                    )
                )
            return HandlerResult(cancel=True, updated_payload={"absorbed": True, "immune_reason": source})

        bus.on(Event.ON_BEFORE_MOVE, absorb, priority=EventPriority.ABILITY, owner=owner)

    return binder


ABILITY_BINDERS[Ability.SAP_SIPPER] = _bind_absorb_boost(Type.GRASS, Stats.ATTACK, 1, "sap_sipper")
ABILITY_BINDERS[Ability.WELL_BAKED_BODY] = _bind_absorb_boost(Type.FIRE, Stats.DEFENCE, 2, "well_baked_body")
ABILITY_BINDERS[Ability.EARTH_EATER] = _bind_type_absorb(Ability.EARTH_EATER, Type.GROUND)


@ability(Ability.DRY_SKIN)
def _bind_dry_skin(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    _bind_type_absorb(Ability.DRY_SKIN, Type.WATER)(bus, pokemon, owner)

    def fire_vulnerability(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.defender is pokemon and payload["move_type"] is Type.FIRE:
            payload.setdefault("final_mods_4096", []).append(5120)
        return None

    def weather_flux(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.actor is not pokemon:
            return None
        weather = effective_weather(context.battle)
        eighth = max(1, pokemon.stat_totals.HP // 8)
        if weather in (Weather.RAIN, Weather.HEAVY_RAIN):
            healed = pokemon.apply_healing(eighth)
            if healed > 0:
                context.log.add(
                    AbilityHealed(
                        side=payload["side_index"], pokemon=pokemon.nickname, ability=Ability.DRY_SKIN, amount=healed
                    )
                )
        elif weather in (Weather.SUN, Weather.HARSH_SUN) and not payload.get("magic_guard", False):
            dealt = pokemon.apply_damage(eighth)
            context.log.add(
                AbilityChipDamage(
                    side=payload["side_index"], pokemon=pokemon.nickname, ability=Ability.DRY_SKIN, amount=dealt
                )
            )
        return None

    bus.on(Event.ON_DAMAGE_CALC, fire_vulnerability, priority=EventPriority.ABILITY, owner=owner)
    bus.on(Event.ON_RESIDUAL, weather_flux, priority=ResidualOrder.WEATHER_ABILITY, owner=owner)


# -- Residual and switch-out abilities -------------------------------------------


@ability(Ability.BAD_DREAMS)
def _bind_bad_dreams(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    def torment_sleepers(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.actor is not pokemon:
            return None
        side_index = payload["side_index"]
        foe = context.battle.sides[1 - side_index].active_pokemon
        if foe.is_fainted() or foe.status is not Status.SLEEP or foe.ability is Ability.MAGIC_GUARD:
            return None
        dealt = foe.apply_damage(max(1, foe.stat_totals.HP // 8))
        context.log.add(
            AbilityChipDamage(side=1 - side_index, pokemon=foe.nickname, ability=Ability.BAD_DREAMS, amount=dealt)
        )
        if foe.is_fainted():
            context.log.add(Fainted(side=1 - side_index, pokemon=foe.nickname))
        return None

    bus.on(Event.ON_RESIDUAL, torment_sleepers, priority=ResidualOrder.BAD_DREAMS, owner=owner)


@ability(Ability.HYDRATION)
def _bind_hydration(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    def cure_in_rain(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.actor is not pokemon or pokemon.status is Status.NONE:
            return None
        if effective_weather(context.battle) not in (Weather.RAIN, Weather.HEAVY_RAIN):
            return None
        pokemon.status = Status.NONE
        pokemon.status_turns = 0
        context.log.add(StatusCleared(side=payload["side_index"], pokemon=pokemon.nickname, clearance="hydration"))
        return None

    bus.on(Event.ON_RESIDUAL, cure_in_rain, priority=ResidualOrder.CURE, owner=owner)


@ability(Ability.ICE_BODY)
def _bind_ice_body(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    def snow_recovery(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.actor is not pokemon or effective_weather(context.battle) is not Weather.SNOW:
            return None
        healed = pokemon.apply_healing(max(1, pokemon.stat_totals.HP // 16))
        if healed > 0:
            context.log.add(
                AbilityHealed(
                    side=payload["side_index"], pokemon=pokemon.nickname, ability=Ability.ICE_BODY, amount=healed
                )
            )
        return None

    bus.on(Event.ON_RESIDUAL, snow_recovery, priority=ResidualOrder.WEATHER_ABILITY, owner=owner)


@ability(Ability.RAIN_DISH)
def _bind_rain_dish(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    def rain_recovery(context: EventContext, payload: Payload) -> HandlerResult | None:
        active_weather = effective_weather(context.battle)
        if context.actor is not pokemon or active_weather not in (Weather.RAIN, Weather.HEAVY_RAIN):
            return None
        healed = pokemon.apply_healing(max(1, pokemon.stat_totals.HP // 16))
        if healed > 0:
            context.log.add(
                AbilityHealed(
                    side=payload["side_index"], pokemon=pokemon.nickname, ability=Ability.RAIN_DISH, amount=healed
                )
            )
        return None

    bus.on(Event.ON_RESIDUAL, rain_recovery, priority=ResidualOrder.WEATHER_ABILITY, owner=owner)


@ability(Ability.POISON_HEAL)
def _bind_poison_heal(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    def heal_from_poison(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.actor is not pokemon or pokemon.status not in (Status.POISON, Status.TOXIC):
            return None
        healed = pokemon.apply_healing(max(1, pokemon.stat_totals.HP // 8))
        if healed > 0:
            context.log.add(
                AbilityHealed(
                    side=payload["side_index"], pokemon=pokemon.nickname, ability=Ability.POISON_HEAL, amount=healed
                )
            )
        return None

    bus.on(Event.ON_RESIDUAL, heal_from_poison, priority=ResidualOrder.POISON_HEAL, owner=owner)


@ability(Ability.REGENERATOR)
def _bind_regenerator(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    def regenerate(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.actor is not pokemon:
            return None
        healed = pokemon.apply_healing(max(1, pokemon.stat_totals.HP // 3))
        if healed > 0:
            context.log.add(
                AbilityHealed(
                    side=payload["side_index"], pokemon=pokemon.nickname, ability=Ability.REGENERATOR, amount=healed
                )
            )
        return None

    bus.on(Event.ON_SWITCH_OUT, regenerate, priority=EventPriority.ABILITY, owner=owner)


@ability(Ability.NATURAL_CURE)
def _bind_natural_cure(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    def cure(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.actor is not pokemon or pokemon.status is Status.NONE:
            return None
        pokemon.status = Status.NONE
        pokemon.status_turns = 0
        context.log.add(StatusCleared(side=payload["side_index"], pokemon=pokemon.nickname, clearance="natural_cure"))
        return None

    bus.on(Event.ON_SWITCH_OUT, cure, priority=EventPriority.ABILITY, owner=owner)


@ability(Ability.GOOD_AS_GOLD)
def _bind_good_as_gold(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    def block_status_moves(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.defender is not pokemon or payload["category"] is not Category.STATUS:
            return None
        context.log.add(
            StatusMoveBlocked(side=payload["defender_index"], pokemon=pokemon.nickname, ability=Ability.GOOD_AS_GOLD)
        )
        return HandlerResult(cancel=True, updated_payload={"absorbed": True, "immune_reason": "good_as_gold"})

    bus.on(Event.ON_BEFORE_MOVE, block_status_moves, priority=EventPriority.ABILITY, owner=owner)


@ability(Ability.UNAWARE)
def _bind_unaware(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    def ignore_stages(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.defender is pokemon:
            return HandlerResult(updated_payload={"ignore_attack_stages": True})
        if context.actor is pokemon:
            return HandlerResult(updated_payload={"ignore_defense_stages": True})
        return None

    bus.on(Event.ON_DAMAGE_CALC, ignore_stages, priority=EventPriority.ABILITY, owner=owner)


@ability(Ability.INFILTRATOR)
def _bind_infiltrator(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    """Screen bypass; the substitute bypass is a core check in the move flow."""

    def pierce_screens(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.actor is pokemon:
            return HandlerResult(updated_payload={"bypass_screens": True})
        return None

    bus.on(Event.ON_DAMAGE_CALC, pierce_screens, priority=EventPriority.ABILITY, owner=owner)


def _bind_type_shifter(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    """Libero / Protean: become the type of the used move, once per switch-in.

    The original typing is captured at wiring time (switch-in) and restored on switch-out.
    """
    original_types = pokemon.types
    shifted = False

    def shift(context: EventContext, payload: Payload) -> HandlerResult | None:
        nonlocal shifted
        move = context.move
        if context.actor is not pokemon or shifted or move is None or move.typeless:
            return None
        if pokemon.types == (move.type, None):
            return None
        pokemon.types = (move.type, None)
        shifted = True
        side_index = next(i for i, s in enumerate(context.battle.sides) if s.active_pokemon is pokemon)
        context.log.add(TypeChanged(side=side_index, pokemon=pokemon.nickname, new_type=move.type))
        return None

    def restore(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.actor is pokemon:
            pokemon.types = original_types
        return None

    bus.on(Event.ON_BEFORE_MOVE, shift, priority=EventPriority.SYSTEM, owner=owner)  # before absorbers see the type
    bus.on(Event.ON_SWITCH_OUT, restore, priority=EventPriority.ABILITY, owner=owner)


ABILITY_BINDERS[Ability.LIBERO] = _bind_type_shifter
ABILITY_BINDERS[Ability.PROTEAN] = _bind_type_shifter


# -- Transfer / forme abilities ---------------------------------------------------

_MULTITYPE_PLATES: dict[Item, Type] = {
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
_RKS_SYSTEM_MEMORIES: dict[Item, Type] = {
    Item.FIGHTING_MEMORY: Type.FIGHTING,
    Item.FLYING_MEMORY: Type.FLYING,
    Item.POISON_MEMORY: Type.POISON,
    Item.GROUND_MEMORY: Type.GROUND,
    Item.ROCK_MEMORY: Type.ROCK,
    Item.BUG_MEMORY: Type.BUG,
    Item.GHOST_MEMORY: Type.GHOST,
    Item.STEEL_MEMORY: Type.STEEL,
    Item.FIRE_MEMORY: Type.FIRE,
    Item.WATER_MEMORY: Type.WATER,
    Item.GRASS_MEMORY: Type.GRASS,
    Item.ELECTRIC_MEMORY: Type.ELECTRIC,
    Item.PSYCHIC_MEMORY: Type.PSYCHIC,
    Item.ICE_MEMORY: Type.ICE,
    Item.DRAGON_MEMORY: Type.DRAGON,
    Item.DARK_MEMORY: Type.DARK,
    Item.FAIRY_MEMORY: Type.FAIRY,
}


def _bind_item_type_shifter(type_by_item: dict[Item, Type]) -> AbilityBinder:
    """Multitype / RKS System: the holder's type tracks its held Plate/Memory (Normal with none).

    Synced on switch-in and re-checked every residual tick, so a Knock Off or Trick mid-battle
    (rare, but possible) reverts or swaps the type rather than leaving it stale.
    """

    def binder(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
        def sync_type(context: EventContext, payload: Payload) -> HandlerResult | None:
            if context.actor is not pokemon:
                return None
            wanted = (type_by_item.get(pokemon.item, Type.NORMAL), None)
            if pokemon.types != wanted:
                pokemon.types = wanted
                side_index = next(i for i, s in enumerate(context.battle.sides) if s.active_pokemon is pokemon)
                context.log.add(TypeChanged(side=side_index, pokemon=pokemon.nickname, new_type=wanted[0]))
            return None

        bus.on(Event.ON_SWITCH_IN, sync_type, priority=EventPriority.ABILITY, owner=owner)
        bus.on(Event.ON_RESIDUAL, sync_type, priority=ResidualOrder.PARADOX, owner=owner)

    return binder


ABILITY_BINDERS[Ability.MULTITYPE] = _bind_item_type_shifter(_MULTITYPE_PLATES)
ABILITY_BINDERS[Ability.RKS_SYSTEM] = _bind_item_type_shifter(_RKS_SYSTEM_MEMORIES)


@ability(Ability.IMPOSTER)
def _bind_imposter(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    def copy_opponent(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.actor is not pokemon:
            return None
        side_index = payload["side_index"]
        opponent = context.battle.sides[1 - side_index].active_pokemon
        if opponent.is_fainted():
            return None
        from battle_sim.engine.transform import transform_into  # lazy: avoids a mechanics->engine cycle

        transform_into(pokemon, side_index, opponent, context.battle, context.log)
        return None

    bus.on(Event.ON_SWITCH_IN, copy_opponent, priority=EventPriority.ABILITY, owner=owner)


def _steal_item(thief: Pokemon, thief_index: int, victim: Pokemon, context: EventContext) -> None:
    from battle_sim.mechanics.effects import rewire_active

    stolen = victim.item
    victim.item = Item.NONE
    victim.item_consumed = True  # losing an item activates Unburden
    victim.stripped_item = stolen  # taken rather than spent, so Nine Lives brings it back
    thief.item = stolen
    rewire_active(context.battle.bus, context.battle.effects, thief)
    rewire_active(context.battle.bus, context.battle.effects, victim)
    context.log.add(ItemStolen(side=thief_index, pokemon=thief.nickname, item=stolen))


@ability(Ability.PICKPOCKET)
def _bind_pickpocket(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    def pickpocket(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.defender is not pokemon or not payload["contact"] or pokemon.is_fainted():
            return None
        attacker = context.actor
        assert attacker is not None  # hit events always carry the attacker
        if pokemon.item is not Item.NONE or attacker.item is Item.NONE or attacker.is_fainted():
            return None
        _steal_item(pokemon, payload["defender_index"], attacker, context)
        return None

    bus.on(Event.ON_AFTER_HIT, pickpocket, priority=EventPriority.ABILITY, owner=owner)


@ability(Ability.MAGICIAN)
def _bind_magician(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    def pilfer(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.actor is not pokemon or payload["dealt"] <= 0 or pokemon.is_fainted():
            return None
        defender = context.defender
        assert defender is not None  # hit events always carry the defender
        if pokemon.item is not Item.NONE or defender.item is Item.NONE:
            return None
        _steal_item(pokemon, payload["attacker_index"], defender, context)
        return None

    bus.on(Event.ON_AFTER_HIT, pilfer, priority=EventPriority.ABILITY, owner=owner)


@ability(Ability.TRACE)
def _bind_trace(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    def trace(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.actor is not pokemon:
            return None
        side_index = payload["side_index"]
        opponent = context.battle.sides[1 - side_index].active_pokemon
        if opponent.is_fainted() or opponent.ability is Ability.NONE:
            return None
        if is_untouchable(opponent.ability):
            return None  # Nine Lives is not copyable: a traced one would give the challenger nine too
        from battle_sim.mechanics.effects import rewire_active

        pokemon.ability = opponent.ability
        rewire_active(context.battle.bus, context.battle.effects, pokemon)
        # The copied ability's own switch-in trigger does not re-fire (this emit already snapshotted handlers).
        context.log.add(AbilityCopied(side=side_index, pokemon=pokemon.nickname, ability=opponent.ability))
        return None

    bus.on(Event.ON_SWITCH_IN, trace, priority=EventPriority.ABILITY, owner=owner)


def _change_forme(pokemon: Pokemon, side_index: int, forme: str, context: EventContext) -> None:
    from battle_sim.database.loader import get_species  # lazy: avoids a mechanics->database import at module load

    species = get_species(forme)
    pokemon.name = species.name
    pokemon.base_stats = species.base_stats
    pokemon.types = species.types
    pokemon.refresh_stats()
    context.log.add(FormeChanged(side=side_index, pokemon=pokemon.nickname, forme=species.name))


@ability(Ability.ZERO_TO_HERO)
def _bind_zero_to_hero(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    def become_hero(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.actor is pokemon and pokemon.name == "Palafin":
            _change_forme(pokemon, payload["side_index"], "Palafin-Hero", context)
        return None

    bus.on(Event.ON_SWITCH_OUT, become_hero, priority=EventPriority.ABILITY, owner=owner)


@ability(Ability.TERA_SHIFT)
def _bind_tera_shift(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    def shift(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.actor is pokemon and pokemon.name == "Terapagos":
            _change_forme(pokemon, payload["side_index"], "Terapagos-Terastal", context)
        return None

    bus.on(Event.ON_SWITCH_IN, shift, priority=EventPriority.ABILITY, owner=owner)


# -- Damage riders and the remaining tail -------------------------------------------


@ability(Ability.SHEER_FORCE)
def _bind_sheer_force(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    """The 1.3x boost; the secondary stripping is gated in the move-effect loop."""

    def boost(context: EventContext, payload: Payload) -> HandlerResult | None:
        move = context.move
        if context.actor is pokemon and move is not None and any(_is_secondary(e) for e in move.effects):
            payload.setdefault("power_mods_4096", []).append(5325)
        return None

    bus.on(Event.ON_DAMAGE_CALC, boost, priority=EventPriority.ABILITY, owner=owner)


def _is_secondary(effect: object) -> bool:
    return bool(getattr(effect, "is_secondary", False))


def _bind_flagged_power_boost(predicate_flag: str, mod_4096: int) -> AbilityBinder:
    """Iron Fist (punch) / Punk Rock (sound): flagged moves hit harder."""

    def binder(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
        def boost(context: EventContext, payload: Payload) -> HandlerResult | None:
            move = context.move
            if context.actor is pokemon and move is not None and getattr(move, predicate_flag):
                payload.setdefault("power_mods_4096", []).append(mod_4096)
            return None

        bus.on(Event.ON_DAMAGE_CALC, boost, priority=EventPriority.ABILITY, owner=owner)

    return binder


ABILITY_BINDERS[Ability.IRON_FIST] = _bind_flagged_power_boost("punching", 4915)
ABILITY_BINDERS[Ability.STRONG_JAW] = _bind_flagged_power_boost("biting", 6144)
ABILITY_BINDERS[Ability.MEGA_LAUNCHER] = _bind_flagged_power_boost("pulse", 6144)


@ability(Ability.TOUGH_CLAWS)
def _bind_tough_claws(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    """Contact is a property of the hit, not the move: Protective Pads and a Punching Glove clear it."""

    def boost(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.actor is pokemon and payload["contact"]:
            payload.setdefault("power_mods_4096", []).append(5325)
        return None

    bus.on(Event.ON_DAMAGE_CALC, boost, priority=EventPriority.ABILITY, owner=owner)


@ability(Ability.SAND_FORCE)
def _bind_sand_force(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    def boost(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.actor is not pokemon or payload["move_type"] not in _SAND_FORCE_TYPES:
            return None
        if effective_weather(context.battle) is Weather.SANDSTORM:
            payload.setdefault("power_mods_4096", []).append(5325)
        return None

    bus.on(Event.ON_DAMAGE_CALC, boost, priority=EventPriority.ABILITY, owner=owner)


@ability(Ability.LIGHTNING_ROD)
def _bind_lightning_rod(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    def absorb_electric(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.defender is not pokemon or payload["move_type"] is not Type.ELECTRIC:
            return None
        delta = pokemon.change_stat_stage(Stats.SP_ATTACK, 1)
        if delta > 0:
            context.log.add(
                StatStageChanged(
                    side=payload["defender_index"],
                    pokemon=pokemon.nickname,
                    stat=Stats.SP_ATTACK,
                    delta=delta,
                    requested=1,
                    source="lightning_rod",
                )
            )
        return HandlerResult(cancel=True, updated_payload={"absorbed": True, "immune_reason": "lightning_rod"})

    bus.on(Event.ON_BEFORE_MOVE, absorb_electric, priority=EventPriority.ABILITY, owner=owner)


@ability(Ability.PARENTAL_BOND)
def _bind_parental_bond(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    """Approximated as a flat 1.25x rather than a real second strike.

    The true ability adds a second hit at 25% power, which totals exactly 1.25x on a single-hit
    damaging move — so total damage is right. What this does NOT reproduce: breaking a Substitute
    and then striking, rolling secondary effects twice, and being blocked twice by Sturdy or a Focus
    Sash. Modelling that needs the dynamic-extra-hit machinery `DamageEffect.multi_hit` doesn't have.
    """

    def boost(context: EventContext, payload: Payload) -> HandlerResult | None:
        move = context.move
        if context.actor is pokemon and move is not None and not _has_multiple_hits(move):
            payload.setdefault("power_mods_4096", []).append(5120)
        return None

    bus.on(Event.ON_DAMAGE_CALC, boost, priority=EventPriority.ABILITY, owner=owner)


@ability(Ability.PUNK_ROCK)
def _bind_punk_rock(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    def sound_tuning(context: EventContext, payload: Payload) -> HandlerResult | None:
        move = context.move
        if move is None or not move.sound:
            return None
        if context.actor is pokemon:
            payload.setdefault("power_mods_4096", []).append(5325)
        if context.defender is pokemon:
            payload.setdefault("final_mods_4096", []).append(2048)
        return None

    bus.on(Event.ON_DAMAGE_CALC, sound_tuning, priority=EventPriority.ABILITY, owner=owner)


@ability(Ability.ELECTROMORPHOSIS)
def _bind_electromorphosis(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    charged = False

    def charge_up(context: EventContext, payload: Payload) -> HandlerResult | None:
        nonlocal charged
        if context.defender is pokemon and payload["dealt"] > 0:
            charged = True
        return None

    def discharge(context: EventContext, payload: Payload) -> HandlerResult | None:
        nonlocal charged
        if context.actor is pokemon and charged and payload["move_type"] is Type.ELECTRIC:
            payload.setdefault("power_mods_4096", []).append(8192)
            charged = False
        return None

    bus.on(Event.ON_AFTER_HIT, charge_up, priority=EventPriority.ABILITY, owner=owner)
    bus.on(Event.ON_DAMAGE_CALC, discharge, priority=EventPriority.ABILITY, owner=owner)


@ability(Ability.WIND_RIDER)
def _bind_wind_rider(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    def ride(context: EventContext, payload: Payload) -> HandlerResult | None:
        move = context.move
        if context.defender is not pokemon or move is None or not move.wind:
            return None
        delta = pokemon.change_stat_stage(Stats.ATTACK, 1)
        if delta > 0:
            context.log.add(
                StatStageChanged(
                    side=payload["defender_index"],
                    pokemon=pokemon.nickname,
                    stat=Stats.ATTACK,
                    delta=delta,
                    requested=1,
                    source="justified",
                )
            )
        return HandlerResult(cancel=True, updated_payload={"absorbed": True, "immune_reason": "wind_rider"})

    bus.on(Event.ON_BEFORE_MOVE, ride, priority=EventPriority.ABILITY, owner=owner)


@ability(Ability.HARVEST)
def _bind_harvest(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    def regrow(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.actor is not pokemon or pokemon.item is not Item.NONE:
            return None
        if not pokemon.item_consumed or pokemon.last_consumed_item is Item.NONE:
            return None
        if "berry" not in pokemon.last_consumed_item.name.lower():
            return None
        in_sun = effective_weather(context.battle) in (Weather.SUN, Weather.HARSH_SUN)
        if in_sun or context.rng.roll_chance(0.5):
            pokemon.item = pokemon.last_consumed_item
            pokemon.item_consumed = False
        return None

    bus.on(Event.ON_RESIDUAL, regrow, priority=ResidualOrder.CURE, owner=owner)


ABILITY_BINDERS[Ability.ROCKY_PAYLOAD] = _bind_type_attack_boost(Type.ROCK)
ABILITY_BINDERS[Ability.TRANSISTOR] = _bind_type_attack_boost(Type.ELECTRIC)


@ability(Ability.BULLETPROOF)
def _bind_bulletproof(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    def block_bullets(context: EventContext, payload: Payload) -> HandlerResult | None:
        move = context.move
        if context.defender is not pokemon or move is None or not move.bullet:
            return None
        context.log.add(
            AbsorbBlocked(side=payload["defender_index"], pokemon=pokemon.nickname, ability=Ability.BULLETPROOF)
        )
        return HandlerResult(cancel=True, updated_payload={"absorbed": True, "immune_reason": "bulletproof"})

    bus.on(Event.ON_BEFORE_MOVE, block_bullets, priority=EventPriority.ABILITY, owner=owner)


@ability(Ability.BATTLE_BOND)
def _bind_battle_bond(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    def bond(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.actor is not pokemon or pokemon.is_fainted() or pokemon.switch_in_boost_used:
            return None
        pokemon.switch_in_boost_used = True  # the bond strengthens once per battle
        apply_stage_changes(
            pokemon,
            payload["attacker_index"],
            {Stats.ATTACK: 1, Stats.SP_ATTACK: 1, Stats.SPEED: 1},
            context.log,
            inflicted_by_opponent=False,
            source="move",
        )
        return None

    bus.on(Event.ON_FAINT, bond, priority=EventPriority.ABILITY, owner=owner)


# -- Coverage-audit repairs, round two (2026-09-08) ---------------------------------
#
# Each of these left its holder battling with no ability at all. Poison Point and Effect Spore reuse
# the contact-status factory above; the rest are their own small handlers.

ABILITY_BINDERS[Ability.POISON_POINT] = _bind_contact_status(Status.POISON)
_SPORE_STATUSES = (Status.POISON, Status.PARALYSIS, Status.SLEEP)


@ability(Ability.EFFECT_SPORE)
def _bind_effect_spore(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    """A contact hit risks poison, paralysis or sleep — 30% split three ways, as Gen 5+ has it."""

    def afflict(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.defender is not pokemon or not payload["contact"]:
            return None
        attacker = context.actor
        assert attacker is not None  # hit events always carry the attacker
        if attacker.is_fainted() or not context.rng.roll_chance(0.3):
            return None
        from battle_sim.engine.status_apply import _apply_main_status  # lazy: avoids a mechanics->engine cycle

        status = _SPORE_STATUSES[context.rng.random_integer(0, len(_SPORE_STATUSES) - 1)]
        teams = (context.battle.sides[0].team, context.battle.sides[1].team)
        _apply_main_status(status, attacker, payload["attacker_index"], context.rng, context.log, teams)
        return None

    bus.on(Event.ON_AFTER_HIT, afflict, priority=EventPriority.ABILITY, owner=owner)


@ability(Ability.FLUFFY)
def _bind_fluffy(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    """Halves contact damage and doubles Fire damage — the two apply together, so a contact Fire
    move lands for exactly its usual amount."""

    def cushion(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.defender is not pokemon:
            return None
        if payload["contact"]:
            payload.setdefault("final_mods_4096", []).append(2048)
        if payload["move_type"] is Type.FIRE:
            payload.setdefault("final_mods_4096", []).append(8192)
        return None

    bus.on(Event.ON_DAMAGE_CALC, cushion, priority=EventPriority.ABILITY, owner=owner)


def _bind_draw_in(drawn: Type, source: StatChangeSource) -> AbilityBinder:
    """Lightning Rod / Storm Drain: the move is absorbed whole and pays a Sp. Atk stage for trying."""

    def binder(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
        def absorb(context: EventContext, payload: Payload) -> HandlerResult | None:
            if context.defender is not pokemon or payload["move_type"] is not drawn:
                return None
            delta = pokemon.change_stat_stage(Stats.SP_ATTACK, 1)
            if delta > 0:
                context.log.add(
                    StatStageChanged(
                        side=payload["defender_index"],
                        pokemon=pokemon.nickname,
                        stat=Stats.SP_ATTACK,
                        delta=delta,
                        requested=1,
                        source=source,
                    )
                )
            return HandlerResult(cancel=True, updated_payload={"absorbed": True, "immune_reason": source})

        bus.on(Event.ON_BEFORE_MOVE, absorb, priority=EventPriority.ABILITY, owner=owner)

    return binder


ABILITY_BINDERS[Ability.STORM_DRAIN] = _bind_draw_in(Type.WATER, "storm_drain")


@ability(Ability.FLOWER_GIFT)
def _bind_flower_gift(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    """Sun raises Cherrim's Attack and Sp. Def by half. The forme it changes into to show that is not
    modelled — the stats are the half anyone notices."""

    def bloom(context: EventContext, payload: Payload) -> HandlerResult | None:
        if effective_weather(context.battle) not in (Weather.SUN, Weather.HARSH_SUN):
            return None
        if context.actor is pokemon and payload["category"] is Category.PHYSICAL:
            payload.setdefault("attack_mods_4096", []).append(6144)
        if context.defender is pokemon and payload["category"] is Category.SPECIAL:
            payload.setdefault("defense_mods_4096", []).append(6144)
        return None

    bus.on(Event.ON_DAMAGE_CALC, bloom, priority=EventPriority.ABILITY, owner=owner)


@ability(Ability.ANALYTIC)
def _bind_analytic(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    """1.3x for moving last, which the opponent having already acted this turn is the record of."""

    def punish_the_slow_turn(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.actor is not pokemon:
            return None
        if context.battle.sides[payload["defender_index"]].acted_this_turn:
            payload.setdefault("power_mods_4096", []).append(5325)
        return None

    bus.on(Event.ON_DAMAGE_CALC, punish_the_slow_turn, priority=EventPriority.ABILITY, owner=owner)


@ability(Ability.SHED_SKIN)
def _bind_shed_skin(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    """A third of the time, whatever it is sloughs off at the end of the turn."""

    def slough(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.actor is not pokemon or pokemon.status is Status.NONE:
            return None
        if not context.rng.roll_chance(1 / 3):
            return None
        pokemon.status = Status.NONE
        pokemon.status_turns = 0
        context.log.add(StatusCleared(side=payload["side_index"], pokemon=pokemon.nickname, clearance="shed_skin"))
        return None

    bus.on(Event.ON_RESIDUAL, slough, priority=ResidualOrder.CURE, owner=owner)


@ability(Ability.AFTERMATH)
def _bind_aftermath(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    """Going down to a contact hit takes a quarter of the attacker with it."""

    def parting_shot(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.defender is not pokemon or not payload["contact"] or not pokemon.is_fainted():
            return None
        attacker = context.actor
        assert attacker is not None  # hit events always carry the attacker
        if attacker.is_fainted() or attacker.ability is Ability.MAGIC_GUARD:
            return None
        chip = attacker.apply_damage(max(1, attacker.stat_totals.HP // 4))
        context.log.add(
            AbilityChipDamage(
                side=payload["attacker_index"], pokemon=attacker.nickname, ability=Ability.AFTERMATH, amount=chip
            )
        )
        return None

    bus.on(Event.ON_AFTER_HIT, parting_shot, priority=EventPriority.ABILITY, owner=owner)


def _bind_flee_at_half(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    """Emergency Exit / Wimp Out: dropping to half health sends its holder running.

    Only the flag is set here. The turn loop pulls the holder once the action has finished resolving,
    exactly as it does for an Eject Pack — a switch part-way through a hit would be a mess.
    """

    def flee(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.defender is not pokemon or pokemon.is_fainted():
            return None
        maximum = pokemon.stat_totals.HP
        if pokemon.live_stats.HP * 2 <= maximum < (pokemon.live_stats.HP + payload["dealt"]) * 2:
            pokemon.flee_pending = True  # crossed the halfway line on this hit, rather than sitting under it
        return None

    bus.on(Event.ON_AFTER_HIT, flee, priority=EventPriority.ABILITY, owner=owner)


ABILITY_BINDERS[Ability.EMERGENCY_EXIT] = _bind_flee_at_half
ABILITY_BINDERS[Ability.WIMP_OUT] = _bind_flee_at_half


# -- What a damage estimate has to know without running the bus ----------------------
# Every absorb above cancels its move from inside an ON_BEFORE_MOVE handler, which is the right
# place for the engine and the wrong place for anyone trying to price a move before playing it:
# `analysis.damage_range` never runs the bus, so it read a Ground move into a Levitate Rotom as a
# clean one-shot and the search picked it four turns running. Same shape as `status_cannot_land` —
# the conditions live once, and both the engine and the estimator read them from here.
_TYPE_ABSORBING_ABILITIES: dict[Ability, Type] = {
    Ability.VOLT_ABSORB: Type.ELECTRIC,
    Ability.LIGHTNING_ROD: Type.ELECTRIC,
    Ability.MOTOR_DRIVE: Type.ELECTRIC,
    Ability.WATER_ABSORB: Type.WATER,
    Ability.STORM_DRAIN: Type.WATER,
    Ability.DRY_SKIN: Type.WATER,
    Ability.SAP_SIPPER: Type.GRASS,
    Ability.FLASH_FIRE: Type.FIRE,
    Ability.WELL_BAKED_BODY: Type.FIRE,
    Ability.EARTH_EATER: Type.GROUND,
}
_MOLD_BREAKERS = frozenset({Ability.MOLD_BREAKER, Ability.TERAVOLT, Ability.TURBOBLAZE})
# Only what the engine actually binds: Solid Rock has no binder here, so mirroring it would put the
# estimator ahead of the engine, which is the same divergence in the opposite direction.
_SUPER_EFFECTIVE_REDUCERS = frozenset({Ability.FILTER, Ability.PRISM_ARMOR})


def ability_absorbs(move_type: Type, move: Move, attacker: Pokemon, defender: Pokemon) -> bool:
    """Whether the defender simply cannot be hurt by this move, ability and item included.

    `move_type` is passed separately because the caller may already have resolved a type-changing
    ability (Aerilate and friends): the absorb applies to what the move has *become*, not what it
    was written as.

    Grounding is asked of the Pokemon rather than matched against Levitate by name, so Air Balloon
    is covered by the same question — it is the same invisible immunity from the estimator's side.
    """
    if attacker.ability in _MOLD_BREAKERS:
        return False  # ignores every ability below, which is the whole of what Mold Breaker does
    if move_type is Type.GROUND and not defender.is_grounded():
        return True
    absorbed = _TYPE_ABSORBING_ABILITIES.get(defender.ability)
    if absorbed is not None and move_type is absorbed:
        return True
    if defender.ability is Ability.WONDER_GUARD and type_effectiveness(move_type, defender.types) < 2:
        return True
    if defender.ability is Ability.SOUNDPROOF and move.sound:
        return True
    return defender.ability is Ability.BULLETPROOF and move.bullet


def static_ability_modifiers(
    move_type: Type, move: Move, base_power: int | None, attacker: Pokemon, defender: Pokemon
) -> dict[str, object]:
    """The ability multipliers a damage estimate can work out on its own, keyed as the payload wants.

    The companion to `static_damage_modifiers` for items, and mirrors of the handlers above: list
    values append to a channel, scalars replace one. Every one of these was invisible to
    `analysis.damage_range`, which is how the search came to price a Huge Power hit at half strength
    and a Multiscale wall at double what it takes.

    Defender abilities are skipped against a mould breaker, exactly as the engine skips them.
    """
    effectiveness = type_effectiveness(move_type, defender.types)
    channels: dict[str, list[int]] = {}
    modifiers: dict[str, object] = {}
    _attacking_side(move_type, move, base_power, attacker, effectiveness, channels, modifiers)
    if attacker.ability not in _MOLD_BREAKERS:
        _defending_side(move_type, move, defender, effectiveness, channels)
    return {**modifiers, **{channel: values for channel, values in channels.items() if values}}


def _attacking_side(
    move_type: Type,
    move: Move,
    base_power: int | None,
    attacker: Pokemon,
    effectiveness: float,
    channels: dict[str, list[int]],
    modifiers: dict[str, object],
) -> None:
    physical = move.category is Category.PHYSICAL
    if physical and attacker.ability in (Ability.HUGE_POWER, Ability.PURE_POWER):
        channels.setdefault("attack_mods_4096", []).append(8192)
    if attacker.ability is Ability.GUTS:
        if physical and attacker.status is not Status.NONE:
            channels.setdefault("attack_mods_4096", []).append(6144)
        # Set whatever the status, exactly as the handler does: Guts ignores the burn's own halving,
        # and without this a burned Guts attacker read as *weaker* rather than half again as strong.
        modifiers["ignore_burn"] = True
    if attacker.ability is Ability.DEFEATIST and attacker.live_stats.HP * 2 <= attacker.stat_totals.HP:
        channels.setdefault("attack_mods_4096", []).append(2048)
    if attacker.ability is Ability.ADAPTABILITY and move_type in attacker.types:
        modifiers["stab_4096"] = 8192
    if attacker.ability is Ability.TECHNICIAN and base_power is not None and base_power <= 60:
        channels.setdefault("power_mods_4096", []).append(6144)
    if attacker.ability is Ability.SHEER_FORCE and any(_is_secondary(effect) for effect in move.effects):
        channels.setdefault("power_mods_4096", []).append(5325)
    if attacker.ability is Ability.TINTED_LENS and 0 < effectiveness < 1:
        channels.setdefault("final_mods_4096", []).append(8192)


def _defending_side(
    move_type: Type, move: Move, defender: Pokemon, effectiveness: float, channels: dict[str, list[int]]
) -> None:
    if defender.ability is Ability.THICK_FAT and move_type in (Type.FIRE, Type.ICE):
        channels.setdefault("attack_mods_4096", []).append(2048)
    if defender.ability is Ability.MULTISCALE and defender.live_stats.HP == defender.stat_totals.HP:
        channels.setdefault("final_mods_4096", []).append(2048)
    if defender.ability in _SUPER_EFFECTIVE_REDUCERS and effectiveness >= 2:
        channels.setdefault("final_mods_4096", []).append(3072)
    if defender.ability is Ability.FUR_COAT and move.category is Category.PHYSICAL:
        channels.setdefault("defense_mods_4096", []).append(8192)


_AURA_ABILITIES = {Ability.DARK_AURA: Type.DARK, Ability.FAIRY_AURA: Type.FAIRY}


def static_field_modifiers(move_type: Type, actives: Sequence[Pokemon]) -> dict[str, object]:
    """Ability multipliers that depend on who else is on the field, not on the attacker alone.

    `static_ability_modifiers` is handed an attacker and a defender, so it structurally cannot see
    an aura -- which is owned by whichever Pokemon happens to be standing there and applies to *both*
    sides' moves of its type. Yveltal's Dark Aura is a 1.33x on every Dark move in the game while it
    is out, and `analysis.damage_range` was pricing all of them at face value.

    Mirrors `_bind_aura`, which is the authority on the numbers.
    """
    if not any(p.ability in _AURA_ABILITIES and _AURA_ABILITIES[p.ability] is move_type for p in actives):
        return {}
    broken = any(p.ability is Ability.AURA_BREAK for p in actives)
    return {"power_mods_4096": [3072 if broken else 5461]}


def field_from_actives(actives: Sequence[Pokemon]) -> tuple[Weather | None, Terrain | None]:
    """The weather and terrain these Pokemon would have set on the way in.

    A state assembled by hand -- the lead matrix's scratch board, say -- never runs the switch-in
    that turns Groudon's Desolate Land on, so every Water move aimed at it scored at full strength.
    Measured: Kyogre's best hit on Groudon read 370 against a true 184.

    Later actives win, matching the engine's ordering when both sides set weather on entry.
    """
    weather: Weather | None = None
    terrain: Terrain | None = None
    for pokemon in actives:
        if pokemon.is_fainted():
            continue
        weather = ABILITY_WEATHER.get(pokemon.ability, weather)
        terrain = ABILITY_TERRAIN.get(pokemon.ability, terrain)
    return weather, terrain


# -- Abilities that rewrite somebody else's ability ---------------------------


def set_ability(
    target: Pokemon,
    target_index: int,
    ability: Ability,
    battle: "BattleState",
    log: "BattleLog",
) -> bool:
    """Give `target` a different ability, wiring it up. False if it refused.

    The one place an ability is rewritten, so the one place that has to check whether it may be. Both
    ends are guarded: an untouchable ability cannot be written *over*, and never gets handed out — a
    Mummy that could take Nine Lives off the butler and a Mummy that could give it away are the same
    hole seen from two sides.

    The refusal is announced rather than silent: an ability that visibly does nothing reads as a bug,
    and a challenger who has just thrown Entrainment at the butler deserves to know why it did not
    take.
    """
    from battle_sim.mechanics.effects import rewire_active

    if is_untouchable(target.ability):
        log.add(AbilityUnchanged(side=target_index, pokemon=target.nickname, ability=target.ability))
        return False
    if is_untouchable(ability) or target.ability is ability:
        return False
    target.ability = ability
    rewire_active(battle.bus, battle.effects, target)
    log.add(AbilityChanged(side=target_index, pokemon=target.nickname, ability=ability))
    return True


@ability(Ability.MUMMY)
def _bind_mummy(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    """Touching a Mummy makes you one. Spreads on contact, and never to something that cannot take it."""

    def infect(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.defender is not pokemon or not payload["contact"]:
            return None
        attacker = context.actor
        assert attacker is not None  # hit events always carry the attacker
        if attacker.is_fainted() or attacker.ability is Ability.MUMMY:
            return None
        set_ability(attacker, payload["attacker_index"], Ability.MUMMY, context.battle, context.log)
        return None

    bus.on(Event.ON_AFTER_HIT, infect, priority=EventPriority.ABILITY, owner=owner)
