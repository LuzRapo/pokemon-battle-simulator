"""End-of-turn residuals. The order is a mechanics contract (see ResidualOrder):

1. Field duration ticks (weather, terrain, pseudo-weather).
2. Per active, one ON_RESIDUAL emit resolved in Showdown's canonical order:
   Magic Guard flag -> sandstorm chip -> recovery items -> Leech Seed ->
   status chip -> Nightmare -> Speed Boost; then faint check and volatile
   ticks.
3. Per side: tailwind and screen ticks.
"""

from battle_sim.mechanics.battle import BattleState, SideState, effective_weather
from battle_sim.mechanics.events import Event, EventBus, EventContext, HandlerResult, Payload, ResidualOrder
from battle_sim.mechanics.log import BattleLog
from battle_sim.models.log_events import (
    Fainted,
    Healed,
    LeechSeedSap,
    PseudoWeatherEnded,
    ResidualDamage,
    ScreenFaded,
    StatusCleared,
    StatusInflicted,
    TailwindFaded,
    TerrainFaded,
    VolatileInflicted,
    WeatherFaded,
)
from battle_sim.models.pokemon import Pokemon
from battle_sim.utils import Ability, ExtraStatus, Status, Terrain, Type, Weather

_SANDSTORM_IMMUNE_TYPES = frozenset({Type.ROCK, Type.GROUND, Type.STEEL})


def _apply_residuals(state: BattleState, log: BattleLog) -> None:
    _tick_field_durations(state, log)
    _ensure_core_residual_handlers(state.bus)
    for side_index, side in enumerate(state.sides):
        active = side.active_pokemon
        if not active.is_fainted():
            ctx = EventContext(rng=state.rng, battle=state, log=log, actor=active)
            state.bus.emit(Event.ON_RESIDUAL, ctx, {"side_index": side_index})
            if active.is_fainted():
                log.add(Fainted(side=side_index, pokemon=active.nickname))
            _tick_volatiles(active, side_index, log)
        _tick_side_durations(side, side_index, log)


def _ensure_core_residual_handlers(bus: EventBus) -> None:
    """Idempotent (bus dedupes by handler identity): the universal residual chips."""
    bus.on(Event.ON_RESIDUAL, _sandstorm_chip, priority=ResidualOrder.WEATHER)
    bus.on(Event.ON_RESIDUAL, _grassy_terrain_heal, priority=ResidualOrder.TERRAIN)
    bus.on(Event.ON_RESIDUAL, _leech_seed_sap, priority=ResidualOrder.LEECH_SEED)
    bus.on(Event.ON_RESIDUAL, _status_chip, priority=ResidualOrder.STATUS)
    bus.on(Event.ON_RESIDUAL, _nightmare, priority=ResidualOrder.NIGHTMARE)
    bus.on(Event.ON_RESIDUAL, _salt_cure, priority=ResidualOrder.SALT_CURE)
    bus.on(Event.ON_RESIDUAL, _curse_chip, priority=ResidualOrder.CURSE)
    bus.on(Event.ON_RESIDUAL, _yawn, priority=ResidualOrder.YAWN)
    bus.on(Event.ON_RESIDUAL, _locked_move, priority=ResidualOrder.LOCKED_MOVE)


def _sandstorm_chip(context: EventContext, payload: Payload) -> HandlerResult | None:
    active = context.actor
    assert active is not None  # ON_RESIDUAL always carries the active pokemon
    if payload.get("magic_guard", False):
        return None
    if effective_weather(context.battle) is not Weather.SANDSTORM:
        return None
    if _SANDSTORM_IMMUNE_TYPES.intersection(t for t in active.types if t is not None):
        return None
    if active.ability in (Ability.SAND_VEIL, Ability.OVERCOAT):
        return None
    dealt = active.apply_damage(max(1, active.stat_totals.HP // 16))
    context.log.add(
        ResidualDamage(side=payload["side_index"], pokemon=active.nickname, source="sandstorm", amount=dealt)
    )
    return None


def _grassy_terrain_heal(context: EventContext, payload: Payload) -> HandlerResult | None:
    active = context.actor
    assert active is not None  # ON_RESIDUAL always carries the active pokemon
    if context.battle.field.terrain is not Terrain.GRASSY or not active.is_grounded():
        return None
    healed = active.apply_healing(max(1, active.stat_totals.HP // 16))
    if healed > 0:
        context.log.add(Healed(side=payload["side_index"], pokemon=active.nickname, amount=healed))
    return None


def _leech_seed_sap(context: EventContext, payload: Payload) -> HandlerResult | None:
    active = context.actor
    assert active is not None  # ON_RESIDUAL always carries the active pokemon
    if ExtraStatus.LEECH_SEED not in active.volatiles or payload.get("magic_guard", False):
        return None
    side_index = payload["side_index"]
    dealt = active.apply_damage(max(1, active.stat_totals.HP // 8))
    context.log.add(LeechSeedSap(side=side_index, pokemon=active.nickname, amount=dealt))
    drainer = context.battle.sides[1 - side_index].active_pokemon
    if not drainer.is_fainted():
        drainer.apply_healing(dealt)
    return None


def _status_chip(context: EventContext, payload: Payload) -> HandlerResult | None:
    active = context.actor
    assert active is not None  # ON_RESIDUAL always carries the active pokemon
    if payload.get("magic_guard", False):
        return None
    if active.ability is Ability.POISON_HEAL and active.status in (Status.POISON, Status.TOXIC):
        return None  # the ability's own handler healed instead
    side_index = payload["side_index"]
    if active.status is Status.BURN:
        dealt = active.apply_damage(max(1, active.stat_totals.HP // 16))
        context.log.add(ResidualDamage(side=side_index, pokemon=active.nickname, source="burn", amount=dealt))
    elif active.status is Status.POISON:
        dealt = active.apply_damage(max(1, active.stat_totals.HP // 8))
        context.log.add(ResidualDamage(side=side_index, pokemon=active.nickname, source="poison", amount=dealt))
    elif active.status is Status.TOXIC:
        active.status_turns += 1
        dealt = active.apply_damage(max(1, active.stat_totals.HP * active.status_turns // 16))
        context.log.add(ResidualDamage(side=side_index, pokemon=active.nickname, source="toxic", amount=dealt))
    return None


def _nightmare(context: EventContext, payload: Payload) -> HandlerResult | None:
    active = context.actor
    assert active is not None  # ON_RESIDUAL always carries the active pokemon
    if ExtraStatus.NIGHTMARE not in active.volatiles:
        return None
    if active.status is not Status.SLEEP:
        del active.volatiles[ExtraStatus.NIGHTMARE]
        return None
    if payload.get("magic_guard", False):
        return None
    dealt = active.apply_damage(max(1, active.stat_totals.HP // 4))
    context.log.add(
        ResidualDamage(side=payload["side_index"], pokemon=active.nickname, source="nightmare", amount=dealt)
    )
    return None


def _salt_cure(context: EventContext, payload: Payload) -> HandlerResult | None:
    active = context.actor
    assert active is not None  # ON_RESIDUAL always carries the active pokemon
    if ExtraStatus.SALT_CURE not in active.volatiles or payload.get("magic_guard", False):
        return None
    fraction = 4 if {Type.WATER, Type.STEEL}.intersection(t for t in active.types if t is not None) else 8
    dealt = active.apply_damage(max(1, active.stat_totals.HP // fraction))
    context.log.add(
        ResidualDamage(side=payload["side_index"], pokemon=active.nickname, source="salt_cure", amount=dealt)
    )
    return None


def _curse_chip(context: EventContext, payload: Payload) -> HandlerResult | None:
    active = context.actor
    assert active is not None  # ON_RESIDUAL always carries the active pokemon
    if ExtraStatus.CURSE not in active.volatiles or payload.get("magic_guard", False):
        return None
    dealt = active.apply_damage(max(1, active.stat_totals.HP // 4))
    context.log.add(ResidualDamage(side=payload["side_index"], pokemon=active.nickname, source="curse", amount=dealt))
    return None


def _yawn(context: EventContext, payload: Payload) -> HandlerResult | None:
    active = context.actor
    assert active is not None  # ON_RESIDUAL always carries the active pokemon
    if ExtraStatus.YAWN not in active.volatiles:
        return None
    active.volatiles[ExtraStatus.YAWN] -= 1
    if active.volatiles[ExtraStatus.YAWN] > 0:
        return None
    del active.volatiles[ExtraStatus.YAWN]
    if active.status is Status.NONE:
        active.status = Status.SLEEP
        active.status_turns = context.rng.random_integer(1, 4)
        context.log.add(StatusInflicted(side=payload["side_index"], pokemon=active.nickname, status=Status.SLEEP))
    return None


def _locked_move(context: EventContext, payload: Payload) -> HandlerResult | None:
    active = context.actor
    assert active is not None  # ON_RESIDUAL always carries the active pokemon
    if ExtraStatus.LOCKED_MOVE not in active.volatiles:
        return None
    active.volatiles[ExtraStatus.LOCKED_MOVE] -= 1
    if active.volatiles[ExtraStatus.LOCKED_MOVE] > 0:
        return None
    del active.volatiles[ExtraStatus.LOCKED_MOVE]
    active.locked_slot = None
    if ExtraStatus.CONFUSION not in active.volatiles:  # fatigue (PS lockedmove onEnd)
        active.volatiles[ExtraStatus.CONFUSION] = context.rng.random_integer(2, 6)
        context.log.add(
            VolatileInflicted(side=payload["side_index"], pokemon=active.nickname, volatile=ExtraStatus.CONFUSION)
        )
    return None


def _tick_field_durations(state: BattleState, log: BattleLog) -> None:
    field = state.field
    if field.weather_turns_left > 0:
        field.weather_turns_left -= 1
        if field.weather_turns_left == 0:
            prior_weather = field.weather
            field.weather = Weather.NONE
            if prior_weather is not Weather.NONE:
                log.add(WeatherFaded(weather=prior_weather))
    if field.terrain_turns_left > 0:
        field.terrain_turns_left -= 1
        if field.terrain_turns_left == 0:
            prior_terrain = field.terrain
            field.terrain = Terrain.NONE
            if prior_terrain is not Terrain.NONE:
                log.add(TerrainFaded(terrain=prior_terrain))
    for pseudo in list(field.pseudo_weather):
        field.pseudo_weather[pseudo] -= 1
        if field.pseudo_weather[pseudo] <= 0:
            del field.pseudo_weather[pseudo]
            log.add(PseudoWeatherEnded(kind=pseudo))


def _tick_volatiles(active: Pokemon, side_index: int, log: BattleLog) -> None:
    active.volatiles.pop(ExtraStatus.FLINCH, None)
    active.volatiles.pop(ExtraStatus.PROTECT, None)
    active.volatiles.pop(ExtraStatus.ENDURE, None)
    if ExtraStatus.PERISH in active.volatiles:
        active.volatiles[ExtraStatus.PERISH] -= 1
        if active.volatiles[ExtraStatus.PERISH] <= 0:
            del active.volatiles[ExtraStatus.PERISH]
            active.apply_damage(active.live_stats.HP)
            log.add(Fainted(side=side_index, pokemon=active.nickname))
    if ExtraStatus.TAUNT in active.volatiles:
        active.volatiles[ExtraStatus.TAUNT] -= 1
        if active.volatiles[ExtraStatus.TAUNT] <= 0:
            del active.volatiles[ExtraStatus.TAUNT]
            log.add(StatusCleared(side=side_index, pokemon=active.nickname, clearance="taunt_ended"))
    if ExtraStatus.ENCORE in active.volatiles:
        active.volatiles[ExtraStatus.ENCORE] -= 1
        if active.volatiles[ExtraStatus.ENCORE] <= 0:
            del active.volatiles[ExtraStatus.ENCORE]
            active.encored_slot = None
            log.add(StatusCleared(side=side_index, pokemon=active.nickname, clearance="encore_ended"))
    if ExtraStatus.DISABLE in active.volatiles:
        active.volatiles[ExtraStatus.DISABLE] -= 1
        if active.volatiles[ExtraStatus.DISABLE] <= 0:
            del active.volatiles[ExtraStatus.DISABLE]
            active.disabled_slot = None
            log.add(StatusCleared(side=side_index, pokemon=active.nickname, clearance="disable_ended"))


def _tick_side_durations(side: SideState, side_index: int, log: BattleLog) -> None:
    if side.wish_turns > 0:
        side.wish_turns -= 1
        if side.wish_turns == 0:
            recipient = side.active_pokemon
            if not recipient.is_fainted():
                healed = recipient.apply_healing(side.wish_pending)
                if healed > 0:
                    log.add(Healed(side=side_index, pokemon=recipient.nickname, amount=healed))
            side.wish_pending = 0
    if side.tailwind_turns > 0:
        side.tailwind_turns -= 1
        if side.tailwind_turns == 0:
            log.add(TailwindFaded(side=side_index))
    for screen in list(side.screens):
        side.screens[screen] -= 1
        if side.screens[screen] <= 0:
            del side.screens[screen]
            log.add(ScreenFaded(side=side_index, screen=screen))
