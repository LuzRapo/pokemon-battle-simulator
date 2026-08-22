from battle_sim.mechanics.battle import BattleState, SideState
from battle_sim.mechanics.items import WEATHER_ROCKS, consume_terrain_seed
from battle_sim.mechanics.log import BattleLog
from battle_sim.models.log_events import (
    HazardsCleared,
    HazardSet,
    MoveFailed,
    PseudoWeatherStarted,
    ScreenFaded,
    ScreenSet,
    StatusCleared,
    TailwindSet,
    TerrainChanged,
    TerrainFaded,
    WeatherChanged,
)
from battle_sim.models.moves import (
    Move,
    PseudoWeatherEffect,
    RemoveHazardsEffect,
    SideConditionEffect,
    TerrainEffect,
    WeatherEffect,
)
from battle_sim.models.pokemon import Pokemon
from battle_sim.utils import ExtraStatus, Hazards, Item, Target, Terrain

_SCREEN_HAZARDS = frozenset({Hazards.REFLECT, Hazards.LIGHT_SCREEN, Hazards.AURORA_VEIL})
_MAX_HAZARD_LAYERS: dict[Hazards, int] = {Hazards.SPIKES: 3, Hazards.TOXIC_SPIKES: 2}


def _apply_field_effect(
    effect: WeatherEffect | TerrainEffect | PseudoWeatherEffect, attacker: Pokemon, state: BattleState, log: BattleLog
) -> None:
    if isinstance(effect, WeatherEffect):
        duration = 8 if attacker.item is WEATHER_ROCKS.get(effect.kind) else (effect.duration_turns or 5)
        state.field.weather = effect.kind
        state.field.weather_turns_left = duration
        log.add(WeatherChanged(weather=effect.kind))
    elif isinstance(effect, TerrainEffect):
        duration = 8 if attacker.item is Item.TERRAIN_EXTENDER else (effect.duration_turns or 5)
        state.field.terrain = effect.kind
        state.field.terrain_turns_left = duration
        log.add(TerrainChanged(terrain=effect.kind))
        for side_index, side in enumerate(state.sides):
            consume_terrain_seed(side.active_pokemon, side_index, effect.kind, log)
    else:
        state.field.pseudo_weather[effect.kind] = effect.duration_turns or 5
        log.add(PseudoWeatherStarted(kind=effect.kind))


def _apply_remove_hazards(
    effect: RemoveHazardsEffect, attacker: Pokemon, attacker_side_index: int, state: BattleState, log: BattleLog
) -> None:
    own_side = state.sides[attacker_side_index]
    _clear_hazards(own_side, attacker_side_index, log)
    if effect.style == "RAPID_SPIN":
        if attacker.volatiles.pop(ExtraStatus.LEECH_SEED, None) is not None:
            log.add(
                StatusCleared(side=attacker_side_index, pokemon=attacker.nickname, clearance="freed_from_leech_seed")
            )
        return
    target_index = 1 - attacker_side_index
    target_side = state.sides[target_index]
    _clear_hazards(target_side, target_index, log)
    for screen in list(target_side.screens):
        del target_side.screens[screen]
        log.add(ScreenFaded(side=target_index, screen=screen))
    if state.field.terrain is not Terrain.NONE:
        prior_terrain = state.field.terrain
        state.field.terrain = Terrain.NONE
        state.field.terrain_turns_left = 0
        log.add(TerrainFaded(terrain=prior_terrain))


def _clear_hazards(side: SideState, side_index: int, log: BattleLog) -> None:
    for hazard in list(side.hazards):
        del side.hazards[hazard]
        log.add(HazardsCleared(side=side_index, hazard=hazard))


def _apply_side_condition(
    effect: SideConditionEffect,
    move: Move,
    attacker: Pokemon,
    attacker_side_index: int,
    state: BattleState,
    log: BattleLog,
) -> None:
    target_index = 1 - attacker_side_index if move.target is Target.OPPONENT_SIDE else attacker_side_index
    target_side = state.sides[target_index]

    if effect.kind is Hazards.TAILWIND:
        if target_side.tailwind_turns > 0:
            log.add(MoveFailed())
            return
        assert effect.duration_turns is not None  # the loader always sets Tailwind's duration
        target_side.tailwind_turns = effect.duration_turns
        log.add(TailwindSet(side=target_index))
        return

    if effect.kind in _SCREEN_HAZARDS:
        target_side.screens[effect.kind] = 8 if attacker.item is Item.LIGHT_CLAY else (effect.duration_turns or 5)
        log.add(ScreenSet(side=target_index, screen=effect.kind))
        return

    max_layers = _MAX_HAZARD_LAYERS.get(effect.kind, 1)
    current = target_side.hazards.get(effect.kind, 0)
    if current >= max_layers:
        log.add(MoveFailed())
        return
    target_side.hazards[effect.kind] = current + 1
    log.add(HazardSet(side=target_index, hazard=effect.kind))
