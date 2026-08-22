from battle_sim.mechanics.battle import BattleState, FieldState, SideState
from battle_sim.models.actions import Action, ActionType
from battle_sim.models.moves import Move
from battle_sim.models.pokemon import Pokemon
from battle_sim.utils import Ability, Category, Item, PseudoWeather, Stats, Status, Type, Weather

_CATEGORY_ORDER: dict[ActionType, int] = {
    ActionType.SWITCH_OUT: 0,
    ActionType.USE_ITEM: 1,
    ActionType.USE_MOVE: 2,
    ActionType.RUN: 3,
}
_WEATHER_SPEED_DOUBLERS: dict[Ability, frozenset[Weather]] = {
    Ability.SWIFT_SWIM: frozenset({Weather.RAIN, Weather.HEAVY_RAIN}),
    Ability.CHLOROPHYLL: frozenset({Weather.SUN, Weather.HARSH_SUN}),
    Ability.SAND_RUSH: frozenset({Weather.SANDSTORM}),
    Ability.SLUSH_RUSH: frozenset({Weather.SNOW}),
}


def effective_speed(pokemon: Pokemon, side: SideState, field: FieldState) -> int:
    speed = pokemon.effective_stat(Stats.SPEED)
    if pokemon.status is Status.PARALYSIS and pokemon.ability is not Ability.QUICK_FEET:
        speed = speed // 2
    if pokemon.ability is Ability.QUICK_FEET and pokemon.status is not Status.NONE:
        speed = speed * 3 // 2
    if field.weather in _WEATHER_SPEED_DOUBLERS.get(pokemon.ability, frozenset()):
        speed *= 2
    if pokemon.ability is Ability.UNBURDEN and pokemon.item is Item.NONE and pokemon.item_consumed:
        speed *= 2
    if pokemon.paradox_boost is Stats.SPEED:
        speed = speed * 3 // 2
    if side.tailwind_turns > 0:
        speed *= 2
    if pokemon.item is Item.CHOICE_SCARF:
        speed = speed * 3 // 2
    return max(1, speed)


def order_actions(actions: dict[int, Action], state: BattleState) -> list[tuple[int, Action]]:
    tie_breakers = {side_index: state.rng.random_probability() for side_index in actions}
    items = list(actions.items())
    items.sort(key=lambda pair: _sort_key(pair[0], pair[1], state, tie_breakers[pair[0]]))
    return items


def _sort_key(
    side_index: int, action: Action, state: BattleState, tie_breaker: float
) -> tuple[int, int, int, int, float]:
    side = state.sides[side_index]
    actor = side.active_pokemon

    category = _CATEGORY_ORDER[action.action]
    in_bracket_jump = 0

    if action.action is ActionType.USE_MOVE:
        slot = action.move
        assert slot is not None
        move = actor.moves[slot]
        assert move is not None
        priority_value = int(move.priority)
        priority_value += _priority_modifiers(actor, move)
        priority = -priority_value
        in_bracket_jump = -_bracket_jump(actor, state)
        if actor.ability is Ability.MYCELIUM_MIGHT and move.category is Category.STATUS:
            in_bracket_jump = 1  # status moves go last within their bracket
    else:
        priority = 0

    speed = effective_speed(actor, side, state.field)
    speed_key = speed if PseudoWeather.TRICK_ROOM in state.field.pseudo_weather else -speed

    return (category, priority, in_bracket_jump, speed_key, tie_breaker)


def _priority_modifiers(actor: Pokemon, move: Move) -> int:
    bonus = 0
    if actor.ability is Ability.PRANKSTER and move.category is Category.STATUS:
        bonus += 1
    if actor.ability is Ability.GALE_WINGS and move.type is Type.FLYING and actor.live_stats.HP == actor.stat_totals.HP:
        bonus += 1
    if actor.ability is Ability.TRIAGE and move.healing:
        bonus += 3
    return bonus


def _bracket_jump(actor: Pokemon, state: BattleState) -> int:
    """Quick Claw / Quick Draw / Custap Berry: a chance to move first within the priority bracket."""
    if actor.item is Item.QUICK_CLAW and state.rng.roll_chance(0.2):
        return 1
    if actor.ability is Ability.QUICK_DRAW and state.rng.roll_chance(0.3):
        return 1
    if actor.item is Item.CUSTAP_BERRY and 4 * actor.live_stats.HP <= actor.stat_totals.HP:
        actor.consume_item()
        return 1
    return 0
