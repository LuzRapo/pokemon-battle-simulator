from battle_sim.mechanics.battle import BattleState, FieldState, SideState
from battle_sim.models.actions import Action, ActionType
from battle_sim.models.moves import Move
from battle_sim.models.pokemon import Pokemon
from battle_sim.utils import (
    Ability,
    Category,
    ExtraStatus,
    Item,
    PseudoWeather,
    Stats,
    Status,
    Terrain,
    Type,
    Weather,
)

_CATEGORY_ORDER: dict[ActionType, int] = {
    ActionType.SWITCH_OUT: 0,
    ActionType.USE_ITEM: 1,
    ActionType.USE_MOVE: 2,
    ActionType.RUN: 3,
}
# Sorts after every move there is. Not a priority value any move can hold — it is the butler being
# made to deal with what was just put in his paws, and he has to be handed it before he can eat it.
_AFTER_EVERY_MOVE = 100
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
    if ExtraStatus.SLOW_START in pokemon.volatiles:
        speed = speed // 2
    if pokemon.ability is Ability.QUICK_FEET and pokemon.status is not Status.NONE:
        speed = speed * 3 // 2
    if field.weather in _WEATHER_SPEED_DOUBLERS.get(pokemon.ability, frozenset()):
        speed *= 2
    if pokemon.ability is Ability.SURGE_SURFER and field.terrain is Terrain.ELECTRIC:
        speed *= 2  # the weather doublers' terrain cousin, and the whole of Alolan Raichu's identity
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
        if move.name == "Pursuit" and _target_is_leaving(side_index, state):
            # Pursuit's whole point: it catches its target on the way out, so it has to resolve
            # ahead of a switch that would otherwise take the target off the field first. Switches
            # already sort before every move, so this is the one thing that has to sort before them.
            category = _CATEGORY_ORDER[ActionType.SWITCH_OUT] - 1
        priority_value = int(move.priority)
        priority_value += _priority_modifiers(actor, move)
        priority = -priority_value
        if _about_to_be_tricked(side_index, state):
            priority = _AFTER_EVERY_MOVE
        in_bracket_jump = -_bracket_jump(actor, state)
        if actor.ability is Ability.MYCELIUM_MIGHT and move.category is Category.STATUS:
            in_bracket_jump = 1  # status moves go last within their bracket
    else:
        priority = 0

    speed = effective_speed(actor, side, state.field)
    speed_key = speed if PseudoWeather.TRICK_ROOM in state.field.pseudo_weather else -speed

    return (category, priority, in_bracket_jump, speed_key, tie_breaker)


def _about_to_be_tricked(side_index: int, state: BattleState) -> bool:
    """Whether this turn's job for the butler is eating whatever is about to be Tricked onto him.

    Read off the *other* side's chosen action, the way Pursuit reads a switch — the trade has to
    land before there is anything to eat, so whatever he had planned waits until after it.

    Only Nine Lives. For everybody else a Trick is an ordinary move resolving in speed order.
    """
    actor = state.sides[side_index].active_pokemon
    if actor.ability is not Ability.NINE_LIVES:
        return False
    return _opponent_is_using(1 - side_index, state, "Trick")


def _opponent_is_using(side_index: int, state: BattleState, move_name: str) -> bool:
    """Read off the side's declared action, exactly as `_target_is_leaving` reads a switch."""
    side = state.sides[side_index]
    chosen = side.chosen_action
    if side.acted_this_turn or chosen is None or chosen.action is not ActionType.USE_MOVE or chosen.move is None:
        return False
    move = side.active_pokemon.moves[chosen.move]
    return move is not None and move.name == move_name


def _target_is_leaving(side_index: int, state: BattleState) -> bool:
    """Whether the opponent has chosen to switch out and has not gone yet."""
    opponent = state.sides[1 - side_index]
    chosen = opponent.chosen_action
    return not opponent.acted_this_turn and chosen is not None and chosen.action is ActionType.SWITCH_OUT


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
