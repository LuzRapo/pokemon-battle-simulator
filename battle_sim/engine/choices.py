"""The authoritative list of actions the engine will accept for a side.

Every action returned here is accepted by step()/apply_forced_switch() without
crashing; submitting anything else is an illegal action and asserts. A returned
action is not guaranteed to *succeed* (a Disabled or Taunt-blocked move wastes
the turn, exactly as the engine resolves it).
"""

from battle_sim.mechanics.battle import BattleState, SideState
from battle_sim.models.actions import Action, ActionType
from battle_sim.models.moves import MoveSlot
from battle_sim.models.pokemon import Pokemon
from battle_sim.utils import Ability, Category, ExtraStatus, Type


def legal_actions(state: BattleState, side_index: int) -> list[Action]:
    side = state.sides[side_index]
    active = side.active_pokemon
    if active.is_fainted() or side.needs_switch:
        return _switch_actions(side)
    if ExtraStatus.LOCKED_MOVE in active.volatiles and active.locked_slot is not None:
        return [_move_action(active, active.locked_slot)]  # a rampage forbids switching
    if ExtraStatus.CHARGING in active.volatiles and active.charging_slot is not None:
        return [_move_action(active, active.charging_slot)]  # committed to releasing the charged move
    if _trapped(active, state.sides[1 - side_index].active_pokemon):
        return _move_actions(active)
    return _move_actions(active) + _switch_actions(side)


def _trapped(active: Pokemon, opponent: Pokemon) -> bool:
    """Shadow Tag / Magnet Pull; ghosts are free to leave (gen 6+)."""
    if opponent.is_fainted() or Type.GHOST in active.types:
        return False
    if opponent.ability is Ability.SHADOW_TAG and active.ability is not Ability.SHADOW_TAG:
        return True
    return opponent.ability is Ability.MAGNET_PULL and Type.STEEL in active.types


def _move_actions(active: Pokemon) -> list[Action]:
    forced = next((lock for lock in (active.choice_locked_move, active.encored_slot) if lock is not None), None)
    slots = [forced] if forced is not None else [s for s in MoveSlot if active.moves[s] is not None]
    usable = [s for s in slots if active.pp[s] > 0 and s is not active.disabled_slot and not _taunt_blocked(active, s)]
    if usable:
        return [_move_action(active, s) for s in usable]
    if forced is not None or all(pp == 0 for pp in active.pp.values()):
        # Constrained into an empty move, or everything exhausted: the engine substitutes Struggle.
        return [_move_action(active, forced if forced is not None else slots[0])]
    # Only Disabled / Taunt-blocked moves remain: the engine accepts them and wastes the turn.
    return [_move_action(active, s) for s in slots if active.pp[s] > 0]


def _taunt_blocked(active: Pokemon, slot: MoveSlot) -> bool:
    move = active.moves[slot]
    assert move is not None  # callers only pass filled slots
    return ExtraStatus.TAUNT in active.volatiles and move.category is Category.STATUS


def _move_action(active: Pokemon, slot: MoveSlot) -> Action:
    move = active.moves[slot]
    assert move is not None  # callers only pass filled slots
    return Action(action=ActionType.USE_MOVE, target=move.target, move=slot)


def _switch_actions(side: SideState) -> list[Action]:
    return [
        Action(action=ActionType.SWITCH_OUT, switch_in=pokemon)
        for index, pokemon in enumerate(side.team)
        if index != side.active[0] and not pokemon.is_fainted()
    ]
