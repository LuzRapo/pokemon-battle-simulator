"""The authoritative list of actions the engine will accept for a side.

Every action returned here is accepted by step()/apply_forced_switch() without
crashing; submitting anything else is an illegal action and asserts. A returned
action is not guaranteed to *succeed* (a Disabled or Taunt-blocked move wastes
the turn, exactly as the engine resolves it).
"""

from dataclasses import replace

from battle_sim.mechanics.battle import BattleState, SideState
from battle_sim.models.actions import Action, ActionType
from battle_sim.models.moves import MoveSlot
from battle_sim.models.pokemon import Pokemon
from battle_sim.utils import Ability, Category, ExtraStatus, Item, Type
from battle_sim.zmoves import z_move_for


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
        return _move_actions(active, side)
    return _move_actions(active, side) + _switch_actions(side)


def _trapped(active: Pokemon, opponent: Pokemon) -> bool:
    """Shadow Tag / Magnet Pull / Arena Trap, and being wrapped; ghosts are free to leave (gen 6+).

    Shed Shell is the item answer to all of it, and the only reason a wall carries one instead of
    Leftovers — a Blissey that cannot escape a Shadow Tag is simply removed from the game.
    """
    if opponent.is_fainted() or Type.GHOST in active.types or active.item is Item.SHED_SHELL:
        return False
    if ExtraStatus.PARTIALLY_TRAPPED in active.volatiles:
        return True  # held until the grip runs out; `residuals._partial_trap` counts it down
    if opponent.ability is Ability.SHADOW_TAG and active.ability is not Ability.SHADOW_TAG:
        return True
    if opponent.ability is Ability.ARENA_TRAP and active.is_grounded():
        return True  # Dugtrio's whole reason for existing, and it was missing from this list
    return opponent.ability is Ability.MAGNET_PULL and Type.STEEL in active.types


def _move_actions(active: Pokemon, side: SideState) -> list[Action]:
    forced = next((lock for lock in (active.choice_locked_move, active.encored_slot) if lock is not None), None)
    slots = [forced] if forced is not None else [s for s in MoveSlot if active.moves[s] is not None]
    usable = [s for s in slots if active.pp[s] > 0 and s is not active.disabled_slot and not _taunt_blocked(active, s)]
    if usable:
        return [a for s in usable for a in _slot_actions(active, side, s)]
    if forced is not None or all(pp == 0 for pp in active.pp.values()):
        # Constrained into an empty move, or everything exhausted: the engine substitutes Struggle.
        return [_move_action(active, forced if forced is not None else slots[0])]
    # Only Disabled / Taunt-blocked moves remain: the engine accepts them and wastes the turn.
    return [_move_action(active, s) for s in slots if active.pp[s] > 0]


def _taunt_blocked(active: Pokemon, slot: MoveSlot) -> bool:
    move = active.moves[slot]
    assert move is not None  # callers only pass filled slots
    return ExtraStatus.TAUNT in active.volatiles and move.category is Category.STATUS


def _slot_actions(active: Pokemon, side: SideState, slot: MoveSlot) -> list[Action]:
    """The plain use of this slot, plus its Z-move if the held crystal can upgrade it.

    Only moves matching the crystal's type qualify, so this adds one or two actions rather than
    doubling the move set the way a Mega variant of every move would.
    """
    plain = _move_action(active, slot)
    if side.has_used_z_move:
        return [plain]
    move = active.moves[slot]
    assert move is not None  # callers only pass filled slots
    if z_move_for(active.item, move) is None:
        return [plain]
    return [plain, replace(plain, z_move=True)]


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
