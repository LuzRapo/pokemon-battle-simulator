from battle_sim.engine.hazards import _apply_entry_hazards
from battle_sim.engine.transform import restore_form
from battle_sim.mechanics.battle import BattleState, SideState
from battle_sim.mechanics.effects import register_ability, register_active, unregister_ability, unregister_active
from battle_sim.mechanics.events import Event, EventContext
from battle_sim.mechanics.log import BattleLog
from battle_sim.models.actions import Action, ActionType
from battle_sim.models.log_events import Healed, Switched
from battle_sim.models.pokemon import Pokemon
from battle_sim.utils import Ability, ExtraStatus, Status


def _execute_switch(state: BattleState, side_index: int, action: Action, log: BattleLog) -> None:
    assert action.switch_in is not None
    side = state.sides[side_index]
    target_index = next(i for i, pokemon in enumerate(side.team) if pokemon is action.switch_in)
    outgoing = side.active_pokemon
    incoming = side.team[target_index]
    if not outgoing.is_fainted():
        ctx = EventContext(rng=state.rng, battle=state, actor=outgoing, log=log)
        state.bus.emit(Event.ON_SWITCH_OUT, ctx, {"side_index": side_index})
    restore_form(outgoing, state)
    outgoing.reset_stat_stages()
    outgoing.volatiles.clear()
    outgoing.choice_locked_move = None
    outgoing.protect_streak = 0
    outgoing.last_move_slot = None
    outgoing.encored_slot = None
    outgoing.disabled_slot = None
    outgoing.locked_slot = None
    outgoing.charging_slot = None
    outgoing.rolling_hits = 0
    if outgoing.status is Status.TOXIC:
        outgoing.status_turns = 0
    unregister_active(state.bus, state.effects, outgoing)
    side.active[0] = target_index
    outgoing.flash_fire_active = False
    outgoing.paradox_boost = None
    outgoing.paradox_from_booster = False
    log.add(Switched(side=side_index, withdrew=outgoing.nickname, sent_out=incoming.nickname))
    incoming.turns_active = 0
    register_active(state.bus, state.effects, incoming)
    if Ability.NEUTRALIZING_GAS in (outgoing.ability, incoming.ability):
        _sync_neutralizing_gas(state)
    _apply_entry_hazards(state, incoming, side_index, log)
    if incoming.is_fainted():
        unregister_active(state.bus, state.effects, incoming)
        return
    _grant_healing_wish(side, side_index, incoming, log)
    if side.pending_substitute > 0 and ExtraStatus.SUBSTITUTE not in incoming.volatiles:
        incoming.volatiles[ExtraStatus.SUBSTITUTE] = side.pending_substitute  # Shed Tail's parting gift
        side.pending_substitute = 0
    ctx = EventContext(rng=state.rng, battle=state, actor=incoming, log=log)
    state.bus.emit(Event.ON_SWITCH_IN, ctx, {"side_index": side_index})


def _sync_neutralizing_gas(state: BattleState) -> None:
    """While a Neutralizing Gas holder is active, every other active's ability handlers are unwired.

    Direct ability checks outside the bus (speed modifiers, recoil exemptions) are not suppressed.
    """
    gas_active = any(
        not side.active_pokemon.is_fainted() and side.active_pokemon.ability is Ability.NEUTRALIZING_GAS
        for side in state.sides
    )
    for side in state.sides:
        active = side.active_pokemon
        if active.is_fainted() or active.ability is Ability.NEUTRALIZING_GAS:
            continue
        if gas_active:
            unregister_ability(state.bus, state.effects, active)
        else:
            register_ability(state.bus, state.effects, active)


def _force_random_switch(state: BattleState, side_index: int, log: BattleLog) -> None:
    """Phazing (Whirlwind / Roar / Dragon Tail / Red Card): a random healthy teammate is dragged in."""
    side = state.sides[side_index]
    bench = [i for i, member in enumerate(side.team) if i != side.active[0] and not member.is_fainted()]
    if not bench:
        return
    target_index = bench[state.rng.random_integer(0, len(bench))]
    _execute_switch(state, side_index, Action(action=ActionType.SWITCH_OUT, switch_in=side.team[target_index]), log)
    side.needs_switch = False  # the drag IS the replacement


def _grant_healing_wish(side: SideState, side_index: int, incoming: Pokemon, log: BattleLog) -> None:
    """A pending Healing Wish waits until a hurt or statused pokemon arrives (PS gen 8+)."""
    if not side.healing_wish_pending:
        return
    hurt = incoming.live_stats.HP < incoming.stat_totals.HP or incoming.status is not Status.NONE
    if not hurt:
        return
    side.healing_wish_pending = False
    healed = incoming.apply_healing(incoming.stat_totals.HP)
    incoming.status = Status.NONE
    incoming.status_turns = 0
    log.add(Healed(side=side_index, pokemon=incoming.nickname, amount=healed))
