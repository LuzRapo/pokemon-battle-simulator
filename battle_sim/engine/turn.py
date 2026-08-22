from battle_sim.engine.actions import _execute_action
from battle_sim.engine.outcome import _update_outcome
from battle_sim.engine.residuals import _apply_residuals
from battle_sim.engine.switching import _execute_switch, _sync_neutralizing_gas
from battle_sim.mechanics.battle import BattleState
from battle_sim.mechanics.events import Event, EventContext
from battle_sim.mechanics.log import BattleLog
from battle_sim.mechanics.priority import effective_speed, order_actions
from battle_sim.models.actions import Action, ActionType
from battle_sim.models.log_events import SelfSwitchPending
from battle_sim.utils import Ability


def step(state: BattleState, actions: dict[int, Action]) -> BattleLog:
    log = BattleLog()
    bus = state.bus
    turn_ctx = EventContext(rng=state.rng, battle=state, log=log)
    choosers = {side_index: side.active_pokemon for side_index, side in enumerate(state.sides)}
    for side_index, side in enumerate(state.sides):
        side.chosen_action = actions[side_index]
        side.acted_this_turn = False
        side.active_pokemon.last_hit_taken = 0
        side.active_pokemon.last_hit_category = None
    if state.turn == 0:
        _send_out_leads(state, log)
    bus.emit(Event.ON_TURN_START, turn_ctx)

    for side_index, action in order_actions(actions, state):
        if state.outcome is not None:
            break
        actor = state.sides[side_index].active_pokemon
        side = state.sides[side_index]
        if actor is not choosers[side_index]:
            continue  # phazed out before acting: the queued move left with the mon that chose it
        if actor.is_fainted() and action.action is not ActionType.SWITCH_OUT:
            continue
        if side.needs_switch and not side.acted_this_turn:
            continue  # ejected before acting: the pending action is forfeited
        _execute_action(state, side_index, action, log)
        side.acted_this_turn = True
        _resolve_eject_packs(state, log)
        _update_outcome(state, log)

    if state.outcome is None:
        _apply_residuals(state, log)
        _update_outcome(state, log)

    bus.emit(Event.ON_TURN_END, turn_ctx)
    state.turn += 1
    return log


def _resolve_eject_packs(state: BattleState, log: BattleLog) -> None:
    """An Eject Pack pulls its holder once the triggering action has resolved."""
    for side_index, side in enumerate(state.sides):
        active = side.active_pokemon
        if not active.eject_pending:
            continue
        active.eject_pending = False
        has_healthy_bench = any(i != side.active[0] and not p.is_fainted() for i, p in enumerate(side.team))
        if active.is_fainted() or not has_healthy_bench:
            continue
        active.consume_item()
        side.needs_switch = True
        log.add(SelfSwitchPending(side=side_index, pokemon=active.nickname))


def _send_out_leads(state: BattleState, log: BattleLog) -> None:
    """Battle start: lead abilities fire in speed order, so the slower weather-setter's weather stands."""
    if any(side.active_pokemon.ability is Ability.NEUTRALIZING_GAS for side in state.sides):
        _sync_neutralizing_gas(state)
    order = sorted(
        (0, 1),
        key=lambda i: effective_speed(state.sides[i].active_pokemon, state.sides[i], state.field),
        reverse=True,
    )
    for side_index in order:
        lead = state.sides[side_index].active_pokemon
        ctx = EventContext(rng=state.rng, battle=state, log=log, actor=lead)
        state.bus.emit(Event.ON_SWITCH_IN, ctx, {"side_index": side_index})


def apply_forced_switch(state: BattleState, side_index: int, action: Action) -> BattleLog:
    if action.action is not ActionType.SWITCH_OUT:
        raise ValueError("Forced switch requires a SWITCH_OUT action.")
    log = BattleLog()
    _execute_switch(state, side_index, action, log)
    state.sides[side_index].needs_switch = False
    _update_outcome(state, log)  # entry hazards can faint the replacement and end the battle
    return log
