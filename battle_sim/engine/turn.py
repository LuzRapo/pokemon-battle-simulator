from collections.abc import Callable

from battle_sim.engine.actions import _execute_action
from battle_sim.engine.outcome import _update_outcome
from battle_sim.engine.residuals import _apply_residuals
from battle_sim.engine.switching import _execute_switch, _sync_neutralizing_gas
from battle_sim.formes import apply_forme, mega_forme
from battle_sim.mechanics.battle import BattleState
from battle_sim.mechanics.effects import rewire_active
from battle_sim.mechanics.events import Event, EventContext
from battle_sim.mechanics.log import BattleLog
from battle_sim.mechanics.priority import effective_speed, order_actions
from battle_sim.models.actions import Action, ActionType
from battle_sim.models.log_events import FormeChanged, SelfSwitchPending
from battle_sim.models.pokemon import Pokemon
from battle_sim.utils import Ability

type SwitchChooser = Callable[[BattleState, int], Action | None]


def step(state: BattleState, actions: dict[int, Action], switch_chooser: SwitchChooser | None = None) -> BattleLog:
    """Resolve one turn. `switch_chooser` is asked for a replacement the instant a pivot move (or
    Eject Pack/Button) forces one — real Pokemon sends U-turn/Volt Switch's user out immediately, so
    a target still waiting to act this turn hits whoever just switched in, not the pivot's user.
    Without it (the default), a forced switch is left for the caller's post-turn replacement loop,
    same as a faint — correct for a faint (nothing is left to threaten the fainted side this turn
    either way) but not for a pivot when its target hasn't acted yet. Returning None from the
    chooser defers that one switch the old way, for a side that cannot answer synchronously (a human
    waiting on a Discord button, say).
    """
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
    _resolve_mega_evolution(state, log)

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
        if switch_chooser is not None:
            _resolve_pending_switches(state, switch_chooser, log)
        _update_outcome(state, log)

    if state.outcome is None:
        _apply_residuals(state, log)
        _update_outcome(state, log)

    _tick_turns_active(state, choosers)

    bus.emit(Event.ON_TURN_END, turn_ctx)
    state.turn += 1
    return log


def _tick_turns_active(state: BattleState, choosers: dict[int, Pokemon]) -> None:
    """One more whole turn survived on the field, for Fake Out/First Impression's switch-in check.

    Skips whoever is active at turn-end but wasn't the one who chose this turn's action — a switch
    (voluntary, forced by a faint, or a self-switch effect) that lands mid-turn didn't get a move
    of its own this turn, so it must still read as freshly switched in when the next turn starts.
    """
    for side_index, side in enumerate(state.sides):
        active = side.active_pokemon
        if not active.is_fainted() and active is choosers[side_index]:
            active.turns_active += 1


def _resolve_mega_evolution(state: BattleState, log: BattleLog) -> None:
    """Mega Evolve any active holding its matching stone, before a single move is ordered.

    Placed before `order_actions` deliberately: `effective_speed` reads `stat_totals` at sort time,
    so the new Speed decides this turn's order — which is what Gen 7 does. A side that chose to
    switch does not Mega Evolve; the two are mutually exclusive in one turn.

    Simplification: a Pokemon Megas at its first opportunity rather than the player choosing when.
    Delaying is occasionally right (Abomasnow-Mega and Slowbro-Mega lose Speed), but making it a
    choice means offering a Mega variant of every move, doubling the branching factor at every
    search node — too expensive for the value it adds.
    """
    for side_index, side in enumerate(state.sides):
        chosen = side.chosen_action
        if side.has_mega_evolved or chosen is None or chosen.action is not ActionType.USE_MOVE:
            continue
        active = side.active_pokemon
        if active.is_fainted():
            continue
        forme = mega_forme(active.name, active.item)
        if forme is None:
            continue
        apply_forme(active, forme)
        rewire_active(state.bus, state.effects, active)  # the new forme's ability replaces the old one's handlers
        side.has_mega_evolved = True
        log.add(FormeChanged(side=side_index, pokemon=active.nickname, forme=active.name))


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


def _resolve_pending_switches(state: BattleState, switch_chooser: SwitchChooser, log: BattleLog) -> None:
    """Send in whoever `switch_chooser` names for any side a pivot (or Eject Pack/Button) just
    forced out — before the loop moves on to an action that should be facing the replacement, not
    the mon that just left. A `None` answer leaves `needs_switch` set, for the caller's own post-turn
    replacement loop to pick up exactly as it always has.
    """
    for side_index, side in enumerate(state.sides):
        if not side.needs_switch:
            continue
        replacement = switch_chooser(state, side_index)
        if replacement is None:
            continue
        _execute_switch(state, side_index, replacement, log)
        side.needs_switch = False


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
