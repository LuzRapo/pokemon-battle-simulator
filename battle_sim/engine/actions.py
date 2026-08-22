from collections.abc import Iterator
from contextlib import contextmanager

from battle_sim.engine.moves import _execute_move
from battle_sim.engine.switching import _execute_switch
from battle_sim.mechanics.battle import BattleState
from battle_sim.mechanics.effects import register_ability, unregister_ability
from battle_sim.mechanics.events import Event, EventContext
from battle_sim.mechanics.log import BattleLog
from battle_sim.models.actions import Action, ActionType
from battle_sim.utils import Ability

_MOLD_BREAKERS = frozenset({Ability.MOLD_BREAKER, Ability.TERAVOLT, Ability.TURBOBLAZE})


def _execute_action(state: BattleState, side_index: int, action: Action, log: BattleLog) -> None:
    actor = state.sides[side_index].active_pokemon
    ctx = EventContext(rng=state.rng, battle=state, actor=actor, action=action, log=log)
    state.bus.emit(Event.ON_BEFORE_ACTION, ctx)

    if action.action is ActionType.SWITCH_OUT:
        _execute_switch(state, side_index, action, log)
    elif action.action is ActionType.USE_MOVE:
        with _mold_breaker_window(state, side_index):
            _execute_move(state, side_index, action, log)

    state.bus.emit(Event.ON_AFTER_ACTION, ctx)


@contextmanager
def _mold_breaker_window(state: BattleState, side_index: int) -> Iterator[None]:
    """Mold Breaker / Teravolt / Turboblaze: the defender's ability handlers are unwired for the move's duration."""
    attacker = state.sides[side_index].active_pokemon
    defender = state.sides[1 - side_index].active_pokemon
    suppress = attacker.ability in _MOLD_BREAKERS and not defender.is_fainted()
    if suppress:
        unregister_ability(state.bus, state.effects, defender)
    try:
        yield
    finally:
        still_active = state.sides[1 - side_index].active_pokemon is defender
        if suppress and still_active and not defender.is_fainted():
            register_ability(state.bus, state.effects, defender)
