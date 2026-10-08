"""A specialist pilot for stall teams, layered over the ordinary search."""

from collections.abc import Sequence

from battle_sim.analysis import is_setting_up, residual_drain, strongest_hit
from battle_sim.engine.status_apply import status_cannot_land
from battle_sim.matchup import set_hazard
from battle_sim.mechanics.battle import BattleState
from battle_sim.models.actions import Action
from battle_sim.models.moves import HealEffect, Move
from battle_sim.models.pokemon import Pokemon
from battle_sim.search import SearchPlayer
from battle_sim.utils import Hazards, Status

# A wall commits to healing while there is still a bar to restore.
_HEAL_BELOW = 0.65
_HEALTHY_ENOUGH_TO_POISON = 0.5  # below this, a target is dying to damage and Toxic is a wasted turn
_HAZARD_CAPS = {Hazards.STEALTH_ROCK: 1, Hazards.SPIKES: 3, Hazards.TOXIC_SPIKES: 2, Hazards.STICKY_WEB: 1}


def _heal_fraction(move: Move) -> float:
    if move.healing:
        return 0.5
    return max((e.fraction for e in move.effects if isinstance(e, HealEffect)), default=0.0)


class StallPlayer(SearchPlayer):
    """Plays the stall plan where it is clear, and searches everywhere else."""

    def choose_action(self, state: BattleState, side_index: int, actions: Sequence[Action]) -> Action:
        side = state.sides[side_index]
        if len(actions) == 1 or side.needs_switch or side.active_pokemon.is_fainted():
            return super().choose_action(state, side_index, actions)
        planned = self._plan(state, side_index, actions)
        return planned if planned is not None else super().choose_action(state, side_index, actions)

    def _moves(self, state: BattleState, side_index: int, actions: Sequence[Action]) -> list[tuple[Move, Action]]:
        active = state.sides[side_index].active_pokemon
        out = []
        for action in actions:
            if action.move is None:
                continue
            move = active.moves[action.move]
            if move is not None:
                out.append((move, action))
        return out

    def _heal(
        self,
        me: Pokemon,
        state: BattleState,
        available: list[tuple[Move, Action]],
        surviving: bool,
        incoming: int,
    ) -> Action | None:
        """Heal while there is still something to heal, and only when it wins the race."""
        if me.live_stats.HP / me.stat_totals.HP >= _HEAL_BELOW:
            return None
        for move, action in available:
            gain = _heal_fraction(move) * me.stat_totals.HP
            if gain > 0 and gain > residual_drain(me, state) and (surviving or gain > incoming):
                return action
        return None

    def _plan(self, state: BattleState, side_index: int, actions: Sequence[Action]) -> Action | None:
        """The first rule that fires, or None to hand the turn back to the search."""
        me = state.sides[side_index].active_pokemon
        them = state.sides[1 - side_index].active_pokemon
        available = self._moves(state, side_index, actions)
        if not available or them.is_fainted():
            return None
        incoming = strongest_hit(them, me, state)
        surviving = me.live_stats.HP - incoming > 0

        # Phaze a sweeper before it is set up enough to matter.
        if is_setting_up(them):
            phaze = next((a for move, a in available if move.force_switch), None)
            if phaze is not None:
                return phaze

        heal = self._heal(me, state, available, surviving, incoming)
        if heal is not None:
            return heal

        # 3. Put the clock on anything that will still be standing to feel it.
        if them.status is Status.NONE and them.live_stats.HP > them.stat_totals.HP * _HEALTHY_ENOUGH_TO_POISON:
            toxic = next(
                (
                    a
                    for move, a in available
                    if move.name == "Toxic" and not status_cannot_land(Status.TOXIC, them, me, state.field)
                ),
                None,
            )
            if toxic is not None and surviving:
                return toxic

        # 4. Hazards, while there is still a bench left to walk into them.
        theirs = state.sides[1 - side_index]
        if surviving and sum(1 for p in theirs.bench if not p.is_fainted()) >= 2:
            for move, action in available:
                hazard = set_hazard(move)
                if hazard is not None and theirs.hazards.get(hazard, 0) < _HAZARD_CAPS.get(hazard, 1):
                    return action
        return None
