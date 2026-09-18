"""A specialist pilot for stall teams, layered over the ordinary search.

Stall is the one archetype our search demonstrably cannot drive. Every defensive backbone finished in
the bottom six of the Anything Goes round-robin; the Elite Four's stall team rated 1866 against its
hyper-offence team's 2482; and on the real ladder the same two compositions go 50.7% and 43.1%, so
the ordering does not merely differ, it inverts. The cause is measured and structural: the search
sees 1.4 turns of a forty-turn game, and a stall win condition pays off ten to twenty turns out.

Raising the coefficients that price those payoffs did not fix it. Two A/B runs of 760 games each came
back at 52.2% and 51.0%, both inside one standard error of nothing, because `genome_prior` blends the
myopic action scorer in at equal weight and its spread across candidate actions (~3.7) dwarfs
anything the position evaluator contributes (0.1-1.0). Evaluator changes get diluted before they
reach the decision.

This takes the other route. It is not a better evaluator; it is a *plan*, expressed as a short ladder
of guarded rules, and it bypasses the scorer entirely rather than trying to shout over it. Each rule
fires only when it is unambiguous, and anything that is not covered falls through to the full search
-- so this keeps the tactical play it is good at and adds only the long-horizon judgement it lacks.
"""

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

# A wall commits to healing here rather than at death's door: recovery has to happen while there is
# still a bar to restore, and waiting for 30% is how a staller loses to a roll it could have avoided.
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

    def _heal(  # noqa: PLR0913
        self,
        me: Pokemon,
        state: BattleState,
        available: list[tuple[Move, Action]],
        surviving: bool,
        incoming: int,
    ) -> Action | None:
        """Heal while there is still something to heal, and only when it wins the race.

        `residual_drain` is the guard that stopped a badly poisoned Moltres roosting eleven turns
        running against a clock it could not out-heal.
        """
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

        # 1. Blow a sweeper out before it is set up enough to matter. Phazing is the one answer a
        #    wall has to a boosted attacker, and the search reaches for it a turn or two too late.
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
