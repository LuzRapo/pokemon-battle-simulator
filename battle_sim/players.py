"""Reference players for the headless runner.

RandomPlayer is the fuzzing/fitness floor. BasicPlayer plays like a person with a damage
calculator open: it weighs each move's expected damage against the opponent's worst-case
answer, takes guaranteed KOs, retreats from hopeless matchups, and spends safe turns on
hazards, status, healing, or setup. It is fully deterministic.
"""

import random
from collections.abc import Sequence
from dataclasses import dataclass

from battle_sim.analysis import damage_range, expected_damage, strongest_hit, usable_moves
from battle_sim.maths.damage import move_effectiveness
from battle_sim.mechanics.battle import BattleState
from battle_sim.mechanics.priority import effective_speed
from battle_sim.models.actions import Action, ActionType
from battle_sim.models.moves import (
    DamageEffect,
    HealEffect,
    InflictStatusEffect,
    Move,
    SideConditionEffect,
    StatStageChangeEffect,
)
from battle_sim.models.pokemon import Pokemon
from battle_sim.models.spec import PokemonSpec
from battle_sim.observation import BeliefSampler
from battle_sim.runner import Player
from battle_sim.teams import build_pokemon
from battle_sim.utils import Hazards, Status

_ENTRY_HAZARDS = frozenset({Hazards.STEALTH_ROCK, Hazards.SPIKES, Hazards.TOXIC_SPIKES, Hazards.STICKY_WEB})

_DEAD_MOVE = 0.01  # below any real chip damage: never pick a no-op over an attack


@dataclass(frozen=True)
class Weights:
    """BasicPlayer's scoring constants, exposed as an evolvable genome.

    Scores are fractions of the opponent's remaining HP a turn is worth; utility plays
    carry fixed weights so they win only when no attack threatens comparable damage.
    Defaults are the hand-tuned stock values.
    """

    guaranteed_ko: float = 10.0
    heal_when_hurt: float = 0.65
    hazard_setup: float = 0.5
    status_spread: float = 0.45
    stat_setup: float = 0.4
    safe_threat_fraction: float = 0.25  # "safe" = the opponent's worst case costs less than this much of our HP
    switch_advantage_margin: float = 0.35


@dataclass
class FixedOrderPlayer:
    """Wraps another player, pinning team preview to that team's own listed order instead of
    letting it choose a lead strategically — for anything (a Mirror Mode duel, a deterministic
    sim setup) that needs both sides to send Pokémon out in the same sequence."""

    inner: Player

    def choose_order(self, own: Sequence[PokemonSpec], opponent: Sequence[PokemonSpec]) -> Sequence[int]:
        return list(range(len(own)))

    def choose_action(self, state: BattleState, side_index: int, actions: Sequence[Action]) -> Action:
        return self.inner.choose_action(state, side_index, actions)

    def bind_belief_sampler(self, sampler: BeliefSampler) -> None:
        """Forwarded so wrapping a `Determinizing` player (a `SearchPlayer`) still gets bound —
        `interactive.py`'s `isinstance(ai, Determinizing)` check only sees this wrapper, not `inner`.
        A no-op for an inner player that never needed one."""
        bind = getattr(self.inner, "bind_belief_sampler", None)
        if bind is not None:
            bind(sampler)


class RandomPlayer:
    """Uniform over legal actions and team orders."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def choose_order(self, own: Sequence[PokemonSpec], opponent: Sequence[PokemonSpec]) -> Sequence[int]:
        order = list(range(len(own)))
        self.rng.shuffle(order)
        return order

    def choose_action(self, state: BattleState, side_index: int, actions: Sequence[Action]) -> Action:
        return self.rng.choice(list(actions))


class BasicPlayer:
    """A deterministic damage-calculator heuristic over the information a human player has."""

    def __init__(self, weights: Weights = Weights()):
        self.weights = weights

    def choose_order(self, own: Sequence[PokemonSpec], opponent: Sequence[PokemonSpec]) -> Sequence[int]:
        ours = [build_pokemon(spec) for spec in own]
        theirs = [build_pokemon(spec) for spec in opponent]
        scores = [sum(_preview_edge(mine, foe) for foe in theirs) for mine in ours]
        return sorted(range(len(own)), key=lambda i: scores[i], reverse=True)

    def choose_action(self, state: BattleState, side_index: int, actions: Sequence[Action]) -> Action:
        me = state.sides[side_index].active_pokemon
        opponent = state.sides[1 - side_index].active_pokemon
        switches = [a for a in actions if a.action is ActionType.SWITCH_OUT]
        moves = [a for a in actions if a.action is ActionType.USE_MOVE]
        if not moves:  # fainted or otherwise forced out: pick the best matchup
            return max(switches, key=lambda a: self._matchup(_required(a.switch_in), opponent, state))

        scored = [(self._move_score(action, me, opponent, state, side_index), action) for action in moves]
        best_score, best_action = max(scored, key=lambda pair: pair[0])

        if switches and best_score < self.weights.guaranteed_ko and self._is_doomed(me, opponent, state, side_index):
            escape = max(switches, key=lambda a: self._matchup(_required(a.switch_in), opponent, state))
            candidate = _required(escape.switch_in)
            if (
                self._matchup(candidate, opponent, state)
                > self._matchup(me, opponent, state) + self.weights.switch_advantage_margin
            ):
                return escape
        return best_action

    def _move_score(self, action: Action, me: Pokemon, opponent: Pokemon, state: BattleState, side_index: int) -> float:
        assert action.move is not None  # USE_MOVE actions always carry a slot
        move = me.moves[action.move]
        assert move is not None  # legal actions only point at filled slots
        if me.pp[action.move] == 0:
            return _DEAD_MOVE  # the engine will substitute Struggle
        low, high = damage_range(move, me, opponent, state)
        accuracy = move.accuracy_probability if move.accuracy_probability is not None else 1.0
        if low >= opponent.live_stats.HP and low > 0:
            return self.weights.guaranteed_ko * accuracy
        if high > 0:
            return min(1.0, (low + high) / 2 / max(1, opponent.live_stats.HP)) * accuracy
        return self._utility_score(move, me, opponent, state, side_index)

    def _utility_score(self, move: Move, me: Pokemon, opponent: Pokemon, state: BattleState, side_index: int) -> float:
        threatened = strongest_hit(opponent, me, state) / me.stat_totals.HP
        if _heals(move) and 2 * me.live_stats.HP < me.stat_totals.HP:
            return self.weights.heal_when_hurt
        if (hazard := _sets_hazard(move)) is not None:
            laid = state.sides[1 - side_index].hazards.get(hazard, 0)
            if laid == 0:
                return self.weights.hazard_setup
        if _spreads_status(move) and opponent.status is Status.NONE:
            return self.weights.status_spread
        if _boosts_self(move) and threatened < self.weights.safe_threat_fraction:
            return self.weights.stat_setup
        return _DEAD_MOVE

    def _is_doomed(self, me: Pokemon, opponent: Pokemon, state: BattleState, side_index: int) -> bool:
        """They can take us out before our best answer lands."""
        incoming = strongest_hit(opponent, me, state)
        if incoming < me.live_stats.HP:
            return False
        my_side, their_side = state.sides[side_index], state.sides[1 - side_index]
        return effective_speed(opponent, their_side, state.field) > effective_speed(me, my_side, state.field)

    def _matchup(self, candidate: Pokemon, opponent: Pokemon, state: BattleState) -> float:
        """Offensive pressure minus the fraction of the candidate the opponent removes per turn."""
        offense = max(
            (expected_damage(move, candidate, opponent, state) for move in usable_moves(candidate)), default=0.0
        )
        incoming = strongest_hit(opponent, candidate, state) if not opponent.is_fainted() else 0
        return offense / max(1, opponent.live_stats.HP) - incoming / max(1, candidate.live_stats.HP)


def _required(pokemon: Pokemon | None) -> Pokemon:
    assert pokemon is not None  # SWITCH_OUT actions always carry the switch-in
    return pokemon


def _preview_edge(mine: Pokemon, foe: Pokemon) -> float:
    """Team-preview matchup proxy: best STAB-and-effectiveness-weighted power, theirs subtracted."""
    return _preview_offense(mine, foe) - _preview_offense(foe, mine)


def _preview_offense(attacker: Pokemon, defender: Pokemon) -> float:
    best = 0.0
    for move in attacker.known_moves():
        damage_effect = next((e for e in move.effects if isinstance(e, DamageEffect)), None)
        if damage_effect is None:
            continue
        power = damage_effect.power if damage_effect.power is not None else 60
        stab = 1.5 if move.type in attacker.types else 1.0
        best = max(best, power * stab * move_effectiveness(move, attacker, defender))
    return best


def _heals(move: Move) -> bool:
    return move.healing or any(isinstance(e, HealEffect) for e in move.effects)


def _sets_hazard(move: Move) -> Hazards | None:
    for effect in move.effects:
        if isinstance(effect, SideConditionEffect) and effect.kind in _ENTRY_HAZARDS:
            return effect.kind
    return None


def _spreads_status(move: Move) -> bool:
    return any(
        isinstance(e, InflictStatusEffect) and isinstance(e.status, Status) and not e.to_self and not e.is_secondary
        for e in move.effects
    )


def _boosts_self(move: Move) -> bool:
    return any(
        isinstance(e, StatStageChangeEffect)
        and e.target == "SELF"
        and not e.is_secondary
        and any(change > 0 for change in e.stages.values())
        for e in move.effects
    )
