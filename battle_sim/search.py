import random
from collections.abc import Sequence
from dataclasses import dataclass, replace

from battle_sim.analysis import EDGE_CAP, exchange_edge, hazard_pressure
from battle_sim.engine import apply_forced_switch, legal_actions, step
from battle_sim.matchup import MatchupPlayer, MatchupWeights
from battle_sim.maths.rng import RNG
from battle_sim.mechanics.battle import BattleState, FieldState, SideState
from battle_sim.models.actions import Action
from battle_sim.models.moves import MoveSet
from battle_sim.models.pokemon import Pokemon
from battle_sim.models.spec import PokemonSpec
from battle_sim.observation import BeliefSampler
from battle_sim.teams import build_pokemon
from battle_sim.utils import Outcome, Status

_MAX_DEPTH = 8
_WIN_VALUE = 100.0  # dwarfs every positional term; material on top keeps "win bigger" preferred
_MIX_FLOOR = 0.1  # post-averaging dust below this never gets sampled


@dataclass(frozen=True)
class SearchProfile:
    """Every knob that shapes one search decision; the genome stays in MatchupWeights."""

    budget: int
    top_k: int = 3
    root_k: int = 5
    chance_samples: int = 2
    determinizations: int = 1


def clone_for_search(state: BattleState, seed: int) -> BattleState:
    """An independent copy the engine can step without touching the original.

    Move objects are shared (immutable database entries); every mutable container is
    copied; the clone gets a fresh RNG so simulated futures never peek at the real
    battle's roll stream.
    """
    mapping: dict[int, Pokemon] = {}
    sides = tuple(_clone_side(side, mapping) for side in state.sides)
    field = state.field
    return BattleState(
        format=state.format,
        turn=state.turn,
        sides=(sides[0], sides[1]),
        field=FieldState(
            weather=field.weather,
            weather_turns_left=field.weather_turns_left,
            terrain=field.terrain,
            terrain_turns_left=field.terrain_turns_left,
            pseudo_weather=dict(field.pseudo_weather),
        ),
        rng=RNG(seed=seed),
        outcome=state.outcome,
        transforms={id(mapping[key]): snapshot for key, snapshot in state.transforms.items()},
    )


def _clone_side(side: SideState, mapping: dict[int, Pokemon]) -> SideState:
    return SideState(
        team=[_clone_pokemon(mon, mapping) for mon in side.team],
        active=list(side.active),
        hazards=dict(side.hazards),
        screens=dict(side.screens),
        tailwind_turns=side.tailwind_turns,
        needs_switch=side.needs_switch,
        chosen_action=None,  # step() reassigns it; a stale one must not leak into the simulation
        acted_this_turn=side.acted_this_turn,
        wish_pending=side.wish_pending,
        wish_turns=side.wish_turns,
        healing_wish_pending=side.healing_wish_pending,
        pending_substitute=side.pending_substitute,
    )


def _clone_pokemon(mon: Pokemon, mapping: dict[int, Pokemon]) -> Pokemon:
    clone = mon.model_copy(
        update={
            "moves": MoveSet(*mon.moves),
            "live_stats": mon.live_stats.model_copy(),
            "stat_stages": mon.stat_stages.model_copy(),
            "volatiles": dict(mon.volatiles),
            "pp": dict(mon.pp),
        }
    )
    mapping[id(mon)] = clone
    return clone


def evaluate_position(state: BattleState, side_index: int, weights: MatchupWeights) -> float:
    """State value in units of pokemon: material differential plus gene-weighted position."""
    material = _material(state.sides[side_index]) - _material(state.sides[1 - side_index])
    if state.outcome is not None:
        if state.outcome is Outcome.DRAW:
            return material
        won = (state.outcome is Outcome.P1_WIN) == (side_index == 0)
        return (_WIN_VALUE if won else -_WIN_VALUE) + material
    mine, theirs = state.sides[side_index], state.sides[1 - side_index]
    value = material
    me, them = mine.active_pokemon, theirs.active_pokemon
    if not me.is_fainted() and not them.is_fainted():
        value += weights.exchange_edge * exchange_edge(me, them, state) / EDGE_CAP
    value += weights.hazard_value * (hazard_pressure(theirs) - hazard_pressure(mine))
    value += weights.timer_value * (_statused(theirs) - _statused(mine)) / 6
    return value


def _material(side: SideState) -> float:
    return sum(p.live_stats.HP / p.stat_totals.HP for p in side.team if not p.is_fainted())


def _statused(side: SideState) -> int:
    return sum(1 for p in side.team if not p.is_fainted() and p.status is not Status.NONE)


def _fainted_count(state: BattleState) -> int:
    return sum(1 for side in state.sides for p in side.team if p.is_fainted())


def solve_zero_sum(matrix: list[list[float]], iterations: int = 64) -> tuple[list[float], float]:
    """Approximate mixed equilibrium of a small zero-sum game by fictitious play.

    Returns the row player's empirical strategy and the game value under both players'
    empirical mixtures. Deterministic: ties break to the lowest index.
    """
    rows, cols = len(matrix), len(matrix[0])
    row_counts, col_counts = [0] * rows, [0] * cols
    row_payoffs, col_payoffs = [0.0] * rows, [0.0] * cols
    row_choice, col_choice = 0, 0
    for _ in range(iterations):
        row_counts[row_choice] += 1
        col_counts[col_choice] += 1
        for j in range(cols):
            col_payoffs[j] += matrix[row_choice][j]
        for i in range(rows):
            row_payoffs[i] += matrix[i][col_choice]
        row_choice = max(range(rows), key=lambda i: row_payoffs[i])
        col_choice = min(range(cols), key=lambda j: col_payoffs[j])
    total = sum(row_counts)
    strategy = [count / total for count in row_counts]
    column_strategy = [count / total for count in col_counts]
    value = sum(strategy[i] * column_strategy[j] * matrix[i][j] for i in range(rows) for j in range(cols))
    return strategy, value


def gated_mixture(matrix: list[list[float]], iterations: int = 256) -> list[float]:
    """The equilibrium row mixture restricted to rows pulling at least half the heaviest row's weight.

    Fictitious play parks real weight only on rows that keep winning best-response rounds:
    genuine 50/50s sit near parity and keep their mix, while a strictly worse row that the
    solver merely visited early never reaches half the best row's weight and is zeroed —
    sampling it would be a free gift to an opponent who never exploits determinism. The
    relative cutoff sidesteps fictitious play's slow value convergence, which makes any
    absolute value tolerance either collapse real coin flips or admit junk.
    """
    strategy, _ = solve_zero_sum(matrix, iterations)
    cutoff = max(strategy) / 2
    gated = [weight if weight >= cutoff else 0.0 for weight in strategy]
    total = sum(gated)
    return [weight / total for weight in gated]


def row_values(matrix: list[list[float]], iterations: int = 256) -> list[float]:
    """Each row's expected value against the column player's equilibrium mixture."""
    transposed = [[-matrix[i][j] for i in range(len(matrix))] for j in range(len(matrix[0]))]
    column, _ = solve_zero_sum(transposed, iterations)
    return [sum(weight * cell for weight, cell in zip(column, row, strict=True)) for row in matrix]


class _BudgetExhausted(Exception):
    pass


class SearchPlayer:
    """The MatchupWeights genome driving a budgeted lookahead; a bigger budget is a smarter player."""

    def __init__(self, weights: MatchupWeights = MatchupWeights(), profile: SearchProfile = SearchProfile(budget=64)):
        self.weights = weights
        self.profile = profile
        self._mine = MatchupPlayer(weights)  # prunes my actions; keeps its per-battle cache
        self._theirs = MatchupPlayer(weights)  # prunes the opponent's, with a separate cache
        self._rollout = MatchupPlayer(weights)  # forced replacements inside simulated lines
        self._sampler: BeliefSampler | None = None
        self._rng = random.Random(0)
        self._spent = 0
        self._sim_index = 0
        self._extend = False  # faint extensions joined the current iteration; off for the depth-1 safety net

    def bind_belief_sampler(self, sampler: BeliefSampler) -> None:
        self._sampler = sampler

    def choose_order(self, own: Sequence[PokemonSpec], opponent: Sequence[PokemonSpec]) -> Sequence[int]:
        """Sample the lead from the equilibrium of the lead-vs-lead edge matrix; rest by edge total."""
        self._rng = random.Random(0)  # battle start: an identical mixing stream for every same-seed replay
        ours = [build_pokemon(spec) for spec in own]
        theirs = [build_pokemon(spec) for spec in opponent]
        scratch = BattleState(sides=(SideState(team=ours), SideState(team=theirs)), rng=RNG(seed=0))
        matrix = [[exchange_edge(mine, foe, scratch) for foe in theirs] for mine in ours]
        lead = self._sample(range(len(own)), gated_mixture(matrix))
        rest = sorted((i for i in range(len(own)) if i != lead), key=lambda i: sum(matrix[i]), reverse=True)
        return [lead, *rest]

    def choose_action(self, state: BattleState, side_index: int, actions: Sequence[Action]) -> Action:
        if len(actions) == 1:
            return actions[0]
        my_scored = self._mine.score_actions(state, side_index, actions)
        fallback = max(my_scored, key=lambda pair: pair[0])[1]
        self._spent = 0
        self._sim_index = 0
        my_side = state.sides[side_index]
        if my_side.needs_switch or my_side.active_pokemon.is_fainted():
            return self._forced_choice(state, side_index, my_scored, fallback)
        my_actions = _top(my_scored, self.profile.root_k)
        views = self._views(state)
        pruned = [(view, self._their_actions(view, side_index)) for view in views]
        mixture: list[float] | None = None
        for depth in range(1, _MAX_DEPTH + 1):
            # The first iteration runs extension-free so it always completes within root_k * top_k *
            # chance_samples simulations: violent positions must never silently degrade to the myopic fallback.
            self._extend = depth > 1
            try:
                strategies = [self._view_strategy(v, side_index, my_actions, theirs, depth) for v, theirs in pruned]
            except _BudgetExhausted:
                break
            mixture = [sum(column) / len(strategies) for column in zip(*strategies, strict=True)]
        if mixture is None:
            return fallback
        return self._sample(my_actions, mixture)

    def action_values(self, state: BattleState, side_index: int, actions: Sequence[Action]) -> list[float]:
        """One value per offered action — unpruned, in evaluate_position units — for grading decisions.

        Turn actions are valued against the opponent's equilibrium mixture at the deepest
        completed iteration; forced replacements are valued by the same unilateral one-ply
        lookahead the player itself uses. Runs on the state as given (no determinization).
        Falls back to the myopic scores only if not even depth 1 fits the budget.
        """
        self._spent = 0
        self._sim_index = 0
        scored = self._mine.score_actions(state, side_index, actions)
        my_side = state.sides[side_index]
        if my_side.needs_switch or my_side.active_pokemon.is_fainted():
            return [
                evaluate_position(self._simulate_switch(state, side_index, action), side_index, self.weights)
                for _, action in scored
            ]
        their_actions = self._their_actions(state, side_index)
        my_actions = list(actions)
        values: list[float] | None = None
        for depth in range(1, _MAX_DEPTH + 1):
            self._extend = depth > 1
            try:
                matrix = self._payoff_matrix(
                    state, side_index, my_actions, their_actions, depth, self.profile.chance_samples
                )
            except _BudgetExhausted:
                break
            values = row_values(matrix)
        return values if values is not None else [score for score, _ in scored]

    def _views(self, state: BattleState) -> list[BattleState]:
        """The modal believed state plus any sampled belief-consistent worlds to hedge over."""
        views = [state]
        if self._sampler is not None:
            views += [self._sampler(self._rng) for _ in range(self.profile.determinizations - 1)]
        return views

    def _their_actions(self, view: BattleState, side_index: int) -> list[Action]:
        """Their plausible responses stay top_k even at the root: the opponent model is the myopic
        scorer, and hedging against columns it would never play just leaks value."""
        scored = self._theirs.score_actions(view, 1 - side_index, legal_actions(view, 1 - side_index))
        return _top(scored, self.profile.top_k)

    def _view_strategy(
        self, view: BattleState, side_index: int, my_actions: list[Action], their_actions: list[Action], depth: int
    ) -> list[float]:
        matrix = self._payoff_matrix(view, side_index, my_actions, their_actions, depth, self.profile.chance_samples)
        return gated_mixture(matrix)

    def _sample[T](self, options: Sequence[T], mixture: Sequence[float]) -> T:
        """One draw from the mixture with dust floored away; the max weight always survives the floor."""
        kept = [(option, weight) for option, weight in zip(options, mixture, strict=True) if weight >= _MIX_FLOOR]
        roll = self._rng.random() * sum(weight for _, weight in kept)
        for option, weight in kept:
            roll -= weight
            if roll <= 0:
                return option
        return kept[-1][0]

    def _forced_choice(
        self, state: BattleState, side_index: int, scored: list[tuple[float, Action]], fallback: Action
    ) -> Action:
        """Mid-turn replacement: a unilateral one-ply lookahead over the offered switches."""
        if self.profile.budget < len(scored):
            return fallback
        best_value, best = -float("inf"), fallback
        for _, action in scored:
            clone = self._simulate_switch(state, side_index, action)
            value = evaluate_position(clone, side_index, self.weights)
            if value > best_value:
                best_value, best = value, action
        return best

    def _simulate_switch(self, state: BattleState, side_index: int, action: Action) -> BattleState:
        clone = self._clone(state)
        apply_forced_switch(clone, side_index, _remap(action, state, clone, side_index))
        return clone

    def _payoff_matrix(
        self,
        state: BattleState,
        side_index: int,
        my_actions: list[Action],
        their_actions: list[Action],
        depth: int,
        samples: int,
    ) -> list[list[float]]:
        return [
            [self._cell_value(state, side_index, mine, theirs, depth, samples) for theirs in their_actions]
            for mine in my_actions
        ]

    def _cell_value(
        self, state: BattleState, side_index: int, my_action: Action, their_action: Action, depth: int, samples: int
    ) -> float:
        total = sum(self._line_value(state, side_index, my_action, their_action, depth) for _ in range(samples))
        return total / samples

    def _line_value(
        self, state: BattleState, side_index: int, my_action: Action, their_action: Action, depth: int
    ) -> float:
        fainted = _fainted_count(state)
        clone = self._clone(state)
        actions = {
            side_index: _remap(my_action, state, clone, side_index),
            1 - side_index: _remap(their_action, state, clone, 1 - side_index),
        }
        step(clone, actions)
        self._settle(clone)
        if clone.outcome is not None:
            return evaluate_position(clone, side_index, self.weights)
        if depth > 1:
            return self._node_value(clone, side_index, depth - 1)
        if self._extend and _fainted_count(clone) > fainted:
            return self._node_value(clone, side_index, 1)  # horizon extension: never evaluate mid-KO-exchange
        return evaluate_position(clone, side_index, self.weights)

    def _node_value(self, state: BattleState, side_index: int, depth: int) -> float:
        mine = MatchupPlayer(self.weights)  # per-node evaluators: clone sides never repeat
        theirs = MatchupPlayer(self.weights)
        my_actions = _top(mine.score_actions(state, side_index, legal_actions(state, side_index)), self.profile.top_k)
        their_actions = _top(
            theirs.score_actions(state, 1 - side_index, legal_actions(state, 1 - side_index)), self.profile.top_k
        )
        matrix = self._payoff_matrix(state, side_index, my_actions, their_actions, depth, samples=1)
        _, value = solve_zero_sum(matrix)
        return value

    def _settle(self, clone: BattleState) -> None:
        """Resolve fainted/forced replacements greedily with the myopic scorer, as the runner would."""
        while clone.outcome is None:
            pending = [i for i in (0, 1) if clone.sides[i].active_pokemon.is_fainted() or clone.sides[i].needs_switch]
            if not pending:
                return
            for i in pending:
                if clone.outcome is not None:
                    return
                choice = self._rollout.choose_action(clone, i, legal_actions(clone, i))
                apply_forced_switch(clone, i, choice)

    def _clone(self, state: BattleState) -> BattleState:
        if self._spent >= self.profile.budget:
            raise _BudgetExhausted
        self._spent += 1
        self._sim_index += 1
        return clone_for_search(state, seed=self._sim_index)


def _top(scored: list[tuple[float, Action]], k: int) -> list[Action]:
    return [action for _, action in sorted(scored, key=lambda pair: pair[0], reverse=True)[:k]]


def _remap(action: Action, original: BattleState, clone: BattleState, side_index: int) -> Action:
    """Point a SWITCH_OUT's switch-in at the clone's copy of the same team slot."""
    if action.switch_in is None:
        return action
    team = original.sides[side_index].team
    index = next(i for i, mon in enumerate(team) if mon is action.switch_in)
    return replace(action, switch_in=clone.sides[side_index].team[index])
