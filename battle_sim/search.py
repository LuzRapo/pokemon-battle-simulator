import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from battle_sim.analysis import (
    EDGE_CAP,
    exchange_edge,
    hazard_pressure,
    hazard_toll,
    residual_pressure,
    setup_potential,
    sweep_threat,
)
from battle_sim.engine import apply_forced_switch, legal_actions, step
from battle_sim.matchup import MatchupPlayer, MatchupWeights, set_hazard
from battle_sim.maths.rng import RNG
from battle_sim.mechanics.abilities import field_from_actives
from battle_sim.mechanics.battle import BattleState, FieldState, SideState
from battle_sim.models.actions import Action, ActionType
from battle_sim.models.moves import MoveSet
from battle_sim.models.pokemon import Pokemon
from battle_sim.models.spec import PokemonSpec
from battle_sim.observation import BeliefSampler
from battle_sim.teams import build_pokemon
from battle_sim.utils import Outcome, Status, Terrain, Weather

_MAX_DEPTH = 8
_WIN_VALUE = 100.0  # dwarfs every positional term; material on top keeps "win bigger" preferred
_MIX_FLOOR = 0.1  # post-averaging dust below this never gets sampled
_TIE_TOLERANCE = 1e-9  # rows this close to the best are the same row as far as a payoff estimate knows

type LeafEvaluator = Callable[[BattleState, int], float]


@dataclass(frozen=True)
class PositionWeights:
    """The coefficients `evaluate_position` needs, split out from the action-scoring genome.

    `MatchupWeights` was serving two unrelated jobs: ranking actions inside one choice set, and
    valuing a position on an absolute scale. Those want different numbers — a coefficient can be
    unidentifiable for ranking (a feature constant across the choice set cancels in a softmax) while
    being live and load-bearing here. Fitting the shared vector to observed play therefore poisoned
    the leaf evaluator with values that were never constrained by the objective. Defaults derive
    from the genome, so an evolved champion behaves exactly as before.
    """

    exchange_edge: float = 0.6
    timer_value: float = 0.5
    # What the hazards already down are worth. No longer derived from the genome: the gene it used
    # to copy is an *action*-ranking coefficient, and fitting one number to both jobs is the exact
    # confusion this class exists to undo.
    #
    # 3.0 is derived, not tuned. `hazard_pressure` reads (layers/4) x (bench/6), so one layer of
    # Stealth Rock against a full bench of five gives 0.208; a neutral rock costs each entrant an
    # eighth of its bar, so five of them walking in once costs 5/8 = 0.625 of a Pokemon. 0.625/0.208
    # is 3.0 -- the weight at which the term is worth the health it will actually take.
    #
    # It replaces 0.594, which priced that same layer at 0.12. Three independent measurements said
    # that was far too low: humans spend 6.5% of their turns on hazards against our 1-4%, they have
    # rocks down by turn 3 against our turn 7, and a logistic fit over 4,970 replays put a layer at
    # ~1.1 Pokemon against the live 0.125. The fit is associational -- the side with rocks up is also
    # the side playing better, so it is an upper bound -- which is why this lands below it.
    hazard_value: float = 3.0
    # What it is worth that one side's active, as boosted, beats a share of the other's whole team.
    # Set above `exchange_edge` on purpose: a sweep is not a better matchup, it is the game ending a
    # few turns later, and the position value has to say so loudly enough that the search will spend
    # a turn breaking it up — attacking into it, phazing it out, or paralysing it — rather than
    # taking the locally cheaper option and being run over. No genome gene backs it; there is no
    # observed-play coefficient to fit here, and deriving one from an action-ranking weight is what
    # `PositionWeights` exists to avoid.
    sweep_threat: float = 1.2
    # What it is worth that the clock is already killing something. `timer_value` scores a status
    # as a flat constant — a Toxic that kills in two turns and one landed this turn are the same
    # number to it — which is exactly the blindness that made every stall team in the Anything Goes
    # round-robin finish in the bottom six. Measured: raising the search budget a thousandfold
    # changed the AI's play in those positions not at all, because the payoff is six turns out and
    # six turns is about nine to the fourth times the cost of two. So it cannot be searched for and
    # has to be evaluated instead. Like `sweep_threat`, no genome gene backs it.
    #
    # 6.0 is not tuning, it is the scale that makes the term commensurate with material: `_material`
    # runs to 6.0 for a full team, and `residual_pressure` reaches 1.0 when every member dies inside
    # the horizon, so a point of pressure buys exactly what a Pokemon's worth of HP buys. Measured
    # behaviour is flat anywhere in 5-10 and gets worse by 20.
    residual_pressure: float = 6.0
    # What an opponent asleep is worth, in Pokemon. The action scorer already ranks Dark Void above
    # an outright kill (1.334 against 1.054) and it was still chosen on fewer than a quarter of
    # turns, because the search overruled it: `_statused` prices a sleeping Pokemon at one flat
    # constant, about 0.13, against the 1.0 of material a knockout banks. The genome said sleep, the
    # search said kill, and the search won.
    #
    # Set at parity with a knockout on purpose. Sleep is not *generally* better than killing —
    # they can switch the sleeper out — but it buys the two free turns a Nasty Plot and a Bad Dreams
    # clock need, and this is the number that lets the search agree with the scorer rather than
    # fight it.
    sleep_value: float = 1.0
    # What a sweep that has not started yet is worth, to whichever side owns it. Held below
    # `sweep_threat` deliberately: a boost still on the drawing board is worth less than one already
    # on the board, and pricing them alike would have the AI set up in front of anything.
    setup_potential: float = 0.8

    @classmethod
    def from_matchup(cls, weights: MatchupWeights) -> "PositionWeights":
        return cls(
            exchange_edge=weights.exchange_edge,
            timer_value=weights.timer_value,
        )


@dataclass(frozen=True)
class SearchProfile:
    """Every knob that shapes one search decision; the genome stays in MatchupWeights."""

    budget: int
    top_k: int = 3
    root_k: int = 5
    chance_samples: int = 2
    determinizations: int = 1
    switch_tax: float = 0.15  # what a switch must earn to be worth the free turn it concedes
    exploit_p: float = 0.0  # weight on the opponent's predicted policy vs their equilibrium mixture
    predict_temperature: float = 1.0  # how sharp that prediction is; -> 0 predicts their best action outright
    # How much of the genome's own ranking is blended into the searched payoffs. 1.0 (equal say to
    # the search's own payoff) is the smallest weight that reliably breaks near-ties the search
    # can't otherwise separate in its own favor — see `with_genome_prior`; 0.5 measurably left some
    # of them unresolved (e.g. a full-HP Rest sampled about as often as a real attack against it).
    genome_prior: float = 1.0
    # How close to the best action's mixture weight a rival has to be before the pick is drawn
    # rather than taken outright. `exploit_p` answers "how exploitable am I in one game"; this
    # answers "how memorisable am I across many", which is a different threat and needs a
    # different knob — a human who replays the same trainer sees a fixed board produce a fixed
    # reply and can learn the whole line. Banding by *mixture weight* keeps the cost near zero:
    # an action the search actually prefers still wins every time, so only the decisions it was
    # already indifferent about move. 0.0 restores the strict argmax.
    tie_band: float = 0.0


def _apply_entry_field(state: BattleState, *actives: Pokemon) -> None:
    """Put the weather and terrain these Pokemon would set onto a hand-built board."""
    weather, terrain = field_from_actives(actives)
    state.field.weather = weather if weather is not None else Weather.NONE
    state.field.terrain = terrain if terrain is not None else Terrain.NONE


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
        has_mega_evolved=side.has_mega_evolved,  # without this every lookahead line would mega again
        has_ultra_bursted=side.has_ultra_bursted,  # likewise, and it is a separate once-per-battle
        has_used_z_move=side.has_used_z_move,  # likewise: one Z-move per line, not per node
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


def evaluate_position(state: BattleState, side_index: int, weights: PositionWeights) -> float:
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
        # Both ways round: a sweep I am set up for is worth as much as one being set up on me.
        value += weights.sweep_threat * (sweep_threat(me, theirs, state) - sweep_threat(them, mine, state))
        # And our own sweep one boost out, so a Dragon Dance is scored against the five behind the
        # Pokemon in front of it rather than only against that one.
        #
        # Deliberately NOT mirrored onto their potential, though theirs is the more interesting half.
        # This applies exactly one boost, so subtracting it credited any defensive counter-boost with
        # permanently solving a sweeper: a Snorlax facing a +3 Volcarona picked Amnesia, because +2
        # Special Defence moved the reading from a one-shot to a two-shot -- and the Volcarona simply
        # Quiver Dances again. Their realised setup is already priced by `sweep_threat` above; their
        # *latent* setup needs a term that models a sweeper reaching its ceiling, not taking one step.
        value += weights.setup_potential * setup_potential(me, theirs, state)
    value += weights.hazard_value * (hazard_pressure(theirs) - hazard_pressure(mine))
    value += weights.timer_value * (_statused(theirs) - _statused(mine)) / 6
    value += weights.sleep_value * (_asleep(theirs) - _asleep(mine))
    # Kept alongside the flat count rather than replacing it: paralysis and a burn's halved attack
    # do no residual damage at all, so a pressure-only term would quietly stop valuing them.
    #
    # One-sided on purpose. Scored both ways, this reads every contact move into a Flame Body or a
    # Static as risking a clock of one's own, and that priced *not attacking* above attacking on
    # exactly the near-ties `genome_prior` exists to protect — the resting Snorlax came straight
    # back. Being on a clock oneself is already covered where it belongs: `matchup._heal_feature`
    # nets `residual_drain` when deciding whether a heal is worth the turn.
    value += weights.residual_pressure * residual_pressure(theirs, state)
    return value


def _material(side: SideState) -> float:
    return sum(p.live_stats.HP / p.stat_totals.HP for p in side.team if not p.is_fainted())


def _asleep(side: SideState) -> int:
    """Members currently losing their turns. The only status that takes an action away outright."""
    return sum(1 for p in side.team if not p.is_fainted() and p.status is Status.SLEEP)


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


def column_equilibrium(matrix: list[list[float]], iterations: int = 256) -> list[float]:
    """The column player's equilibrium mixture: their own solve of the game they are losing."""
    transposed = [[-matrix[i][j] for i in range(len(matrix))] for j in range(len(matrix[0]))]
    column, _ = solve_zero_sum(transposed, iterations)
    return column


def row_values(matrix: list[list[float]], iterations: int = 256) -> list[float]:
    """Each row's expected value against the column player's equilibrium mixture."""
    return _values_against(matrix, column_equilibrium(matrix, iterations))


def _values_against(matrix: list[list[float]], column: Sequence[float]) -> list[float]:
    return [sum(weight * cell for weight, cell in zip(column, row, strict=True)) for row in matrix]


def with_genome_prior(
    matrix: list[list[float]], my_actions: Sequence[Action], scores: dict[int, float], weight: float
) -> list[list[float]]:
    """Blend the genome's own ranking into the payoffs, so what the search cannot separate falls its way.

    A depth-limited payoff estimate can only resolve differences bigger than its own noise, and
    below that its ordering is arbitrary — which is how an Arceus recovers into a losing race, or a
    Miltank with one attack left spends five turns on a move that cannot do anything. Both were
    ranked correctly by the genome and overturned by the search on margins under 0.05. Adding a
    fraction of the genome's score to every cell in a row leaves any real searched edge intact and
    hands the near-ties to the scorer that knows an attack from an inert move.
    """
    if weight <= 0:
        return matrix
    return [
        [cell + weight * scores[id(action)] for cell in row] for row, action in zip(matrix, my_actions, strict=True)
    ]


def taxed(matrix: list[list[float]], my_actions: Sequence[Action], switch_tax: float) -> list[list[float]]:
    """The payoff matrix with a standing charge on switching.

    Switching hands the opponent a free turn, and a search that prices it at nothing will shuffle
    two pokemon back and forth forever when neither attacking nor advancing looks better — which is
    exactly what happens against an evasive staller, where attacks miss and switching resets the
    toxic counter, so doing nothing scores best on every axis the evaluator can see. The tax is what
    makes a switch have to *earn* its turn; a genuine pivot clears it easily.
    """
    if switch_tax <= 0:
        return matrix
    return [
        [value - switch_tax if action.action is ActionType.SWITCH_OUT else value for value in row]
        for action, row in zip(my_actions, matrix, strict=True)
    ]


def exploit_mixture(matrix: list[list[float]], predicted: Sequence[float], exploit_p: float) -> list[float]:
    """Best response to an opponent who plays their predicted policy `exploit_p` of the time.

    Equilibrium play is unexploitable and therefore leaves value on the table against anyone who
    is not playing equilibrium — which is every human and most bots. Blending their predicted
    policy into the column we answer buys that value back, and `exploit_p` is the dial: 0 is pure
    equilibrium, 1 is a naive best response that a deliberate opponent could farm. The blend is
    the practical form of a restricted Nash response, not its linear program: we hold the
    adversarial part fixed at the equilibrium mixture rather than letting it re-optimise.
    """
    equilibrium = column_equilibrium(matrix)
    blended = [(1 - exploit_p) * eq + exploit_p * guess for eq, guess in zip(equilibrium, predicted, strict=True)]
    values = _values_against(matrix, blended)
    best = max(values)
    tied = [1.0 if value >= best - _TIE_TOLERANCE else 0.0 for value in values]
    return [weight / sum(tied) for weight in tied]


class _BudgetExhausted(Exception):
    pass


class SearchPlayer:
    """The MatchupWeights genome driving a budgeted lookahead; a bigger budget is a smarter player."""

    def __init__(
        self,
        weights: MatchupWeights = MatchupWeights(),
        profile: SearchProfile = SearchProfile(budget=64),
        leaf_evaluator: LeafEvaluator | None = None,
        position_weights: PositionWeights | None = None,
        opponent_model: MatchupWeights | None = None,
        seed: int = 0,
    ):
        """`leaf_evaluator` replaces `evaluate_position`'s *non-terminal* scoring only — a resolved
        outcome always goes through the hardcoded win/draw/loss handling below, since a learned
        evaluator trained on live decision points never saw a post-game state during training.

        `position_weights` defaults to the genome's own values, so an evolved champion is unchanged.
        Pass it explicitly when the genome was fitted for action ranking rather than position value.

        `opponent_model` is the genome we believe the *opponent* scores actions with — the job the
        replay-fitted weights are actually good at, having lost badly as a policy. Without one the
        search answers the equilibrium mixture, which is unexploitable and therefore never punishes
        a predictable opponent; `SearchProfile.exploit_p` decides how far to trust it.

        `seed` fixes the mixing stream every draw comes from — the lead sample and any near-tie
        pick. It defaults to 0 so a bare SearchPlayer is reproducible, but a caller that plays the
        same human repeatedly must pass a per-battle value: without one the lead mixture, however
        wide, resolves to the same draw in every battle it ever plays.
        """
        self._seed = seed
        self.weights = weights
        self.position_weights = position_weights or PositionWeights.from_matchup(weights)
        self.profile = profile
        self._opponent_model = MatchupPlayer(opponent_model) if opponent_model is not None else None
        self._leaf_evaluator = leaf_evaluator
        self._mine = MatchupPlayer(weights)  # prunes my actions; keeps its per-battle cache
        self._genome_hazard_value = weights.hazard_value if weights is not None else MatchupWeights().hazard_value
        self._theirs = MatchupPlayer(weights)  # prunes the opponent's, with a separate cache
        self._rollout = MatchupPlayer(weights)  # forced replacements inside simulated lines
        self._sampler: BeliefSampler | None = None
        self._rng = random.Random(seed)
        self._spent = 0
        self._sim_index = 0
        self._extend = False  # faint extensions joined the current iteration; off for the depth-1 safety net

    def _leaf_value(self, state: BattleState, side_index: int) -> float:
        if state.outcome is not None or self._leaf_evaluator is None:
            return evaluate_position(state, side_index, self.position_weights)
        return self._leaf_evaluator(state, side_index)

    def bind_belief_sampler(self, sampler: BeliefSampler) -> None:
        self._sampler = sampler

    def choose_order(self, own: Sequence[PokemonSpec], opponent: Sequence[PokemonSpec]) -> Sequence[int]:
        """Sample the lead from the equilibrium of the lead-vs-lead edge matrix; rest by edge total."""
        self._rng = random.Random(self._seed)  # battle start: one mixing stream per seed, so a replay repeats
        ours = [build_pokemon(spec) for spec in own]
        theirs = [build_pokemon(spec) for spec in opponent]
        scratch = BattleState(sides=(SideState(team=ours), SideState(team=theirs)), rng=RNG(seed=0))
        # Hand-built boards never run the switch-in, so nothing here would have turned on Groudon's
        # Desolate Land or Rayquaza's Delta Stream. Without it Kyogre's best hit on Groudon scored
        # 370 against a true 184, and the lead matrix read a Primal as a bad lead on that arithmetic.
        _apply_entry_field(scratch, ours[0], theirs[0])
        matrix = [[self._lead_matchup(mine, foe, scratch) for foe in theirs] for mine in ours]
        # A lead is not only the fight it wins. Scored on `exchange_edge` alone this picked the
        # biggest attacker every time -- across all thirteen Anything Goes teams carrying Stealth
        # Rock it never once led the setter, so the rocks waited for that Pokemon to arrive some
        # other way and went up on turn 13 against a human median of turn 3. What a lead can bank on
        # the first turn belongs in the matrix, because no other member can bank it as cheaply.
        for row, mine in zip(matrix, ours, strict=True):
            banked = self._lead_hazard_value(mine, scratch.sides[1])
            if banked:
                row[:] = [cell + banked for cell in row]
        lead = self._sample(range(len(own)), gated_mixture(matrix))
        rest = sorted((i for i in range(len(own)) if i != lead), key=lambda i: sum(matrix[i]), reverse=True)
        return [lead, *rest]

    def _lead_matchup(self, mine: Pokemon, foe: Pokemon, scratch: BattleState) -> float:
        """Lead-vs-lead edge with whatever field these two would actually create between them."""
        _apply_entry_field(scratch, mine, foe)
        return exchange_edge(mine, foe, scratch)

    def _lead_hazard_value(self, mine: Pokemon, theirs: SideState) -> float:
        """What this Pokemon would bank by spending turn one on the best hazard it carries.

        Scaled by the *genome's* hazard coefficient rather than the position evaluator's, which is
        not a detail. The lead matrix is denominated in exchange edges -- turns-to-kill differences,
        capped at four -- and `PositionWeights.hazard_value` is denominated in Pokemon. Adding one to
        the other was a unit error that went unnoticed while both numbers happened to be near 0.6.
        Once the position weight was raised to 3.0 on the evidence, the bonus reached 4.5 against
        lead spreads of about two, and every team led its hazard setter against every opponent --
        collapsing `gated_mixture` to a single row and making the opening perfectly memorisable,
        which is the one thing the lead mixture exists to prevent.
        """
        best = 0.0
        for move in mine.moves.to_list():
            hazard = None if move is None else set_hazard(move)
            if hazard is None:
                continue
            gained = hazard_toll(theirs, extra=hazard, attacker=mine) - hazard_toll(theirs, attacker=mine)
            best = max(best, gained)
        return best * self._genome_hazard_value

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
        genome_scores = {id(action): score for score, action in my_scored}
        views = self._views(state)
        pruned = [(view, self._their_actions(view, side_index)) for view in views]
        mixture: list[float] | None = None
        for depth in range(1, _MAX_DEPTH + 1):
            # The first iteration runs extension-free so it always completes within root_k * top_k *
            # chance_samples simulations: violent positions must never silently degrade to the myopic fallback.
            self._extend = depth > 1
            try:
                strategies = [
                    self._view_strategy(v, side_index, my_actions, theirs, depth, genome_scores) for v, theirs in pruned
                ]
            except _BudgetExhausted:
                break
            mixture = [sum(column) / len(strategies) for column in zip(*strategies, strict=True)]
        if mixture is None:
            return fallback
        return self._pick(my_actions, mixture)

    def _pick(self, actions: Sequence[Action], mixture: Sequence[float]) -> Action:
        """The mixture's best action, except among those it rates as near-equal — see `tie_band`.

        Contenders are drawn proportionally rather than uniformly, so even inside the band the
        action the search liked most is still the one it plays most often.
        """
        best = max(mixture)
        if self.profile.tie_band <= 0.0:
            return actions[mixture.index(best)]
        floor = best - self.profile.tie_band
        contenders = [(action, weight) for action, weight in zip(actions, mixture, strict=True) if weight >= floor]
        if len(contenders) == 1:
            return contenders[0][0]
        return self._sample([action for action, _ in contenders], [weight for _, weight in contenders])

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
                self._leaf_value(self._simulate_switch(state, side_index, action), side_index) for _, action in scored
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
            values = row_values(taxed(matrix, my_actions, self.profile.switch_tax))
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
        self,
        view: BattleState,
        side_index: int,
        my_actions: list[Action],
        their_actions: list[Action],
        depth: int,
        scores: dict[int, float],
    ) -> list[float]:
        matrix = self._payoff_matrix(view, side_index, my_actions, their_actions, depth, self.profile.chance_samples)
        matrix = taxed(matrix, my_actions, self.profile.switch_tax)
        matrix = with_genome_prior(matrix, my_actions, scores, self.profile.genome_prior)
        if self.profile.exploit_p > 0 and self._opponent_model is not None:
            predicted = self._predicted_column(view, side_index, their_actions)
            return exploit_mixture(matrix, predicted, self.profile.exploit_p)
        return gated_mixture(matrix)

    def _predicted_column(self, view: BattleState, side_index: int, their_actions: Sequence[Action]) -> list[float]:
        """What we expect the opponent to actually play, as a distribution over the matrix columns.

        Temperature 1 is right for a genome fitted as a conditional logit, whose scores already
        live on the scale where exponentiating them gives the choice probability. A genome tuned
        for ranking does not: its scores are spread far too narrowly, and softmaxing them raw
        returns something close to uniform, which predicts nothing. Sharpen for those.
        """
        assert self._opponent_model is not None  # only reached from the exploit branch
        scored = self._opponent_model.score_actions(view, 1 - side_index, their_actions)
        by_action = {id(action): score for score, action in scored}
        temperature = self.profile.predict_temperature
        scores = [by_action[id(action)] / temperature for action in their_actions]
        highest = max(scores)
        weights = [math.exp(score - highest) for score in scores]  # shifted: exp overflows on raw scores
        total = sum(weights)
        return [weight / total for weight in weights]

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
        """Mid-turn replacement: a unilateral one-ply lookahead over the offered switches.

        The leaf value's only per-candidate signal is `exchange_edge`, a ceil()-quantised turns
        differential (see `_switch_features`'s docstring) that regularly ties switch-ins with very
        different real matchups — an immunity and a 2x weakness against the same attacker land in
        the same bucket as often as not. The myopic switch score already carries the finer,
        unquantised signal built for exactly that blind spot (`incoming_damage_taken`,
        `incoming_outspeeds`, ...), so it breaks the tie here the same way it breaks near-ties
        during search, via `with_genome_prior`.
        """
        if self.profile.budget < len(scored):
            return fallback
        best_value, best = -float("inf"), fallback
        for genome_score, action in scored:
            clone = self._simulate_switch(state, side_index, action)
            value = self._leaf_value(clone, side_index) + self.profile.genome_prior * genome_score
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
        # A pivot's switch is resolved the instant it's forced, same as the runner does for a real
        # battle — otherwise the rollout would see the pivot's original mon take a hit that, in a
        # real game, its replacement takes instead, and undervalue pivoting out of a bad matchup.
        step(clone, actions, switch_chooser=lambda st, i: self._rollout.choose_action(st, i, legal_actions(st, i)))
        self._settle(clone)
        if clone.outcome is not None:
            return self._leaf_value(clone, side_index)
        if depth > 1:
            return self._node_value(clone, side_index, depth - 1)
        if self._extend and _fainted_count(clone) > fainted:
            return self._node_value(clone, side_index, 1)  # horizon extension: never evaluate mid-KO-exchange
        return self._leaf_value(clone, side_index)

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
