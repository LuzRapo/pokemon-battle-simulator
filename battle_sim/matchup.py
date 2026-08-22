import math
from collections.abc import Sequence
from dataclasses import dataclass

from battle_sim.analysis import (
    EDGE_CAP,
    HAZARD_LAYER_CAPS,
    bootless_healthy,
    damage_range,
    entry_hazard_chip,
    exchange_edge,
    exchange_edge_from,
    hazard_pressure,
)
from battle_sim.maths.rng import RNG
from battle_sim.mechanics.battle import BattleState, SideState
from battle_sim.mechanics.priority import effective_speed
from battle_sim.models.actions import Action, ActionType
from battle_sim.models.moves import (
    CodedEffect,
    CodedMoveKind,
    HealEffect,
    InflictStatusEffect,
    Move,
    MoveSlot,
    RemoveHazardsEffect,
    SideConditionEffect,
    StatStageChangeEffect,
)
from battle_sim.models.pokemon import Pokemon
from battle_sim.models.spec import PokemonSpec
from battle_sim.teams import build_pokemon
from battle_sim.utils import CHOICE_ITEMS, Category, Hazards, Item, Stats, Status

_ITEM_GRABBERS = frozenset({CodedMoveKind.KNOCK_OFF_ITEM, CodedMoveKind.TRICK})
_DEAD_MOVE = 0.01  # below any scored play: never pick a no-op when something scores
_FODDER_THREAT_FRACTION = 0.2  # "setup fodder": their best hit costs less than this much of our HP
_CATEGORY_STAT = {Category.PHYSICAL: Stats.ATTACK, Category.SPECIAL: Stats.SP_ATTACK}


@dataclass(frozen=True)
class MatchupWeights:
    """The genome: one gene per Smogon concept. Defaults are hand-set priors, not tuned."""

    ko_now: float = 6.0
    hko_progress: float = 1.0
    exchange_edge: float = 0.6
    timer_value: float = 0.5
    para_speed_control: float = 0.5
    burn_disable: float = 0.5
    knock_progress: float = 0.4
    hazard_value: float = 0.6
    hazard_removal: float = 0.4
    heal_turns: float = 0.7
    entry_cost: float = 0.8
    setup_value: float = 0.6
    fodder_exploit: float = 0.4
    wincon_preservation: float = 0.7
    matchup_gain: float = 0.8
    tempo_cost: float = 0.5
    phaze_value: float = 0.4
    lock_risk: float = 0.8


@dataclass(frozen=True)
class _Offense:
    """One attacker's options against one defender, summarized in a single pass."""

    best_expected: float  # accuracy-weighted midpoint of the best move
    strongest: int  # biggest max-roll hit: what the defender must plan for
    best_category: Category | None


@dataclass(frozen=True)
class _Turn:
    """Everything about the live exchange that every action score reads."""

    state: BattleState
    side_index: int
    me: Pokemon
    opponent: Pokemon
    wincon: Pokemon
    mine: _Offense
    theirs: _Offense
    edge: float
    opponent_faster: bool


class MatchupPlayer:
    """Scores every legal action as a weighted sum of matchup features; fully deterministic."""

    def __init__(self, weights: MatchupWeights = MatchupWeights()):
        self.weights = weights
        self._battle_side: SideState | None = None
        self._matrix: dict[tuple[int, int], _Offense] = {}

    def choose_order(self, own: Sequence[PokemonSpec], opponent: Sequence[PokemonSpec]) -> Sequence[int]:
        ours = [build_pokemon(spec) for spec in own]
        theirs = [build_pokemon(spec) for spec in opponent]
        scratch = BattleState(sides=(SideState(team=ours), SideState(team=theirs)), rng=RNG(seed=0))
        totals = [sum(exchange_edge(mine, foe, scratch) for foe in theirs) for mine in ours]
        return sorted(range(len(own)), key=lambda i: totals[i], reverse=True)

    def choose_action(self, state: BattleState, side_index: int, actions: Sequence[Action]) -> Action:
        return max(self.score_actions(state, side_index, actions), key=lambda pair: pair[0])[1]

    def score_actions(
        self, state: BattleState, side_index: int, actions: Sequence[Action]
    ) -> list[tuple[float, Action]]:
        """Every offered action with its myopic score; SearchPlayer prunes with these."""
        self._ensure_battle(state.sides[side_index])
        turn = self._turn(state, side_index)
        voluntary = not turn.me.is_fainted() and not state.sides[side_index].needs_switch
        scored: list[tuple[float, Action]] = []
        for action in actions:
            if action.action is ActionType.SWITCH_OUT:
                incoming = action.switch_in
                assert incoming is not None  # SWITCH_OUT actions always carry the switch-in
                scored.append((self._switch_score(incoming, turn, voluntary), action))
            else:
                scored.append((self._move_score(action, turn), action))
        return scored

    def _turn(self, state: BattleState, side_index: int) -> _Turn:
        my_side, their_side = state.sides[side_index], state.sides[1 - side_index]
        me, opponent = my_side.active_pokemon, their_side.active_pokemon
        mine, theirs = _offense(me, opponent, state), _offense(opponent, me, state)
        faster = effective_speed(me, my_side, state.field) >= effective_speed(opponent, their_side, state.field)
        edge = exchange_edge_from(
            me.live_stats.HP, opponent.live_stats.HP, mine.best_expected, theirs.best_expected, faster
        )
        return _Turn(
            state=state,
            side_index=side_index,
            me=me,
            opponent=opponent,
            wincon=self._wincon(state, side_index),
            mine=mine,
            theirs=theirs,
            edge=edge,
            opponent_faster=not faster,
        )

    # -- Move scoring ---------------------------------------------------------------

    def _move_score(self, action: Action, turn: _Turn) -> float:
        assert action.move is not None  # USE_MOVE actions always carry a slot
        move = turn.me.moves[action.move]
        assert move is not None  # legal actions only point at filled slots
        if turn.me.pp[action.move] == 0:
            return _DEAD_MOVE  # the engine will substitute Struggle
        w = self.weights
        score = w.exchange_edge * turn.edge / EDGE_CAP  # staying in means accepting this exchange
        low, high = damage_range(move, turn.me, turn.opponent, turn.state)
        score += self._damage_features(move, low, high, turn)
        effects = self._effect_features(move, turn)
        threatened = turn.theirs.strongest / turn.me.stat_totals.HP
        if effects > 0 and high == 0 and threatened < _FODDER_THREAT_FRACTION:
            effects += w.fodder_exploit  # a useful free turn they can't punish: classic setup fodder
        score += effects
        if turn.me is turn.wincon and turn.edge < 0:
            score -= w.wincon_preservation  # the closer is losing this exchange: staying in risks the game plan
        return score

    def _damage_features(self, move: Move, low: int, high: int, turn: _Turn) -> float:
        w = self.weights
        accuracy = move.accuracy_probability if move.accuracy_probability is not None else 1.0
        expected = (low + high) / 2 * accuracy
        if expected <= 0:
            return 0.0
        score = 0.0
        if low >= turn.opponent.live_stats.HP and low > 0:
            score += w.ko_now * accuracy
        turns = math.ceil(turn.opponent.live_stats.HP / expected)
        score += w.hko_progress / turns
        if turn.me.item in CHOICE_ITEMS:
            score += w.lock_risk * self._lock_risk(move, turn)
        return score

    def _lock_risk(self, move: Move, turn: _Turn) -> float:
        """Fraction of the opposing team that takes nothing from this move: the choice-lock trap."""
        their_team = [p for p in turn.state.sides[1 - turn.side_index].team if not p.is_fainted()]
        immune = sum(1 for member in their_team if damage_range(move, turn.me, member, turn.state)[1] == 0)
        return -immune / len(their_team)

    def _effect_features(self, move: Move, turn: _Turn) -> float:
        w = self.weights
        score = 0.0
        score += self._status_features(move, turn)
        score += w.hazard_value * self._hazard_feature(move, turn)
        score += w.hazard_removal * self._removal_feature(move, turn)
        score += w.knock_progress * self._knock_feature(move, turn)
        score += w.heal_turns * self._heal_feature(move, turn)
        score += w.setup_value * self._setup_feature(move, turn)
        score += w.phaze_value * self._phaze_feature(move, turn)
        return score

    def _status_features(self, move: Move, turn: _Turn) -> float:
        status = _inflicted_status(move)
        if status is None or turn.opponent.status is not Status.NONE:
            return 0.0
        w = self.weights
        survival = (
            math.ceil(turn.opponent.live_stats.HP / turn.mine.best_expected)
            if turn.mine.best_expected > 0
            else math.inf
        )
        score = w.timer_value * min(survival, 6.0) / 6
        if status is Status.PARALYSIS and turn.opponent_faster:
            score += w.para_speed_control
        if status is Status.BURN and turn.theirs.best_category is Category.PHYSICAL:
            score += w.burn_disable
        return score

    def _hazard_feature(self, move: Move, turn: _Turn) -> float:
        hazard = _set_hazard(move)
        if hazard is None:
            return 0.0
        their_side = turn.state.sides[1 - turn.side_index]
        if their_side.hazards.get(hazard, 0) >= HAZARD_LAYER_CAPS.get(hazard, 1):
            return 0.0
        return bootless_healthy(their_side) / 6

    def _removal_feature(self, move: Move, turn: _Turn) -> float:
        removal = next((e for e in move.effects if isinstance(e, RemoveHazardsEffect)), None)
        if removal is None:
            return 0.0
        my_side, their_side = turn.state.sides[turn.side_index], turn.state.sides[1 - turn.side_index]
        relief = hazard_pressure(my_side)
        if removal.style == "DEFOG":
            relief -= hazard_pressure(their_side)  # Defog clears our own offense too
        return relief

    def _knock_feature(self, move: Move, turn: _Turn) -> float:
        grabs = any(isinstance(e, CodedEffect) and e.kind in _ITEM_GRABBERS for e in move.effects)
        if not grabs or turn.opponent.item is Item.NONE:
            return 0.0
        their_side = turn.state.sides[1 - turn.side_index]
        if turn.opponent.item is Item.HEAVY_DUTY_BOOTS and their_side.hazards:
            return 1.0  # stripping Boots re-enables the hazard tax: the highest-value Knock
        return 0.6

    def _heal_feature(self, move: Move, turn: _Turn) -> float:
        if not _heals(move):
            return 0.0
        fraction = next((e.fraction for e in move.effects if isinstance(e, HealEffect)), 0.5)
        healed = min(turn.me.stat_totals.HP - turn.me.live_stats.HP, fraction * turn.me.stat_totals.HP)
        turns_gained = healed / max(1, turn.theirs.strongest)
        return min(turns_gained, 2.0) / 2

    def _setup_feature(self, move: Move, turn: _Turn) -> float:
        """Boost value as exchange turns shaved off: +1 Atk means nothing if it doesn't change the HKO count."""
        if turn.mine.best_expected <= 0 or turn.mine.best_category not in _CATEGORY_STAT:
            return 0.0
        stat = _CATEGORY_STAT[turn.mine.best_category]
        gain = _self_boost(move, stat)
        if gain == 0:
            return 0.0
        current = turn.me.stat_stages[stat]
        boosted = turn.mine.best_expected * _stage_multiplier(min(6, current + gain)) / _stage_multiplier(current)
        turns_now = math.ceil(turn.opponent.live_stats.HP / turn.mine.best_expected)
        turns_after = math.ceil(turn.opponent.live_stats.HP / boosted)
        return min(turns_now - turns_after, 3) / 3

    def _phaze_feature(self, move: Move, turn: _Turn) -> float:
        if not move.force_switch:
            return 0.0
        their_side = turn.state.sides[1 - turn.side_index]
        if not any(not p.is_fainted() and p is not turn.opponent for p in their_side.team):
            return 0.0  # nothing to drag in
        boosts = sum(max(0, turn.opponent.stat_stages[stat]) for stat in _CATEGORY_STAT.values()) / 4
        chip = 0.5 if their_side.hazards and bootless_healthy(their_side) > 0 else 0.0
        return min(boosts + chip, 1.5)

    # -- Switch scoring -------------------------------------------------------------

    def _switch_score(self, incoming: Pokemon, turn: _Turn, voluntary: bool) -> float:
        w = self.weights
        incoming_edge = self._cached_edge(incoming, turn.opponent, turn.state)
        gain = incoming_edge / EDGE_CAP
        score = w.matchup_gain * gain
        if voluntary:
            score -= w.matchup_gain * turn.edge / EDGE_CAP + w.tempo_cost
            if turn.me is turn.wincon and turn.edge < 0:
                score += w.wincon_preservation  # pull the game plan out of a losing exchange
        score -= w.entry_cost * entry_hazard_chip(incoming, turn.state.sides[turn.side_index]) / incoming.stat_totals.HP
        if incoming is turn.wincon and incoming_edge < 0:
            score -= w.wincon_preservation  # don't feed the closer to a bad matchup
        return score

    # -- Matchup matrix and win condition ---------------------------------------------

    def _ensure_battle(self, own: SideState) -> None:
        """Reset the offense cache when a new battle starts; the own side is its identity."""
        if self._battle_side is not own:
            self._battle_side = own
            self._matrix = {}

    def _pair_offense(self, attacker: Pokemon, defender: Pokemon, state: BattleState) -> _Offense:
        key = (id(attacker), id(defender))
        cached = self._matrix.get(key)
        if cached is None:
            cached = _offense(attacker, defender, state)
            self._matrix[key] = cached
        return cached

    def _cached_edge(self, mine: Pokemon, theirs: Pokemon, state: BattleState) -> float:
        """Live-HP exchange edge from full-strength cached offenses: exact for benched, unstaged pokemon."""
        my_best = self._pair_offense(mine, theirs, state).best_expected
        their_best = self._pair_offense(theirs, mine, state).best_expected
        faster = mine.stat_totals.SPEED >= theirs.stat_totals.SPEED
        return exchange_edge_from(mine.live_stats.HP, theirs.live_stats.HP, my_best, their_best, faster)

    def _wincon(self, state: BattleState, side_index: int) -> Pokemon:
        """The teammate with the best full-strength matchup spread against their survivors."""
        survivors = [p for p in state.sides[1 - side_index].team if not p.is_fainted()]
        candidates = [p for p in state.sides[side_index].team if not p.is_fainted()]
        return max(candidates, key=lambda mine: sum(self._cached_edge(mine, foe, state) for foe in survivors))


def _offense(attacker: Pokemon, defender: Pokemon, state: BattleState) -> _Offense:
    best_expected, strongest, best_category = 0.0, 0, None
    for slot in MoveSlot:
        move = attacker.moves[slot]
        if move is None or attacker.pp[slot] == 0:
            continue
        low, high = damage_range(move, attacker, defender, state)
        accuracy = move.accuracy_probability if move.accuracy_probability is not None else 1.0
        expected = (low + high) / 2 * accuracy
        if expected > best_expected:
            best_expected, best_category = expected, move.category
        strongest = max(strongest, high)
    return _Offense(best_expected=best_expected, strongest=strongest, best_category=best_category)


def _inflicted_status(move: Move) -> Status | None:
    for effect in move.effects:
        if (
            isinstance(effect, InflictStatusEffect)
            and isinstance(effect.status, Status)
            and not effect.to_self
            and not effect.is_secondary
        ):
            return effect.status
    return None


def _set_hazard(move: Move) -> Hazards | None:
    for effect in move.effects:
        if isinstance(effect, SideConditionEffect) and effect.kind in HAZARD_LAYER_CAPS:
            return effect.kind
    return None


def _heals(move: Move) -> bool:
    return move.healing or any(isinstance(e, HealEffect) for e in move.effects)


def _self_boost(move: Move, stat: Stats) -> int:
    return sum(
        effect.stages.get(stat, 0)
        for effect in move.effects
        if isinstance(effect, StatStageChangeEffect) and effect.target == "SELF" and not effect.is_secondary
    )


def _stage_multiplier(stage: int) -> float:
    return (2 + stage) / 2 if stage >= 0 else 2 / (2 - stage)
