import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

from battle_sim.analysis import (
    EDGE_CAP,
    HAZARD_LAYER_CAPS,
    bootless_bench,
    damage_range,
    entry_hazard_chip,
    exchange_edge,
    exchange_edge_from,
    hazard_pressure,
    hazard_toll,
    posterior_threat,
    residual_drain,
)
from battle_sim.engine.power import coded_move_fails
from battle_sim.engine.status_apply import status_cannot_land
from battle_sim.maths.rng import RNG
from battle_sim.mechanics.battle import BattleState, SideState
from battle_sim.mechanics.priority import effective_speed
from battle_sim.models.actions import Action, ActionType
from battle_sim.models.moves import (
    CodedEffect,
    CodedMoveKind,
    DamageEffect,
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
from battle_sim.utils import CHOICE_ITEMS, Ability, Category, ExtraStatus, Hazards, Item, Stats, Status, Type
from battle_sim.zmoves import z_move_for

_ITEM_GRABBERS = frozenset({CodedMoveKind.KNOCK_OFF_ITEM, CodedMoveKind.TRICK})
_DEAD_MOVE_MARGIN = 1e-6  # a dead action scores just below the worst real one, never a fixed constant
_FODDER_THREAT_FRACTION = 0.2  # "setup fodder": their best hit costs less than this much of our HP
_CATEGORY_STAT = {Category.PHYSICAL: Stats.ATTACK, Category.SPECIAL: Stats.SP_ATTACK}
_BOOST_STATS = (Stats.ATTACK, Stats.SP_ATTACK, Stats.SPEED)
_RELIEF_STATS = (Stats.ATTACK, Stats.DEFENCE, Stats.SP_ATTACK, Stats.SP_DEFENCE, Stats.SPEED)
# What escaping each volatile is worth, relative to one another; switching clears all of them.
_BOOST_CAP = 6.0  # what `_positive_boosts` is scaled against: two stats maxed is already decisive
_SLEEP_TURNS = 2.0  # Gen 7 sleeps for one to three turns; the middle is what to plan around
_FREE_TURN_SETUP = 0.85  # a boost nobody can punish is nearly the best a turn can do
_DENIAL_BASE = 0.4  # shutting down a threat that has not started yet is still worth a turn
# The volatiles that take an opponent's turn away rather than damaging them.
_DENIAL_VOLATILES = frozenset({ExtraStatus.TAUNT, ExtraStatus.ENCORE, ExtraStatus.DISABLE})
_VOLATILE_RELIEF: dict[ExtraStatus, float] = {
    ExtraStatus.CONFUSION: 1.0,
    ExtraStatus.LEECH_SEED: 0.8,
    ExtraStatus.TAUNT: 0.5,
    ExtraStatus.ENCORE: 0.5,
}


GENE_NAMES: tuple[str, ...] = (
    "ko_now",
    "hko_progress",
    "exchange_edge",
    "timer_value",
    "para_speed_control",
    "sleep_tempo",
    "burn_disable",
    "knock_progress",
    "hazard_value",
    "hazard_removal",
    "heal_turns",
    "entry_cost",
    "setup_value",
    "fodder_exploit",
    "wincon_preservation",
    "wincon_support",
    "matchup_gain",
    "tempo_cost",
    "phaze_value",
    "lock_risk",
    "incoming_damage_taken",
    "incoming_damage_dealt",
    "incoming_outspeeds",
    "lock_relief",
    "volatile_relief",
    "debuff_relief",
    "setup_concession",
    "setup_denial",
)


@dataclass(frozen=True)
class MatchupWeights:
    """The genome: one gene per Smogon concept. Defaults are hand-set priors, not tuned."""

    ko_now: float = 6.0
    hko_progress: float = 1.0
    exchange_edge: float = 0.6
    timer_value: float = 0.5
    para_speed_control: float = 0.5
    # Outspeeding something and taking its next two turns away is among the most valuable things a turn buys.
    sleep_tempo: float = 1.2
    burn_disable: float = 0.5
    knock_progress: float = 0.4
    hazard_value: float = 0.6
    hazard_removal: float = 0.4
    heal_turns: float = 0.7
    entry_cost: float = 0.8
    setup_value: float = 0.6
    fodder_exploit: float = 0.4
    wincon_preservation: float = 0.7
    wincon_support: float = 0.7  # damage toward whichever survivor best checks my current wincon
    matchup_gain: float = 0.8
    tempo_cost: float = 0.5
    phaze_value: float = 0.4
    lock_risk: float = 0.8
    incoming_damage_taken: float = 0.8  # their best hit on the switch-in, as a fraction of its HP (unquantised)
    incoming_damage_dealt: float = 0.8  # the switch-in's best hit on them, as a fraction of their HP
    incoming_outspeeds: float = 0.3  # the switch-in moves first against their active
    lock_relief: float = 0.4  # switching frees a choice-locked active from a move it is stuck with
    volatile_relief: float = 0.4  # switching clears confusion / Leech Seed / Taunt / trapping
    debuff_relief: float = 0.4  # switching resets the active's negative stat stages
    setup_concession: float = 0.6  # staying in hands them a free turn to boost or stall behind recovery
    setup_denial: float = 0.6  # the switch-in threatens them enough to deny that free turn

    def as_vector(self) -> tuple[float, ...]:
        """Listed explicitly, in GENE_NAMES order, so no reflection is needed to score."""
        return (
            self.ko_now,
            self.hko_progress,
            self.exchange_edge,
            self.timer_value,
            self.para_speed_control,
            self.sleep_tempo,
            self.burn_disable,
            self.knock_progress,
            self.hazard_value,
            self.hazard_removal,
            self.heal_turns,
            self.entry_cost,
            self.setup_value,
            self.fodder_exploit,
            self.wincon_preservation,
            self.wincon_support,
            self.matchup_gain,
            self.tempo_cost,
            self.phaze_value,
            self.lock_risk,
            self.incoming_damage_taken,
            self.incoming_damage_dealt,
            self.incoming_outspeeds,
            self.lock_relief,
            self.volatile_relief,
            self.debuff_relief,
            self.setup_concession,
            self.setup_denial,
        )

    @classmethod
    def from_vector(cls, vector: Sequence[float]) -> "MatchupWeights":
        values = tuple(vector)
        assert len(values) == len(GENE_NAMES)
        return cls(*values)


@dataclass(frozen=True)
class ActionFeatures:
    """One action's unweighted feature per gene, so that `score == dot(features, weights)`."""

    ko_now: float = 0.0
    hko_progress: float = 0.0
    exchange_edge: float = 0.0
    timer_value: float = 0.0
    para_speed_control: float = 0.0
    sleep_tempo: float = 0.0
    burn_disable: float = 0.0
    knock_progress: float = 0.0
    hazard_value: float = 0.0
    hazard_removal: float = 0.0
    heal_turns: float = 0.0
    entry_cost: float = 0.0
    setup_value: float = 0.0
    fodder_exploit: float = 0.0
    wincon_preservation: float = 0.0
    wincon_support: float = 0.0
    matchup_gain: float = 0.0
    tempo_cost: float = 0.0
    phaze_value: float = 0.0
    lock_risk: float = 0.0
    incoming_damage_taken: float = 0.0
    incoming_damage_dealt: float = 0.0
    incoming_outspeeds: float = 0.0
    lock_relief: float = 0.0
    volatile_relief: float = 0.0
    debuff_relief: float = 0.0
    setup_concession: float = 0.0
    setup_denial: float = 0.0

    def as_vector(self) -> tuple[float, ...]:
        return (
            self.ko_now,
            self.hko_progress,
            self.exchange_edge,
            self.timer_value,
            self.para_speed_control,
            self.sleep_tempo,
            self.burn_disable,
            self.knock_progress,
            self.hazard_value,
            self.hazard_removal,
            self.heal_turns,
            self.entry_cost,
            self.setup_value,
            self.fodder_exploit,
            self.wincon_preservation,
            self.wincon_support,
            self.matchup_gain,
            self.tempo_cost,
            self.phaze_value,
            self.lock_risk,
            self.incoming_damage_taken,
            self.incoming_damage_dealt,
            self.incoming_outspeeds,
            self.lock_relief,
            self.volatile_relief,
            self.debuff_relief,
            self.setup_concession,
            self.setup_denial,
        )

    def score(self, weights: MatchupWeights) -> float:
        return sum(f * w for f, w in zip(self.as_vector(), weights.as_vector(), strict=True))


@dataclass(frozen=True)
class _Offense:
    """One attacker's options against one defender, summarized in a single pass."""

    best_expected: float  # accuracy-weighted midpoint of the best move
    threat_expected: float  # the same, averaged over a *believed* set: see analysis.posterior_threat
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
        featured = self.feature_actions(state, side_index, actions)
        scored = [(None if f is None else f.score(self.weights), action) for f, action in featured]
        real_scores = [score for score, _ in scored if score is not None]
        dead_score = min(real_scores) - _DEAD_MOVE_MARGIN if real_scores else 0.0
        return [(dead_score if score is None else score, action) for score, action in scored]

    def feature_actions(
        self, state: BattleState, side_index: int, actions: Sequence[Action]
    ) -> list[tuple[ActionFeatures | None, Action]]:
        """Every offered action with its feature vector, or None where no linear score applies."""
        self._ensure_battle(state.sides[side_index])
        turn = self._turn(state, side_index)
        voluntary = not turn.me.is_fainted() and not state.sides[side_index].needs_switch
        out: list[tuple[ActionFeatures | None, Action]] = []
        for action in actions:
            if action.action is ActionType.SWITCH_OUT:
                incoming = action.switch_in
                assert incoming is not None  # SWITCH_OUT actions always carry the switch-in
                out.append((self._switch_features(incoming, turn, voluntary), action))
            elif action.move is not None and turn.me.pp[action.move] == 0:
                out.append((None, action))
            else:
                out.append((self._move_features(action, turn), action))
        return out

    def _turn(self, state: BattleState, side_index: int) -> _Turn:
        my_side, their_side = state.sides[side_index], state.sides[1 - side_index]
        me, opponent = my_side.active_pokemon, their_side.active_pokemon
        mine, theirs = _offense(me, opponent, state), _offense(opponent, me, state)
        faster = effective_speed(me, my_side, state.field) >= effective_speed(opponent, their_side, state.field)
        # Our own set is known, so use our best move; theirs is believed, so use the softened threat.
        edge = exchange_edge_from(
            me.live_stats.HP, opponent.live_stats.HP, mine.best_expected, theirs.threat_expected, faster
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

    def _move_features(self, action: Action, turn: _Turn) -> ActionFeatures | None:
        assert action.move is not None  # USE_MOVE actions always carry a slot
        move = turn.me.moves[action.move]
        assert move is not None  # legal actions only point at filled slots
        if action.z_move:
            # Feature the Z upgrade, not the slot, or the plain move always ties it and the crystal sits unused.
            upgraded = z_move_for(turn.me.item, move)
            assert upgraded is not None  # legal_actions only offers the variant when it resolves
            move = upgraded
        if move.reflectable and turn.opponent.ability is Ability.MAGIC_BOUNCE:
            # Worse than useless: a bounced status lands on its own user.
            return None
        if coded_move_fails(move, turn.me, turn.opponent, turn.state):
            # A move whose condition fails (Dream Eater on an awake target and so on) is a dead action.
            return None
        low, high = damage_range(move, turn.me, turn.opponent, turn.state)
        damage = self._damage_features(move, low, high, turn)
        effects = self._effect_features(move, turn, low, high)
        threatened = turn.theirs.strongest / turn.me.stat_totals.HP
        # The fodder bonus keys on any effect feature firing, which keeps the score linear in the weights.
        fodder = 1.0 if _any_effect(effects) and high == 0 and threatened < _FODDER_THREAT_FRACTION else 0.0
        losing_the_wincon = turn.me is turn.wincon and turn.edge < 0
        return replace(
            effects,
            ko_now=damage.ko_now,
            hko_progress=damage.hko_progress,
            lock_risk=damage.lock_risk,
            exchange_edge=turn.edge / EDGE_CAP,  # staying in means accepting this exchange
            fodder_exploit=fodder,
            setup_concession=-self._setup_concession(turn),
            # the closer is losing this exchange: staying in risks the game plan
            wincon_preservation=-1.0 if losing_the_wincon else 0.0,
            wincon_support=self._wincon_support_feature(low, high, turn),
        )

    def _setup_concession(self, turn: _Turn) -> float:
        """The mirror of `fodder_exploit`: how free a turn they get if we stay in."""
        if not self._opponent_gains_from_a_free_turn(turn):
            return 0.0
        if turn.mine.best_expected <= 0:
            return 1.0  # we cannot touch them at all: every turn we stay is theirs
        turns_to_ko = math.ceil(turn.opponent.live_stats.HP / turn.mine.best_expected)
        return 0.0 if turns_to_ko <= 2 else min(turns_to_ko - 2, 4) / 4

    def _damage_features(self, move: Move, low: int, high: int, turn: _Turn) -> ActionFeatures:
        accuracy = move.accuracy_probability if move.accuracy_probability is not None else 1.0
        expected = (low + high) / 2 * accuracy
        if expected <= 0:
            return ActionFeatures()
        turns = math.ceil(turn.opponent.live_stats.HP / expected)
        return ActionFeatures(
            ko_now=accuracy if low >= turn.opponent.live_stats.HP and low > 0 else 0.0,
            hko_progress=1 / turns,
            lock_risk=self._lock_risk(move, turn) if turn.me.item in CHOICE_ITEMS else 0.0,
        )

    def _lock_risk(self, move: Move, turn: _Turn) -> float:
        """Fraction of the opposing team that takes nothing from this move: the choice-lock trap."""
        their_team = [p for p in turn.state.sides[1 - turn.side_index].team if not p.is_fainted()]
        immune = sum(1 for member in their_team if damage_range(move, turn.me, member, turn.state)[1] == 0)
        return -immune / len(their_team)

    def _effect_features(self, move: Move, turn: _Turn, low: int = 0, high: int = 0) -> ActionFeatures:
        """`low`/`high` are this move's expected hit, needed only for the drain half of `_heal_feature`."""
        status = self._status_features(move, turn)
        return replace(
            status,
            hazard_value=self._hazard_feature(move, turn),
            hazard_removal=self._removal_feature(move, turn),
            knock_progress=self._knock_feature(move, turn),
            heal_turns=self._heal_feature(move, turn, low, high),
            setup_value=self._setup_feature(move, turn),
            phaze_value=self._phaze_feature(move, turn),
            setup_denial=self._denial_feature(move, turn),
            timer_value=max(status.timer_value, self._seed_feature(move, turn)),
        )

    def _denial_feature(self, move: Move, turn: _Turn) -> float:
        """What taking their plan away is worth: Haze, Taunt, Encore and Disable."""
        theirs = _positive_boosts(turn.opponent)
        if _resets_stat_stages(move):
            # Haze clears both sides, so it is only worth something when they are the ones ahead.
            return max(0.0, (theirs - _positive_boosts(turn.me)) / _BOOST_CAP)
        blocked = _inflicted_volatile(move)
        if blocked not in _DENIAL_VOLATILES:
            return 0.0
        if blocked in turn.opponent.volatiles:
            return 0.0  # already taunted/encored; a second one buys nothing
        if not self._opponent_gains_from_a_free_turn(turn) and theirs == 0:
            return 0.0  # nothing to shut down: they have no setup and no recovery to deny
        # Worth more against something already partway through setting up than against a fresh one.
        return min(1.0, _DENIAL_BASE + theirs / _BOOST_CAP)

    def _seed_feature(self, move: Move, turn: _Turn) -> float:
        """Leech Seed as the timer it is — the same clock Toxic starts, plus the HP it hands back."""
        if _inflicted_volatile(move) is not ExtraStatus.LEECH_SEED:
            return 0.0
        if ExtraStatus.LEECH_SEED in turn.opponent.volatiles or Type.GRASS in turn.opponent.types:
            return 0.0  # already seeded, or a Grass type, which cannot be
        survival = (
            math.ceil(turn.opponent.live_stats.HP / turn.mine.best_expected)
            if turn.mine.best_expected > 0
            else math.inf
        )
        return min(survival, 6.0) / 6

    def _status_features(self, move: Move, turn: _Turn) -> ActionFeatures:
        status = _inflicted_status(move)
        if status is None or turn.opponent.status is not Status.NONE:
            return ActionFeatures()
        if status_cannot_land(status, turn.opponent, turn.me, turn.state.field):
            # A status the target is immune to never starts its timer, so it is worth nothing.
            return ActionFeatures()
        survival = (
            math.ceil(turn.opponent.live_stats.HP / turn.mine.best_expected)
            if turn.mine.best_expected > 0
            else math.inf
        )
        return ActionFeatures(
            timer_value=min(survival, 6.0) / 6,
            para_speed_control=1.0 if status is Status.PARALYSIS and turn.opponent_faster else 0.0,
            burn_disable=1.0 if status is Status.BURN and turn.theirs.best_category is Category.PHYSICAL else 0.0,
            sleep_tempo=self._sleep_tempo(status, turn),
        )

    def _sleep_tempo(self, status: Status, turn: _Turn) -> float:
        """What taking their turns away is worth -- the one thing sleep does and nothing else models."""
        if status is not Status.SLEEP:
            return 0.0
        mine_left = max(1, turn.me.live_stats.HP)
        return min(1.0, turn.theirs.best_expected * _SLEEP_TURNS / mine_left)

    def _hazard_feature(self, move: Move, turn: _Turn) -> float:
        hazard = set_hazard(move)
        if hazard is None:
            return 0.0
        their_side = turn.state.sides[1 - turn.side_index]
        if their_side.hazards.get(hazard, 0) >= HAZARD_LAYER_CAPS.get(hazard, 1):
            return 0.0
        # The marginal toll of this layer, not a count of who is left to walk into it.
        mine = turn.state.sides[turn.side_index].active_pokemon
        with_it = hazard_toll(their_side, extra=hazard, attacker=mine, state=turn.state)
        return with_it - hazard_toll(their_side, attacker=mine, state=turn.state)

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

    def _heal_feature(self, move: Move, turn: _Turn, low: int = 0, high: int = 0) -> float:
        """What recovery is worth once the residuals have taken their cut."""
        if not _heals(move):
            return 0.0
        healed = _drain_heal(move, turn, low, high)
        if healed is None:  # an ordinary recovery move: a flat fraction of its own maximum
            fraction = next((e.fraction for e in move.effects if isinstance(e, HealEffect)), 0.5)
            healed = fraction * turn.me.stat_totals.HP
        elif healed < 0:
            # Liquid Ooze turns the drain into damage, as the engine's `damage_apply` does.
            return max(-1.0, healed / max(1, turn.me.stat_totals.HP))
        healed = min(turn.me.stat_totals.HP - turn.me.live_stats.HP, healed)
        kept = healed - residual_drain(turn.me, turn.state)
        if kept <= 0:
            return 0.0  # the toll meets or beats the heal: a race recovery cannot win
        turns_gained = kept / max(1, turn.theirs.strongest)
        return min(turns_gained, 2.0) / 2

    def _setup_feature(self, move: Move, turn: _Turn) -> float:
        """Boost value as exchange turns shaved off, plus a bonus when a speed boost flips who moves first."""
        free = _cannot_act(turn.opponent)
        if not free and turn.theirs.threat_expected >= turn.me.live_stats.HP:
            return 0.0
        speed_bonus = self._speed_flip_bonus(move, turn)
        if turn.mine.best_expected <= 0 or turn.mine.best_category not in _CATEGORY_STAT:
            return _with_free_turn(speed_bonus, free, move)
        stat = _CATEGORY_STAT[turn.mine.best_category]
        gain = _self_boost(move, stat)
        if gain == 0:
            return _with_free_turn(speed_bonus, free, move)
        current = turn.me.stat_stages[stat]
        boosted = turn.mine.best_expected * _stage_multiplier(min(6, current + gain)) / _stage_multiplier(current)
        turns_now = math.ceil(turn.opponent.live_stats.HP / turn.mine.best_expected)
        turns_after = math.ceil(turn.opponent.live_stats.HP / boosted)
        return _with_free_turn(min(turns_now - turns_after, 3) / 3 + speed_bonus, free, move)

    def _speed_flip_bonus(self, move: Move, turn: _Turn) -> float:
        """1.0 when a speed boost changes who acts first against the current opponent, else 0.0."""
        gain = _self_boost(move, Stats.SPEED)
        if gain <= 0:
            return 0.0
        my_side, their_side = turn.state.sides[turn.side_index], turn.state.sides[1 - turn.side_index]
        current_speed = effective_speed(turn.me, my_side, turn.state.field)
        their_speed = effective_speed(turn.opponent, their_side, turn.state.field)
        if current_speed > their_speed:
            return 0.0
        stage = turn.me.stat_stages[Stats.SPEED]
        boosted_speed = current_speed * _stage_multiplier(min(6, stage + gain)) / _stage_multiplier(stage)
        return 1.0 if boosted_speed > their_speed else 0.0

    def _wincon_support_feature(self, low: int, high: int, turn: _Turn) -> float:
        """Damage toward the opponent's specific check to my wincon, as a share of its HP."""
        if turn.me is turn.wincon:
            return 0.0  # the wincon's own offense is already priced by ko_now/hko_progress
        checker = self._wincon_checker(turn)
        if checker is None or checker is not turn.opponent:
            return 0.0
        return _damage_share((low + high) / 2, checker, cap=1.0)

    def _wincon_checker(self, turn: _Turn) -> Pokemon | None:
        """The opposing survivor with the worst matchup edge against my wincon."""
        my_side, their_side = turn.state.sides[turn.side_index], turn.state.sides[1 - turn.side_index]
        candidates = [p for p in their_side.team if not p.is_fainted()]
        if not candidates:
            return None
        return min(candidates, key=lambda foe: self._cached_edge(turn.wincon, foe, turn.state, my_side, their_side))

    def _phaze_feature(self, move: Move, turn: _Turn) -> float:
        if not move.force_switch:
            return 0.0
        their_side = turn.state.sides[1 - turn.side_index]
        if not any(not p.is_fainted() and p is not turn.opponent for p in their_side.team):
            return 0.0  # nothing to drag in
        boosts = sum(max(0, turn.opponent.stat_stages[stat]) for stat in _CATEGORY_STAT.values()) / 4
        chip = 0.5 if their_side.hazards and bootless_bench(their_side) > 0 else 0.0
        return min(boosts + chip, 1.5)

    # -- Switch scoring -------------------------------------------------------------

    def _switch_features(self, incoming: Pokemon, turn: _Turn, voluntary: bool) -> ActionFeatures:
        my_side, their_side = turn.state.sides[turn.side_index], turn.state.sides[1 - turn.side_index]
        incoming_edge = self._cached_edge(incoming, turn.opponent, turn.state, my_side, their_side)
        matchup_gain = incoming_edge / EDGE_CAP
        wincon = 0.0
        if voluntary:
            matchup_gain -= turn.edge / EDGE_CAP  # what we give up by leaving this exchange
            if turn.me is turn.wincon and turn.edge < 0:
                wincon += 1.0  # pull the game plan out of a losing exchange
        if incoming is turn.wincon and incoming_edge < 0:
            wincon -= 1.0  # don't feed the closer to a bad matchup
        # `matchup_gain` is one quantised turns differential, too coarse to separate switch-ins.
        their_hit = self._pair_offense(turn.opponent, incoming, turn.state).threat_expected
        my_hit = self._pair_offense(incoming, turn.opponent, turn.state).best_expected
        outspeeds = effective_speed(incoming, my_side, turn.state.field) > effective_speed(
            turn.opponent, their_side, turn.state.field
        )
        return ActionFeatures(
            matchup_gain=matchup_gain,
            tempo_cost=-1.0 if voluntary else 0.0,
            entry_cost=-entry_hazard_chip(incoming, my_side) / incoming.stat_totals.HP,
            wincon_preservation=wincon,
            incoming_damage_taken=-min(their_hit / incoming.stat_totals.HP, 2.0),
            incoming_damage_dealt=_damage_share(my_hit, turn.opponent, cap=2.0),
            incoming_outspeeds=1.0 if outspeeds else 0.0,
            lock_relief=self._lock_relief(turn) if voluntary else 0.0,
            volatile_relief=self._volatile_relief(turn) if voluntary else 0.0,
            debuff_relief=self._debuff_relief(turn) if voluntary else 0.0,
            setup_denial=self._setup_denial(incoming, turn),
        )

    def _lock_relief(self, turn: _Turn) -> float:
        """Leaving frees a choice-locked active. Worth more the worse the move it is stuck with."""
        locked = turn.me.choice_locked_move
        if locked is None:
            return 0.0
        move = turn.me.moves[locked]
        if move is None:
            return 1.0  # locked into an empty slot: the engine will substitute Struggle
        stuck = damage_range(move, turn.me, turn.opponent, turn.state)[1]
        best = turn.mine.strongest
        return 1.0 if best <= 0 else 1.0 - min(stuck / best, 1.0)

    def _volatile_relief(self, turn: _Turn) -> float:
        """Switching clears every volatile; confusion and Leech Seed are the ones worth paying for."""
        return sum(_VOLATILE_RELIEF.get(volatile, 0.0) for volatile in turn.me.volatiles)

    def _debuff_relief(self, turn: _Turn) -> float:
        """Summed negative stat stages on the active, unquantised so `ceil()` cannot swallow it."""
        dropped = -sum(min(0, turn.me.stat_stages[stat]) for stat in _RELIEF_STATS)
        return min(dropped, 6) / 6

    def _setup_denial(self, incoming: Pokemon, turn: _Turn) -> float:
        """How much bringing this pokemon in denies them the free turn described by `_setup_concession`."""
        if not self._opponent_gains_from_a_free_turn(turn):
            return 0.0
        my_hit = self._pair_offense(incoming, turn.opponent, turn.state).best_expected
        return _damage_share(my_hit, turn.opponent, cap=1.0)

    def _opponent_gains_from_a_free_turn(self, turn: _Turn) -> bool:
        """Do they hold a boosting move, or recovery that outpaces us? Their set is believed, not known."""
        boosts = any(
            _self_boost(move, stat) > 0 for move in _known_moves(turn.opponent) for stat in _CATEGORY_STAT.values()
        )
        heals = any(_heals(move) for move in _known_moves(turn.opponent))
        return boosts or heals

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

    def _cached_edge(
        self, mine: Pokemon, theirs: Pokemon, state: BattleState, my_side: SideState, their_side: SideState
    ) -> float:
        """Live-HP exchange edge from full-strength cached offenses: exact for benched, unstaged pokemon."""
        my_best = self._pair_offense(mine, theirs, state).best_expected
        their_threat = self._pair_offense(theirs, mine, state).threat_expected
        faster = effective_speed(mine, my_side, state.field) >= effective_speed(theirs, their_side, state.field)
        return exchange_edge_from(mine.live_stats.HP, theirs.live_stats.HP, my_best, their_threat, faster)

    def _wincon(self, state: BattleState, side_index: int) -> Pokemon:
        """The teammate with the best full-strength matchup spread against their survivors."""
        my_side, their_side = state.sides[side_index], state.sides[1 - side_index]
        survivors = [p for p in their_side.team if not p.is_fainted()]
        candidates = [p for p in my_side.team if not p.is_fainted()]
        return max(
            candidates,
            key=lambda mine: sum(self._cached_edge(mine, foe, state, my_side, their_side) for foe in survivors),
        )


def _damage_share(damage: float, target: Pokemon, cap: float) -> float:
    """Damage as a share of what the target has left, capped; 0.0 once it has nothing left."""
    if target.live_stats.HP <= 0:
        return 0.0
    return min(damage / target.live_stats.HP, cap)


def _known_moves(pokemon: Pokemon) -> list[Move]:
    return [move for slot in MoveSlot if (move := pokemon.moves[slot]) is not None]


def _any_effect(features: ActionFeatures) -> bool:
    """Whether this move does anything beyond damage: the fodder bonus needs a free useful turn."""
    return any(value != 0.0 for value in features.as_vector())


def _offense(attacker: Pokemon, defender: Pokemon, state: BattleState) -> _Offense:
    """Both reads of one attacker: its best modelled move and the expected threat over its possible sets."""
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
    return _Offense(
        best_expected=best_expected,
        threat_expected=posterior_threat(attacker, defender, state),
        strongest=strongest,
        best_category=best_category,
    )


def _positive_boosts(pokemon: Pokemon) -> float:
    """How far ahead this Pokemon's offensive and speed stages have put it."""
    return float(sum(max(0, pokemon.stat_stages[stat]) for stat in _BOOST_STATS))


def _resets_stat_stages(move: Move) -> bool:
    return any(isinstance(effect, CodedEffect) and effect.kind is CodedMoveKind.HAZE for effect in move.effects)


def _inflicted_volatile(move: Move) -> ExtraStatus | None:
    """The volatile this move puts on the *target*, if any."""
    for effect in move.effects:
        if (
            isinstance(effect, InflictStatusEffect)
            and isinstance(effect.status, ExtraStatus)
            and not effect.to_self
            and not effect.is_secondary
        ):
            return effect.status
    return None


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


def set_hazard(move: Move) -> Hazards | None:
    for effect in move.effects:
        if isinstance(effect, SideConditionEffect) and effect.kind in HAZARD_LAYER_CAPS:
            return effect.kind
    return None


def _cannot_act(pokemon: Pokemon) -> bool:
    """Whether this Pokemon is certain to lose its next turn."""
    if pokemon.status is Status.SLEEP:
        return pokemon.status_turns > 1
    return pokemon.status is Status.FREEZE


def _with_free_turn(value: float, free: bool, move: Move) -> float:
    """A boost taken against something that cannot act is worth a floor, whatever it shaves."""
    if not free or not any(_self_boost(move, stat) > 0 for stat in _BOOST_STATS):
        return value
    return min(1.0, max(value, _FREE_TURN_SETUP))


def _heals(move: Move) -> bool:
    return move.healing or any(isinstance(e, HealEffect) for e in move.effects)


def _drain_heal(move: Move, turn: _Turn, low: int, high: int) -> float | None:
    """What a draining move gives its user, or None if it is not one."""
    drained = next((e.drain_percent for e in move.effects if isinstance(e, DamageEffect) and e.drain_percent), None)
    if drained is None:
        return None
    dealt = min((low + high) / 2, turn.opponent.live_stats.HP)  # no draining off HP that is not there
    healed = drained * dealt
    return -healed if turn.opponent.ability is Ability.LIQUID_OOZE else healed


def _self_boost(move: Move, stat: Stats) -> int:
    return sum(
        effect.stages.get(stat, 0)
        for effect in move.effects
        if isinstance(effect, StatStageChangeEffect) and effect.target == "SELF" and not effect.is_secondary
    )


def _stage_multiplier(stage: int) -> float:
    return (2 + stage) / 2 if stage >= 0 else 2 / (2 - stage)
