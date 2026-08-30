import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import replace
from typing import Final

from battle_sim.engine.damage_apply import _fixed_amount
from battle_sim.engine.power import effective_power, move_type_override, payload_overrides
from battle_sim.maths.damage import calculate_damage
from battle_sim.maths.rng import RNG
from battle_sim.mechanics.battle import BattleState, SideState
from battle_sim.mechanics.priority import effective_speed
from battle_sim.models.moves import DamageEffect, FixedDamageEffect, Move, MoveSlot
from battle_sim.models.pokemon import BelievedSet, Pokemon
from battle_sim.models.type_matchups import type_effectiveness
from battle_sim.utils import Ability, Hazards, Item, Type

_MIN_ROLL = 85
_MAX_ROLL = 100
_SPIKES_CHIP = {1: 1 / 8, 2: 1 / 6, 3: 1 / 4}
EDGE_CAP = 4.0  # exchange edges live in [-EDGE_CAP, EDGE_CAP]
# `damage_range` fixes both the crit and the roll, so `calculate_damage` never touches this: its
# `rng` reads are guarded by `if is_crit is None` and `if random_roll is None`. Constructing one per
# call seeded a fresh Mersenne Twister ~200k times per search decision, ~11% of search time, for an
# object that is never used. Shared because it is only ever an unused argument.
_UNUSED_RNG: Final = RNG(seed=0)
HAZARD_LAYER_CAPS = {Hazards.STEALTH_ROCK: 1, Hazards.SPIKES: 3, Hazards.TOXIC_SPIKES: 2, Hazards.STICKY_WEB: 1}


def damage_range(move: Move, attacker: Pokemon, defender: Pokemon, state: BattleState) -> tuple[int, int]:
    """(min, max) damage of one use against the defender, crits excluded; (0, 0) if it cannot hurt."""
    damage_effect = next((e for e in move.effects if isinstance(e, DamageEffect)), None)
    if damage_effect is None:
        fixed_effect = next((e for e in move.effects if isinstance(e, FixedDamageEffect)), None)
        if fixed_effect is None:
            return 0, 0
        amount = _fixed_amount(fixed_effect, attacker, defender)
        return (amount, amount) if amount is not None else (0, 0)

    # Same type resolution the live engine applies before it ever calculates damage (Aerilate,
    # Judgment, Weather Ball, ...) -- without this the scorer reads a converted move's pre-
    # conversion type, both for effectiveness and for the "-ate" abilities' own power boost below.
    override_type = move_type_override(move, attacker, state)
    if override_type is not None and move.type is not override_type:
        move = replace(move, type=override_type)

    payload = {
        **payload_overrides(move, attacker, defender),
        "power_override": effective_power(move, damage_effect, attacker, defender, state),
    }
    if defender.ability is Ability.UNAWARE and attacker.ability is not Ability.MOLD_BREAKER:
        payload["ignore_attack_stages"] = True
    if attacker.ability is Ability.UNAWARE:
        payload["ignore_defense_stages"] = True
    defender_side = _side_of(defender, state)
    low = calculate_damage(
        attacker,
        defender,
        move,
        state.field,
        defender_side,
        rng=_UNUSED_RNG,
        is_crit=False,
        random_roll=_MIN_ROLL,
        modifiers=dict(payload),
    )
    high = calculate_damage(
        attacker,
        defender,
        move,
        state.field,
        defender_side,
        rng=_UNUSED_RNG,
        is_crit=False,
        random_roll=_MAX_ROLL,
        modifiers=dict(payload),
    )
    if damage_effect.multi_hit is not None:
        hits_low, hits_high = damage_effect.multi_hit
        low, high = low * hits_low, high * hits_high
    return low, high


def expected_damage(move: Move, attacker: Pokemon, defender: Pokemon, state: BattleState) -> float:
    """Accuracy-weighted midpoint of the damage range."""
    low, high = damage_range(move, attacker, defender, state)
    accuracy = move.accuracy_probability if move.accuracy_probability is not None else 1.0
    return (low + high) / 2 * accuracy


def usable_moves(pokemon: Pokemon) -> list[Move]:
    return [move for slot in MoveSlot if (move := pokemon.moves[slot]) is not None and pokemon.pp[slot] > 0]


def strongest_hit(attacker: Pokemon, defender: Pokemon, state: BattleState) -> int:
    """The worst case a defender must plan for: the attacker's biggest max-roll hit."""
    return max((damage_range(move, attacker, defender, state)[1] for move in usable_moves(attacker)), default=0)


def best_expected_damage(attacker: Pokemon, defender: Pokemon, state: BattleState) -> float:
    return max((expected_damage(move, attacker, defender, state) for move in usable_moves(attacker)), default=0.0)


def posterior_threat(attacker: Pokemon, defender: Pokemon, state: BattleState) -> float:
    """Expected incoming damage from an attacker whose set we only *believe*, not know.

    Scoring one guessed set commits to a single guess and inherits its whole error when the guess
    is wrong. The honest quantity is the expectation of the per-set best move over the posterior:
    an attacker picks the best move it *has*, so each candidate is credited with its own coverage
    and no more, and the reading degrades gracefully instead of tracking one arbitrary set.

    Measured on the full-dex pool (`battle_sim/calibration.py`) this cuts per-position error
    against the attacker's true best hit by ~18% with nothing revealed — the state most switch
    decisions are made in — and matches the single-set estimator once reveals pin the set down.

    A pokemon whose set is known (our own, or a sampled determinization) carries no posterior and
    falls through to `best_expected_damage`, which is then exactly right.
    """
    if attacker.believed_sets is None:
        return best_expected_damage(attacker, defender, state)
    total = 0.0
    for (item, ability), candidates in _by_equipment(attacker.believed_sets).items():
        # Damage depends on the attacker's item and ability but not on which other moves it holds,
        # so one variant prices the group, and each distinct move is calculated once however many
        # candidate sets share it — which most of them do, the sets differing by a slot or two.
        worn = (item, ability) == (attacker.item, attacker.ability)
        variant = attacker if worn else _equipped(attacker, item, ability)
        distinct = {move.name: move for candidate in candidates for move in candidate.moves}
        priced = {name: expected_damage(move, variant, defender, state) for name, move in distinct.items()}
        total += sum(candidate.weight * max(priced[move.name] for move in candidate.moves) for candidate in candidates)
    return total


def _by_equipment(sets: Sequence[BelievedSet]) -> dict[tuple[Item, Ability], list[BelievedSet]]:
    grouped: dict[tuple[Item, Ability], list[BelievedSet]] = defaultdict(list)
    for believed in sets:
        grouped[believed.item, believed.ability].append(believed)
    return grouped


def _equipped(attacker: Pokemon, item: Item, ability: Ability) -> Pokemon:
    """The attacker as it would be holding one candidate's item and ability.

    `stat_totals` survives the copy because nothing it depends on changed; recomputing it per
    candidate would cost more than the damage calculations this exists to serve.

    Only the effects `damage_range` reads directly move with the swap — Mold Breaker against
    Unaware, and whatever `payload_overrides`/`effective_power` consult. Choice Band and its
    kind bind handlers to the damage event bus, which no estimator in this module runs, so they
    are invisible here whether or not the candidate holds them.
    """
    return attacker.model_copy(update={"item": item, "ability": ability})


def survival_turns(defender: Pokemon, attacker: Pokemon, state: BattleState) -> float:
    """Turns the defender survives the attacker's expected hits; inf when it cannot be hurt."""
    best = posterior_threat(attacker, defender, state)
    if best <= 0:
        return math.inf
    return math.ceil(defender.live_stats.HP / best)


def exchange_edge(mine: Pokemon, theirs: Pokemon, state: BattleState) -> float:
    """Speed-ordered turns-to-KO differential in [-EDGE_CAP, EDGE_CAP]: positive means mine wins the 1v1.

    The pro framing: "I 2HKO, they 3HKO me, and I'm faster" — integer exchange turns plus
    a half-turn credit for acting first.
    """
    my_speed = effective_speed(mine, _side_of(mine, state), state.field)
    their_speed = effective_speed(theirs, _side_of(theirs, state), state.field)
    return exchange_edge_from(
        my_hp=mine.live_stats.HP,
        their_hp=theirs.live_stats.HP,
        my_best=posterior_threat(mine, theirs, state),
        their_best=posterior_threat(theirs, mine, state),
        faster=my_speed >= their_speed,
    )


def exchange_edge_from(my_hp: int, their_hp: int, my_best: float, their_best: float, faster: bool) -> float:
    """The exchange_edge formula over precomputed best-expected damages."""
    my_ttk = math.ceil(their_hp / my_best) if my_best > 0 else math.inf
    their_ttk = math.ceil(my_hp / their_best) if their_best > 0 else math.inf
    if math.isinf(my_ttk) and math.isinf(their_ttk):
        return 0.0
    return max(-EDGE_CAP, min(EDGE_CAP, their_ttk - my_ttk + (0.5 if faster else -0.5)))


def entry_hazard_chip(incoming: Pokemon, side: SideState) -> int:
    """Estimated HP an entrant loses to the hazards on its side; the player-facing approximation
    of the engine's entry sequence (Toxic Spikes status and ability absorbers are not counted)."""
    if incoming.item is Item.HEAVY_DUTY_BOOTS or incoming.ability is Ability.MAGIC_GUARD:
        return 0
    max_hp = incoming.stat_totals.HP
    chip = 0.0
    if side.hazards.get(Hazards.STEALTH_ROCK, 0) > 0:
        chip += max_hp / 8 * type_effectiveness(Type.ROCK, incoming.types)
    layers = side.hazards.get(Hazards.SPIKES, 0)
    if layers > 0 and incoming.is_grounded():
        chip += max_hp * _SPIKES_CHIP[min(layers, 3)]
    return int(chip)


def bootless_healthy(side: SideState) -> int:
    return sum(1 for p in side.team if not p.is_fainted() and p.item is not Item.HEAVY_DUTY_BOOTS)


def hazard_pressure(side: SideState) -> float:
    """How much the hazards on this side hurt its own bootless members, normalized to [0, 1]."""
    layers = sum(min(count, HAZARD_LAYER_CAPS.get(kind, 1)) for kind, count in side.hazards.items())
    if layers == 0:
        return 0.0
    return min(layers, 4) / 4 * bootless_healthy(side) / 6


def _side_of(pokemon: Pokemon, state: BattleState) -> SideState:
    return next(side for side in state.sides if any(member is pokemon for member in side.team))
