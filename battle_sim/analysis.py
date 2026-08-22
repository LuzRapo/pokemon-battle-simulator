import math

from battle_sim.engine.damage_apply import _fixed_amount
from battle_sim.engine.power import effective_power, payload_overrides
from battle_sim.maths.damage import calculate_damage
from battle_sim.maths.rng import RNG
from battle_sim.mechanics.battle import BattleState, SideState
from battle_sim.mechanics.priority import effective_speed
from battle_sim.models.moves import DamageEffect, FixedDamageEffect, Move, MoveSlot
from battle_sim.models.pokemon import Pokemon
from battle_sim.models.type_matchups import type_effectiveness
from battle_sim.utils import Ability, Hazards, Item, Type

_MIN_ROLL = 85
_MAX_ROLL = 100
_SPIKES_CHIP = {1: 1 / 8, 2: 1 / 6, 3: 1 / 4}
EDGE_CAP = 4.0  # exchange edges live in [-EDGE_CAP, EDGE_CAP]
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
        rng=RNG(seed=0),
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
        rng=RNG(seed=0),
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


def survival_turns(defender: Pokemon, attacker: Pokemon, state: BattleState) -> float:
    """Turns the defender survives the attacker's best expected hits; inf when it cannot be hurt."""
    best = best_expected_damage(attacker, defender, state)
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
        my_best=best_expected_damage(mine, theirs, state),
        their_best=best_expected_damage(theirs, mine, state),
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
