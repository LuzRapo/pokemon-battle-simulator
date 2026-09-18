import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Final

from battle_sim.engine.damage_apply import _fixed_amount
from battle_sim.engine.power import effective_power, move_type_override, payload_overrides
from battle_sim.maths.damage import calculate_damage
from battle_sim.maths.rng import RNG
from battle_sim.mechanics.abilities import ability_absorbs, static_ability_modifiers, static_field_modifiers
from battle_sim.mechanics.battle import BattleState, SideState, effective_weather
from battle_sim.mechanics.items import static_damage_modifiers
from battle_sim.mechanics.priority import effective_speed
from battle_sim.models.moves import DamageEffect, FixedDamageEffect, Move, MoveSlot, StatStageChangeEffect
from battle_sim.models.pokemon import BelievedSet, Pokemon
from battle_sim.models.type_matchups import type_effectiveness
from battle_sim.utils import Ability, ExtraStatus, Hazards, Item, Stats, Status, Type, Weather

_MIN_ROLL = 85
_MAX_ROLL = 100
_SPIKES_CHIP = {1: 1 / 8, 2: 1 / 6, 3: 1 / 4}
# One neutral Stealth Rock entry costs an eighth of a bar. Dividing the toll through by it keeps
# `hazard_toll` on the same scale as the plain headcount it replaces, so the genome weight fitted
# against that headcount still means what it meant.
_NEUTRAL_ROCK_DIVISOR = 8.0
# Both in whole bench members, on the same scale as the headcount: taking a Focus Sash off a
# Marshadow, or making a switch-in die to the hit it was going to survive, is worth about as much
# as one more body having to walk into the rocks at all.
_FREE_SURVIVAL_DENIAL = 1.0
_BREAKPOINT_CONVERSION = 1.0
EDGE_CAP = 4.0  # exchange edges live in [-EDGE_CAP, EDGE_CAP]
# `damage_range` fixes both the crit and the roll, so `calculate_damage` never touches this: its
# `rng` reads are guarded by `if is_crit is None` and `if random_roll is None`. Constructing one per
# call seeded a fresh Mersenne Twister ~200k times per search decision, ~11% of search time, for an
# object that is never used. Shared because it is only ever an unused argument.
_UNUSED_RNG: Final = RNG(seed=0)
_SANDSTORM_IMMUNE_TYPES = frozenset({Type.ROCK, Type.GROUND, Type.STEEL})
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

    # Asked after the type override, so an absorb answers the type the move actually arrives as.
    # The engine cancels these from an event handler, which this estimator never runs — without the
    # check a Ground move into a Levitate target priced as a clean one-shot, and the search picked
    # it over and over because by its own arithmetic it was the winning move.
    if ability_absorbs(move.type, move, attacker, defender):
        return 0, 0

    payload: dict[str, object] = {
        **payload_overrides(move, attacker, defender),
        "power_override": effective_power(move, damage_effect, attacker, defender, state),
    }
    # The item and ability multipliers the engine applies from the damage bus, which this never
    # runs. A list value appends to its channel; a scalar replaces one outright.
    static = {
        **static_damage_modifiers(move.type, move.category, attacker, defender, damage_effect.power),
        **static_ability_modifiers(move.type, move, damage_effect.power, attacker, defender),
        # Auras belong to whoever is standing on the field rather than to either party in this
        # exchange, so they cannot come from the attacker/defender pair above.
        **static_field_modifiers(move.type, [side.active_pokemon for side in state.sides]),
    }
    for channel, value in static.items():
        if not isinstance(value, list):
            payload[channel] = value
            continue
        existing = payload.get(channel)
        payload[channel] = [*existing, *value] if isinstance(existing, list) else list(value)
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
    return int(incoming.stat_totals.HP * entry_hazard_share(incoming, side.hazards))


def entry_hazard_share(incoming: Pokemon, hazards: Mapping[Hazards, int]) -> float:
    """The same toll as a share of the entrant's maximum HP, against an arbitrary hazard layout.

    Taking the layout rather than the side is what lets a scorer ask "what would this be worth if I
    set another layer", which is the question `_hazard_feature` needs answered and could not ask.
    """
    if incoming.item is Item.HEAVY_DUTY_BOOTS or incoming.ability is Ability.MAGIC_GUARD:
        return 0.0
    share = 0.0
    if hazards.get(Hazards.STEALTH_ROCK, 0) > 0:
        share += type_effectiveness(Type.ROCK, incoming.types) / 8
    layers = hazards.get(Hazards.SPIKES, 0)
    if layers > 0 and incoming.is_grounded():
        share += _SPIKES_CHIP[min(layers, 3)]
    return share


def has_free_survival(pokemon: Pokemon) -> bool:
    """Whether this Pokemon is currently guaranteed to live through one hit it otherwise would not.

    Both mechanisms need full HP, which is exactly why any chip at all switches them off.
    """
    if pokemon.live_stats.HP < pokemon.stat_totals.HP:
        return False
    return pokemon.item is Item.FOCUS_SASH or pokemon.ability is Ability.STURDY


def _reaches_a_breakpoint(attacker: Pokemon, victim: Pokemon, share: float, state: BattleState) -> bool:
    """Whether entry chip is what turns this attacker's non-kill into a kill."""
    best = best_expected_damage(attacker, victim, state)
    return best < victim.live_stats.HP <= best + share * victim.stat_totals.HP


def hazard_toll(
    side: SideState,
    extra: Hazards | None = None,
    attacker: Pokemon | None = None,
    state: BattleState | None = None,
) -> float:
    """What the hazards on this side will actually cost the members still to come in.

    Counted in Pokemon rather than in layers, and scaled so that a full bench of neutral targets
    reads the same as the headcount this replaces -- the point is not to move the average, it is to
    stop a Stealth Rock against a bench of Ho-Oh and Yveltal scoring the same as one against a bench
    of Steel types. Four-times-weak is eight times the toll of a double resist, and the old count
    could not tell them apart at all.

    The two terms that matter most are not proportional to HP at all, which is why counting chip
    alone still left the AI setting rocks on one turn in a hundred against a human's one in fifteen:

      * a Focus Sash or a Sturdy is a *guaranteed survival*, and any chip deletes it outright. That
        is not an eighth of a health bar, it is the difference between a Marshadow that takes two
        hits and one that takes one.
      * chip that drops a target under what the attacker already hits for converts a 2HKO into a
        OHKO, which does not cost them 12% of a Pokemon, it costs them the turn they were going to
        get back.

    Both are threshold effects, so both are counted in whole bodies rather than in health. Passing
    `attacker` and `state` enables the second; without them only the first is available.

    `extra` adds one hypothetical layer, so the marginal worth of setting a hazard is the difference
    between asking with it and asking without.
    """
    hazards: dict[Hazards, int] = dict(side.hazards)
    if extra is not None:
        hazards[extra] = hazards.get(extra, 0) + 1
    health, bodies = 0.0, 0.0
    for member in side.bench:
        if member.is_fainted() or member.item is Item.HEAVY_DUTY_BOOTS:
            continue
        share = entry_hazard_share(member, hazards)
        if share <= 0:
            continue
        health += share
        if has_free_survival(member):
            bodies += _FREE_SURVIVAL_DENIAL
        elif attacker is not None and state is not None and _reaches_a_breakpoint(attacker, member, share, state):
            bodies += _BREAKPOINT_CONVERSION
    return (health * _NEUTRAL_ROCK_DIVISOR + bodies) / 6


_SWEEP_STATS = (Stats.ATTACK, Stats.SP_ATTACK, Stats.SPEED)
_TWO_SHOT_WEIGHT = 0.5  # outsped and 2HKO'd is halfway to being swept, and the half worth acting on
# Toxic from full HP kills on the sixth tick; looking two turns past that covers the burn and
# Leech Seed cases without pretending to know what a battle looks like ten turns from now.
_RESIDUAL_HORIZON = 8
_BENCH_HORIZON = 3  # a benched member's clock is stopped; this is what it pays on its way back in
_RESIDUAL_DISCOUNT = 0.7  # per turn: what damage this far out is worth against a board that will have moved
_SETUP_DISCOUNT = 0.6  # a boost one turn away is worth less than one already on the board
_BOOST_CEILING = 6


def is_setting_up(pokemon: Pokemon) -> bool:
    """Whether this Pokemon has boosted anything that would let it run through a team."""
    return any(pokemon.stat_stages[stat] > 0 for stat in _SWEEP_STATS)


def sweep_threat(attacker: Pokemon, defenders: SideState, state: BattleState) -> float:
    """How much of `defenders` this attacker, as it currently stands, simply beats: outspeeding and
    killing in one. Zero to one.

    A position was worth material, hazards, status, and the edge between the two Pokemon presently
    facing each other — that last one being the only place a stat boost showed up at all. So a
    Volcarona two Quiver Dances deep read as "a slightly worse exchange" rather than "this now beats
    five of the six behind it", and the search would keep handing it turns to carry on. Being run
    over by a sweeper is a whole-team fact and it needs a whole-team number.

    Only asked of a Pokemon that has actually boosted something — the gate is what keeps this off
    the hot path, since every ordinary position skips it and pays nothing.

    Paralysis falls out of this for free rather than needing a rule: it halves the sweeper's speed
    inside `effective_speed`, so the same term that says "this is about to sweep me" also says
    "and paralysing it would stop that", which is the reading that makes speed control worth a turn.
    """
    if not is_setting_up(attacker):
        return 0.0
    return _beaten_share(attacker, _side_of(attacker, state), defenders, state)


def best_setup_boost(pokemon: Pokemon) -> dict[Stats, int]:
    """The stat stages the best setup move this Pokemon carries would put on itself in one use.

    "Best" is simply the largest total across the sweeping stats, which is what separates a Quiver
    Dance from a lone Iron Defense without needing a list of move names to maintain.
    """
    best: dict[Stats, int] = {}
    for move in usable_moves(pokemon):
        for effect in move.effects:
            if not isinstance(effect, StatStageChangeEffect) or effect.target != "SELF":
                continue
            if effect.is_secondary or effect.probability < 1.0:
                continue
            gained = {stat: n for stat, n in effect.stages.items() if n > 0 and stat in _SWEEP_STATS}
            if sum(gained.values()) > sum(best.values()):
                best = gained
    return best


def setup_potential(attacker: Pokemon, defenders: SideState, state: BattleState) -> float:
    """What this Pokemon would beat *after* the setup move it is carrying, zero to one.

    `sweep_threat` only fires once something has already boosted, so the search saw a sweep coming
    exactly one turn before it arrived and never saw the reason to spend a turn heading it off --
    nor the reason to spend a turn setting up itself, since a Dragon Dance was scored against the
    Pokemon in front of it rather than against the five behind. Both sides of that are the same
    number: what the board looks like one boost from now.

    Discounted because it is not free -- the boost still has to be granted a turn, and the holder
    has to survive being handed it.
    """
    boost = best_setup_boost(attacker)
    if not boost:
        return 0.0
    # On a copy: this is a hypothetical, and the last term that measured a sweeper against live
    # state made healing look like sweep prevention. Nothing here may touch the real Pokemon.
    hypothetical = attacker.model_copy(deep=True)
    for stat, stages in boost.items():
        hypothetical.stat_stages[stat] = min(_BOOST_CEILING, hypothetical.stat_stages[stat] + stages)
    return _SETUP_DISCOUNT * _beaten_share(hypothetical, _side_of(attacker, state), defenders, state)


def _beaten_share(attacker: Pokemon, attacker_side: SideState, defenders: SideState, state: BattleState) -> float:
    """Share of `defenders` this attacker outspeeds and kills, graded by how few hits it needs."""
    standing = [p for p in defenders.team if not p.is_fainted()]
    if not standing:
        return 0.0
    speed = effective_speed(attacker, attacker_side, state.field)
    beaten = 0.0
    for defender in standing:
        if speed <= effective_speed(defender, defenders, state.field):
            continue  # it gets a turn back, so this is a fight rather than a sweep
        hit = best_expected_damage(attacker, defender, state)
        if hit <= 0:
            continue
        # Measured against full health, not what is left. Against current HP this reads chip damage
        # as being swept, which makes *healing* look like sweep prevention: a Snorlax at full HP
        # facing a +3 Volcarona picked Rest — healing nothing — over a Body Slam that could have
        # paralysed the thing, because topping up was the cheapest way to make this number fall.
        # The question here is whether that attacker can plough through a healthy team, which is a
        # fact about the attacker; how battered they are already is what material is for.
        #
        # Graded rather than a cliff, too. Counting only outright one-shots showed no danger at all
        # until the boost that made it lethal, so a sweeper could climb to +2 unopposed and the
        # alarm sounded on the turn it was already too late. Two turns per kill is still a sweep.
        if hit >= defender.stat_totals.HP:
            beaten += 1.0
        elif hit * 2 >= defender.stat_totals.HP:
            beaten += _TWO_SHOT_WEIGHT
    return beaten / len(standing)


def residual_drain(pokemon: Pokemon, state: BattleState) -> int:
    """Net HP this Pokemon expects to lose to residuals next turn — negative if it gains.

    Recovery is only worth what it *keeps*, and nothing scoring a heal knew that. A badly poisoned
    Moltres roosted eleven turns running while the toxic counter climbed past what Roost restores,
    dealt no damage the whole time, and died in a race it could not win from a winning position.

    Toxic is quoted at its *next* tick rather than its current one, which is both correct (the
    counter increments before it bites) and self-limiting: the number grows every turn it is asked,
    so the value of healing decays on its own instead of holding steady forever.

    Mirrors `engine.residuals`, which is the authority on every formula here.
    """
    return _residual_drain(pokemon, state, pokemon.status_turns)


def _residual_drain(pokemon: Pokemon, state: BattleState, toxic_counter: int) -> int:
    """`residual_drain` with the toxic counter supplied rather than read off the Pokemon.

    Projecting a poisoning forward means asking the same question about a counter that has not
    happened yet, and the formula must not be written twice for the two callers to drift apart.
    """
    if pokemon.ability is Ability.MAGIC_GUARD:
        return 0
    max_hp = pokemon.stat_totals.HP
    drain = 0
    poisoned = pokemon.status in (Status.POISON, Status.TOXIC)
    if not (poisoned and pokemon.ability is Ability.POISON_HEAL):
        if pokemon.status is Status.BURN:
            drain += max(1, max_hp // 16)
        elif pokemon.status is Status.POISON:
            drain += max(1, max_hp // 8)
        elif pokemon.status is Status.TOXIC:
            drain += max(1, max_hp * (toxic_counter + 1) // 16)
    if ExtraStatus.LEECH_SEED in pokemon.volatiles:
        drain += max(1, max_hp // 8)
    if effective_weather(state) is Weather.SANDSTORM:
        immune_type = _SANDSTORM_IMMUNE_TYPES.intersection(t for t in pokemon.types if t is not None)
        if not immune_type and pokemon.ability not in (Ability.SAND_VEIL, Ability.OVERCOAT):
            drain += max(1, max_hp // 16)
    if pokemon.item in (Item.LEFTOVERS, Item.BLACK_SLUDGE):
        drain -= max(1, max_hp // 16)  # ticking the other way, and it can turn a losing race
    return drain


def projected_residual_loss(
    pokemon: Pokemon, state: BattleState, turns: int, toxic_counter: int | None = None
) -> float:
    """Share of this Pokemon's *current* HP the residuals take over `turns` turns, in [0, 1].

    Standing still, doing nothing. This is the piece the search cannot reach: a Toxic landed now
    kills in six turns, and a six-turn lookahead costs about nine to the fourth times what a
    two-turn one does, so the horizon is never getting there by searching. Extrapolating the formula
    arithmetically costs nothing and tells the leaf evaluator the same thing.

    Later turns are discounted, and that is what separates a clock from ordinary chip. Undiscounted,
    eight turns of burn on a bulky Pokemon totals half a health bar, which priced avoiding a *possible*
    burn as highly as landing a real attack -- a Snorlax went straight back to resting through a
    Volcarona sweep rather than risk Flame Body. Discounting says the obvious thing instead: damage
    eight turns away is worth much less than damage next turn, because eight turns from now the
    battle will not look like this. Toxic survives it because it accelerates, which is the whole
    point; flat chip does not.

    Net healing returns 0.0 rather than a negative: a Leftovers holder out-ticking its burn is not
    under pressure, and how comfortably it wins that race is not what this measures.
    """
    max_hp = pokemon.stat_totals.HP
    start = pokemon.live_stats.HP
    if start <= 0 or max_hp <= 0:
        return 0.0
    counter = pokemon.status_turns if toxic_counter is None else toxic_counter
    hp = start
    value = 0.0
    for turn in range(turns):
        drain = _residual_drain(pokemon, state, counter)
        discount = _RESIDUAL_DISCOUNT**turn
        if hp - drain <= 0:  # it dies here; the rest of the bar is what that is worth
            return min(1.0, value + discount * hp / start)
        hp = min(max_hp, hp - drain)
        value += discount * drain / start
        counter += 1
    return min(1.0, max(0.0, value))


def residual_pressure(side: SideState, state: BattleState, horizon: int = _RESIDUAL_HORIZON) -> float:
    """How much of this side's material the clock takes if nobody intervenes, normalized to [0, 1].

    Only the active Pokemon is actually on the clock -- residuals never touch the bench -- so it
    gets the full horizon and its real toxic counter. Benched members still matter, because a
    poisoned team has to keep sending poisoned Pokemon back in, but they get a short window and a
    counter of zero: switching out resets it (`engine.switching`), so their clock restarts from
    scratch every time they return.
    """
    active = side.active_pokemon
    total = 0.0
    if not active.is_fainted():
        total += projected_residual_loss(active, state, horizon)
    for member in side.bench:
        if not member.is_fainted():
            total += projected_residual_loss(member, state, _BENCH_HORIZON, toxic_counter=0)
    return total / 6


def bootless_bench(side: SideState) -> int:
    """Members that could still be made to walk into hazards, and would pay for it.

    Hazards only ever charge a Pokemon on the way in, so whoever is already standing there never
    pays and must not be counted. Counting them was worth a fraction of a point per layer, which was
    enough for the search to keep setting Spikes against an opponent down to their last Pokemon:
    three turns of a won game spent on a toll nobody could ever be charged.

    Heavy-Duty Boots holders step over hazards, and are left out for the same reason.
    """
    return sum(1 for p in side.bench if not p.is_fainted() and p.item is not Item.HEAVY_DUTY_BOOTS)


def hazard_pressure(side: SideState) -> float:
    """How much the hazards on this side hurt whoever still has to come in, normalized to [0, 1]."""
    layers = sum(min(count, HAZARD_LAYER_CAPS.get(kind, 1)) for kind, count in side.hazards.items())
    if layers == 0:
        return 0.0
    return min(layers, 4) / 4 * bootless_bench(side) / 6


def _side_of(pokemon: Pokemon, state: BattleState) -> SideState:
    return next(side for side in state.sides if any(member is pokemon for member in side.team))
