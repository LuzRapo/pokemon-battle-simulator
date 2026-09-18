import pytest

from battle_sim.analysis import (
    best_expected_damage,
    best_setup_boost,
    damage_range,
    expected_damage,
    has_free_survival,
    hazard_toll,
    is_setting_up,
    posterior_threat,
    projected_residual_loss,
    setup_potential,
    sweep_threat,
)
from battle_sim.database.loader import get_move, get_species
from battle_sim.maths.rng import RNG
from battle_sim.mechanics.battle import BattleState, FieldState, SideState, effective_weather
from battle_sim.models.moves import MoveSet
from battle_sim.models.pokemon import BelievedSet, Pokemon
from battle_sim.models.stats import BaseStats, EVs, IVs
from battle_sim.search import PositionWeights, _apply_entry_field, evaluate_position
from battle_sim.utils import Ability, Hazards, Item, Nature, Status, Type, Weather

TACKLE = get_move("Tackle")
EMBER = get_move("Ember")
SPLASH = get_move("Splash")
FLAMETHROWER = get_move("Flamethrower")


def _mk(types: tuple[Type, Type | None] = (Type.NORMAL, None), item: Item = Item.NONE) -> Pokemon:
    return Pokemon(
        name="TestMon",
        nickname="X",
        level=50,
        base_stats=BaseStats(HP=100, ATTACK=100, DEFENCE=100, SP_ATTACK=100, SP_DEFENCE=100, SPEED=100),
        effort_values=EVs(),
        individual_values=IVs(),
        types=types,
        moves=MoveSet(TACKLE, EMBER, SPLASH, None),
        nature=Nature.HARDY,
        item=item,
    )


def _battle(attacker: Pokemon, defender: Pokemon) -> BattleState:
    return BattleState(
        sides=(SideState(team=[defender]), SideState(team=[attacker])), rng=RNG(seed=0), field=FieldState()
    )


def _set(weight: float, *moves, item: Item = Item.NONE, ability: Ability = Ability.NONE) -> BelievedSet:
    return BelievedSet(weight=weight, moves=tuple(moves), item=item, ability=ability)


def test_a_known_set_is_read_as_its_own_best_move():
    attacker, defender = _mk(), _mk()
    state = _battle(attacker, defender)
    assert attacker.believed_sets is None
    assert posterior_threat(attacker, defender, state) == best_expected_damage(attacker, defender, state)


def test_threat_is_the_weighted_mean_of_each_candidates_own_best_move():
    attacker, defender = _mk(), _mk(types=(Type.GRASS, None))
    state = _battle(attacker, defender)
    tackle = expected_damage(TACKLE, attacker, defender, state)
    ember = expected_damage(EMBER, attacker, defender, state)
    attacker.believed_sets = (_set(0.75, TACKLE, SPLASH), _set(0.25, EMBER, SPLASH))
    assert posterior_threat(attacker, defender, state) == pytest.approx(0.75 * tackle + 0.25 * ember)


def test_a_candidate_is_credited_only_with_the_coverage_it_owns():
    """The point of the estimator: one set holding both moves reads hotter than a mix holding one each."""
    attacker, defender = _mk(), _mk(types=(Type.GRASS, None))
    state = _battle(attacker, defender)
    attacker.believed_sets = (_set(1.0, TACKLE, EMBER),)
    both = posterior_threat(attacker, defender, state)
    attacker.believed_sets = (_set(0.5, TACKLE, SPLASH), _set(0.5, EMBER, SPLASH))
    split = posterior_threat(attacker, defender, state)
    assert both > split


def test_a_candidates_ability_is_priced_into_its_own_damage():
    """Mold Breaker ignores the Unaware wall's stat-stage immunity; a candidate without it does not.

    Item and most ability multipliers are bound to the damage event bus, which `damage_range`
    never runs — so the abilities it reads directly are what this can assert.
    """
    attacker, defender = _mk(), _mk()
    attacker.stat_stages.ATTACK = 2
    defender.ability = Ability.UNAWARE
    state = _battle(attacker, defender)
    attacker.believed_sets = (_set(1.0, TACKLE, ability=Ability.NONE),)
    ignored = posterior_threat(attacker, defender, state)
    attacker.believed_sets = (_set(1.0, TACKLE, ability=Ability.MOLD_BREAKER),)
    counted = posterior_threat(attacker, defender, state)
    assert counted > ignored
    assert attacker.ability is Ability.NONE  # pricing a candidate must not mutate the pokemon


def test_an_empty_candidate_pool_threatens_nothing():
    attacker, defender = _mk(), _mk()
    state = _battle(attacker, defender)
    attacker.believed_sets = (_set(1.0, SPLASH),)
    assert posterior_threat(attacker, defender, state) == 0.0


# -- Sweep threat -------------------------------------------------------------------
# A boost used to show up only in the edge between the two Pokemon presently facing each other, so
# a sweeper mid-setup read as "a slightly worse matchup" rather than "this beats the five behind it".


def _sweeper() -> Pokemon:
    fast = BaseStats(HP=100, ATTACK=180, DEFENCE=80, SP_ATTACK=80, SP_DEFENCE=80, SPEED=150)
    return Pokemon(
        name="TestMon",
        nickname="Sweeper",
        level=50,
        base_stats=fast,
        effort_values=EVs(),
        individual_values=IVs(),
        types=(Type.NORMAL, None),
        moves=MoveSet(TACKLE, EMBER, SPLASH, None),
        nature=Nature.HARDY,
        item=Item.NONE,
    )


def _victim(nickname: str, speed: int = 60) -> Pokemon:
    frail = BaseStats(HP=60, ATTACK=80, DEFENCE=60, SP_ATTACK=60, SP_DEFENCE=60, SPEED=speed)
    return Pokemon(
        name="TestMon",
        nickname=nickname,
        level=50,
        base_stats=frail,
        effort_values=EVs(),
        individual_values=IVs(),
        types=(Type.NORMAL, None),
        moves=MoveSet(TACKLE, EMBER, SPLASH, None),
        nature=Nature.HARDY,
        item=Item.NONE,
    )


def _facing(sweeper: Pokemon, victims: list[Pokemon]) -> BattleState:
    return BattleState(sides=(SideState(team=[sweeper]), SideState(team=victims)), rng=RNG(seed=0), field=FieldState())


def test_an_unboosted_pokemon_is_not_a_sweep_threat() -> None:
    """The gate: every ordinary position skips this and pays nothing for it."""
    sweeper = _sweeper()
    state = _facing(sweeper, [_victim(f"V{i}") for i in range(4)])
    assert not is_setting_up(sweeper)
    assert sweep_threat(sweeper, state.sides[1], state) == 0.0


def test_the_threat_rises_as_it_sets_up() -> None:
    """And rises *before* the boost that makes it lethal — counting only one-shots meant a sweeper
    could climb to +2 unopposed and the alarm sounded on the turn it was already too late."""
    sweeper = _sweeper()
    state = _facing(sweeper, [_victim(f"V{i}") for i in range(4)])
    readings = []
    for stage in (1, 2, 3):
        sweeper.stat_stages.ATTACK = stage
        sweeper.stat_stages.SPEED = stage
        readings.append(sweep_threat(sweeper, state.sides[1], state))
    assert readings[0] > 0.0  # danger registers at +1, not only at the end
    assert readings == sorted(readings)
    assert readings[-1] == 1.0


def test_something_it_cannot_outspeed_is_not_being_swept() -> None:
    """Outspeeding is the whole difference between a sweep and a fight: they get a turn back."""
    sweeper = _sweeper()
    sweeper.stat_stages.ATTACK = 3
    quick = [_victim(f"Q{i}", speed=200) for i in range(4)]
    state = _facing(sweeper, quick)
    assert sweep_threat(sweeper, state.sides[1], state) == 0.0


def test_paralysis_cuts_a_sweep_it_actually_outruns() -> None:
    """Speed control falls out of the same term rather than needing a rule of its own: halving the
    sweeper's speed is read directly by `effective_speed`."""
    sweeper = _sweeper()
    sweeper.stat_stages.ATTACK = 3
    middling = [_victim(f"M{i}", speed=110) for i in range(4)]
    state = _facing(sweeper, middling)
    before = sweep_threat(sweeper, state.sides[1], state)

    sweeper.status = Status.PARALYSIS
    assert before > 0.0
    assert sweep_threat(sweeper, state.sides[1], state) < before


def test_a_position_is_worth_less_to_whoever_is_about_to_be_swept() -> None:
    """Each side against itself, before and after the setup — not against each other. One Pokemon
    facing four is three material down whatever it has boosted to, and rightly remains the worse
    position; what has to move is how each side rates the *same* board once the boosts land.
    """
    sweeper = _sweeper()
    state = _facing(sweeper, [_victim(f"V{i}") for i in range(4)])
    weights = PositionWeights()
    calm_defender = evaluate_position(state, 1, weights)
    calm_sweeper = evaluate_position(state, 0, weights)

    sweeper.stat_stages.ATTACK = 3
    sweeper.stat_stages.SPEED = 3

    assert evaluate_position(state, 1, weights) < calm_defender  # the defender sees the danger
    assert evaluate_position(state, 0, weights) > calm_sweeper  # and the sweeper sees the prize


# -- Ability absorbs ----------------------------------------------------------------
# Reported from match 3992ab76: the AI's Groudon threw Precipice Blades into a Levitate Rotom-Heat
# four turns running. Not a prediction — its own estimator read the move as 432-508 damage against
# 125 HP, because every absorb is an ON_BEFORE_MOVE handler and this estimator never runs the bus.


def _species(name: str, moves: list[str], ability: Ability, item: Item = Item.NONE) -> Pokemon:
    spec = get_species(name)
    return Pokemon(
        name=spec.name,
        nickname=spec.name,
        level=50,
        base_stats=spec.base_stats,
        effort_values=EVs(ATTACK=252),
        individual_values=IVs(),
        types=spec.types,
        moves=MoveSet(*[get_move(m) for m in moves], *[None] * (4 - len(moves))),
        nature=Nature.HARDY,
        ability=ability,
        item=item,
    )


@pytest.mark.parametrize(
    ("defender", "ability", "move"),
    [
        ("Rotom-Heat", Ability.LEVITATE, "Earthquake"),
        ("Heatran", Ability.FLASH_FIRE, "Flamethrower"),
        ("Vaporeon", Ability.WATER_ABSORB, "Surf"),
        ("Jolteon", Ability.VOLT_ABSORB, "Thunderbolt"),
        ("Azumarill", Ability.SAP_SIPPER, "Giga Drain"),
        ("Chesnaught", Ability.BULLETPROOF, "Shadow Ball"),
        ("Exploud", Ability.SOUNDPROOF, "Boomburst"),
    ],
)
def test_an_absorbed_move_is_priced_at_nothing(defender: str, ability: Ability, move: str) -> None:
    target = _species(defender, ["Tackle"], ability)
    attacker = _species("Groudon", [move], Ability.DROUGHT)
    state = BattleState(
        sides=(SideState(team=[attacker]), SideState(team=[target])), rng=RNG(seed=0), field=FieldState()
    )
    assert damage_range(get_move(move), attacker, target, state) == (0, 0)


def test_mold_breaker_still_punches_through_every_absorb() -> None:
    """The estimator must not overcorrect: ignoring these abilities is the whole of what it does."""
    target = _species("Rotom-Heat", ["Tackle"], Ability.LEVITATE)
    attacker = _species("Groudon", ["Earthquake"], Ability.MOLD_BREAKER)
    state = BattleState(
        sides=(SideState(team=[attacker]), SideState(team=[target])), rng=RNG(seed=0), field=FieldState()
    )
    low, _ = damage_range(get_move("Earthquake"), attacker, target, state)
    assert low > 0


def test_an_air_balloon_is_read_as_the_same_immunity() -> None:
    """Grounding is asked of the Pokemon rather than matched against Levitate by name, so the item
    that does the same thing is covered by the same question."""
    floating = _species("Snorlax", ["Tackle"], Ability.IMMUNITY, item=Item.AIR_BALLOON)
    attacker = _species("Groudon", ["Earthquake"], Ability.DROUGHT)
    state = BattleState(
        sides=(SideState(team=[attacker]), SideState(team=[floating])), rng=RNG(seed=0), field=FieldState()
    )
    assert damage_range(get_move("Earthquake"), attacker, floating, state) == (0, 0)


def test_the_move_that_actually_hits_is_now_the_one_that_scores() -> None:
    """The match position itself: Fire Punch is resisted and unexciting, but it is not zero."""
    rotom = _species("Rotom-Heat", ["Will-O-Wisp"], Ability.LEVITATE, item=Item.CHOICE_SPECS)
    groudon = _species("Groudon", ["Precipice Blades", "Fire Punch", "Rock Polish", "Swords Dance"], Ability.DROUGHT)
    state = BattleState(sides=(SideState(team=[groudon]), SideState(team=[rotom])), rng=RNG(seed=0), field=FieldState())
    blades = expected_damage(get_move("Precipice Blades"), groudon, rotom, state)
    punch = expected_damage(get_move("Fire Punch"), groudon, rotom, state)
    assert blades == 0.0
    assert punch > 0.0
    assert best_expected_damage(groudon, rotom, state) == pytest.approx(punch)


# -- Item multipliers ---------------------------------------------------------------
# Also bus handlers, also invisible. Measured against the engine over 2,000 random hits, an Eviolite
# defender took 0.73x what was predicted and a Choice Band attacker dealt 1.25x — and Eviolite was
# in 40% of the sample, so this was quietly wrong in a large share of every decision made.


def _built(name: str, item: Item = Item.NONE) -> Pokemon:
    """Through `build_pokemon`, because `fully_evolved` comes from the species and Eviolite reads it."""
    import random as _random

    from battle_sim.setgen import random_set
    from battle_sim.teams import build_pokemon

    pokemon = build_pokemon(random_set(name, _random.Random(1)))
    pokemon.item = item
    return pokemon


def _hit(attacker: Pokemon, defender: Pokemon, move: str) -> tuple[int, int]:
    state = BattleState(
        sides=(SideState(team=[attacker]), SideState(team=[defender])), rng=RNG(seed=0), field=FieldState()
    )
    return damage_range(get_move(move), attacker, defender, state)


@pytest.mark.parametrize(
    ("item", "move", "ratio"),
    [(Item.CHOICE_BAND, "Body Slam", 1.5), (Item.CHOICE_SPECS, "Flamethrower", 1.5), (Item.LIFE_ORB, "Body Slam", 1.3)],
)
def test_an_attackers_item_raises_the_estimate(item: Item, move: str, ratio: float) -> None:
    defender = _built("Snorlax")
    bare = _hit(_built("Groudon"), defender, move)
    held = _hit(_built("Groudon", item), defender, move)
    assert held[0] / bare[0] == pytest.approx(ratio, abs=0.06)


def test_eviolite_is_read_only_on_something_that_can_still_evolve() -> None:
    attacker = _built("Groudon")
    unevolved = _built("Chansey")
    assert not unevolved.fully_evolved
    bare = _hit(attacker, unevolved, "Body Slam")
    unevolved.item = Item.EVIOLITE
    assert _hit(attacker, unevolved, "Body Slam")[0] / bare[0] == pytest.approx(1 / 1.5, abs=0.06)

    grown = _built("Snorlax")
    assert grown.fully_evolved
    plain = _hit(attacker, grown, "Body Slam")
    grown.item = Item.EVIOLITE
    assert _hit(attacker, grown, "Body Slam") == plain  # the item does nothing here, and must read as nothing


def test_assault_vest_only_blunts_special_damage() -> None:
    attacker = _built("Groudon")
    defender = _built("Snorlax")
    physical = _hit(attacker, defender, "Body Slam")
    special = _hit(attacker, defender, "Flamethrower")
    defender.item = Item.ASSAULT_VEST
    assert _hit(attacker, defender, "Body Slam") == physical
    assert _hit(attacker, defender, "Flamethrower")[0] < special[0]


# -- Ability multipliers ------------------------------------------------------------
# The second tranche of bus-only effects. Every one of these was invisible, so the search priced a
# Huge Power hit at half strength, a Multiscale wall at double what it takes, and — worst of the
# lot — a burned Guts attacker as *weaker*, having applied the burn without the immunity to it.


def _ability_hit(move: str, attacker_ability: Ability, defender_ability: Ability, defender: str = "Snorlax"):
    attacker, target = _built("Groudon"), _built(defender)
    attacker.ability, target.ability = attacker_ability, defender_ability
    return _hit(attacker, target, move)


@pytest.mark.parametrize(
    ("move", "ability", "ratio", "defender"),
    [
        ("Earthquake", Ability.ADAPTABILITY, 4 / 3, "Snorlax"),  # STAB 1.5 -> 2
        ("Bullet Punch", Ability.TECHNICIAN, 1.5, "Snorlax"),
        ("Fire Punch", Ability.SHEER_FORCE, 1.3, "Snorlax"),
        ("Body Slam", Ability.HUGE_POWER, 2.0, "Snorlax"),
        ("Fire Punch", Ability.TINTED_LENS, 2.0, "Vaporeon"),  # genuinely resisted, not immune
    ],
)
def test_an_attackers_ability_raises_the_estimate(move: str, ability: Ability, ratio: float, defender: str) -> None:
    plain = _ability_hit(move, Ability.DROUGHT, Ability.IMMUNITY, defender)
    boosted = _ability_hit(move, ability, Ability.IMMUNITY, defender)
    assert boosted[0] / plain[0] == pytest.approx(ratio, abs=0.08)


@pytest.mark.parametrize(
    ("move", "ability", "ratio", "defender"),
    [
        ("Fire Punch", Ability.THICK_FAT, 0.5, "Snorlax"),
        ("Body Slam", Ability.FUR_COAT, 0.5, "Snorlax"),
        ("Body Slam", Ability.MULTISCALE, 0.5, "Snorlax"),
        ("Earthquake", Ability.FILTER, 0.75, "Heatran"),
    ],
)
def test_a_defenders_ability_lowers_the_estimate(move: str, ability: Ability, ratio: float, defender: str) -> None:
    plain = _ability_hit(move, Ability.DROUGHT, Ability.IMMUNITY, defender)
    blunted = _ability_hit(move, Ability.DROUGHT, ability, defender)
    assert blunted[0] / plain[0] == pytest.approx(ratio, abs=0.08)


def test_a_defenders_ability_is_ignored_by_a_mould_breaker() -> None:
    plain = _ability_hit("Body Slam", Ability.DROUGHT, Ability.IMMUNITY)
    assert _ability_hit("Body Slam", Ability.MOLD_BREAKER, Ability.FUR_COAT) == plain


def test_guts_reads_a_burn_as_a_boost_rather_than_a_penalty() -> None:
    """It was inverted, not merely missing: the estimate applied the burn's Attack halving and not
    Guts' immunity to it, so a burned Guts attacker priced at half strength instead of half again."""
    attacker, defender = _built("Groudon"), _built("Snorlax")
    defender.ability = Ability.IMMUNITY
    attacker.ability = Ability.GUTS
    healthy = _hit(attacker, defender, "Body Slam")
    attacker.status = Status.BURN
    burned = _hit(attacker, defender, "Body Slam")
    assert burned[0] > healthy[0]
    assert burned[0] / healthy[0] == pytest.approx(1.5, abs=0.08)


def test_toxic_converts_inside_the_horizon_but_flat_chip_does_not():
    """The whole point of the term: a clock that kills scores far above chip that merely nibbles.

    Toxic escalates (1/16, 2/16, 3/16 ...) and takes a full bar on the sixth tick, so it survives
    the per-turn discount. A burn at a flat 1/16 never gets there, and pricing the two the same is
    what `timer_value`'s flat count was doing.
    """
    poisoned, burned = _mk(), _mk()
    poisoned.status, burned.status = Status.TOXIC, Status.BURN
    state = _battle(poisoned, _mk())
    toxic_value = projected_residual_loss(poisoned, state, turns=8)
    burn_value = projected_residual_loss(burned, state, turns=8)
    assert toxic_value > 2 * burn_value
    assert toxic_value > 0.3  # discounted, a kill six turns out is still worth a third of a bar


def test_a_leftovers_holder_out_ticking_its_burn_is_under_no_pressure():
    """Net healing is 0.0, not a negative — comfort is not what this measures."""
    healthy = _mk(item=Item.LEFTOVERS)
    healthy.status = Status.BURN  # 1/16 out, 1/16 back in
    assert projected_residual_loss(healthy, _battle(healthy, _mk()), turns=8) == 0.0


def test_the_clock_is_scored_on_them_and_not_on_us():
    """One-sided by design. Scored both ways, every contact move into a Flame Body read as risking
    a clock of one's own, which priced not attacking above attacking — see PositionWeights."""
    mine, theirs = _mk(), _mk()
    mine.status = Status.TOXIC
    state = BattleState(sides=(SideState(team=[mine]), SideState(team=[theirs])), rng=RNG(seed=0), field=FieldState())
    weights = PositionWeights(residual_pressure=6.0)
    my_own_clock = evaluate_position(state, 0, weights)
    assert my_own_clock == evaluate_position(state, 0, PositionWeights(residual_pressure=0.0))
    theirs.status = Status.TOXIC
    assert evaluate_position(state, 0, weights) > my_own_clock


def test_rocks_are_worth_more_against_a_bench_that_is_weak_to_them():
    """The headcount this replaces could not tell a Ho-Oh from a Ferrothorn, and priced Stealth Rock
    the same against both. Four-times-weak pays half a bar on the way in; a double resist pays a
    sixteenth."""
    flying = _mk(types=(Type.FIRE, Type.FLYING))
    steel = _mk(types=(Type.STEEL, Type.GROUND))
    against_flying = SideState(team=[_mk(), flying, flying])
    against_steel = SideState(team=[_mk(), steel, steel])
    rocks = Hazards.STEALTH_ROCK
    weak_toll = hazard_toll(against_flying, extra=rocks) - hazard_toll(against_flying)
    resist_toll = hazard_toll(against_steel, extra=rocks) - hazard_toll(against_steel)
    assert weak_toll > 7 * resist_toll


def test_a_neutral_bench_still_reads_as_the_headcount_it_replaces():
    """The scaling exists so the fitted `hazard_value` gene keeps meaning what it meant."""
    side = SideState(team=[_mk(), _mk(), _mk(), _mk(), _mk(), _mk()])  # five on the bench, all neutral
    marginal = hazard_toll(side, extra=Hazards.STEALTH_ROCK) - hazard_toll(side)
    assert marginal == pytest.approx(5 / 6)


def test_boots_holders_are_not_counted_as_walking_into_anything():
    booted = _mk(item=Item.HEAVY_DUTY_BOOTS)
    side = SideState(team=[_mk(), booted, booted])
    assert hazard_toll(side, extra=Hazards.STEALTH_ROCK) == 0.0


def test_rocks_are_worth_most_against_a_guaranteed_survival():
    """A Focus Sash is not an eighth of a health bar, it is a free hit — and any chip deletes it.
    Counting only HP is what left the AI setting rocks on one turn in a hundred."""
    sashed = _mk(item=Item.FOCUS_SASH)
    plain = _mk()
    with_sash = SideState(team=[_mk(), sashed])
    without = SideState(team=[_mk(), plain])
    rocks = Hazards.STEALTH_ROCK
    denied = hazard_toll(with_sash, extra=rocks) - hazard_toll(with_sash)
    ordinary = hazard_toll(without, extra=rocks) - hazard_toll(without)
    # The denial is priced as one more body walking in, so against a lone neutral target it doubles.
    assert denied == pytest.approx(2 * ordinary)


def test_a_sash_already_broken_is_not_denied_twice():
    """Both mechanisms need full HP, so a chipped holder has nothing left to take away."""
    sashed = _mk(item=Item.FOCUS_SASH)
    sashed.live_stats.HP -= 1
    assert not has_free_survival(sashed)
    side = SideState(team=[_mk(), sashed])
    plain = SideState(team=[_mk(), _mk()])
    rocks = Hazards.STEALTH_ROCK
    assert hazard_toll(side, extra=rocks) - hazard_toll(side) == pytest.approx(
        hazard_toll(plain, extra=rocks) - hazard_toll(plain)
    )


def test_an_aura_boosts_its_type_for_whoever_is_attacking():
    """Dark Aura is owned by whatever is standing on the field, not by either party in the exchange,
    so `static_ability_modifiers` — which only ever sees an attacker and a defender — could not
    apply it. Every Dark move in the game was priced at face value while a Yveltal was out."""
    attacker, defender = _mk(types=(Type.DARK, None)), _mk()
    attacker.moves = MoveSet(get_move("Crunch"), None, None, None)
    state = _battle(attacker, defender)
    plain = best_expected_damage(attacker, defender, state)
    defender.ability = Ability.DARK_AURA  # the *opponent's* aura still boosts our Dark move
    assert best_expected_damage(attacker, defender, state) > plain


def test_a_primals_weather_is_on_the_board_the_lead_matrix_scores():
    """A hand-built state never runs the switch-in that turns Desolate Land on, so Water moves at a
    Groudon scored at full strength — 370 against a true 184 — and the lead matrix read a Primal as
    a bad lead on that arithmetic."""
    groudon = _mk(types=(Type.GROUND, None))
    groudon.ability = Ability.DESOLATE_LAND
    state = _battle(groudon, _mk())
    assert effective_weather(state) is Weather.NONE
    _apply_entry_field(state, groudon)
    assert effective_weather(state) is Weather.HARSH_SUN


def test_a_boost_is_scored_against_the_whole_team_not_the_pokemon_in_front():
    """`sweep_threat` fires only once something has already boosted, so a Dragon Dance was worth
    whatever it did to the current matchup and nothing else. This is what it is worth one boost out."""
    sweeper = _sweeper()
    sweeper.moves = MoveSet(TACKLE, get_move("Dragon Dance"), None, None)
    state = _facing(sweeper, [_victim(f"V{i}", speed=140) for i in range(4)])
    assert sweep_threat(sweeper, state.sides[1], state) == 0.0  # nothing boosted yet
    assert setup_potential(sweeper, state.sides[1], state) > 0.0


def test_reading_the_potential_does_not_boost_the_real_pokemon():
    """It is a hypothetical. The last term that measured a sweeper against live state is what made
    healing look like sweep prevention."""
    sweeper = _sweeper()
    sweeper.moves = MoveSet(TACKLE, get_move("Dragon Dance"), None, None)
    state = _facing(sweeper, [_victim("V", speed=140)])
    setup_potential(sweeper, state.sides[1], state)
    assert sweeper.stat_stages.ATTACK == 0
    assert sweeper.stat_stages.SPEED == 0


def test_a_purely_defensive_boost_is_not_a_sweep_setup():
    """Amnesia raises Special Defence, which wins no races and kills nothing."""
    wall = _sweeper()
    wall.moves = MoveSet(TACKLE, get_move("Amnesia"), None, None)
    assert best_setup_boost(wall) == {}
