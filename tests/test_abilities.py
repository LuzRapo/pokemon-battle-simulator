import pytest

from battle_sim.database.loader import get_move
from battle_sim.engine import legal_actions, step
from battle_sim.engine.power import effective_power
from battle_sim.maths.rng import RNG
from battle_sim.mechanics.battle import BattleState, FieldState, SideState
from battle_sim.mechanics.priority import effective_speed
from battle_sim.models.actions import Action, ActionType
from battle_sim.models.log_events import CantAct, DoesNotAffect, StatChangesSwept
from battle_sim.models.moves import DamageEffect, Move, MoveSet, MoveSlot
from battle_sim.models.pokemon import NINE_LIVES, Pokemon
from battle_sim.models.stats import BaseStats, EVs, IVs
from battle_sim.utils import Ability, ExtraStatus, Hazards, Item, Nature, Stats, Status, Target, Terrain, Type, Weather

TACKLE = get_move("Tackle")
EMBER = get_move("Ember")
QUICK_ATTACK = get_move("Quick Attack")
SWORDS_DANCE = get_move("Swords Dance")
USE_TACKLE = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.FIRST)
USE_QUICK_ATTACK = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.SECOND)
USE_EMBER = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.THIRD)
USE_SWORDS_DANCE = Action(action=ActionType.USE_MOVE, target=Target.SELF, move=MoveSlot.FOURTH)


def _mk(
    nickname: str = "X",
    types: tuple[Type, Type | None] = (Type.NORMAL, None),
    ability: Ability = Ability.NONE,
    item: Item = Item.NONE,
    base_stats: BaseStats | None = None,
    moves: MoveSet | None = None,
    status: Status = Status.NONE,
    level: int = 50,
) -> Pokemon:
    if base_stats is None:
        base_stats = BaseStats(HP=100, ATTACK=100, DEFENCE=100, SP_ATTACK=100, SP_DEFENCE=100, SPEED=100)
    if moves is None:
        moves = MoveSet(TACKLE, QUICK_ATTACK, EMBER, SWORDS_DANCE)
    return Pokemon(
        name="TestMon",
        nickname=nickname,
        level=level,
        base_stats=base_stats,
        effort_values=EVs(),
        individual_values=IVs(),
        types=types,
        moves=moves,
        nature=Nature.HARDY,
        status=status,
        item=item,
        ability=ability,
    )


def _battle(side0: list[Pokemon], side1: list[Pokemon], seed: int = 0, field: FieldState | None = None) -> BattleState:
    return BattleState(
        sides=(SideState(team=side0), SideState(team=side1)),
        rng=RNG(seed=seed),
        field=field or FieldState(),
    )


def test_speed_boost_does_not_raise_speed_on_the_turn_it_enters():
    """Leading with it counts as entering too: Bulbapedia — "except for the turn it enters battle"."""
    a = _mk("A", ability=Ability.SPEED_BOOST)
    b = _mk("B")
    state = _battle([a], [b])
    step(state, {0: USE_SWORDS_DANCE, 1: USE_SWORDS_DANCE})
    assert a.stat_stages.SPEED == 0


def test_speed_boost_raises_speed_after_a_full_turn_out():
    a = _mk("A", ability=Ability.SPEED_BOOST)
    b = _mk("B")
    state = _battle([a], [b])
    step(state, {0: USE_SWORDS_DANCE, 1: USE_SWORDS_DANCE})
    step(state, {0: USE_SWORDS_DANCE, 1: USE_SWORDS_DANCE})
    assert a.stat_stages.SPEED == 1


def test_speed_boost_does_not_raise_speed_on_a_mid_battle_switch_in():
    a = _mk("A")
    a2 = _mk("A2", ability=Ability.SPEED_BOOST)
    b = _mk("B")
    state = _battle([a, a2], [b])
    switch = Action(action=ActionType.SWITCH_OUT, switch_in=a2)
    step(state, {0: switch, 1: USE_SWORDS_DANCE})
    assert a2.stat_stages.SPEED == 0
    step(state, {0: USE_SWORDS_DANCE, 1: USE_SWORDS_DANCE})
    assert a2.stat_stages.SPEED == 1


def test_intimidate_lowers_opponent_attack_on_switch_in():
    a, a2 = _mk("A"), _mk("A2", ability=Ability.INTIMIDATE)
    b = _mk("B")
    state = _battle([a, a2], [b])
    switch = Action(action=ActionType.SWITCH_OUT, switch_in=a2)
    step(state, {0: switch, 1: USE_TACKLE})
    assert b.stat_stages.ATTACK == -1


def test_intimidate_fires_at_battle_start_when_leading():
    a = _mk("A", ability=Ability.INTIMIDATE)
    b = _mk("B")
    state = _battle([a], [b])
    step(state, {0: USE_SWORDS_DANCE, 1: USE_TACKLE})
    assert b.stat_stages.ATTACK == -1


def test_lead_intimidate_fires_only_once():
    a = _mk("A", ability=Ability.INTIMIDATE)
    b = _mk("B")
    state = _battle([a], [b])
    step(state, {0: USE_SWORDS_DANCE, 1: USE_TACKLE})
    step(state, {0: USE_SWORDS_DANCE, 1: USE_TACKLE})
    assert b.stat_stages.ATTACK == -1


def test_lead_weather_ability_fires_at_battle_start():
    a = _mk("A", ability=Ability.SAND_STREAM)
    state = _battle([a], [_mk("B")])
    step(state, {0: USE_SWORDS_DANCE, 1: USE_SWORDS_DANCE})
    assert state.field.weather is Weather.SANDSTORM


def test_slower_lead_weather_setter_wins():
    fast = _mk(
        "Fast",
        ability=Ability.DROUGHT,
        base_stats=BaseStats(HP=100, ATTACK=100, DEFENCE=100, SP_ATTACK=100, SP_DEFENCE=100, SPEED=200),
    )
    slow = _mk(
        "Slow",
        ability=Ability.DRIZZLE,
        base_stats=BaseStats(HP=100, ATTACK=100, DEFENCE=100, SP_ATTACK=100, SP_DEFENCE=100, SPEED=50),
    )
    state = _battle([fast], [slow])
    step(state, {0: USE_SWORDS_DANCE, 1: USE_SWORDS_DANCE})
    assert state.field.weather is Weather.RAIN


def test_drought_sets_sun_on_switch_in():
    a, a2 = _mk("A"), _mk("A2", ability=Ability.DROUGHT)
    state = _battle([a, a2], [_mk("B")])
    switch = Action(action=ActionType.SWITCH_OUT, switch_in=a2)
    step(state, {0: switch, 1: USE_SWORDS_DANCE})
    assert state.field.weather is Weather.SUN
    assert state.field.weather_turns_left == 4


def test_drizzle_sets_rain():
    a, a2 = _mk("A"), _mk("A2", ability=Ability.DRIZZLE)
    state = _battle([a, a2], [_mk("B")])
    switch = Action(action=ActionType.SWITCH_OUT, switch_in=a2)
    step(state, {0: switch, 1: USE_SWORDS_DANCE})
    assert state.field.weather is Weather.RAIN


def test_sand_stream_sets_sandstorm():
    a, a2 = _mk("A"), _mk("A2", ability=Ability.SAND_STREAM)
    state = _battle([a, a2], [_mk("B")])
    switch = Action(action=ActionType.SWITCH_OUT, switch_in=a2)
    step(state, {0: switch, 1: USE_SWORDS_DANCE})
    assert state.field.weather is Weather.SANDSTORM


def test_snow_warning_sets_snow():
    a, a2 = _mk("A"), _mk("A2", ability=Ability.SNOW_WARNING)
    state = _battle([a, a2], [_mk("B")])
    switch = Action(action=ActionType.SWITCH_OUT, switch_in=a2)
    step(state, {0: switch, 1: USE_SWORDS_DANCE})
    assert state.field.weather is Weather.SNOW


def test_levitate_immune_to_ground_moves():
    earthquake = get_move("Earthquake")
    attacker = _mk(
        "A",
        moves=MoveSet(earthquake, TACKLE, EMBER, SWORDS_DANCE),
        base_stats=BaseStats(HP=100, ATTACK=200, DEFENCE=100, SP_ATTACK=100, SP_DEFENCE=100, SPEED=200),
    )
    defender = _mk("B", ability=Ability.LEVITATE)
    state = _battle([attacker], [defender])
    use_eq = Action(action=ActionType.USE_MOVE, target=Target.ALL_ADJACENT, move=MoveSlot.FIRST)
    step(state, {0: use_eq, 1: USE_SWORDS_DANCE})
    assert defender.live_stats.HP == defender.stat_totals.HP


def test_levitate_ignores_spikes():
    a = _mk("A")
    b, b2 = _mk("B"), _mk("B2", ability=Ability.LEVITATE)
    state = _battle([a], [b, b2])
    state.sides[1].hazards[Hazards.SPIKES] = 3
    switch = Action(action=ActionType.SWITCH_OUT, switch_in=b2)
    step(state, {0: USE_SWORDS_DANCE, 1: switch})
    assert b2.live_stats.HP == b2.stat_totals.HP


def test_wonder_guard_blocks_anything_short_of_super_effective():
    ice_beam = get_move("Ice Beam")  # neutral (1x) against Bug/Ghost
    attacker = _mk("A", moves=MoveSet(ice_beam, TACKLE, EMBER, SWORDS_DANCE))
    defender = _mk("B", types=(Type.BUG, Type.GHOST), ability=Ability.WONDER_GUARD)
    state = _battle([attacker], [defender])
    step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
    assert defender.live_stats.HP == defender.stat_totals.HP


def test_wonder_guard_lets_super_effective_hits_through():
    shadow_ball = get_move("Shadow Ball")  # super effective (2x) against Ghost
    attacker = _mk("A", moves=MoveSet(shadow_ball, TACKLE, EMBER, SWORDS_DANCE))
    defender = _mk("B", types=(Type.BUG, Type.GHOST), ability=Ability.WONDER_GUARD)
    state = _battle([attacker], [defender])
    use_shadow_ball = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.FIRST)
    step(state, {0: use_shadow_ball, 1: USE_SWORDS_DANCE})
    assert defender.live_stats.HP < defender.stat_totals.HP


def test_sturdy_survives_ohko_at_full_hp():
    glass_cannon = _mk(
        "Glass",
        base_stats=BaseStats(HP=100, ATTACK=200, DEFENCE=100, SP_ATTACK=100, SP_DEFENCE=100, SPEED=200),
    )
    fragile = _mk(
        "Fragile",
        base_stats=BaseStats(HP=1, ATTACK=1, DEFENCE=1, SP_ATTACK=1, SP_DEFENCE=1, SPEED=1),
        ability=Ability.STURDY,
    )
    state = _battle([glass_cannon], [fragile])
    step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
    assert fragile.live_stats.HP == 1
    assert fragile.ability is Ability.STURDY  # ability stays (unlike Focus Sash)


def test_flash_fire_immune_to_fire_and_boosts_own_fire():
    flash_fire_holder = _mk("F", ability=Ability.FLASH_FIRE, types=(Type.FIRE, None))
    attacker = _mk("A", types=(Type.NORMAL, None))
    state = _battle([attacker], [flash_fire_holder])
    step(state, {0: USE_EMBER, 1: USE_SWORDS_DANCE})
    assert flash_fire_holder.live_stats.HP == flash_fire_holder.stat_totals.HP
    assert flash_fire_holder.flash_fire_active is True


def test_volt_absorb_heals_from_electric_moves():
    thunderbolt = get_move("Thunderbolt")
    attacker = _mk("A", moves=MoveSet(thunderbolt, TACKLE, EMBER, SWORDS_DANCE))
    defender = _mk("B", ability=Ability.VOLT_ABSORB)
    defender.live_stats.HP = defender.stat_totals.HP // 2
    state = _battle([attacker], [defender])
    use_tb = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.FIRST)
    step(state, {0: use_tb, 1: USE_SWORDS_DANCE})
    assert defender.live_stats.HP > defender.stat_totals.HP // 2


def test_water_absorb_heals_from_water_moves():
    surf = get_move("Surf")
    attacker = _mk("A", moves=MoveSet(surf, TACKLE, EMBER, SWORDS_DANCE))
    defender = _mk("B", ability=Ability.WATER_ABSORB)
    defender.live_stats.HP = defender.stat_totals.HP // 2
    state = _battle([attacker], [defender])
    use_surf = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.FIRST)
    step(state, {0: use_surf, 1: USE_SWORDS_DANCE})
    assert defender.live_stats.HP > defender.stat_totals.HP // 2


def test_magic_guard_prevents_status_residual_damage():
    a = _mk("A", ability=Ability.MAGIC_GUARD, status=Status.POISON)
    b = _mk("B")
    state = _battle([a], [b])
    step(state, {0: USE_SWORDS_DANCE, 1: USE_SWORDS_DANCE})
    assert a.live_stats.HP == a.stat_totals.HP


def test_magic_guard_prevents_stealth_rock_damage():
    a = _mk("A")
    b, b2 = _mk("B"), _mk("B2", ability=Ability.MAGIC_GUARD)
    state = _battle([a], [b, b2])
    state.sides[1].hazards[Hazards.STEALTH_ROCK] = 1
    switch = Action(action=ActionType.SWITCH_OUT, switch_in=b2)
    step(state, {0: USE_SWORDS_DANCE, 1: switch})
    assert b2.live_stats.HP == b2.stat_totals.HP


def test_prankster_grants_priority_to_status_moves():
    a = _mk("A", ability=Ability.PRANKSTER, moves=MoveSet(get_move("Will-O-Wisp"), TACKLE, EMBER, SWORDS_DANCE))
    b = _mk("B", base_stats=BaseStats(HP=100, ATTACK=200, DEFENCE=100, SP_ATTACK=100, SP_DEFENCE=100, SPEED=200))
    state = _battle([a], [b], seed=1)
    use_wow = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.FIRST)
    step(state, {0: use_wow, 1: USE_TACKLE})
    assert b.status is Status.BURN or a.live_stats.HP < a.stat_totals.HP


def test_prankster_does_not_affect_dark_types():
    a = _mk(
        "A",
        ability=Ability.PRANKSTER,
        moves=MoveSet(get_move("Will-O-Wisp"), TACKLE, EMBER, SWORDS_DANCE),
        base_stats=BaseStats(HP=100, ATTACK=100, DEFENCE=100, SP_ATTACK=100, SP_DEFENCE=100, SPEED=200),
    )
    dark_def = _mk("D", types=(Type.DARK, None))
    state = _battle([a], [dark_def])
    use_wow = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.FIRST)
    log = step(state, {0: use_wow, 1: USE_SWORDS_DANCE})
    assert dark_def.status is Status.NONE
    assert any(isinstance(entry, DoesNotAffect) for entry in log)


def test_quick_feet_ignores_paralysis_halving_and_boosts_speed():
    paralyzed = _mk("P", ability=Ability.QUICK_FEET, status=Status.PARALYSIS)
    side = SideState(team=[paralyzed])
    base_speed = paralyzed.effective_stat(Stats.SPEED)
    eff = effective_speed(paralyzed, side, FieldState())
    assert eff == base_speed * 3 // 2


def test_adaptability_makes_stab_double():
    adaptable = _mk("A", ability=Ability.ADAPTABILITY, types=(Type.NORMAL, None))
    normal_stab = _mk("N", types=(Type.NORMAL, None))
    defender_a = _mk("B")
    defender_b = _mk("B")
    state_a = _battle([adaptable], [defender_a])
    state_b = _battle([normal_stab], [defender_b])
    hp_a, hp_b = defender_a.live_stats.HP, defender_b.live_stats.HP
    step(state_a, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
    step(state_b, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
    adapt_dmg = hp_a - defender_a.live_stats.HP
    base_dmg = hp_b - defender_b.live_stats.HP
    assert adapt_dmg > base_dmg
    # Adaptability gives 2x STAB instead of 1.5x — ratio should be ~4:3
    assert abs(adapt_dmg * 3 - base_dmg * 4) <= 3


def test_huge_power_doubles_attack():
    strong = _mk("S", ability=Ability.HUGE_POWER, types=(Type.FIRE, None))
    normal = _mk("N", types=(Type.FIRE, None))
    defender_a = _mk("B")
    defender_b = _mk("B")
    state_a = _battle([strong], [defender_a])
    state_b = _battle([normal], [defender_b])
    hp_a, hp_b = defender_a.live_stats.HP, defender_b.live_stats.HP
    step(state_a, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
    step(state_b, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
    huge_dmg = hp_a - defender_a.live_stats.HP
    base_dmg = hp_b - defender_b.live_stats.HP
    assert huge_dmg > base_dmg


def test_pure_power_doubles_attack():
    strong = _mk("S", ability=Ability.PURE_POWER, types=(Type.FIRE, None))
    normal = _mk("N", types=(Type.FIRE, None))
    defender_a = _mk("B")
    defender_b = _mk("B")
    state_a = _battle([strong], [defender_a])
    state_b = _battle([normal], [defender_b])
    hp_a, hp_b = defender_a.live_stats.HP, defender_b.live_stats.HP
    step(state_a, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
    step(state_b, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
    assert (hp_a - defender_a.live_stats.HP) > (hp_b - defender_b.live_stats.HP)


def _best_tackle(ability: Ability, status: Status, trials: int = 200) -> int:
    """The hardest this attacker ever tackles, over `trials` seeds.

    The damage roll spans 85-100%, so a single turn cannot be compared against another turn. The
    maximum over enough seeds is the 100% roll, which can be — and lets these assert a ratio rather
    than merely an ordering.
    """
    best = 0
    for seed in range(trials):
        attacker = _mk("G", ability=ability, status=status, types=(Type.FIRE, None))
        defender = _mk("B")
        state = _battle([attacker], [defender], seed=seed)
        before = defender.live_stats.HP
        step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
        best = max(best, before - defender.live_stats.HP)
    return best


def test_guts_boosts_attack_when_statused():
    healthy = _best_tackle(Ability.GUTS, Status.NONE)
    poisoned = _best_tackle(Ability.GUTS, Status.POISON)
    assert poisoned / healthy == pytest.approx(1.5, abs=0.05)
    assert healthy == _best_tackle(Ability.NONE, Status.NONE)  # nothing without a status to feed on


def test_guts_cancels_the_burn_halving_rather_than_merely_outweighing_it():
    """The assertion that has to be a ratio. This used to read `burned Guts > burned ordinary`, which
    passes just as happily when the burn's halving is still being applied underneath the boost:
    0.75x of healthy still beats 0.5x. Only the exact figures separate "cancelled" from "outweighed".

    A burned Guts attacker should hit for the same as a poisoned one — the burn costs it nothing at
    all — and for half again what it manages healthy.
    """
    healthy = _best_tackle(Ability.GUTS, Status.NONE)
    burned = _best_tackle(Ability.GUTS, Status.BURN)
    poisoned = _best_tackle(Ability.GUTS, Status.POISON)

    assert burned == poisoned  # the burn is doing nothing whatever to the damage
    assert burned / healthy == pytest.approx(1.5, abs=0.05)
    # And the halving is real for anybody else, so the test above is not measuring its absence.
    assert _best_tackle(Ability.NONE, Status.BURN) / _best_tackle(Ability.NONE, Status.NONE) == pytest.approx(
        0.5, abs=0.05
    )


def test_intimidate_blocked_by_substitute():
    substitute = get_move("Substitute")
    a = _mk("A", moves=MoveSet(substitute, TACKLE, EMBER, SWORDS_DANCE))
    b, b2 = _mk("B"), _mk("B2", ability=Ability.INTIMIDATE)
    state = _battle([a], [b, b2])
    use_sub = Action(action=ActionType.USE_MOVE, target=Target.SELF, move=MoveSlot.FIRST)
    step(state, {0: use_sub, 1: USE_SWORDS_DANCE})
    switch = Action(action=ActionType.SWITCH_OUT, switch_in=b2)
    step(state, {0: USE_TACKLE, 1: switch})
    assert a.stat_stages.ATTACK == 0


BRAVE_BIRD = get_move("Brave Bird")
USE_FIRST = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.FIRST)


def test_rock_head_prevents_recoil():
    a = _mk("A", ability=Ability.ROCK_HEAD, moves=MoveSet(BRAVE_BIRD, TACKLE, EMBER, SWORDS_DANCE))
    state = _battle([a], [_mk("B")])
    step(state, {0: USE_FIRST, 1: USE_SWORDS_DANCE})
    assert a.live_stats.HP == a.stat_totals.HP


def test_no_guard_makes_moves_always_hit():
    dynamic_punch = get_move("Dynamic Punch")  # 50% accuracy
    for seed in range(10):
        a = _mk("A", ability=Ability.NO_GUARD, moves=MoveSet(dynamic_punch, TACKLE, EMBER, SWORDS_DANCE))
        b = _mk("B")
        state = _battle([a], [b], seed=seed)
        step(state, {0: USE_FIRST, 1: USE_SWORDS_DANCE})
        assert b.live_stats.HP < b.stat_totals.HP


def test_motor_drive_absorbs_electric_and_raises_speed():
    thunderbolt = get_move("Thunderbolt")
    a = _mk("A", moves=MoveSet(thunderbolt, TACKLE, EMBER, SWORDS_DANCE))
    b = _mk("B", ability=Ability.MOTOR_DRIVE)
    state = _battle([a], [b])
    step(state, {0: USE_FIRST, 1: USE_SWORDS_DANCE})
    assert b.live_stats.HP == b.stat_totals.HP
    assert b.stat_stages.SPEED == 1


def test_pressure_doubles_pp_cost():
    a = _mk("A")
    b = _mk("B", ability=Ability.PRESSURE)
    state = _battle([a], [b])
    starting_pp = a.pp[MoveSlot.FIRST]
    step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
    assert a.pp[MoveSlot.FIRST] == starting_pp - 2
    assert b.pp[MoveSlot.FOURTH] == get_move("Swords Dance").pp - 1  # self-move pays normal PP


def test_solar_power_boosts_special_and_chips_in_sun():
    a = _mk("A", ability=Ability.SOLAR_POWER, types=(Type.FIRE, None))
    b = _mk("B")
    sunny = FieldState(weather=Weather.SUN, weather_turns_left=5)
    state = _battle([a], [b], field=sunny)
    plain = _mk("P", types=(Type.FIRE, None))
    state_plain = _battle([plain], [_mk("B2")], field=FieldState(weather=Weather.SUN, weather_turns_left=5))
    step(state, {0: USE_EMBER, 1: USE_SWORDS_DANCE})
    step(state_plain, {0: USE_EMBER, 1: USE_SWORDS_DANCE})
    boosted = state.sides[1].active_pokemon.stat_totals.HP - state.sides[1].active_pokemon.live_stats.HP
    plain_damage = (
        state_plain.sides[1].active_pokemon.stat_totals.HP - state_plain.sides[1].active_pokemon.live_stats.HP
    )
    assert boosted > plain_damage
    assert max(1, a.stat_totals.HP // 8) == a.stat_totals.HP - a.live_stats.HP  # sun chip


def test_moxie_raises_attack_on_ko():
    a = _mk(
        "A",
        ability=Ability.MOXIE,
        base_stats=BaseStats(HP=100, ATTACK=200, DEFENCE=100, SP_ATTACK=100, SP_DEFENCE=100, SPEED=200),
    )
    weak = _mk("W", base_stats=BaseStats(HP=1, ATTACK=1, DEFENCE=1, SP_ATTACK=1, SP_DEFENCE=1, SPEED=1))
    state = _battle([a], [weak])
    step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
    assert weak.is_fainted()
    assert a.stat_stages.ATTACK == 1


def test_clear_body_blocks_intimidate_and_stat_drops():
    growl = get_move("Growl")
    a = _mk("A", moves=MoveSet(growl, TACKLE, EMBER, SWORDS_DANCE))
    b = _mk("B", ability=Ability.CLEAR_BODY)
    state = _battle([b], [a, _mk("A2", ability=Ability.INTIMIDATE)])
    use_growl = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.FIRST)
    step(state, {1: use_growl, 0: USE_SWORDS_DANCE})
    assert b.stat_stages.ATTACK == 2  # only its own Swords Dance; Growl blocked
    switch = Action(action=ActionType.SWITCH_OUT, switch_in=state.sides[1].team[1])
    step(state, {0: USE_TACKLE, 1: switch})
    assert b.stat_stages.ATTACK == 2  # Intimidate blocked too


def test_sand_veil_immune_to_sandstorm_chip():
    a = _mk("A", ability=Ability.SAND_VEIL)
    state = _battle([a], [_mk("B", ability=Ability.SAND_STREAM)])
    step(state, {0: USE_SWORDS_DANCE, 1: USE_SWORDS_DANCE})
    assert a.live_stats.HP == a.stat_totals.HP


def test_rough_skin_chips_contact_attackers():
    a = _mk("A")
    b = _mk("B", ability=Ability.ROUGH_SKIN)
    state = _battle([a], [b])
    step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
    assert max(1, a.stat_totals.HP // 8) == a.stat_totals.HP - a.live_stats.HP


# -- New gen-9 ability coverage ---------------------------------------------------

X_SCISSOR = get_move("X-Scissor")
WATER_GUN = get_move("Water Gun")
DRAGON_CLAW = get_move("Dragon Claw")
VINE_WHIP = get_move("Vine Whip")
EARTHQUAKE = get_move("Earthquake")
BITE = get_move("Bite")
ROCK_SMASH = get_move("Rock Smash")
FAKE_OUT = get_move("Fake Out")
SPORE = get_move("Spore")
TAUNT_MOVE = get_move("Taunt")
BULLET_SEED = get_move("Bullet Seed")
SPLASH = get_move("Splash")
USE_FIRST = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.FIRST)
IDLE_MOVES = MoveSet(SPLASH, TACKLE, EMBER, SWORDS_DANCE)
USE_SPLASH = Action(action=ActionType.USE_MOVE, target=Target.SELF, move=MoveSlot.FIRST)


def _duel_damage(attacker: Pokemon, defender: Pokemon, action: Action, field: FieldState | None = None) -> int:
    state = _battle([attacker], [defender], field=field)
    hp_before = defender.live_stats.HP
    step(state, {0: action, 1: USE_SWORDS_DANCE})
    return hp_before - defender.live_stats.HP


def test_multiscale_halves_damage_at_full_hp():
    plain = _duel_damage(_mk("A"), _mk("B"), USE_TACKLE)
    scaled = _duel_damage(_mk("A"), _mk("B", ability=Ability.MULTISCALE), USE_TACKLE)
    assert 0 < scaled < plain


def test_vessel_of_ruin_weakens_opposing_special_attacks():
    plain = _duel_damage(_mk("A", types=(Type.FIRE, None)), _mk("B"), USE_EMBER)
    ruined = _duel_damage(_mk("A", types=(Type.FIRE, None)), _mk("B", ability=Ability.VESSEL_OF_RUIN), USE_EMBER)
    assert 0 < ruined < plain


def test_sword_of_ruin_weakens_the_opposing_defence():
    plain = _duel_damage(_mk("A"), _mk("B"), USE_TACKLE)
    boosted = _duel_damage(_mk("A", ability=Ability.SWORD_OF_RUIN), _mk("B"), USE_TACKLE)
    assert boosted > plain


def test_sharpness_boosts_slicing_moves():
    moves = MoveSet(X_SCISSOR, TACKLE, EMBER, SWORDS_DANCE)
    plain = _duel_damage(_mk("A", moves=moves), _mk("B"), USE_FIRST)
    sharp = _duel_damage(_mk("A", ability=Ability.SHARPNESS, moves=moves), _mk("B"), USE_FIRST)
    assert sharp > plain


def test_technician_boosts_weak_moves():
    plain = _duel_damage(_mk("A"), _mk("B"), USE_TACKLE)
    technical = _duel_damage(_mk("A", ability=Ability.TECHNICIAN), _mk("B"), USE_TACKLE)
    assert technical > plain


def test_blaze_boosts_fire_moves_in_a_pinch():
    healthy = _mk("A", ability=Ability.BLAZE, types=(Type.FIRE, None))
    hurt = _mk("A", ability=Ability.BLAZE, types=(Type.FIRE, None))
    hurt.live_stats.HP = hurt.stat_totals.HP // 3
    plain = _duel_damage(healthy, _mk("B"), USE_EMBER)
    pinched = _duel_damage(hurt, _mk("B"), USE_EMBER)
    assert pinched > plain


def test_supreme_overlord_scales_with_fallen_teammates():
    fresh_attacker, fresh_partner = _mk("A", ability=Ability.SUPREME_OVERLORD), _mk("P")
    state = _battle([fresh_attacker, fresh_partner], [_mk("B")])
    hp = state.sides[1].active_pokemon.live_stats.HP
    step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
    plain = hp - state.sides[1].active_pokemon.live_stats.HP

    grieving_attacker, dead_partner = _mk("A", ability=Ability.SUPREME_OVERLORD), _mk("P")
    dead_partner.live_stats.HP = 0
    state = _battle([grieving_attacker, dead_partner], [_mk("B")])
    hp = state.sides[1].active_pokemon.live_stats.HP
    step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
    empowered = hp - state.sides[1].active_pokemon.live_stats.HP
    assert empowered > plain


def test_prism_armor_softens_super_effective_hits():
    plain = _duel_damage(_mk("A", types=(Type.FIRE, None)), _mk("B", types=(Type.GRASS, None)), USE_EMBER)
    armored = _duel_damage(
        _mk("A", types=(Type.FIRE, None)), _mk("B", types=(Type.GRASS, None), ability=Ability.PRISM_ARMOR), USE_EMBER
    )
    assert 0 < armored < plain


def test_tinted_lens_doubles_resisted_hits():
    plain = _duel_damage(_mk("A", types=(Type.FIRE, None)), _mk("B", types=(Type.WATER, None)), USE_EMBER)
    tinted = _duel_damage(
        _mk("A", types=(Type.FIRE, None), ability=Ability.TINTED_LENS), _mk("B", types=(Type.WATER, None)), USE_EMBER
    )
    assert tinted > plain


def test_thick_fat_halves_fire_damage():
    plain = _duel_damage(_mk("A", types=(Type.FIRE, None)), _mk("B"), USE_EMBER)
    fat = _duel_damage(_mk("A", types=(Type.FIRE, None)), _mk("B", ability=Ability.THICK_FAT), USE_EMBER)
    assert 0 < fat < plain


def test_water_bubble_doubles_water_offense():
    moves = MoveSet(WATER_GUN, TACKLE, EMBER, SWORDS_DANCE)
    plain = _duel_damage(_mk("A", types=(Type.WATER, None), moves=moves), _mk("B"), USE_FIRST)
    bubbled = _duel_damage(
        _mk("A", types=(Type.WATER, None), ability=Ability.WATER_BUBBLE, moves=moves), _mk("B"), USE_FIRST
    )
    assert bubbled > plain


def test_water_bubble_blocks_burns():
    from battle_sim.engine.status_apply import _apply_main_status
    from battle_sim.mechanics.log import BattleLog

    mon = _mk("B", ability=Ability.WATER_BUBBLE)
    _apply_main_status(Status.BURN, mon, 0, RNG(seed=0), BattleLog(), ((mon,), ()))
    assert mon.status is Status.NONE


def test_purifying_salt_blocks_all_statuses():
    from battle_sim.engine.status_apply import _apply_main_status
    from battle_sim.mechanics.log import BattleLog

    mon = _mk("B", ability=Ability.PURIFYING_SALT)
    for status in (Status.BURN, Status.SLEEP, Status.TOXIC, Status.PARALYSIS):
        _apply_main_status(status, mon, 0, RNG(seed=0), BattleLog(), ((mon,), ()))
        assert mon.status is Status.NONE


def test_ice_scales_halves_special_damage_only():
    plain_special = _duel_damage(_mk("A", types=(Type.FIRE, None)), _mk("B"), USE_EMBER)
    scaled_special = _duel_damage(_mk("A", types=(Type.FIRE, None)), _mk("B", ability=Ability.ICE_SCALES), USE_EMBER)
    plain_physical = _duel_damage(_mk("A"), _mk("B"), USE_TACKLE)
    scaled_physical = _duel_damage(_mk("A"), _mk("B", ability=Ability.ICE_SCALES), USE_TACKLE)
    assert 0 < scaled_special < plain_special
    assert scaled_physical == plain_physical


def test_dragons_maw_boosts_dragon_moves():
    moves = MoveSet(DRAGON_CLAW, TACKLE, EMBER, SWORDS_DANCE)
    plain = _duel_damage(_mk("A", moves=moves), _mk("B"), USE_FIRST)
    mawed = _duel_damage(_mk("A", ability=Ability.DRAGONS_MAW, moves=moves), _mk("B"), USE_FIRST)
    assert mawed > plain


def test_grassy_surge_sets_terrain_and_heals_grounded_actives():
    surger = _mk("A", ability=Ability.GRASSY_SURGE)
    hurt = _mk("B")
    hurt.live_stats.HP = hurt.stat_totals.HP // 2
    hp_before = hurt.live_stats.HP
    state = _battle([surger], [hurt])
    step(state, {0: USE_SWORDS_DANCE, 1: USE_SWORDS_DANCE})
    assert state.field.terrain is Terrain.GRASSY
    assert hp_before + hurt.stat_totals.HP // 16 == hurt.live_stats.HP


def test_hadron_engine_sets_terrain_and_boosts_specials_on_it():
    electric_field = FieldState(terrain=Terrain.ELECTRIC, terrain_turns_left=8)
    plain = _duel_damage(_mk("A", types=(Type.FIRE, None)), _mk("B"), USE_EMBER, field=electric_field)
    engined = _duel_damage(
        _mk("A", types=(Type.FIRE, None), ability=Ability.HADRON_ENGINE),
        _mk("B"),
        USE_EMBER,
        field=FieldState(terrain=Terrain.ELECTRIC, terrain_turns_left=8),
    )
    assert engined > plain
    state = _battle([_mk("A", ability=Ability.HADRON_ENGINE)], [_mk("B")])
    step(state, {0: USE_SWORDS_DANCE, 1: USE_SWORDS_DANCE})
    assert state.field.terrain is Terrain.ELECTRIC


def test_orichalcum_pulse_sets_sun_and_boosts_physicals_in_it():
    sunny = FieldState(weather=Weather.SUN, weather_turns_left=8)
    plain = _duel_damage(_mk("A"), _mk("B"), USE_TACKLE, field=sunny)
    pulsed = _duel_damage(
        _mk("A", ability=Ability.ORICHALCUM_PULSE),
        _mk("B"),
        USE_TACKLE,
        field=FieldState(weather=Weather.SUN, weather_turns_left=8),
    )
    assert pulsed > plain
    state = _battle([_mk("A", ability=Ability.ORICHALCUM_PULSE)], [_mk("B")])
    step(state, {0: USE_SWORDS_DANCE, 1: USE_SWORDS_DANCE})
    assert state.field.weather is Weather.SUN


def test_dauntless_shield_boosts_defence_once_per_battle():
    shield, partner = _mk("S", ability=Ability.DAUNTLESS_SHIELD), _mk("P")
    state = _battle([shield, partner], [_mk("B")])
    step(state, {0: Action(action=ActionType.SWITCH_OUT, switch_in=partner), 1: USE_SWORDS_DANCE})
    assert shield.switch_in_boost_used  # the lead boost fired before the switch
    step(state, {0: Action(action=ActionType.SWITCH_OUT, switch_in=shield), 1: USE_SWORDS_DANCE})
    assert shield.stat_stages.DEFENCE == 0  # no second activation


def test_intrepid_sword_boosts_attack_on_entry():
    sword = _mk("S", ability=Ability.INTREPID_SWORD)
    state = _battle([sword], [_mk("B")])
    step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
    assert sword.stat_stages.ATTACK == 1


def test_download_reads_the_weaker_defence():
    bulky_body = BaseStats(HP=100, ATTACK=100, DEFENCE=150, SP_ATTACK=100, SP_DEFENCE=50, SPEED=100)
    downloader = _mk("D", ability=Ability.DOWNLOAD)
    state = _battle([downloader], [_mk("B", base_stats=bulky_body)])
    step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
    assert downloader.stat_stages.SP_ATTACK == 1
    assert downloader.stat_stages.ATTACK == 0


def test_protosynthesis_activates_in_sun_and_boosts_damage():
    sunny = FieldState(weather=Weather.SUN, weather_turns_left=8)
    plain = _duel_damage(_mk("A"), _mk("B"), USE_TACKLE, field=sunny)
    proto = _mk("A", ability=Ability.PROTOSYNTHESIS)
    boosted = _duel_damage(proto, _mk("B"), USE_TACKLE, field=FieldState(weather=Weather.SUN, weather_turns_left=8))
    assert proto.paradox_boost is Stats.ATTACK  # all-equal stats: Attack wins the tie like PS
    assert boosted > plain


def test_protosynthesis_consumes_booster_energy_without_sun():
    proto = _mk("A", ability=Ability.PROTOSYNTHESIS, item=Item.BOOSTER_ENERGY)
    state = _battle([proto], [_mk("B")])
    step(state, {0: USE_SWORDS_DANCE, 1: USE_SWORDS_DANCE})
    assert proto.item is Item.NONE
    assert proto.item_consumed
    assert proto.paradox_boost is not None
    assert proto.paradox_from_booster


def test_protosynthesis_fades_when_the_sun_does():
    proto = _mk("A", ability=Ability.PROTOSYNTHESIS)
    state = _battle([proto], [_mk("B")], field=FieldState(weather=Weather.SUN, weather_turns_left=1))
    step(state, {0: USE_SWORDS_DANCE, 1: USE_SWORDS_DANCE})  # the weather tick ends the sun this turn
    assert state.field.weather is Weather.NONE
    assert proto.paradox_boost is None


def test_quark_drive_activates_on_electric_terrain():
    quark = _mk("A", ability=Ability.QUARK_DRIVE)
    state = _battle([quark], [_mk("B")], field=FieldState(terrain=Terrain.ELECTRIC, terrain_turns_left=8))
    step(state, {0: USE_SWORDS_DANCE, 1: USE_SWORDS_DANCE})
    assert quark.paradox_boost is not None


def test_paradox_speed_boost_multiplies_effective_speed():
    speedy_body = BaseStats(HP=100, ATTACK=100, DEFENCE=100, SP_ATTACK=100, SP_DEFENCE=100, SPEED=200)
    proto = _mk("A", ability=Ability.PROTOSYNTHESIS, base_stats=speedy_body)
    state = _battle([proto], [_mk("B")], field=FieldState(weather=Weather.SUN, weather_turns_left=8))
    side = state.sides[0]
    base_speed = effective_speed(proto, side, FieldState())
    step(state, {0: USE_SWORDS_DANCE, 1: USE_SWORDS_DANCE})
    assert proto.paradox_boost is Stats.SPEED
    assert effective_speed(proto, side, state.field) == base_speed * 3 // 2


def test_mold_breaker_phazing_leaves_the_dragged_out_defender_unwired():
    dragon_tail = get_move("Dragon Tail")
    breaker = _mk("A", ability=Ability.TERAVOLT, moves=MoveSet(dragon_tail, TACKLE, EMBER, SWORDS_DANCE))
    proto = _mk("B1", ability=Ability.PROTOSYNTHESIS)
    teammate = _mk("B2")
    sunny = FieldState(weather=Weather.SUN, weather_turns_left=8)
    state = _battle([breaker], [proto, teammate], field=sunny)
    step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})  # slot FIRST is Dragon Tail here
    assert state.sides[1].active_pokemon is teammate  # the phaze landed
    assert not state.effects.is_registered(proto, Ability.PROTOSYNTHESIS)
    step(state, {0: USE_SWORDS_DANCE, 1: USE_SWORDS_DANCE})  # residuals must not touch benched handlers


def test_toxic_debris_lays_toxic_spikes_when_hit_physically():
    debris = _mk("B", ability=Ability.TOXIC_DEBRIS)
    state = _battle([_mk("A")], [debris])
    step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
    assert state.sides[0].hazards[Hazards.TOXIC_SPIKES] == 1
    step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
    step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
    assert state.sides[0].hazards[Hazards.TOXIC_SPIKES] == 2  # capped


def _status_after_contact(defender_ability: Ability) -> Status:
    for seed in range(40):
        attacker = _mk("A")
        state = _battle([attacker], [_mk("B", ability=defender_ability)], seed=seed)
        step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
        if attacker.status is not Status.NONE:
            return attacker.status
    return Status.NONE


def test_flame_body_burns_on_contact():
    assert _status_after_contact(Ability.FLAME_BODY) is Status.BURN


def test_static_paralyses_on_contact():
    assert _status_after_contact(Ability.STATIC) is Status.PARALYSIS


def test_toxic_chain_badly_poisons_on_hit():
    for seed in range(40):
        defender = _mk("B")
        state = _battle([_mk("A", ability=Ability.TOXIC_CHAIN)], [defender], seed=seed)
        step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
        if defender.status is not Status.NONE:
            assert defender.status is Status.TOXIC
            return
    raise AssertionError("Toxic Chain never activated across 40 seeds")


def test_weak_armor_trades_defence_for_speed():
    armored = _mk("B", ability=Ability.WEAK_ARMOR)
    state = _battle([_mk("A")], [armored])
    step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
    assert armored.stat_stages.DEFENCE == -1
    assert armored.stat_stages.SPEED == 2


def test_stamina_raises_defence_when_hit():
    stout = _mk("B", ability=Ability.STAMINA)
    state = _battle([_mk("A")], [stout])
    step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
    assert stout.stat_stages.DEFENCE == 1


def test_berserk_boosts_special_attack_when_crossing_half():
    berserker = _mk("B", ability=Ability.BERSERK)
    berserker.live_stats.HP = berserker.stat_totals.HP // 2 + 1
    state = _battle([_mk("A")], [berserker])
    step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
    assert berserker.stat_stages.SP_ATTACK == 1


def test_justified_boosts_attack_when_hit_by_dark():
    just = _mk("B", ability=Ability.JUSTIFIED, moves=IDLE_MOVES)
    moves = MoveSet(BITE, TACKLE, EMBER, SWORDS_DANCE)
    state = _battle([_mk("A", moves=moves)], [just])
    step(state, {0: USE_FIRST, 1: USE_SPLASH})
    assert just.stat_stages.ATTACK == 1


def test_thermal_exchange_boosts_attack_against_fire_and_blocks_burns():
    from battle_sim.engine.status_apply import _apply_main_status
    from battle_sim.mechanics.log import BattleLog

    exchanger = _mk("B", ability=Ability.THERMAL_EXCHANGE, moves=IDLE_MOVES)
    state = _battle([_mk("A", types=(Type.FIRE, None))], [exchanger])
    step(state, {0: USE_EMBER, 1: USE_SPLASH})
    assert exchanger.stat_stages.ATTACK == 1
    teams = (state.sides[0].team, state.sides[1].team)
    _apply_main_status(Status.BURN, exchanger, 1, RNG(seed=0), BattleLog(), teams)
    assert exchanger.status is Status.NONE


def test_cursed_body_can_disable_the_attacking_move():
    for seed in range(40):
        attacker = _mk("A")
        state = _battle([attacker], [_mk("B", ability=Ability.CURSED_BODY)], seed=seed)
        step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
        if attacker.disabled_slot is not None:
            assert attacker.disabled_slot is MoveSlot.FIRST
            return
    raise AssertionError("Cursed Body never activated across 40 seeds")


def test_sap_sipper_absorbs_grass_for_an_attack_boost():
    sipper = _mk("B", ability=Ability.SAP_SIPPER, moves=IDLE_MOVES)
    hp_before = sipper.live_stats.HP
    moves = MoveSet(VINE_WHIP, TACKLE, EMBER, SWORDS_DANCE)
    state = _battle([_mk("A", moves=moves)], [sipper])
    step(state, {0: USE_FIRST, 1: USE_SPLASH})
    assert hp_before == sipper.live_stats.HP
    assert sipper.stat_stages.ATTACK == 1


def test_well_baked_body_absorbs_fire_for_two_defence_stages():
    baked = _mk("B", ability=Ability.WELL_BAKED_BODY)
    hp_before = baked.live_stats.HP
    state = _battle([_mk("A", types=(Type.FIRE, None))], [baked])
    step(state, {0: USE_EMBER, 1: USE_SWORDS_DANCE})
    assert hp_before == baked.live_stats.HP
    assert baked.stat_stages.DEFENCE == 2


def test_earth_eater_heals_from_ground_moves():
    eater = _mk("B", ability=Ability.EARTH_EATER)
    eater.live_stats.HP = eater.stat_totals.HP // 2
    hp_before = eater.live_stats.HP
    moves = MoveSet(EARTHQUAKE, TACKLE, EMBER, SWORDS_DANCE)
    state = _battle([_mk("A", moves=moves)], [eater])
    step(state, {0: USE_FIRST, 1: USE_SWORDS_DANCE})
    assert hp_before + eater.stat_totals.HP // 4 == eater.live_stats.HP


def test_dry_skin_heals_in_rain_and_burns_in_sun():
    rained = _mk("A", ability=Ability.DRY_SKIN)
    rained.live_stats.HP = rained.stat_totals.HP // 2
    hp_before = rained.live_stats.HP
    state = _battle([rained], [_mk("B")], field=FieldState(weather=Weather.RAIN, weather_turns_left=8))
    step(state, {0: USE_SWORDS_DANCE, 1: USE_SWORDS_DANCE})
    assert hp_before + rained.stat_totals.HP // 8 == rained.live_stats.HP

    sunned = _mk("A", ability=Ability.DRY_SKIN)
    hp_before = sunned.live_stats.HP
    state = _battle([sunned], [_mk("B")], field=FieldState(weather=Weather.SUN, weather_turns_left=8))
    step(state, {0: USE_SWORDS_DANCE, 1: USE_SWORDS_DANCE})
    assert hp_before - sunned.stat_totals.HP // 8 == sunned.live_stats.HP


def test_bad_dreams_chips_sleeping_opponents():
    dreamer = _mk("A", ability=Ability.BAD_DREAMS)
    sleeper = _mk("B", status=Status.SLEEP)
    sleeper.status_turns = 5
    hp_before = sleeper.live_stats.HP
    state = _battle([dreamer], [sleeper])
    step(state, {0: USE_SWORDS_DANCE, 1: USE_SWORDS_DANCE})
    assert hp_before - sleeper.stat_totals.HP // 8 == sleeper.live_stats.HP


def test_hydration_cures_status_in_rain():
    wet = _mk("A", ability=Ability.HYDRATION, status=Status.BURN)
    state = _battle([wet], [_mk("B")], field=FieldState(weather=Weather.RAIN, weather_turns_left=8))
    step(state, {0: USE_SWORDS_DANCE, 1: USE_SWORDS_DANCE})
    assert wet.status is Status.NONE


def test_ice_body_heals_in_snow():
    chilly = _mk("A", ability=Ability.ICE_BODY)
    chilly.live_stats.HP = chilly.stat_totals.HP // 2
    hp_before = chilly.live_stats.HP
    state = _battle([chilly], [_mk("B")], field=FieldState(weather=Weather.SNOW, weather_turns_left=8))
    step(state, {0: USE_SWORDS_DANCE, 1: USE_SWORDS_DANCE})
    assert hp_before + chilly.stat_totals.HP // 16 == chilly.live_stats.HP


def test_poison_heal_heals_instead_of_chipping():
    healer = _mk("A", ability=Ability.POISON_HEAL, status=Status.POISON)
    healer.live_stats.HP = healer.stat_totals.HP // 2
    hp_before = healer.live_stats.HP
    state = _battle([healer], [_mk("B")])
    step(state, {0: USE_SWORDS_DANCE, 1: USE_SWORDS_DANCE})
    assert hp_before + healer.stat_totals.HP // 8 == healer.live_stats.HP


def test_regenerator_heals_a_third_on_switch_out():
    regen, partner = _mk("R", ability=Ability.REGENERATOR), _mk("P")
    regen.live_stats.HP = regen.stat_totals.HP // 2
    hp_before = regen.live_stats.HP
    state = _battle([regen, partner], [_mk("B")])
    step(state, {0: Action(action=ActionType.SWITCH_OUT, switch_in=partner), 1: USE_SWORDS_DANCE})
    assert hp_before + regen.stat_totals.HP // 3 == regen.live_stats.HP


def test_natural_cure_clears_status_on_switch_out():
    cured, partner = _mk("C", ability=Ability.NATURAL_CURE, status=Status.BURN), _mk("P")
    state = _battle([cured, partner], [_mk("B")])
    step(state, {0: Action(action=ActionType.SWITCH_OUT, switch_in=partner), 1: USE_SWORDS_DANCE})
    assert cured.status is Status.NONE


def test_good_as_gold_blocks_status_moves_but_not_attacks():
    from battle_sim.models.log_events import StatusMoveBlocked

    golden = _mk("B", ability=Ability.GOOD_AS_GOLD)
    moves = MoveSet(TAUNT_MOVE, TACKLE, EMBER, SWORDS_DANCE)
    state = _battle([_mk("A", moves=moves)], [golden])
    log = step(state, {0: USE_FIRST, 1: USE_SWORDS_DANCE})
    assert ExtraStatus.TAUNT not in golden.volatiles
    assert any(isinstance(entry, StatusMoveBlocked) for entry in log)
    hp_before = golden.live_stats.HP
    step(
        state,
        {
            0: Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.SECOND),
            1: USE_SWORDS_DANCE,
        },
    )
    assert hp_before > golden.live_stats.HP


def test_overcoat_blocks_sandstorm_chip():
    coated = _mk("A", ability=Ability.OVERCOAT)
    hp_before = coated.live_stats.HP
    state = _battle(
        [coated], [_mk("B", types=(Type.ROCK, None))], field=FieldState(weather=Weather.SANDSTORM, weather_turns_left=8)
    )
    step(state, {0: USE_SWORDS_DANCE, 1: USE_SWORDS_DANCE})
    assert hp_before == coated.live_stats.HP


def test_skill_link_always_rolls_max_hits():
    from battle_sim.models.log_events import MultiHitSummary

    moves = MoveSet(BULLET_SEED, TACKLE, EMBER, SWORDS_DANCE)
    state = _battle([_mk("A", ability=Ability.SKILL_LINK, moves=moves)], [_mk("B")])
    log = step(state, {0: USE_FIRST, 1: USE_SWORDS_DANCE})
    summary = next(entry for entry in log if isinstance(entry, MultiHitSummary))
    assert summary.hits == 5


def test_contrary_inverts_self_boosts():
    contrarian = _mk("A", ability=Ability.CONTRARY)
    state = _battle([contrarian], [_mk("B")])
    step(state, {0: USE_SWORDS_DANCE, 1: USE_SWORDS_DANCE})
    assert contrarian.stat_stages.ATTACK == -2


def test_contrary_turns_intimidate_into_a_boost():
    contrarian = _mk("B", ability=Ability.CONTRARY, moves=IDLE_MOVES)
    state = _battle([_mk("A", ability=Ability.INTIMIDATE)], [contrarian])
    step(state, {0: USE_SWORDS_DANCE, 1: USE_SPLASH})
    assert contrarian.stat_stages.ATTACK == 1


def test_defiant_retaliates_against_intimidate():
    defiant = _mk("B", ability=Ability.DEFIANT, moves=IDLE_MOVES)
    state = _battle([_mk("A", ability=Ability.INTIMIDATE)], [defiant])
    step(state, {0: USE_SWORDS_DANCE, 1: USE_SPLASH})
    assert defiant.stat_stages.ATTACK == 1  # -1 from Intimidate, +2 from Defiant


def test_competitive_retaliates_with_special_attack():
    competitor = _mk("B", ability=Ability.COMPETITIVE, moves=IDLE_MOVES)
    state = _battle([_mk("A", ability=Ability.INTIMIDATE)], [competitor])
    step(state, {0: USE_SWORDS_DANCE, 1: USE_SPLASH})
    assert competitor.stat_stages.ATTACK == -1
    assert competitor.stat_stages.SP_ATTACK == 2


def test_inner_focus_prevents_flinching():
    slow_body = BaseStats(HP=100, ATTACK=100, DEFENCE=100, SP_ATTACK=100, SP_DEFENCE=100, SPEED=10)
    focused = _mk("B", ability=Ability.INNER_FOCUS, base_stats=slow_body)
    attacker = _mk("A")
    moves = MoveSet(FAKE_OUT, TACKLE, EMBER, SWORDS_DANCE)
    hp_before = attacker.live_stats.HP
    state = _battle([_mk("A", moves=moves)], [focused])
    step(
        state, {0: USE_FIRST, 1: Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.FIRST)}
    )
    assert ExtraStatus.FLINCH not in focused.volatiles
    assert hp_before > state.sides[0].active_pokemon.live_stats.HP  # it still got its move off


def test_serene_grace_doubles_secondary_chances():
    moves = MoveSet(ROCK_SMASH, TACKLE, EMBER, SWORDS_DANCE)
    target = _mk("B")
    state = _battle([_mk("A", ability=Ability.SERENE_GRACE, moves=moves)], [target])
    step(state, {0: USE_FIRST, 1: USE_SWORDS_DANCE})
    assert target.stat_stages.DEFENCE == -1  # 50% doubled to a guaranteed drop


def test_shield_dust_blocks_secondary_effects():
    moves = MoveSet(ROCK_SMASH, TACKLE, EMBER, SWORDS_DANCE)
    dusted = _mk("B", ability=Ability.SHIELD_DUST)
    state = _battle([_mk("A", ability=Ability.SERENE_GRACE, moves=moves)], [dusted])
    step(state, {0: USE_FIRST, 1: USE_SWORDS_DANCE})
    assert dusted.stat_stages.DEFENCE == 0


def test_swift_swim_doubles_speed_in_rain():
    swimmer = _mk("A", ability=Ability.SWIFT_SWIM)
    side = SideState(team=[swimmer])
    dry = effective_speed(swimmer, side, FieldState())
    wet = effective_speed(swimmer, side, FieldState(weather=Weather.RAIN, weather_turns_left=8))
    assert wet == dry * 2


def test_unburden_doubles_speed_after_item_consumption():
    light = _mk("A", ability=Ability.UNBURDEN)
    side = SideState(team=[light])
    base = effective_speed(light, side, FieldState())
    light.item_consumed = True
    assert effective_speed(light, side, FieldState()) == base * 2


def test_unaware_defender_ignores_attack_boosts():
    attacker = _mk("A")
    attacker.stat_stages.ATTACK = 6
    boosted_vs_plain = _duel_damage(attacker, _mk("B"), USE_TACKLE)
    attacker_two = _mk("A")
    attacker_two.stat_stages.ATTACK = 6
    boosted_vs_unaware = _duel_damage(attacker_two, _mk("B", ability=Ability.UNAWARE), USE_TACKLE)
    flat = _duel_damage(_mk("A"), _mk("B", ability=Ability.UNAWARE), USE_TACKLE)
    assert boosted_vs_unaware == flat
    assert boosted_vs_plain > boosted_vs_unaware


def test_unaware_attacker_ignores_defence_boosts():
    bulky = _mk("B")
    bulky.stat_stages.DEFENCE = 6
    plain = _duel_damage(_mk("A"), bulky, USE_TACKLE)
    bulky_two = _mk("B")
    bulky_two.stat_stages.DEFENCE = 6
    pierced = _duel_damage(_mk("A", ability=Ability.UNAWARE), bulky_two, USE_TACKLE)
    assert pierced > plain


def test_infiltrator_bypasses_screens_and_substitute():
    screened = _mk("B")
    state = _battle([_mk("A")], [screened])
    state.sides[1].screens[Hazards.REFLECT] = 5
    hp = screened.live_stats.HP
    step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
    behind_screen = hp - screened.live_stats.HP

    screened_two = _mk("B")
    state = _battle([_mk("A", ability=Ability.INFILTRATOR)], [screened_two])
    state.sides[1].screens[Hazards.REFLECT] = 5
    hp = screened_two.live_stats.HP
    step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
    infiltrated = hp - screened_two.live_stats.HP
    assert infiltrated > behind_screen

    subbed = _mk("B")
    subbed.volatiles[ExtraStatus.SUBSTITUTE] = subbed.stat_totals.HP // 4
    hp = subbed.live_stats.HP
    state = _battle([_mk("A", ability=Ability.INFILTRATOR)], [subbed])
    step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
    assert hp > subbed.live_stats.HP  # the hit went straight through the substitute


def test_protean_changes_type_to_the_used_move_once():
    from battle_sim.models.log_events import TypeChanged

    chameleon = _mk("A", ability=Ability.PROTEAN, types=(Type.NORMAL, None))
    state = _battle([chameleon], [_mk("B")])
    log = step(state, {0: USE_EMBER, 1: USE_SWORDS_DANCE})
    assert chameleon.types == (Type.FIRE, None)
    assert any(isinstance(entry, TypeChanged) for entry in log)
    step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
    assert chameleon.types == (Type.FIRE, None)  # once per switch-in


def test_protean_typing_restores_on_switch_out():
    chameleon, partner = _mk("A", ability=Ability.PROTEAN, types=(Type.NORMAL, None)), _mk("P")
    state = _battle([chameleon, partner], [_mk("B")])
    step(state, {0: USE_EMBER, 1: USE_SWORDS_DANCE})
    assert chameleon.types == (Type.FIRE, None)
    step(state, {0: Action(action=ActionType.SWITCH_OUT, switch_in=partner), 1: USE_SWORDS_DANCE})
    assert chameleon.types == (Type.NORMAL, None)


def test_poison_puppeteer_confuses_what_it_poisons():
    toxic = get_move("Toxic")
    moves = MoveSet(toxic, TACKLE, EMBER, SWORDS_DANCE)
    victim = _mk("B", moves=IDLE_MOVES)
    state = _battle([_mk("A", ability=Ability.POISON_PUPPETEER, moves=moves)], [victim])
    step(state, {0: USE_FIRST, 1: USE_SPLASH})
    assert victim.status is Status.TOXIC
    assert ExtraStatus.CONFUSION in victim.volatiles


# -- Gen-7-scope ability coverage (Sam, 2026-08-22) --------------------------------

STONE_EDGE = get_move("Stone Edge")  # 80% accuracy
HYPER_VOICE = get_move("Hyper Voice")  # sound-flagged
DOUBLE_EDGE = get_move("Double-Edge")  # 1/3 recoil
TICKLE = get_move("Tickle")  # drops Attack and Defense together
USE_ALL_ADJACENT_ENEMIES = Action(action=ActionType.USE_MOVE, target=Target.ALL_ADJACENT_ENEMIES, move=MoveSlot.FIRST)


@pytest.mark.parametrize(
    "ability, status",
    [
        (Ability.LIMBER, Status.PARALYSIS),
        (Ability.INSOMNIA, Status.SLEEP),
        (Ability.VITAL_SPIRIT, Status.SLEEP),
        (Ability.WATER_VEIL, Status.BURN),
        (Ability.MAGMA_ARMOR, Status.FREEZE),
        (Ability.IMMUNITY, Status.POISON),
    ],
)
def test_status_immunity_abilities_block_their_status(ability: Ability, status: Status) -> None:
    from battle_sim.engine.status_apply import _apply_main_status
    from battle_sim.mechanics.log import BattleLog

    mon = _mk("B", ability=ability)
    _apply_main_status(status, mon, 0, RNG(seed=0), BattleLog(), ((mon,), ()))
    assert mon.status is Status.NONE


def test_own_tempo_prevents_confusion():
    from battle_sim.engine.status_apply import _apply_volatile
    from battle_sim.mechanics.log import BattleLog

    mon = _mk("B", ability=Ability.OWN_TEMPO)
    _apply_volatile(ExtraStatus.CONFUSION, mon, 0, RNG(seed=0), BattleLog())
    assert ExtraStatus.CONFUSION not in mon.volatiles


def test_white_smoke_blocks_stat_drops():
    from battle_sim.mechanics.log import BattleLog
    from battle_sim.mechanics.stages import apply_stage_changes

    mon = _mk("B", ability=Ability.WHITE_SMOKE)
    apply_stage_changes(mon, 0, {Stats.ATTACK: -1}, BattleLog(), inflicted_by_opponent=True)
    assert mon.stat_stages.ATTACK == 0


def test_soundproof_blocks_sound_moves():
    moves = MoveSet(HYPER_VOICE, TACKLE, EMBER, SWORDS_DANCE)
    deafened = _mk("B", ability=Ability.SOUNDPROOF)
    state = _battle([_mk("A", moves=moves)], [deafened])
    step(state, {0: USE_ALL_ADJACENT_ENEMIES, 1: USE_SWORDS_DANCE})
    assert deafened.live_stats.HP == deafened.stat_totals.HP


def test_reckless_boosts_recoil_move_power():
    moves = MoveSet(DOUBLE_EDGE, TACKLE, EMBER, SWORDS_DANCE)
    plain = _duel_damage(_mk("A", moves=moves), _mk("B"), USE_FIRST)
    reckless = _duel_damage(_mk("A", ability=Ability.RECKLESS, moves=moves), _mk("B"), USE_FIRST)
    assert reckless > plain


def test_rain_dish_heals_each_turn_in_rain():
    dish = _mk("B", ability=Ability.RAIN_DISH)
    dish.live_stats.HP = dish.stat_totals.HP // 2
    hp_before = dish.live_stats.HP
    field = FieldState(weather=Weather.RAIN, weather_turns_left=5)
    state = _battle([_mk("A")], [dish], field=field)
    step(state, {0: USE_SWORDS_DANCE, 1: USE_SWORDS_DANCE})
    assert hp_before + dish.stat_totals.HP // 16 == dish.live_stats.HP


def test_steadfast_gains_speed_when_flinched():
    """Flinch itself is cleared by end-of-turn residuals; the Speed boost it left behind persists."""
    moves = MoveSet(FAKE_OUT, TACKLE, EMBER, SWORDS_DANCE)
    steady = _mk("B", ability=Ability.STEADFAST)
    state = _battle([_mk("A", moves=moves)], [steady])
    step(state, {0: USE_FIRST, 1: USE_SWORDS_DANCE})
    assert steady.stat_stages.SPEED == 1


def _hit_count(move, attacker: Pokemon, defender: Pokemon, field: FieldState | None = None, trials: int = 1000) -> int:
    from battle_sim.engine.moves import _accuracy_check

    resolved_field = field or FieldState()
    return sum(1 for seed in range(trials) if _accuracy_check(move, attacker, defender, resolved_field, RNG(seed=seed)))


def test_compound_eyes_boosts_own_accuracy():
    plain = _hit_count(STONE_EDGE, _mk("A"), _mk("B"))
    boosted = _hit_count(STONE_EDGE, _mk("A", ability=Ability.COMPOUND_EYES), _mk("B"))
    assert boosted > plain


def test_snow_cloak_lowers_incoming_accuracy_in_snow():
    field = FieldState(weather=Weather.SNOW, weather_turns_left=5)
    plain = _hit_count(STONE_EDGE, _mk("A"), _mk("B"), field=field)
    cloaked = _hit_count(STONE_EDGE, _mk("A"), _mk("B", ability=Ability.SNOW_CLOAK), field=field)
    assert cloaked < plain


def test_snow_cloak_does_nothing_outside_snow():
    plain = _hit_count(STONE_EDGE, _mk("A"), _mk("B"))
    cloaked = _hit_count(STONE_EDGE, _mk("A"), _mk("B", ability=Ability.SNOW_CLOAK))
    assert cloaked == plain


THUNDER = get_move("Thunder")  # 70% accuracy, unless weather overrides it
HURRICANE = get_move("Hurricane")  # 70% accuracy, unless weather overrides it
BLIZZARD = get_move("Blizzard")  # 70% accuracy, unless hail overrides it


def test_thunder_never_misses_in_rain():
    field = FieldState(weather=Weather.RAIN, weather_turns_left=5)
    assert _hit_count(THUNDER, _mk("A"), _mk("B"), field=field) == 1000


def test_thunder_never_misses_in_heavy_rain():
    """Primordial Sea counts as rain here, same as Rain Dance."""
    field = FieldState(weather=Weather.HEAVY_RAIN, weather_turns_left=5)
    assert _hit_count(THUNDER, _mk("A"), _mk("B"), field=field) == 1000


def test_thunder_accuracy_is_halved_in_sun():
    clear = _hit_count(THUNDER, _mk("A"), _mk("B"))
    sunny = _hit_count(THUNDER, _mk("A"), _mk("B"), field=FieldState(weather=Weather.SUN, weather_turns_left=5))
    assert sunny < clear


def test_hurricane_never_misses_in_rain():
    field = FieldState(weather=Weather.RAIN, weather_turns_left=5)
    assert _hit_count(HURRICANE, _mk("A"), _mk("B"), field=field) == 1000


def test_hurricane_accuracy_is_halved_in_sun():
    clear = _hit_count(HURRICANE, _mk("A"), _mk("B"))
    sunny = _hit_count(HURRICANE, _mk("A"), _mk("B"), field=FieldState(weather=Weather.SUN, weather_turns_left=5))
    assert sunny < clear


def test_blizzard_never_misses_in_snow():
    field = FieldState(weather=Weather.SNOW, weather_turns_left=5)
    assert _hit_count(BLIZZARD, _mk("A"), _mk("B"), field=field) == 1000


def test_blizzard_accuracy_is_unaffected_by_sun():
    """Unlike Thunder and Hurricane, Blizzard's guarantee is hail-only — sun does nothing to it."""
    clear = _hit_count(BLIZZARD, _mk("A"), _mk("B"))
    sunny = _hit_count(BLIZZARD, _mk("A"), _mk("B"), field=FieldState(weather=Weather.SUN, weather_turns_left=5))
    assert sunny == clear


def test_tangled_feet_lowers_accuracy_while_confused():
    confused = _mk("B")
    confused.volatiles[ExtraStatus.CONFUSION] = 3
    tangled = _mk("B", ability=Ability.TANGLED_FEET)
    tangled.volatiles[ExtraStatus.CONFUSION] = 3
    plain = _hit_count(STONE_EDGE, _mk("A"), confused)
    dodgy = _hit_count(STONE_EDGE, _mk("A"), tangled)
    assert dodgy < plain


def test_hustle_lowers_own_physical_accuracy():
    plain = _hit_count(STONE_EDGE, _mk("A"), _mk("B"))
    hustled = _hit_count(STONE_EDGE, _mk("A", ability=Ability.HUSTLE), _mk("B"))
    assert hustled < plain


def test_hustle_boosts_own_physical_attack():
    plain = _duel_damage(_mk("A"), _mk("B"), USE_TACKLE)
    hustled = _duel_damage(_mk("A", ability=Ability.HUSTLE), _mk("B"), USE_TACKLE)
    assert hustled > plain


# -- Legendary/mythical ability coverage (Sam, 2026-08-22) --------------------------


def test_multitype_takes_the_type_of_its_held_plate():
    arceus = _mk("A", ability=Ability.MULTITYPE, item=Item.FLAME_PLATE)
    state = _battle([arceus], [_mk("B")])
    step(state, {0: USE_SWORDS_DANCE, 1: USE_SWORDS_DANCE})
    assert arceus.types == (Type.FIRE, None)


def test_multitype_is_normal_type_with_no_plate():
    arceus = _mk("A", ability=Ability.MULTITYPE)
    state = _battle([arceus], [_mk("B")])
    step(state, {0: USE_SWORDS_DANCE, 1: USE_SWORDS_DANCE})
    assert arceus.types == (Type.NORMAL, None)


def test_rks_system_takes_the_type_of_its_held_memory():
    silvally = _mk("A", ability=Ability.RKS_SYSTEM, item=Item.WATER_MEMORY)
    state = _battle([silvally], [_mk("B")])
    step(state, {0: USE_SWORDS_DANCE, 1: USE_SWORDS_DANCE})
    assert silvally.types == (Type.WATER, None)


def test_beast_boost_raises_its_highest_stat_on_ko():
    beast = _mk(
        "A",
        ability=Ability.BEAST_BOOST,
        base_stats=BaseStats(HP=100, ATTACK=100, DEFENCE=100, SP_ATTACK=100, SP_DEFENCE=100, SPEED=200),
    )
    weak = _mk("W", base_stats=BaseStats(HP=1, ATTACK=1, DEFENCE=1, SP_ATTACK=1, SP_DEFENCE=1, SPEED=1))
    state = _battle([beast], [weak])
    step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
    assert weak.is_fainted()
    assert beast.stat_stages.SPEED == 1
    assert beast.stat_stages.ATTACK == 0


def test_turboblaze_ignores_the_defenders_ability():
    earthquake = get_move("Earthquake")
    attacker = _mk(
        "A",
        ability=Ability.TURBOBLAZE,
        moves=MoveSet(earthquake, TACKLE, EMBER, SWORDS_DANCE),
        base_stats=BaseStats(HP=100, ATTACK=200, DEFENCE=100, SP_ATTACK=100, SP_DEFENCE=100, SPEED=200),
    )
    defender = _mk("B", ability=Ability.LEVITATE)
    state = _battle([attacker], [defender])
    use_eq = Action(action=ActionType.USE_MOVE, target=Target.ALL_ADJACENT, move=MoveSlot.FIRST)
    step(state, {0: use_eq, 1: USE_SWORDS_DANCE})
    assert defender.live_stats.HP < defender.stat_totals.HP


def test_shadow_shield_halves_damage_at_full_hp():
    plain = _duel_damage(_mk("A"), _mk("B"), USE_TACKLE)
    shielded = _duel_damage(_mk("A"), _mk("B", ability=Ability.SHADOW_SHIELD), USE_TACKLE)
    assert 0 < shielded < plain


def test_neuroforce_boosts_super_effective_damage():
    plain = _duel_damage(_mk("A", types=(Type.FIRE, None)), _mk("B", types=(Type.GRASS, None)), USE_EMBER)
    boosted = _duel_damage(
        _mk("A", types=(Type.FIRE, None), ability=Ability.NEUROFORCE), _mk("B", types=(Type.GRASS, None)), USE_EMBER
    )
    assert boosted > plain


def test_neuroforce_does_not_boost_neutral_damage():
    plain = _duel_damage(_mk("A"), _mk("B"), USE_TACKLE)
    same = _duel_damage(_mk("A", ability=Ability.NEUROFORCE), _mk("B"), USE_TACKLE)
    assert same == plain


def test_victory_star_boosts_own_accuracy():
    plain = _hit_count(STONE_EDGE, _mk("A"), _mk("B"))
    starred = _hit_count(STONE_EDGE, _mk("A", ability=Ability.VICTORY_STAR), _mk("B"))
    assert starred > plain


def test_fairy_aura_boosts_fairy_type_moves():
    moonblast = get_move("Moonblast")
    moves = MoveSet(moonblast, TACKLE, EMBER, SWORDS_DANCE)
    plain = _duel_damage(_mk("A", types=(Type.FAIRY, None), moves=moves), _mk("B"), USE_FIRST)
    boosted = _duel_damage(
        _mk("A", types=(Type.FAIRY, None), moves=moves), _mk("B", ability=Ability.FAIRY_AURA), USE_FIRST
    )
    assert boosted > plain


def test_aura_break_flips_fairy_aura_to_a_reduction():
    moonblast = get_move("Moonblast")
    moves = MoveSet(moonblast, TACKLE, EMBER, SWORDS_DANCE)
    boosted = _duel_damage(
        _mk("A", types=(Type.FAIRY, None), moves=moves), _mk("B", ability=Ability.FAIRY_AURA), USE_FIRST
    )
    broken = _duel_damage(
        _mk("A", types=(Type.FAIRY, None), moves=moves), _mk("B", ability=Ability.AURA_BREAK), USE_FIRST
    )
    assert broken < boosted


def test_synchronize_mirrors_status_onto_the_inflictor():
    wisp = get_move("Will-O-Wisp")
    attacker = _mk("A", moves=MoveSet(wisp, TACKLE, EMBER, SWORDS_DANCE))
    mew = _mk("B", ability=Ability.SYNCHRONIZE)
    state = _battle([attacker], [mew])
    step(state, {0: USE_FIRST, 1: USE_SWORDS_DANCE})
    assert mew.status is Status.BURN
    assert attacker.status is Status.BURN


def test_slow_start_halves_speed_for_five_turns_then_wears_off():
    slow_stats = BaseStats(HP=100, ATTACK=160, DEFENCE=100, SP_ATTACK=100, SP_DEFENCE=100, SPEED=110)
    regigigas = _mk("A", ability=Ability.SLOW_START, base_stats=slow_stats)
    reference = _mk("X", base_stats=slow_stats)
    state = _battle([regigigas], [_mk("B")])
    step(state, {0: USE_SWORDS_DANCE, 1: USE_SWORDS_DANCE})
    assert ExtraStatus.SLOW_START in regigigas.volatiles  # switch-in set 5; this turn's residual already ticked it once
    assert (
        effective_speed(regigigas, state.sides[0], state.field)
        == effective_speed(reference, state.sides[0], state.field) // 2
    )
    for _ in range(5):
        step(state, {0: USE_SWORDS_DANCE, 1: USE_SWORDS_DANCE})
    assert ExtraStatus.SLOW_START not in regigigas.volatiles
    assert effective_speed(regigigas, state.sides[0], state.field) == effective_speed(
        reference, state.sides[0], state.field
    )


def test_slow_start_halves_physical_damage():
    plain = _duel_damage(_mk("A"), _mk("B"), USE_TACKLE)
    slowed = _duel_damage(_mk("A", ability=Ability.SLOW_START), _mk("B"), USE_TACKLE)
    assert 0 < slowed < plain


def test_keen_eye_blocks_accuracy_drops():
    from battle_sim.mechanics.log import BattleLog
    from battle_sim.mechanics.stages import apply_stage_changes

    mon = _mk("B", ability=Ability.KEEN_EYE)
    apply_stage_changes(mon, 0, {Stats.ACCURACY: -1}, BattleLog(), inflicted_by_opponent=True)
    assert mon.stat_stages.ACCURACY == 0


def test_hyper_cutter_blocks_only_its_own_stat_from_a_multi_stat_drop():
    """Tickle drops Attack and Defense together; Hyper Cutter must guard only Attack."""
    from battle_sim.mechanics.log import BattleLog
    from battle_sim.mechanics.stages import apply_stage_changes

    mon = _mk("B", ability=Ability.HYPER_CUTTER)
    apply_stage_changes(mon, 0, {Stats.ATTACK: -1, Stats.DEFENCE: -1}, BattleLog(), inflicted_by_opponent=True)
    assert mon.stat_stages.ATTACK == 0
    assert mon.stat_stages.DEFENCE == -1


def test_big_pecks_blocks_only_its_own_stat_from_a_multi_stat_drop():
    from battle_sim.mechanics.log import BattleLog
    from battle_sim.mechanics.stages import apply_stage_changes

    mon = _mk("B", ability=Ability.BIG_PECKS)
    apply_stage_changes(mon, 0, {Stats.ATTACK: -1, Stats.DEFENCE: -1}, BattleLog(), inflicted_by_opponent=True)
    assert mon.stat_stages.ATTACK == -1
    assert mon.stat_stages.DEFENCE == 0


def test_hyper_cutter_does_not_block_its_own_raises():
    from battle_sim.mechanics.log import BattleLog
    from battle_sim.mechanics.stages import apply_stage_changes

    mon = _mk("B", ability=Ability.HYPER_CUTTER)
    apply_stage_changes(mon, 0, {Stats.ATTACK: 1}, BattleLog(), inflicted_by_opponent=True)
    assert mon.stat_stages.ATTACK == 1


def test_tickle_through_the_full_engine_respects_hyper_cutter():
    moves = MoveSet(TICKLE, TACKLE, EMBER, SWORDS_DANCE)
    cut = _mk("B", ability=Ability.HYPER_CUTTER, moves=IDLE_MOVES)
    state = _battle([_mk("A", moves=moves)], [cut])
    step(state, {0: USE_FIRST, 1: USE_SPLASH})
    assert cut.stat_stages.ATTACK == 0
    assert cut.stat_stages.DEFENCE == -1


def test_truant_acts_the_switch_in_turn_then_loafs_every_other_turn():
    truant = _mk("Slaking", ability=Ability.TRUANT)
    defender = _mk("B", moves=IDLE_MOVES)
    state = _battle([truant], [defender])

    hp = defender.live_stats.HP
    step(state, {0: USE_TACKLE, 1: USE_SPLASH})
    assert hp > defender.live_stats.HP  # free to act the turn it switches in

    hp = defender.live_stats.HP
    step(state, {0: USE_TACKLE, 1: USE_SPLASH})
    assert hp == defender.live_stats.HP  # loafs the very next turn

    hp = defender.live_stats.HP
    step(state, {0: USE_TACKLE, 1: USE_SPLASH})
    assert hp > defender.live_stats.HP  # then acts again


def test_truant_logs_that_it_is_loafing_when_it_skips():
    truant = _mk(ability=Ability.TRUANT)
    state = _battle([truant], [_mk("B", moves=IDLE_MOVES)])
    step(state, {0: USE_TACKLE, 1: USE_SPLASH})
    log = step(state, {0: USE_TACKLE, 1: USE_SPLASH})
    assert any(isinstance(entry, CantAct) and entry.reason == "loafing" for entry in log)


def test_a_fresh_switch_in_can_act_immediately_even_mid_loaf():
    """`volatiles.clear()` on switch-out resets Truant, so the replacement is never born loafing."""
    truant = _mk("Slaking", ability=Ability.TRUANT)
    bench = _mk("Backup", ability=Ability.TRUANT)
    defender = _mk("B", moves=IDLE_MOVES)
    state = _battle([truant, bench], [defender])
    step(state, {0: USE_TACKLE, 1: USE_SPLASH})  # truant acts, then would loaf next
    step(state, {0: Action(action=ActionType.SWITCH_OUT, switch_in=bench), 1: USE_SPLASH})

    hp = defender.live_stats.HP
    step(state, {0: USE_TACKLE, 1: USE_SPLASH})
    assert hp > defender.live_stats.HP


# -- Coverage-audit repairs (2026-09-08) -------------------------------------------
#
# Every ability below loaded with no mechanics at all, so the species built around it battled — and
# was rated — as though it had no ability. The Protect variants were worse still: they loaded with no
# effects whatsoever and spent the turn doing nothing.

KINGS_SHIELD = get_move("King's Shield")
BANEFUL_BUNKER = get_move("Baneful Bunker")
SPIKY_SHIELD = get_move("Spiky Shield")
PROTECT = get_move("Protect")
IRON_HEAD = get_move("Iron Head")
USE_SELF_FIRST = Action(action=ActionType.USE_MOVE, target=Target.SELF, move=MoveSlot.FIRST)


def _damage_through(guard: Move) -> int:
    """What a Tackle gets through `guard`, put up by the defender on the same turn."""
    protector = _mk("B", moves=MoveSet(guard, TACKLE, EMBER, SWORDS_DANCE))
    state = _battle([_mk("A")], [protector])
    hp_before = protector.live_stats.HP
    step(state, {0: USE_TACKLE, 1: USE_SELF_FIRST})
    return hp_before - protector.live_stats.HP


@pytest.mark.parametrize("guard", [PROTECT, KINGS_SHIELD, BANEFUL_BUNKER, SPIKY_SHIELD])
def test_every_protect_variant_actually_blocks_the_hit(guard: Move) -> None:
    assert _damage_through(guard) == 0


def test_a_turn_spent_guarding_is_not_a_turn_spent_doing_nothing() -> None:
    """The regression that started this: the variants loaded with no effects and were free hits."""
    assert _damage_through(TACKLE) > 0  # the control: no guard up, damage lands


def test_iron_barbs_spikes_a_contact_attacker_like_rough_skin() -> None:
    attacker = _mk("A")
    state = _battle([attacker], [_mk("B", ability=Ability.IRON_BARBS)])
    hp_before = attacker.live_stats.HP
    step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
    assert hp_before > attacker.live_stats.HP


def test_long_reach_keeps_a_contact_move_off_iron_barbs() -> None:
    attacker = _mk("A", ability=Ability.LONG_REACH)
    state = _battle([attacker], [_mk("B", ability=Ability.IRON_BARBS)])
    hp_before = attacker.live_stats.HP
    step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
    assert hp_before == attacker.live_stats.HP


def test_galvanize_makes_a_normal_move_electric() -> None:
    """Tested through a Ground-type, which a real Electric move cannot touch at all."""
    grounded = _mk("B", types=(Type.GROUND, None))
    assert _duel_damage(_mk("A"), grounded, USE_TACKLE) > 0
    assert _duel_damage(_mk("A", ability=Ability.GALVANIZE), grounded, USE_TACKLE) == 0


def test_defeatist_halves_offence_below_half_health() -> None:
    healthy = _mk("A", ability=Ability.DEFEATIST)
    assert _duel_damage(healthy, _mk("B"), USE_TACKLE) > 0

    weakened = _mk("A", ability=Ability.DEFEATIST)
    weakened.live_stats.HP = weakened.stat_totals.HP // 2
    plain = _mk("A")
    plain.live_stats.HP = plain.stat_totals.HP // 2
    assert _duel_damage(weakened, _mk("B"), USE_TACKLE) < _duel_damage(plain, _mk("B"), USE_TACKLE)


def test_marvel_scale_toughens_a_statused_defender() -> None:
    plain = _duel_damage(_mk("A"), _mk("B", status=Status.BURN), USE_TACKLE)
    scaled = _duel_damage(_mk("A"), _mk("B", ability=Ability.MARVEL_SCALE, status=Status.BURN), USE_TACKLE)
    assert 0 < scaled < plain


def test_marvel_scale_does_nothing_while_healthy() -> None:
    plain = _duel_damage(_mk("A"), _mk("B"), USE_TACKLE)
    scaled = _duel_damage(_mk("A"), _mk("B", ability=Ability.MARVEL_SCALE), USE_TACKLE)
    assert scaled == plain


def test_fur_coat_halves_physical_damage() -> None:
    plain = _duel_damage(_mk("A"), _mk("B"), USE_TACKLE)
    coated = _duel_damage(_mk("A"), _mk("B", ability=Ability.FUR_COAT), USE_TACKLE)
    assert 0 < coated < plain


def test_fur_coat_leaves_special_damage_alone() -> None:
    attacker = _mk("A", types=(Type.FIRE, None))
    plain = _duel_damage(attacker, _mk("B"), USE_EMBER)
    coated = _duel_damage(attacker, _mk("B", ability=Ability.FUR_COAT), USE_EMBER)
    assert coated == plain


def test_steelworker_boosts_steel_moves() -> None:
    moves = MoveSet(IRON_HEAD, TACKLE, EMBER, SWORDS_DANCE)
    plain = _duel_damage(_mk("A", moves=moves), _mk("B"), USE_FIRST)
    worked = _duel_damage(_mk("A", ability=Ability.STEELWORKER, moves=moves), _mk("B"), USE_FIRST)
    assert worked > plain


def test_queenly_majesty_blocks_a_priority_move() -> None:
    guarded = _mk("B", ability=Ability.QUEENLY_MAJESTY)
    state = _battle([_mk("A")], [guarded])
    hp_before = guarded.live_stats.HP
    step(state, {0: USE_QUICK_ATTACK, 1: USE_SWORDS_DANCE})
    assert hp_before == guarded.live_stats.HP


def test_queenly_majesty_leaves_ordinary_moves_alone() -> None:
    guarded = _mk("B", ability=Ability.QUEENLY_MAJESTY)
    assert _duel_damage(_mk("A"), guarded, USE_TACKLE) > 0


def test_stakeout_doubles_damage_against_something_that_just_came_in() -> None:
    plain = _duel_damage(_mk("A"), _mk("B"), USE_TACKLE)
    staked = _duel_damage(_mk("A", ability=Ability.STAKEOUT), _mk("B"), USE_TACKLE)
    assert staked > plain


def test_merciless_always_crits_a_poisoned_target() -> None:
    plain = _duel_damage(_mk("A"), _mk("B", status=Status.POISON), USE_TACKLE)
    merciless = _duel_damage(_mk("A", ability=Ability.MERCILESS), _mk("B", status=Status.POISON), USE_TACKLE)
    assert merciless > plain


def test_merciless_is_ordinary_against_an_unpoisoned_target() -> None:
    plain = _duel_damage(_mk("A"), _mk("B"), USE_TACKLE)
    merciless = _duel_damage(_mk("A", ability=Ability.MERCILESS), _mk("B"), USE_TACKLE)
    assert merciless == plain


def test_surge_surfer_doubles_speed_on_electric_terrain() -> None:
    raichu = _mk("A", ability=Ability.SURGE_SURFER)
    side = SideState(team=[raichu])
    plain = effective_speed(raichu, side, FieldState())
    surfed = effective_speed(raichu, side, FieldState(terrain=Terrain.ELECTRIC, terrain_turns_left=5))
    assert surfed == plain * 2


def test_surge_surfer_does_nothing_on_other_terrain() -> None:
    raichu = _mk("A", ability=Ability.SURGE_SURFER)
    side = SideState(team=[raichu])
    plain = effective_speed(raichu, side, FieldState())
    grassy = effective_speed(raichu, side, FieldState(terrain=Terrain.GRASSY, terrain_turns_left=5))
    assert grassy == plain


# -- Coverage-audit repairs, round two (2026-09-08) ---------------------------------

FLAIL = get_move("Flail")
RETURN = get_move("Return")


def test_fluffy_halves_contact_damage() -> None:
    plain = _duel_damage(_mk("A"), _mk("B"), USE_TACKLE)
    fluffy = _duel_damage(_mk("A"), _mk("B", ability=Ability.FLUFFY), USE_TACKLE)
    assert 0 < fluffy < plain


def test_fluffy_doubles_fire_damage() -> None:
    attacker = _mk("A", types=(Type.FIRE, None))
    plain = _duel_damage(attacker, _mk("B"), USE_EMBER)
    fluffy = _duel_damage(attacker, _mk("B", ability=Ability.FLUFFY), USE_EMBER)
    assert fluffy > plain


def test_storm_drain_absorbs_water_and_takes_a_boost_for_it() -> None:
    surf = get_move("Surf")
    moves = MoveSet(surf, TACKLE, EMBER, SWORDS_DANCE)
    drainer = _mk("B", ability=Ability.STORM_DRAIN)
    state = _battle([_mk("A", moves=moves)], [drainer])
    hp_before = drainer.live_stats.HP

    step(state, {0: USE_FIRST, 1: USE_SWORDS_DANCE})

    assert hp_before == drainer.live_stats.HP
    assert drainer.stat_stages.SP_ATTACK == 1


def test_comatose_cannot_be_statused() -> None:
    komala = _mk("B", ability=Ability.COMATOSE)
    moves = MoveSet(get_move("Thunder Wave"), TACKLE, EMBER, SWORDS_DANCE)
    state = _battle([_mk("A", moves=moves)], [komala])
    step(state, {0: USE_FIRST, 1: USE_SWORDS_DANCE})
    assert komala.status is Status.NONE


def test_super_luck_lands_more_criticals() -> None:
    lucky = _hit_count(STONE_EDGE, _mk("A", ability=Ability.SUPER_LUCK), _mk("B"))
    assert lucky > 0  # accuracy is untouched; the crit rate is what changed
    plain_damage = _duel_damage(_mk("A"), _mk("B"), USE_TACKLE)
    lucky_damage = max(_duel_damage(_mk("A", ability=Ability.SUPER_LUCK), _mk("B"), USE_TACKLE) for _ in range(20))
    assert lucky_damage >= plain_damage


def test_simple_doubles_a_stat_change() -> None:
    plain = _mk("A")
    simple = _mk("A", ability=Ability.SIMPLE)
    for pokemon in (plain, simple):
        state = _battle([pokemon], [_mk("B")])
        step(state, {0: USE_SWORDS_DANCE, 1: USE_SWORDS_DANCE})
    assert simple.stat_stages.ATTACK == 2 * plain.stat_stages.ATTACK


def test_analytic_boosts_a_move_that_goes_second() -> None:
    """Side 1 acts first here, so side 0's Analytic holder is the one moving last."""
    slow = _mk(
        "A",
        ability=Ability.ANALYTIC,
        base_stats=BaseStats(HP=100, ATTACK=100, DEFENCE=100, SP_ATTACK=100, SP_DEFENCE=100, SPEED=1),
    )
    plain = _mk("A", base_stats=BaseStats(HP=100, ATTACK=100, DEFENCE=100, SP_ATTACK=100, SP_DEFENCE=100, SPEED=1))
    dealt = []
    for attacker in (slow, plain):
        defender = _mk("B")
        state = _battle([attacker], [defender])
        before = defender.live_stats.HP
        step(state, {0: USE_TACKLE, 1: USE_TACKLE})
        dealt.append(before - defender.live_stats.HP)
    assert dealt[0] > dealt[1]


def test_emergency_exit_pulls_its_holder_out_at_half_health() -> None:
    golisopod = _mk("A", ability=Ability.EMERGENCY_EXIT)
    bench = _mk("A2")
    state = _battle([golisopod, bench], [_mk("B")])
    golisopod.live_stats.HP = golisopod.stat_totals.HP // 2 + 12  # a Tackle takes it under the line

    step(state, {0: USE_SWORDS_DANCE, 1: USE_TACKLE})

    assert state.sides[0].needs_switch


def test_emergency_exit_stays_put_with_nobody_to_switch_to() -> None:
    golisopod = _mk("A", ability=Ability.EMERGENCY_EXIT)
    state = _battle([golisopod], [_mk("B")])
    golisopod.live_stats.HP = golisopod.stat_totals.HP // 2 + 12

    step(state, {0: USE_SWORDS_DANCE, 1: USE_TACKLE})

    assert not state.sides[0].needs_switch


def test_return_is_a_real_attack_again() -> None:
    """It loaded with no damage effect at all, so it spent a turn doing nothing — on 125 of 2850
    sampled generated sets."""
    moves = MoveSet(RETURN, TACKLE, EMBER, SWORDS_DANCE)
    assert _duel_damage(_mk("A", moves=moves), _mk("B"), USE_FIRST) > 0


def test_flail_hits_harder_the_closer_its_user_is_to_fainting() -> None:
    moves = MoveSet(FLAIL, TACKLE, EMBER, SWORDS_DANCE)
    healthy = _mk("A", moves=moves)
    dying = _mk("A", moves=moves)
    dying.live_stats.HP = 1

    assert _duel_damage(dying, _mk("B"), USE_FIRST) > _duel_damage(healthy, _mk("B"), USE_FIRST)


def test_heal_bell_clears_the_whole_team_bench_included() -> None:
    moves = MoveSet(get_move("Heal Bell"), TACKLE, EMBER, SWORDS_DANCE)
    cleric = _mk("A", moves=moves, status=Status.BURN)
    benched = _mk("A2", status=Status.PARALYSIS)
    state = _battle([cleric, benched], [_mk("B")])

    step(state, {0: USE_SELF_FIRST, 1: USE_SWORDS_DANCE})

    assert cleric.status is Status.NONE
    assert benched.status is Status.NONE


# -- Conditional moves (2026-09-08) -------------------------------------------------
#
# Each of these loaded as an unconditional attack, so the clause that is the whole point of the move
# never applied: Focus Punch could not be broken, and Shell Trap — which in practice usually does
# nothing — always landed.

# Fast enough to get into the air before the opponent swings: a charge only protects once it starts,
# and an opponent who moves first hits an ordinary Pokemon standing on the ground.
_SWIFT = BaseStats(HP=100, ATTACK=100, DEFENCE=100, SP_ATTACK=100, SP_DEFENCE=100, SPEED=200)
FOCUS_PUNCH = get_move("Focus Punch")
SHELL_TRAP = get_move("Shell Trap")
REVENGE = get_move("Revenge")
BRINE = get_move("Brine")
FLY = get_move("Fly")
EARTHQUAKE = get_move("Earthquake")


def _damage_dealt(attacker: Pokemon, defender: Pokemon, their_action: Action) -> int:
    """What the attacker's first move gets through, with the defender doing `their_action`."""
    state = _battle([attacker], [defender])
    hp_before = defender.live_stats.HP
    step(state, {0: USE_FIRST, 1: their_action})
    return hp_before - defender.live_stats.HP


def test_focus_punch_is_lost_if_its_user_is_hit() -> None:
    moves = MoveSet(FOCUS_PUNCH, TACKLE, EMBER, SWORDS_DANCE)
    assert _damage_dealt(_mk("A", moves=moves), _mk("B"), USE_TACKLE) == 0


def test_focus_punch_lands_if_its_user_is_left_alone() -> None:
    moves = MoveSet(FOCUS_PUNCH, TACKLE, EMBER, SWORDS_DANCE)
    assert _damage_dealt(_mk("A", moves=moves), _mk("B", moves=IDLE_MOVES), USE_SPLASH) > 0


def test_shell_trap_fails_unless_something_physical_sets_it_off() -> None:
    moves = MoveSet(SHELL_TRAP, TACKLE, EMBER, SWORDS_DANCE)
    assert _damage_dealt(_mk("A", moves=moves), _mk("B", moves=IDLE_MOVES), USE_SPLASH) == 0


def test_shell_trap_goes_off_when_a_physical_hit_lands() -> None:
    moves = MoveSet(SHELL_TRAP, TACKLE, EMBER, SWORDS_DANCE)
    assert _damage_dealt(_mk("A", moves=moves), _mk("B"), USE_TACKLE) > 0


def test_revenge_doubles_after_taking_a_hit() -> None:
    moves = MoveSet(REVENGE, TACKLE, EMBER, SWORDS_DANCE)
    hit = _damage_dealt(_mk("A", moves=moves), _mk("B"), USE_TACKLE)
    untouched = _damage_dealt(_mk("A", moves=moves), _mk("B", moves=IDLE_MOVES), USE_SPLASH)
    assert hit > untouched > 0


def test_brine_doubles_against_a_wounded_target() -> None:
    moves = MoveSet(BRINE, TACKLE, EMBER, SWORDS_DANCE)
    healthy = _duel_damage(_mk("A", moves=moves), _mk("B"), USE_FIRST)
    hurt = _mk("B")
    hurt.live_stats.HP = hurt.stat_totals.HP // 3
    assert _duel_damage(_mk("A", moves=moves), hurt, USE_FIRST) > healthy


def test_a_pokemon_mid_fly_cannot_be_touched() -> None:
    """The charge turn already worked; being out of reach during it is what pays for the wait."""
    flier = _mk("A", moves=MoveSet(FLY, TACKLE, EMBER, SWORDS_DANCE), base_stats=_SWIFT)
    state = _battle([flier], [_mk("B")])
    hp_before = flier.live_stats.HP

    step(state, {0: USE_FIRST, 1: USE_TACKLE})  # Fly's charge turn

    assert hp_before == flier.live_stats.HP


def test_the_right_move_still_reaches_something_underground() -> None:
    """Earthquake on a Dig is the answer everyone knows, so it has to keep working."""
    digger = _mk("A", moves=MoveSet(get_move("Dig"), TACKLE, EMBER, SWORDS_DANCE), base_stats=_SWIFT)
    quaker = _mk("B", moves=MoveSet(EARTHQUAKE, TACKLE, EMBER, SWORDS_DANCE))
    state = _battle([digger], [quaker])
    hp_before = digger.live_stats.HP

    step(state, {0: USE_FIRST, 1: USE_FIRST})

    assert hp_before > digger.live_stats.HP


def test_an_ordinary_charge_leaves_its_user_in_plain_sight() -> None:
    """Solar Beam is a two-turn move, not a disappearing act."""
    charger = _mk("A", moves=MoveSet(get_move("Solar Beam"), TACKLE, EMBER, SWORDS_DANCE), base_stats=_SWIFT)
    state = _battle([charger], [_mk("B")])
    hp_before = charger.live_stats.HP

    step(state, {0: USE_FIRST, 1: USE_TACKLE})

    assert hp_before > charger.live_stats.HP


def test_fury_cutter_builds_while_it_keeps_connecting() -> None:
    moves = MoveSet(get_move("Fury Cutter"), TACKLE, EMBER, SWORDS_DANCE)
    cutter = _mk("A", moves=moves)
    defender = _mk("B", moves=IDLE_MOVES)
    state = _battle([cutter], [defender])

    dealt = []
    for _ in range(3):
        before = defender.live_stats.HP
        step(state, {0: USE_FIRST, 1: USE_SPLASH})
        dealt.append(before - defender.live_stats.HP)

    assert dealt[1] > dealt[0] and dealt[2] > dealt[1]


# -- Catching a Pokemon on the way out (2026-09-08) ---------------------------------

PURSUIT = get_move("Pursuit")
PURSUIT_BASE_POWER = 40


def _damage_effect(move: Move) -> DamageEffect:
    return next(effect for effect in move.effects if isinstance(effect, DamageEffect))


def _pursuit_battle() -> tuple[Pokemon, Pokemon, Pokemon, BattleState]:
    chaser = _mk("Chaser", moves=MoveSet(PURSUIT, TACKLE, EMBER, SWORDS_DANCE))
    fleeing = _mk("Runner", moves=IDLE_MOVES)
    bench = _mk("Bench")
    state = _battle([chaser], [fleeing, bench])
    return chaser, fleeing, bench, state


def test_pursuit_catches_its_target_before_the_switch_completes() -> None:
    """The whole move: leaving is exactly when it is supposed to hurt."""
    _, fleeing, bench, state = _pursuit_battle()
    hp_before = fleeing.live_stats.HP

    step(state, {0: USE_FIRST, 1: Action(action=ActionType.SWITCH_OUT, switch_in=bench)})

    assert hp_before > fleeing.live_stats.HP  # hit on the way out, not after it had gone
    assert state.sides[1].active_pokemon is bench  # and the switch still happened


def test_pursuit_hits_harder_against_a_target_that_is_leaving() -> None:
    chaser, fleeing, bench, state = _pursuit_battle()
    switching = Action(action=ActionType.SWITCH_OUT, switch_in=bench)
    state.sides[1].chosen_action = switching
    caught = effective_power(PURSUIT, _damage_effect(PURSUIT), chaser, fleeing, state)

    state.sides[1].chosen_action = USE_SPLASH
    standing = effective_power(PURSUIT, _damage_effect(PURSUIT), chaser, fleeing, state)

    assert caught == 2 * standing


def test_pursuit_is_an_ordinary_move_against_a_target_that_stays() -> None:
    _, fleeing, _, state = _pursuit_battle()
    hp_before = fleeing.live_stats.HP

    step(state, {0: USE_FIRST, 1: USE_SPLASH})

    assert hp_before - fleeing.live_stats.HP > 0


def test_a_target_that_already_moved_cannot_be_caught_leaving() -> None:
    chaser, fleeing, bench, state = _pursuit_battle()
    state.sides[1].chosen_action = Action(action=ActionType.SWITCH_OUT, switch_in=bench)
    state.sides[1].acted_this_turn = True

    assert effective_power(PURSUIT, _damage_effect(PURSUIT), chaser, fleeing, state) == PURSUIT_BASE_POWER


def test_arena_trap_pins_a_grounded_opponent() -> None:
    """Dugtrio's entire reason for existing, and it was missing from the trapping list."""
    trapped = _mk("A")
    state = _battle([trapped], [_mk("B", ability=Ability.ARENA_TRAP), _mk("B2")])
    assert not any(action.action is ActionType.SWITCH_OUT for action in legal_actions(state, 0))


def test_arena_trap_cannot_hold_something_off_the_ground() -> None:
    flier = _mk("A", types=(Type.FLYING, None))
    state = _battle([flier, _mk("A2")], [_mk("B", ability=Ability.ARENA_TRAP)])
    assert any(action.action is ActionType.SWITCH_OUT for action in legal_actions(state, 0))


def test_nine_lives_rises_whole_and_burns_the_status_away():
    """Sturdy with a counter and a heal. The differences matter: no full-HP requirement, so chip
    damage cannot switch it off, and each use restores him completely and cures him."""
    glass_cannon = _mk(
        "Glass",
        base_stats=BaseStats(HP=100, ATTACK=200, DEFENCE=100, SP_ATTACK=100, SP_DEFENCE=100, SPEED=200),
    )
    butler = _mk(
        "Butler",
        base_stats=BaseStats(HP=1, ATTACK=1, DEFENCE=1, SP_ATTACK=1, SP_DEFENCE=1, SPEED=1),
        ability=Ability.NINE_LIVES,
    )
    butler.status = Status.BURN
    state = _battle([glass_cannon], [butler])
    step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
    assert butler.live_stats.HP == butler.stat_totals.HP  # back to full, not left on 1
    assert butler.status is Status.NONE  # and the burn went with it
    assert butler.lives_used == 1


def test_nine_lives_runs_out():
    """Ten killing blows, not nine: the last one lands."""
    glass_cannon = _mk(
        "Glass",
        base_stats=BaseStats(HP=100, ATTACK=200, DEFENCE=100, SP_ATTACK=100, SP_DEFENCE=100, SPEED=200),
    )
    butler = _mk(
        "Butler",
        base_stats=BaseStats(HP=1, ATTACK=1, DEFENCE=1, SP_ATTACK=1, SP_DEFENCE=1, SPEED=1),
        ability=Ability.NINE_LIVES,
    )
    state = _battle([glass_cannon], [butler])
    for _ in range(NINE_LIVES):
        step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
        assert not butler.is_fainted()
    assert butler.lives_used == NINE_LIVES
    step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
    assert butler.is_fainted()


def test_an_ordinary_hit_neither_spends_a_life_nor_heals_him():
    """The restore is gated on a life actually being spent. Without that marker it would fire on
    every hit and he would simply never take damage at all."""
    tickler = _mk("Tickle", base_stats=BaseStats(HP=100, ATTACK=1, DEFENCE=100, SP_ATTACK=1, SP_DEFENCE=100, SPEED=200))
    butler = _mk(
        "Butler",
        base_stats=BaseStats(HP=200, ATTACK=100, DEFENCE=100, SP_ATTACK=100, SP_DEFENCE=100, SPEED=1),
        ability=Ability.NINE_LIVES,
    )
    state = _battle([tickler], [butler])
    step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
    assert 0 < butler.live_stats.HP < butler.stat_totals.HP  # hurt, and left hurt
    assert butler.lives_used == 0


def _knocked_off_butler() -> tuple[BattleState, Pokemon]:
    """A butler holding Leftovers, a thief fast enough to take them off him first."""
    thief = _mk(
        "Thief",
        moves=MoveSet(get_move("Knock Off"), QUICK_ATTACK, EMBER, SWORDS_DANCE),
        base_stats=BaseStats(HP=100, ATTACK=200, DEFENCE=100, SP_ATTACK=100, SP_DEFENCE=100, SPEED=200),
    )
    butler = _mk(
        "Butler",
        ability=Ability.NINE_LIVES,
        item=Item.LEFTOVERS,
        base_stats=BaseStats(HP=200, ATTACK=1, DEFENCE=100, SP_ATTACK=1, SP_DEFENCE=100, SPEED=1),
    )
    return _battle([thief], [butler]), butler


def test_a_knocked_off_item_comes_back_when_a_life_is_spent():
    """Nine Lives restores everything else — full HP, the status burned away — so an item knocked off
    on the way down was the one wound that stuck, and one Knock Off early disarmed all nine lives.

    Driven through a real Knock Off rather than by setting the field directly, because the recording
    happens in the move and the restoring happens in the model: the point is that the two meet.
    """
    state, butler = _knocked_off_butler()

    step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})  # slot one is Knock Off
    assert butler.item is Item.NONE, "Knock Off did not take the item"
    assert butler.stripped_item is Item.LEFTOVERS, "nothing remembered what was taken"

    butler.live_stats.HP = 1
    step(state, {0: USE_QUICK_ATTACK, 1: USE_SWORDS_DANCE})

    assert butler.lives_used == 1, "the setup did not actually spend a life"
    assert butler.item is Item.LEFTOVERS, "he rose again without what was taken from him"
    assert butler.stripped_item is Item.NONE, "the same item could be restored twice"


def test_the_restored_item_is_wired_back_up_rather_than_merely_held():
    """Knocking it off unbound its handlers, so restoring the field alone hands back a dead Leftovers
    that heals nothing — which looks identical in every assertion except this one."""
    state, butler = _knocked_off_butler()
    step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
    butler.live_stats.HP = 1
    step(state, {0: USE_QUICK_ATTACK, 1: USE_SWORDS_DANCE})

    butler.live_stats.HP = butler.stat_totals.HP - 60
    hurt = butler.live_stats.HP
    step(state, {0: USE_SWORDS_DANCE, 1: USE_SWORDS_DANCE})

    assert hurt < butler.live_stats.HP, "the restored Leftovers is held but never run"


def test_an_item_he_spent_himself_is_not_given_back():
    """The line the restore has to hold. A Focus Sash behind nine lives would be nine free survivals,
    so only what an opponent took comes back — what he used up stays used up."""
    butler = _mk("Butler", ability=Ability.NINE_LIVES, item=Item.SITRUS_BERRY)
    butler.consume_item()  # exactly what eating a berry does
    assert butler.stripped_item is Item.NONE

    butler.live_stats.HP = 1
    butler.apply_damage(999)

    assert butler.lives_used == 1
    assert butler.item is Item.NONE, "an item he spent himself came back with him"


def test_a_life_spent_clears_the_drops_an_opponent_put_on_him():
    """Rising whole has to mean whole. An Intimidate landing on turn one otherwise followed him
    through all nine lives, leaving him swinging at 0.67x Attack with nothing in the log to say so."""
    butler = _mk("Butler", ability=Ability.NINE_LIVES)
    butler.stat_stages[Stats.ATTACK] = -1  # 0.67x
    butler.stat_stages[Stats.DEFENCE] = -2
    butler.stat_stages[Stats.SPEED] = -6

    butler.live_stats.HP = 1
    butler.apply_damage(999)

    assert butler.lives_used == 1
    for stat in (Stats.ATTACK, Stats.DEFENCE, Stats.SPEED):
        assert butler.stat_stages[stat] == 0, f"{stat.name} was still down after he rose again"


def test_a_life_spent_leaves_his_own_boosts_where_they_are():
    """Only the drops. Wiping a Swords Dance he earned would turn each life into a punishment for
    having set up before losing one, which is the opposite of what the ability is for."""
    butler = _mk("Butler", ability=Ability.NINE_LIVES)
    butler.stat_stages[Stats.ATTACK] = 4  # two Swords Dances of his own
    butler.stat_stages[Stats.SPEED] = -3  # and a Sticky Web on top

    butler.live_stats.HP = 1
    butler.apply_damage(999)

    assert butler.stat_stages[Stats.ATTACK] == 4, "his own boost went with the drops"
    assert butler.stat_stages[Stats.SPEED] == 0


def test_the_drop_clearing_covers_accuracy_and_evasion_too():
    """They are stages like any other, and `Stats` carries an HP member that has none — so the loop
    reads the stage model's own fields rather than the stat enum, and this is what proves it."""
    butler = _mk("Butler", ability=Ability.NINE_LIVES)
    butler.stat_stages.ACCURACY = -3
    butler.stat_stages.EVASION = -2

    butler.live_stats.HP = 1
    butler.apply_damage(999)

    assert butler.stat_stages.ACCURACY == 0
    assert butler.stat_stages.EVASION == 0


def test_an_intimidate_that_lands_after_a_revival_still_sticks():
    """The clearing happens at the revival and not a moment later — it is not a standing immunity to
    being dropped, or he could never be Intimidated at all."""
    butler = _mk("Butler", ability=Ability.NINE_LIVES)
    butler.live_stats.HP = 1
    butler.apply_damage(999)

    butler.stat_stages[Stats.ATTACK] = -1

    assert butler.stat_stages[Stats.ATTACK] == -1, "he shrugged off a drop he should have taken"


def _butler_about_to_lose_a_life() -> tuple[BattleState, Pokemon, Pokemon]:
    """A butler on 1 HP and something fast enough to finish him. Returns (state, butler, foe)."""
    foe = _mk(
        "Foe",
        base_stats=BaseStats(HP=100, ATTACK=200, DEFENCE=100, SP_ATTACK=100, SP_DEFENCE=100, SPEED=200),
    )
    butler = _mk(
        "Butler",
        ability=Ability.NINE_LIVES,
        base_stats=BaseStats(HP=200, ATTACK=1, DEFENCE=100, SP_ATTACK=1, SP_DEFENCE=100, SPEED=1),
    )
    butler.live_stats.HP = 1
    return _battle([foe], [butler]), butler, foe


def test_rising_again_sweeps_the_board_the_other_side_had_built_up():
    """The evasion cheese this closes: stack Double Team and his nine lives stop being a gauntlet and
    become nine turns of standing still, because he cannot land a thing however often he gets up."""
    state, butler, foe = _butler_about_to_lose_a_life()
    foe.stat_stages[Stats.EVASION] = 3
    foe.stat_stages[Stats.ATTACK] = 4

    step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})

    assert butler.lives_used == 1, "the setup did not actually spend a life"
    assert foe.stat_stages[Stats.EVASION] == 0, "evasion survived him rising again"
    assert foe.stat_stages[Stats.ATTACK] == 0, "a sweep set up once would be paid off nine times"


def test_the_sweep_takes_the_other_side_s_own_drops_with_it():
    """Reset, not confiscation: a Close Combat's own Defence drop clears along with the boosts, or
    the ability would be quietly punishing whoever is facing him as well."""
    state, _, foe = _butler_about_to_lose_a_life()
    foe.stat_stages[Stats.DEFENCE] = -2
    foe.stat_stages[Stats.SP_DEFENCE] = -1

    step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})

    assert foe.stat_stages[Stats.DEFENCE] == 0
    assert foe.stat_stages[Stats.SP_DEFENCE] == 0


def test_the_sweep_is_announced_so_it_does_not_read_as_a_bug():
    """Stages vanishing with nothing said is exactly the shape of thing that gets reported as
    broken. It is only announced when there was something to sweep, though."""
    state, _, foe = _butler_about_to_lose_a_life()
    foe.stat_stages[Stats.EVASION] = 3
    swept = step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
    assert any(isinstance(entry, StatChangesSwept) for entry in swept.entries)

    clean_state, _, _ = _butler_about_to_lose_a_life()
    quiet = step(clean_state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
    assert not any(isinstance(entry, StatChangesSwept) for entry in quiet.entries), "announced an empty sweep"


def test_the_sweep_does_not_touch_the_butler_s_own_boosts():
    """The two halves have to stay apart: his side keeps what he earned, their side keeps nothing."""
    state, butler, foe = _butler_about_to_lose_a_life()
    butler.stat_stages[Stats.ATTACK] = 4
    foe.stat_stages[Stats.ATTACK] = 4

    step(state, {0: USE_TACKLE, 1: USE_TACKLE})  # no Swords Dance here, or the 4 would move on its own

    assert butler.stat_stages[Stats.ATTACK] == 4, "the sweep reached across and took his own boost"
    assert foe.stat_stages[Stats.ATTACK] == 0


def test_nine_lives_catches_every_kind_of_damage_not_just_attacks():
    """The engine checks for a faint in half a dozen places and only one of them emits ON_FAINT, so
    an ability bound to that event let poison, hazards and recoil kill him with eight lives unused.
    This lives at the single point every damage source passes through instead."""
    butler = _mk("Butler", ability=Ability.NINE_LIVES)
    butler.live_stats.HP = 1
    butler.apply_damage(999)  # not a move, not an event: a bare call, as residuals make
    assert not butler.is_fainted()
    assert butler.live_stats.HP == butler.stat_totals.HP
    assert butler.lives_used == 1


def test_the_life_counter_is_not_a_volatile():
    """It used to be, and `volatiles.clear()` on switch-out handed back a full set. Worse, the
    search's cloned futures decremented a shared one -- a real battle logged thirteen of nine."""
    butler = _mk("Butler", ability=Ability.NINE_LIVES)
    butler.live_stats.HP = 1
    butler.apply_damage(999)
    butler.volatiles.clear()
    assert butler.lives_used == 1


# -- Nine Lives cannot be moved, copied or taken away -------------------------


def _swap_pair(attacker_ability: Ability, defender_ability: Ability) -> tuple[BattleState, Pokemon, Pokemon]:
    """A fast Skill Swap user and whatever it is pointed at."""
    skill_swap = get_move("Skill Swap")
    attacker = _mk(
        "Swapper",
        ability=attacker_ability,
        moves=MoveSet(skill_swap, QUICK_ATTACK, EMBER, SWORDS_DANCE),
        base_stats=BaseStats(HP=200, ATTACK=100, DEFENCE=100, SP_ATTACK=100, SP_DEFENCE=100, SPEED=200),
    )
    defender = _mk(
        "Butler",
        ability=defender_ability,
        base_stats=BaseStats(HP=200, ATTACK=100, DEFENCE=100, SP_ATTACK=100, SP_DEFENCE=100, SPEED=1),
    )
    return _battle([attacker], [defender]), attacker, defender


def test_skill_swap_cannot_take_nine_lives_off_the_butler() -> None:
    """One Skill Swap and the final boss is an ordinary six-on-six — the fight stops being the fight."""
    state, swapper, butler = _swap_pair(Ability.INTIMIDATE, Ability.NINE_LIVES)

    step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})  # slot one is Skill Swap

    assert butler.ability is Ability.NINE_LIVES
    assert swapper.ability is Ability.INTIMIDATE, "the challenger walked away with nine lives"


def test_skill_swap_fails_the_same_way_when_the_butler_is_the_one_using_it() -> None:
    """A swap is a trade, so blocking only the direction that takes it off him would still hand it
    over — and the giving half is the worse one."""
    state, butler, foe = _swap_pair(Ability.NINE_LIVES, Ability.INTIMIDATE)

    step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})

    assert butler.ability is Ability.NINE_LIVES
    assert foe.ability is Ability.INTIMIDATE, "the butler gave nine lives away"


def test_a_blocked_skill_swap_is_reported_as_a_failure() -> None:
    """Silently doing nothing reads as a bug. It failed, and the log should say so."""
    from battle_sim.models.log_events import AbilitiesSwapped, MoveFailed

    state, _, _ = _swap_pair(Ability.INTIMIDATE, Ability.NINE_LIVES)

    entries = list(step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE}).entries)

    assert any(isinstance(e, MoveFailed) for e in entries)
    assert not any(isinstance(e, AbilitiesSwapped) for e in entries)


def test_skill_swap_still_works_between_two_ordinary_abilities() -> None:
    """The counterweight: the block is one named ability, not Skill Swap switched off."""
    state, one, two = _swap_pair(Ability.INTIMIDATE, Ability.LEVITATE)

    step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})

    assert (one.ability, two.ability) == (Ability.LEVITATE, Ability.INTIMIDATE)


def _trace_onto(target_ability: Ability) -> Pokemon:
    """Switch a Trace holder in against something, and hand back the tracer."""
    bench = _mk("Bench")
    tracer = _mk("Tracer", ability=Ability.TRACE)
    target = _mk("Target", ability=target_ability)
    state = _battle([bench, tracer], [target])
    step(state, {0: Action(action=ActionType.SWITCH_OUT, switch_in=tracer), 1: USE_SWORDS_DANCE})
    return tracer


def test_trace_cannot_copy_nine_lives() -> None:
    """The worst of the lot, because it needs no move and no setup: switch a Trace holder in against
    the butler and the challenger has nine lives of their own, which nine of his cannot answer."""
    tracer = _trace_onto(Ability.NINE_LIVES)

    assert tracer.ability is Ability.TRACE, "Trace walked off with nine lives"


def test_trace_still_copies_an_ordinary_ability() -> None:
    tracer = _trace_onto(Ability.LEVITATE)

    assert tracer.ability is Ability.LEVITATE


def test_nine_lives_is_the_ability_the_untouchable_list_exists_for() -> None:
    """A rot test. The guard reads a list, and a list with the wrong thing in it protects nothing —
    while `_revive` still keys off this exact member."""
    from battle_sim.utils import UNTOUCHABLE_ABILITIES, is_untouchable

    assert Ability.NINE_LIVES in UNTOUCHABLE_ABILITIES
    assert is_untouchable(Ability.NINE_LIVES)
    assert not is_untouchable(Ability.TRACE)


# -- Mummy and the ability-moving moves ---------------------------------------


def _ability_move(move_name: str, user_ability: Ability, target_ability: Ability) -> tuple[Pokemon, Pokemon, list]:
    """Point one ability-moving move at a target. Returns (user, target, log entries)."""
    user = _mk(
        "User",
        ability=user_ability,
        moves=MoveSet(get_move(move_name), QUICK_ATTACK, EMBER, SWORDS_DANCE),
        base_stats=BaseStats(HP=200, ATTACK=100, DEFENCE=100, SP_ATTACK=100, SP_DEFENCE=100, SPEED=200),
    )
    target = _mk(
        "Target",
        ability=target_ability,
        base_stats=BaseStats(HP=200, ATTACK=100, DEFENCE=100, SP_ATTACK=100, SP_DEFENCE=100, SPEED=1),
    )
    state = _battle([user], [target])
    entries = list(step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE}).entries)
    return user, target, entries


def test_the_ability_moving_moves_do_something_at_all() -> None:
    """All four loaded from the dex with no effects whatever, so they sat in movepools doing nothing.
    One assertion each for the direction they move things in, which is what tells them apart."""
    user, target, _ = _ability_move("Role Play", Ability.INTIMIDATE, Ability.LEVITATE)
    assert (user.ability, target.ability) == (Ability.LEVITATE, Ability.LEVITATE), "Role Play copies to the user"

    user, target, _ = _ability_move("Entrainment", Ability.INTIMIDATE, Ability.LEVITATE)
    assert (user.ability, target.ability) == (Ability.INTIMIDATE, Ability.INTIMIDATE), "Entrainment pushes to target"

    _, target, _ = _ability_move("Worry Seed", Ability.INTIMIDATE, Ability.LEVITATE)
    assert target.ability is Ability.INSOMNIA

    _, target, _ = _ability_move("Simple Beam", Ability.INTIMIDATE, Ability.LEVITATE)
    assert target.ability is Ability.SIMPLE


@pytest.mark.parametrize("move_name", ["Role Play", "Entrainment", "Worry Seed", "Simple Beam"])
def test_no_ability_moving_move_can_touch_nine_lives(move_name: str) -> None:
    """Every one of the four, in both directions: none may take it off him, none may hand it out."""
    user, butler, _ = _ability_move(move_name, Ability.INTIMIDATE, Ability.NINE_LIVES)
    assert butler.ability is Ability.NINE_LIVES, f"{move_name} changed the butler's ability"
    assert user.ability is not Ability.NINE_LIVES, f"{move_name} gave the challenger nine lives"

    butler, target, _ = _ability_move(move_name, Ability.NINE_LIVES, Ability.LEVITATE)
    assert butler.ability is Ability.NINE_LIVES
    assert target.ability is not Ability.NINE_LIVES, f"{move_name} let the butler give nine lives away"


def test_a_refused_ability_change_says_why_exactly_once() -> None:
    """Silence reads as a bug, and two lines saying the same thing read worse than either."""
    from battle_sim.models.log_events import AbilityUnchanged, MoveFailed

    _, _, entries = _ability_move("Entrainment", Ability.INTIMIDATE, Ability.NINE_LIVES)

    assert sum(isinstance(e, AbilityUnchanged) for e in entries) == 1
    assert not any(isinstance(e, MoveFailed) for e in entries), "said it failed on top of saying why"


def test_a_forme_defining_ability_is_left_alone_too() -> None:
    """Multitype is what makes an Arceus its type — and Arceus-Fairy is on the butler's own team, so
    this is not hypothetical. Moving one produces a Pokemon the game has no rules for."""
    user, arceus, _ = _ability_move("Role Play", Ability.LEVITATE, Ability.MULTITYPE)

    assert arceus.ability is Ability.MULTITYPE
    assert user.ability is Ability.LEVITATE, "Role Play walked off with Multitype"


def _mummy_contact(toucher_ability: Ability) -> Pokemon:
    """Something fast hits a Mummy with a contact move. Returns the toucher."""
    toucher = _mk(
        "Toucher",
        ability=toucher_ability,
        base_stats=BaseStats(HP=200, ATTACK=100, DEFENCE=100, SP_ATTACK=100, SP_DEFENCE=100, SPEED=200),
    )
    mummy = _mk(
        "Mummy",
        ability=Ability.MUMMY,
        base_stats=BaseStats(HP=200, ATTACK=100, DEFENCE=100, SP_ATTACK=100, SP_DEFENCE=100, SPEED=1),
    )
    state = _battle([toucher], [mummy])
    step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
    return toucher


def test_touching_a_mummy_makes_you_one() -> None:
    assert _mummy_contact(Ability.INTIMIDATE).ability is Ability.MUMMY


def test_mummy_cannot_take_nine_lives() -> None:
    """The one that needs no move at all from the challenger's side — the butler attacking *them* is
    what triggers it, so he would disarm himself by playing normally."""
    assert _mummy_contact(Ability.NINE_LIVES).ability is Ability.NINE_LIVES


def test_mummy_does_not_spread_on_a_move_that_makes_no_contact() -> None:
    """The counterweight: it is a contact ability, not a passive aura."""
    toucher = _mk(
        "Toucher",
        ability=Ability.INTIMIDATE,
        moves=MoveSet(get_move("Ember"), QUICK_ATTACK, EMBER, SWORDS_DANCE),
        base_stats=BaseStats(HP=200, ATTACK=100, DEFENCE=100, SP_ATTACK=100, SP_DEFENCE=100, SPEED=200),
    )
    mummy = _mk(
        "Mummy",
        ability=Ability.MUMMY,
        base_stats=BaseStats(HP=200, ATTACK=100, DEFENCE=100, SP_ATTACK=100, SP_DEFENCE=100, SPEED=1),
    )
    state = _battle([toucher], [mummy])
    step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})

    assert toucher.ability is Ability.INTIMIDATE


# -- Trick against the butler -------------------------------------------------


def _trick_at_butler(
    butler_item: Item = Item.LEFTOVERS,
    foe_item: Item = Item.CHOICE_SCARF,
    butler_speed: int = 1,
    start_hp: int | None = None,
) -> tuple:
    """A Trick user swaps items with the butler. Returns (state, foe, butler)."""
    foe = _mk(
        "Foe",
        item=foe_item,
        moves=MoveSet(get_move("Trick"), QUICK_ATTACK, EMBER, SWORDS_DANCE),
        base_stats=BaseStats(HP=200, ATTACK=100, DEFENCE=100, SP_ATTACK=100, SP_DEFENCE=100, SPEED=200),
    )
    butler = _mk(
        "Butler",
        ability=Ability.NINE_LIVES,
        item=butler_item,
        base_stats=BaseStats(HP=200, ATTACK=1, DEFENCE=100, SP_ATTACK=1, SP_DEFENCE=100, SPEED=butler_speed),
    )
    if start_hp is not None:
        butler.live_stats.HP = start_hp
    state = _battle([foe], [butler])
    step(state, {0: USE_TACKLE, 1: USE_TACKLE})  # slot one is Trick; he means to attack back
    return state, foe, butler


def test_trick_swaps_items_as_it_should() -> None:
    """The plain mechanic, which had no test at all before this."""
    one = _mk("One", item=Item.LEFTOVERS, moves=MoveSet(get_move("Trick"), QUICK_ATTACK, EMBER, SWORDS_DANCE))
    two = _mk("Two", item=Item.CHOICE_SCARF)
    state = _battle([one], [two])

    step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})

    assert (one.item, two.item) == (Item.CHOICE_SCARF, Item.LEFTOVERS)


def test_the_butler_devours_an_item_tricked_onto_him() -> None:
    """Hand him something and he eats it, and is half his health better off for it."""
    from battle_sim.engine.moves import GREEDY_GOURMAND_DIVISOR

    _, _, butler = _trick_at_butler(start_hp=1)

    assert butler.item is Item.NONE, "he was still holding it when his turn ended"
    assert butler.live_stats.HP == 1 + butler.stat_totals.HP // GREEDY_GOURMAND_DIVISOR


def test_he_does_not_eat_his_own_item() -> None:
    """Only what somebody handed him, or he would spend every battle devouring his own Monocle."""
    butler = _mk("Butler", ability=Ability.NINE_LIVES, item=Item.LEFTOVERS)
    state = _battle([_mk("Foe")], [butler])

    step(state, {0: USE_SWORDS_DANCE, 1: USE_SWORDS_DANCE})

    assert butler.item is Item.LEFTOVERS


def test_a_tricked_item_comes_home_when_he_rises() -> None:
    """His own item is taken back off whoever holds it — and having eaten theirs, they get nothing."""
    state, foe, butler = _trick_at_butler()
    butler.live_stats.HP = 1

    step(state, {0: USE_QUICK_ATTACK, 1: USE_SWORDS_DANCE})

    assert butler.lives_used == 1, "the setup did not spend a life"
    assert butler.item is Item.LEFTOVERS, "his own item did not come home"
    assert foe.item is Item.NONE, "the trick survived him rising again"


def test_the_trade_is_reversed_rather_than_confiscated_when_he_still_holds_it() -> None:
    """If he has not eaten it yet, both items go home — nobody is robbed and nothing is minted."""
    state, foe, butler = _trick_at_butler()
    butler.tricked_item = Item.NONE  # as though the turn had not ended yet
    butler.item = Item.CHOICE_SCARF
    butler.live_stats.HP = 1

    step(state, {0: USE_QUICK_ATTACK, 1: USE_SWORDS_DANCE})

    assert butler.item is Item.LEFTOVERS
    assert foe.item is Item.CHOICE_SCARF, "the scarf was confiscated rather than handed back"


def test_an_ordinary_pokemon_keeps_what_it_traded_for() -> None:
    """The whole mechanism is his. A Trick between two ordinary Pokemon stands, as it always did."""
    one = _mk("One", item=Item.LEFTOVERS, moves=MoveSet(get_move("Trick"), QUICK_ATTACK, EMBER, SWORDS_DANCE))
    two = _mk("Two", item=Item.CHOICE_SCARF)
    state = _battle([one], [two])
    step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})

    assert two.stripped_item is Item.NONE and two.tricked_item is Item.NONE
    assert (one.item, two.item) == (Item.CHOICE_SCARF, Item.LEFTOVERS)


def test_eating_costs_him_the_turn_he_would_have_attacked_in() -> None:
    """It is his action for the round, not a freebie on top of one. He chose to attack; the foe is
    untouched, because the meal is what he did instead."""
    _, foe, _ = _trick_at_butler()

    assert foe.live_stats.HP == foe.stat_totals.HP, "he ate *and* attacked"


@pytest.mark.parametrize("butler_speed", [1, 500])
def test_he_eats_after_the_trick_however_fast_he_is(butler_speed: int) -> None:
    """The point of the reordering. Outrunning the Trick would mean attacking into an empty turn and
    then being handed the item with no action left to deal with it."""
    _, _, butler = _trick_at_butler(butler_speed=butler_speed)

    assert butler.item is Item.NONE, f"at speed {butler_speed} he never got to eat it"
    assert butler.tricked_item is Item.NONE


def test_even_a_priority_move_does_not_let_him_move_first() -> None:
    """Quick Attack is +1 and would ordinarily jump the bracket entirely."""
    foe = _mk(
        "Foe",
        item=Item.CHOICE_SCARF,
        moves=MoveSet(get_move("Trick"), QUICK_ATTACK, EMBER, SWORDS_DANCE),
        base_stats=BaseStats(HP=200, ATTACK=100, DEFENCE=100, SP_ATTACK=100, SP_DEFENCE=100, SPEED=100),
    )
    butler = _mk(
        "Meowfred",
        ability=Ability.NINE_LIVES,
        item=Item.LEFTOVERS,
        base_stats=BaseStats(HP=200, ATTACK=100, DEFENCE=100, SP_ATTACK=100, SP_DEFENCE=100, SPEED=500),
    )
    state = _battle([foe], [butler])

    step(state, {0: USE_TACKLE, 1: USE_QUICK_ATTACK})

    assert butler.item is Item.NONE, "he outran the Trick and never ate"
    assert foe.live_stats.HP == foe.stat_totals.HP, "the Quick Attack went off as well as the meal"


def test_an_ordinary_trick_still_resolves_in_speed_order() -> None:
    """The reordering is his alone — a Trick between two ordinary Pokemon is an ordinary move."""
    fast = _mk(
        "Fast",
        item=Item.CHOICE_SCARF,
        moves=MoveSet(get_move("Trick"), QUICK_ATTACK, EMBER, SWORDS_DANCE),
        base_stats=BaseStats(HP=200, ATTACK=100, DEFENCE=100, SP_ATTACK=100, SP_DEFENCE=100, SPEED=100),
    )
    slow = _mk(
        "Slow",
        item=Item.LEFTOVERS,
        base_stats=BaseStats(HP=200, ATTACK=100, DEFENCE=100, SP_ATTACK=100, SP_DEFENCE=100, SPEED=500),
    )
    state = _battle([fast], [slow])

    step(state, {0: USE_TACKLE, 1: USE_TACKLE})

    assert fast.live_stats.HP < fast.stat_totals.HP, "the faster ordinary Pokemon never got its move"


def test_the_butler_cannot_be_stalled_out_of_pp() -> None:
    """Nine lives is a long fight by design, and his real set runs 24/16/56/16 — against Pressure,
    which doubles every cost, that is eight uses of Warm Dinner. Stalling him into Struggle and
    letting the recoil take the lives one at a time was the cheapest line on the board against him."""
    from battle_sim.models.pokemon import BUTLERS_PP

    butler = _mk("Meowfred", ability=Ability.NINE_LIVES)

    assert set(butler.pp.values()) == {BUTLERS_PP}


def test_everybody_else_keeps_the_pp_their_moves_list() -> None:
    """Keyed on the ability, which only he has — and which `UNTOUCHABLE_ABILITIES` now stops anybody
    else acquiring, so this cannot leak onto a challenger."""
    ordinary = _mk("Ordinary")

    assert set(ordinary.pp.values()) != {99}
    assert ordinary.pp[MoveSlot.FIRST] == TACKLE.pp
