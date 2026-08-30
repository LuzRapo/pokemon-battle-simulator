import pytest

from battle_sim.database.loader import get_move
from battle_sim.engine import step
from battle_sim.maths.rng import RNG
from battle_sim.mechanics.battle import BattleState, FieldState, SideState
from battle_sim.mechanics.priority import effective_speed
from battle_sim.models.actions import Action, ActionType
from battle_sim.models.log_events import CantAct, DoesNotAffect
from battle_sim.models.moves import MoveSet, MoveSlot
from battle_sim.models.pokemon import Pokemon
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


def test_speed_boost_raises_speed_at_end_of_turn():
    a = _mk("A", ability=Ability.SPEED_BOOST)
    b = _mk("B")
    state = _battle([a], [b])
    step(state, {0: USE_SWORDS_DANCE, 1: USE_SWORDS_DANCE})
    assert a.stat_stages.SPEED == 1


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


def test_guts_boosts_attack_when_statused_and_ignores_burn_halving():
    burned_guts = _mk("G", ability=Ability.GUTS, status=Status.BURN, types=(Type.FIRE, None))
    burned_normal = _mk("N", status=Status.BURN, types=(Type.FIRE, None))
    defender_a = _mk("B")
    defender_b = _mk("B")
    state_a = _battle([burned_guts], [defender_a])
    state_b = _battle([burned_normal], [defender_b])
    hp_a, hp_b = defender_a.live_stats.HP, defender_b.live_stats.HP
    step(state_a, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
    step(state_b, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
    assert (hp_a - defender_a.live_stats.HP) > (hp_b - defender_b.live_stats.HP)


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
    _apply_main_status(Status.BURN, mon, 0, RNG(seed=0), BattleLog())
    assert mon.status is Status.NONE


def test_purifying_salt_blocks_all_statuses():
    from battle_sim.engine.status_apply import _apply_main_status
    from battle_sim.mechanics.log import BattleLog

    mon = _mk("B", ability=Ability.PURIFYING_SALT)
    for status in (Status.BURN, Status.SLEEP, Status.TOXIC, Status.PARALYSIS):
        _apply_main_status(status, mon, 0, RNG(seed=0), BattleLog())
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
    _apply_main_status(Status.BURN, exchanger, 1, RNG(seed=0), BattleLog())
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
    _apply_main_status(status, mon, 0, RNG(seed=0), BattleLog())
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
