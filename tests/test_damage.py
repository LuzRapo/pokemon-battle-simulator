import pytest

from battle_sim.database.loader import get_move
from battle_sim.maths.damage import calculate_damage
from battle_sim.maths.rng import RNG
from battle_sim.mechanics.battle import FieldState, SideState
from battle_sim.models.moves import MoveSet
from battle_sim.models.pokemon import Pokemon
from battle_sim.models.stats import BaseStats, EVs, IVs
from battle_sim.models.type_matchups import TypePair
from battle_sim.utils import Ability, ExtraStatus, Hazards, Item, Nature, Status, Terrain, Type, Weather

TACKLE = get_move("Tackle")
EMBER = get_move("Ember")
QUICK_ATTACK = get_move("Quick Attack")
SWORDS_DANCE = get_move("Swords Dance")


def make_pokemon(
    types: TypePair = (Type.FIRE, None),
    level: int = 100,
    base_stats: BaseStats | None = None,
    nature: Nature = Nature.HARDY,
    status: Status = Status.NONE,
    atk_stage: int = 0,
    spa_stage: int = 0,
    def_stage: int = 0,
    spd_stage: int = 0,
) -> Pokemon:
    if base_stats is None:
        base_stats = BaseStats(HP=100, ATTACK=100, DEFENCE=100, SP_ATTACK=100, SP_DEFENCE=100, SPEED=100)
    pokemon = Pokemon(
        name="TestMon",
        nickname="TestMon",
        level=level,
        base_stats=base_stats,
        effort_values=EVs(),
        individual_values=IVs(),
        types=types,
        moves=MoveSet(TACKLE, EMBER, QUICK_ATTACK, SWORDS_DANCE),
        nature=nature,
        status=status,
    )
    pokemon.stat_stages.ATTACK = atk_stage
    pokemon.stat_stages.SP_ATTACK = spa_stage
    pokemon.stat_stages.DEFENCE = def_stage
    pokemon.stat_stages.SP_DEFENCE = spd_stage
    return pokemon


def _damage(
    attacker: Pokemon,
    defender: Pokemon,
    move=TACKLE,
    field: FieldState | None = None,
    screens: dict[Hazards, int] | None = None,
    is_crit: bool = False,
    random_roll: int = 100,
    target_count: int = 1,
) -> int:
    side = SideState(team=[defender], screens=screens or {})
    return calculate_damage(
        attacker,
        defender,
        move,
        field or FieldState(),
        side,
        rng=RNG(seed=0),
        is_crit=is_crit,
        random_roll=random_roll,
        target_count=target_count,
    )


def test_base_formula_no_modifiers():
    attacker = make_pokemon(types=(Type.FIRE, None))
    defender = make_pokemon(types=(Type.NORMAL, None))
    assert _damage(attacker, defender) == 35


def test_minimum_random_roll():
    attacker = make_pokemon(types=(Type.FIRE, None))
    defender = make_pokemon(types=(Type.NORMAL, None))
    assert _damage(attacker, defender, random_roll=85) == 29


def test_default_random_roll_is_within_canonical_range():
    attacker = make_pokemon(types=(Type.FIRE, None))
    defender = make_pokemon(types=(Type.NORMAL, None))
    rolls = {
        calculate_damage(
            attacker,
            defender,
            TACKLE,
            FieldState(),
            SideState(team=[defender]),
            rng=RNG(seed=seed),
            is_crit=False,
        )
        for seed in range(500)
    }
    assert min(rolls) == 29
    assert max(rolls) == 35


def test_stab_applies_for_same_type():
    attacker = make_pokemon(types=(Type.NORMAL, None))
    defender = make_pokemon(types=(Type.FIRE, None))
    assert _damage(attacker, defender) == 52


def test_no_stab_for_different_type():
    attacker = make_pokemon(types=(Type.FIRE, None))
    defender = make_pokemon(types=(Type.NORMAL, None))
    assert _damage(attacker, defender) == 35


def test_critical_hit_multiplies_damage():
    attacker = make_pokemon(types=(Type.FIRE, None))
    defender = make_pokemon(types=(Type.NORMAL, None))
    assert _damage(attacker, defender, is_crit=True) == 52


def test_critical_ignores_negative_attacker_stages():
    attacker = make_pokemon(types=(Type.FIRE, None), atk_stage=-3)
    defender = make_pokemon(types=(Type.NORMAL, None))
    assert _damage(attacker, defender, is_crit=False) < _damage(
        make_pokemon(types=(Type.FIRE, None)), defender, is_crit=False
    )
    assert _damage(attacker, defender, is_crit=True) == 52


def test_critical_ignores_positive_defender_stages():
    attacker = make_pokemon(types=(Type.FIRE, None))
    defender = make_pokemon(types=(Type.NORMAL, None), def_stage=3)
    assert _damage(attacker, defender, is_crit=False) < _damage(
        attacker, make_pokemon(types=(Type.NORMAL, None)), is_crit=False
    )
    assert _damage(attacker, defender, is_crit=True) == 52


def test_burn_halves_physical_damage():
    attacker = make_pokemon(types=(Type.FIRE, None), status=Status.BURN)
    defender = make_pokemon(types=(Type.NORMAL, None))
    assert _damage(attacker, defender) == 17


def test_burn_does_not_halve_special_damage():
    attacker = make_pokemon(types=(Type.WATER, None), status=Status.BURN)
    defender = make_pokemon(types=(Type.NORMAL, None))
    burned = _damage(attacker, defender, move=EMBER)
    healthy_attacker = make_pokemon(types=(Type.WATER, None))
    healthy = _damage(healthy_attacker, defender, move=EMBER)
    assert burned == healthy


def test_defender_burn_does_not_affect_damage():
    attacker = make_pokemon(types=(Type.FIRE, None))
    defender = make_pokemon(types=(Type.NORMAL, None), status=Status.BURN)
    assert _damage(attacker, defender) == 35


def test_type_immune_returns_zero():
    attacker = make_pokemon(types=(Type.FIRE, None))
    defender = make_pokemon(types=(Type.GHOST, None))
    assert _damage(attacker, defender) == 0


def test_super_effective_doubles_damage():
    attacker = make_pokemon(types=(Type.FIRE, None))
    defender = make_pokemon(types=(Type.NORMAL, None), atk_stage=0)
    fire_def = make_pokemon(types=(Type.GRASS, None))
    assert _damage(attacker, fire_def, move=EMBER) == _damage(attacker, defender, move=EMBER) * 2


def test_resisted_halves_damage():
    attacker = make_pokemon(types=(Type.FIRE, None))
    fire_def = make_pokemon(types=(Type.FIRE, None))
    normal_def = make_pokemon(types=(Type.NORMAL, None))
    resisted = _damage(attacker, fire_def, move=EMBER)
    neutral = _damage(attacker, normal_def, move=EMBER)
    assert resisted * 2 == neutral


def test_status_move_returns_zero():
    attacker = make_pokemon(types=(Type.NORMAL, None))
    defender = make_pokemon(types=(Type.NORMAL, None))
    assert _damage(attacker, defender, move=SWORDS_DANCE) == 0


def test_rain_boosts_water_moves():
    surf = get_move("Surf")
    attacker = make_pokemon(types=(Type.FIRE, None))
    defender = make_pokemon(types=(Type.NORMAL, None))
    rain_field = FieldState(weather=Weather.RAIN, weather_turns_left=5)
    boosted = _damage(attacker, defender, move=surf, field=rain_field)
    base = _damage(attacker, defender, move=surf)
    assert boosted > base
    assert abs(boosted * 2 - base * 3) <= 3


def test_rain_halves_fire_moves():
    attacker = make_pokemon(types=(Type.FIRE, None))
    defender = make_pokemon(types=(Type.NORMAL, None))
    rain_field = FieldState(weather=Weather.RAIN, weather_turns_left=5)
    in_rain = _damage(attacker, defender, move=EMBER, field=rain_field)
    clear = _damage(attacker, defender, move=EMBER)
    assert in_rain < clear
    assert abs(in_rain * 2 - clear) <= 2


def test_sun_boosts_fire_moves():
    attacker = make_pokemon(types=(Type.FIRE, None))
    defender = make_pokemon(types=(Type.NORMAL, None))
    sun_field = FieldState(weather=Weather.SUN, weather_turns_left=5)
    sunny = _damage(attacker, defender, move=EMBER, field=sun_field)
    clear = _damage(attacker, defender, move=EMBER)
    assert sunny > clear


def test_sun_halves_water_moves():
    surf = get_move("Surf")
    attacker = make_pokemon(types=(Type.FIRE, None))
    defender = make_pokemon(types=(Type.NORMAL, None))
    sun_field = FieldState(weather=Weather.SUN, weather_turns_left=5)
    in_sun = _damage(attacker, defender, move=surf, field=sun_field)
    clear = _damage(attacker, defender, move=surf)
    assert in_sun < clear


def test_electric_terrain_boosts_electric_for_grounded_attacker():
    thunderbolt = get_move("Thunderbolt")
    attacker = make_pokemon(types=(Type.NORMAL, None))
    defender = make_pokemon(types=(Type.NORMAL, None))
    terrain_field = FieldState(terrain=Terrain.ELECTRIC, terrain_turns_left=5)
    on_terrain = _damage(attacker, defender, move=thunderbolt, field=terrain_field)
    off_terrain = _damage(attacker, defender, move=thunderbolt)
    assert on_terrain > off_terrain


def test_electric_terrain_does_not_boost_flying_attacker():
    thunderbolt = get_move("Thunderbolt")
    flying_attacker = make_pokemon(types=(Type.NORMAL, Type.FLYING))
    defender = make_pokemon(types=(Type.NORMAL, None))
    terrain_field = FieldState(terrain=Terrain.ELECTRIC, terrain_turns_left=5)
    on_terrain = _damage(flying_attacker, defender, move=thunderbolt, field=terrain_field)
    off_terrain = _damage(flying_attacker, defender, move=thunderbolt)
    assert on_terrain == off_terrain


def test_misty_terrain_halves_dragon_against_grounded_defender():
    dragon_pulse = get_move("Dragon Pulse")
    attacker = make_pokemon(types=(Type.NORMAL, None))
    grounded_defender = make_pokemon(types=(Type.NORMAL, None))
    field = FieldState(terrain=Terrain.MISTY, terrain_turns_left=5)
    weakened = _damage(attacker, grounded_defender, move=dragon_pulse, field=field)
    full = _damage(attacker, grounded_defender, move=dragon_pulse)
    assert weakened < full


def test_misty_terrain_does_not_affect_flying_defender():
    dragon_pulse = get_move("Dragon Pulse")
    attacker = make_pokemon(types=(Type.NORMAL, None))
    flying_defender = make_pokemon(types=(Type.NORMAL, Type.FLYING))
    field = FieldState(terrain=Terrain.MISTY, terrain_turns_left=5)
    assert _damage(attacker, flying_defender, move=dragon_pulse, field=field) == _damage(
        attacker, flying_defender, move=dragon_pulse
    )


def test_electric_terrain_does_not_boost_levitate_attacker():
    thunderbolt = get_move("Thunderbolt")
    levitating_attacker = make_pokemon(types=(Type.NORMAL, None))
    levitating_attacker.ability = Ability.LEVITATE
    defender = make_pokemon(types=(Type.NORMAL, None))
    terrain_field = FieldState(terrain=Terrain.ELECTRIC, terrain_turns_left=5)
    on_terrain = _damage(levitating_attacker, defender, move=thunderbolt, field=terrain_field)
    off_terrain = _damage(levitating_attacker, defender, move=thunderbolt)
    assert on_terrain == off_terrain


def test_misty_terrain_does_not_protect_air_balloon_defender():
    dragon_pulse = get_move("Dragon Pulse")
    attacker = make_pokemon(types=(Type.NORMAL, None))
    balloon_defender = make_pokemon(types=(Type.NORMAL, None))
    balloon_defender.item = Item.AIR_BALLOON
    field = FieldState(terrain=Terrain.MISTY, terrain_turns_left=5)
    assert _damage(attacker, balloon_defender, move=dragon_pulse, field=field) == _damage(
        attacker, balloon_defender, move=dragon_pulse
    )


def test_reflect_halves_physical_damage():
    attacker = make_pokemon(types=(Type.FIRE, None))
    defender = make_pokemon(types=(Type.NORMAL, None))
    with_reflect = _damage(attacker, defender, screens={Hazards.REFLECT: 5})
    without = _damage(attacker, defender)
    assert with_reflect == 17
    assert without == 35


def test_reflect_does_not_halve_special_damage():
    attacker = make_pokemon(types=(Type.WATER, None))
    defender = make_pokemon(types=(Type.NORMAL, None))
    with_reflect = _damage(attacker, defender, move=EMBER, screens={Hazards.REFLECT: 5})
    without = _damage(attacker, defender, move=EMBER)
    assert with_reflect == without


def test_light_screen_halves_special_damage():
    attacker = make_pokemon(types=(Type.WATER, None))
    defender = make_pokemon(types=(Type.NORMAL, None))
    with_ls = _damage(attacker, defender, move=EMBER, screens={Hazards.LIGHT_SCREEN: 5})
    without = _damage(attacker, defender, move=EMBER)
    assert with_ls < without


def test_aurora_veil_halves_both_categories():
    attacker_phys = make_pokemon(types=(Type.FIRE, None))
    attacker_spec = make_pokemon(types=(Type.WATER, None))
    defender = make_pokemon(types=(Type.NORMAL, None))
    av = {Hazards.AURORA_VEIL: 5}
    assert _damage(attacker_phys, defender, screens=av) < _damage(attacker_phys, defender)
    assert _damage(attacker_spec, defender, move=EMBER, screens=av) < _damage(attacker_spec, defender, move=EMBER)


def test_crit_ignores_screens():
    attacker = make_pokemon(types=(Type.FIRE, None))
    defender = make_pokemon(types=(Type.NORMAL, None))
    with_reflect_crit = _damage(attacker, defender, screens={Hazards.REFLECT: 5}, is_crit=True)
    crit_no_reflect = _damage(attacker, defender, is_crit=True)
    assert with_reflect_crit == crit_no_reflect


def test_spread_target_reduction():
    attacker = make_pokemon(types=(Type.FIRE, None))
    defender = make_pokemon(types=(Type.NORMAL, None))
    single = _damage(attacker, defender)
    spread = _damage(attacker, defender, target_count=2)
    assert spread < single


def test_damage_is_at_least_one_when_attacked():
    weak_attacker = make_pokemon(
        types=(Type.FIRE, None),
        base_stats=BaseStats(HP=1, ATTACK=4, DEFENCE=1, SP_ATTACK=1, SP_DEFENCE=1, SPEED=1),
        level=1,
    )
    bulky_defender = make_pokemon(
        types=(Type.FIRE, None),
        base_stats=BaseStats(HP=255, ATTACK=1, DEFENCE=255, SP_ATTACK=1, SP_DEFENCE=255, SPEED=1),
        level=100,
    )
    assert _damage(weak_attacker, bulky_defender, random_roll=85) >= 1


@pytest.mark.parametrize(
    "crit_stage, expected_rate, tolerance",
    [
        (0, 1 / 24, 0.02),
        (1, 1 / 8, 0.03),
        (2, 1 / 2, 0.04),
        (3, 1.0, 0.0),
    ],
)
def test_crit_rate_per_stage(crit_stage, expected_rate, tolerance):
    from battle_sim.maths.damage import _crit_chance

    chance = _crit_chance(crit_stage)
    assert abs(chance - expected_rate) <= tolerance or chance == expected_rate


def test_crit_rolled_via_rng_distribution():
    attacker = make_pokemon(types=(Type.FIRE, None))
    defender = make_pokemon(types=(Type.NORMAL, None))
    crits = 0
    trials = 4000
    for seed in range(trials):
        side = SideState(team=[defender])
        normal = calculate_damage(
            attacker,
            defender,
            TACKLE,
            FieldState(),
            side,
            rng=RNG(seed=seed),
            is_crit=False,
            random_roll=100,
        )
        rolled = calculate_damage(
            attacker,
            defender,
            TACKLE,
            FieldState(),
            side,
            rng=RNG(seed=seed),
            random_roll=100,
        )
        if rolled > normal:
            crits += 1
    rate = crits / trials
    assert 1 / 30 < rate < 1 / 18, f"Crit rate {rate} not near 1/24 over {trials} trials"


def test_identified_ghost_takes_normal_damage():
    attacker = make_pokemon(types=(Type.FIRE, None))
    plain_ghost = make_pokemon(types=(Type.GHOST, None))
    identified_ghost = make_pokemon(types=(Type.GHOST, None))
    identified_ghost.volatiles[ExtraStatus.IDENTIFIED] = 1
    assert _damage(attacker, plain_ghost) == 0
    assert _damage(attacker, identified_ghost) > 0


def test_identified_ghost_takes_fighting_damage():
    fighting_attacker = make_pokemon(types=(Type.FIGHTING, None))
    identified_ghost = make_pokemon(types=(Type.GHOST, None))
    identified_ghost.volatiles[ExtraStatus.IDENTIFIED] = 1
    fighting_move = get_move("Karate Chop")
    assert _damage(fighting_attacker, identified_ghost, move=fighting_move) > 0


def test_miracle_eye_psychic_hits_dark():
    attacker = make_pokemon(types=(Type.NORMAL, None))
    plain_dark = make_pokemon(types=(Type.DARK, None))
    miracle_eyed_dark = make_pokemon(types=(Type.DARK, None))
    miracle_eyed_dark.volatiles[ExtraStatus.MIRACLE_EYE] = 1
    psychic_move = get_move("Psychic")
    assert _damage(attacker, plain_dark, move=psychic_move) == 0
    assert _damage(attacker, miracle_eyed_dark, move=psychic_move) > 0


def test_identified_does_not_grant_normal_to_dark_immunity_bypass():
    attacker = make_pokemon(types=(Type.PSYCHIC, None))
    dark_def = make_pokemon(types=(Type.DARK, None))
    dark_def.volatiles[ExtraStatus.IDENTIFIED] = 1
    psychic_move = get_move("Psychic")
    assert _damage(attacker, dark_def, move=psychic_move) == 0
