from battle_sim.database.loader import get_move
from battle_sim.engine import step
from battle_sim.maths.rng import RNG
from battle_sim.mechanics.battle import BattleState, FieldState, SideState
from battle_sim.mechanics.priority import effective_speed
from battle_sim.models.actions import Action, ActionType
from battle_sim.models.log_events import MoveUsed
from battle_sim.models.moves import MoveSet, MoveSlot
from battle_sim.models.pokemon import Pokemon
from battle_sim.models.stats import BaseStats, EVs, IVs
from battle_sim.utils import Ability, Hazards, Item, Nature, Stats, Status, Target, Type

TACKLE = get_move("Tackle")
EMBER = get_move("Ember")
QUICK_ATTACK = get_move("Quick Attack")
SWORDS_DANCE = get_move("Swords Dance")
USE_TACKLE = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.FIRST)
USE_EMBER = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.THIRD)
USE_SWORDS_DANCE = Action(action=ActionType.USE_MOVE, target=Target.SELF, move=MoveSlot.FOURTH)


def _mk(
    nickname: str = "X",
    types: tuple[Type, Type | None] = (Type.NORMAL, None),
    item: Item = Item.NONE,
    base_stats: BaseStats | None = None,
    moves: MoveSet | None = None,
    status: Status = Status.NONE,
    ability: Ability = Ability.NONE,
) -> Pokemon:
    if base_stats is None:
        base_stats = BaseStats(HP=100, ATTACK=100, DEFENCE=100, SP_ATTACK=100, SP_DEFENCE=100, SPEED=100)
    if moves is None:
        moves = MoveSet(TACKLE, QUICK_ATTACK, EMBER, SWORDS_DANCE)
    return Pokemon(
        name="TestMon",
        nickname=nickname,
        level=50,
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


def _battle(side0: list[Pokemon], side1: list[Pokemon], seed: int = 0) -> BattleState:
    return BattleState(
        sides=(SideState(team=side0), SideState(team=side1)),
        rng=RNG(seed=seed),
    )


def _damage_tackle(attacker_item: Item) -> int:
    a = _mk("A", types=(Type.FIRE, None), item=attacker_item)
    b = _mk("B", types=(Type.NORMAL, None))
    state = _battle([a], [b])
    hp_before = b.live_stats.HP
    step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
    return hp_before - b.live_stats.HP


def test_life_orb_boosts_damage_by_about_30_percent():
    base = _damage_tackle(Item.NONE)
    boosted = _damage_tackle(Item.LIFE_ORB)
    assert boosted > base
    assert abs(boosted * 10 - base * 13) <= 5


def test_life_orb_inflicts_recoil_to_attacker():
    a = _mk("A", types=(Type.FIRE, None), item=Item.LIFE_ORB)
    b = _mk("B")
    state = _battle([a], [b])
    max_hp = a.stat_totals.HP
    hp_before = a.live_stats.HP
    step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
    assert hp_before - a.live_stats.HP == max(1, max_hp // 10)


def test_choice_band_boosts_physical_damage_by_about_50_percent():
    base = _damage_tackle(Item.NONE)
    boosted = _damage_tackle(Item.CHOICE_BAND)
    assert boosted > base
    assert abs(boosted * 2 - base * 3) <= 3


def test_choice_band_locks_holder_into_first_used_move():
    a = _mk("A", item=Item.CHOICE_BAND)
    b = _mk("B")
    state = _battle([a], [b])
    step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
    assert a.choice_locked_move is MoveSlot.FIRST


def test_choice_band_coerces_subsequent_action_to_locked_move():
    a = _mk("A", item=Item.CHOICE_BAND, types=(Type.FIRE, None))
    b = _mk("B")
    state = _battle([a], [b])
    step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
    assert a.choice_locked_move is MoveSlot.FIRST
    hp_before = b.live_stats.HP
    log = step(state, {0: USE_EMBER, 1: USE_SWORDS_DANCE})
    used = [entry.move for entry in log if isinstance(entry, MoveUsed)]
    assert "Tackle" in used
    assert "Ember" not in used
    assert hp_before - b.live_stats.HP > 0


def test_choice_lock_clears_on_switch_out():
    a = _mk("A", item=Item.CHOICE_BAND)
    a2 = _mk("A2")
    state = _battle([a, a2], [_mk("B")])
    step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
    assert a.choice_locked_move is MoveSlot.FIRST
    switch = Action(action=ActionType.SWITCH_OUT, switch_in=a2)
    step(state, {0: switch, 1: USE_SWORDS_DANCE})
    assert a.choice_locked_move is None


def test_choice_specs_locks_into_first_special_move():
    a = _mk("A", item=Item.CHOICE_SPECS, types=(Type.WATER, None))
    b = _mk("B")
    state = _battle([a], [b])
    step(state, {0: USE_EMBER, 1: USE_SWORDS_DANCE})
    assert a.choice_locked_move is MoveSlot.THIRD


def test_choice_scarf_locks_after_first_move():
    a = _mk("A", item=Item.CHOICE_SCARF)
    b = _mk("B")
    state = _battle([a], [b])
    step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
    assert a.choice_locked_move is MoveSlot.FIRST


def test_choice_band_does_not_boost_special_damage():
    a_band = _mk("A", types=(Type.WATER, None), item=Item.CHOICE_BAND)
    a_none = _mk("A", types=(Type.WATER, None), item=Item.NONE)
    b1 = _mk("B")
    b2 = _mk("B")
    state_band = _battle([a_band], [b1])
    state_none = _battle([a_none], [b2])
    hp1, hp2 = b1.live_stats.HP, b2.live_stats.HP
    step(state_band, {0: USE_EMBER, 1: USE_SWORDS_DANCE})
    step(state_none, {0: USE_EMBER, 1: USE_SWORDS_DANCE})
    assert (hp1 - b1.live_stats.HP) == (hp2 - b2.live_stats.HP)


def test_choice_specs_boosts_special_damage():
    a_specs = _mk("A", types=(Type.WATER, None), item=Item.CHOICE_SPECS)
    a_none = _mk("A", types=(Type.WATER, None), item=Item.NONE)
    b1 = _mk("B")
    b2 = _mk("B")
    state_specs = _battle([a_specs], [b1])
    state_none = _battle([a_none], [b2])
    hp1, hp2 = b1.live_stats.HP, b2.live_stats.HP
    step(state_specs, {0: USE_EMBER, 1: USE_SWORDS_DANCE})
    step(state_none, {0: USE_EMBER, 1: USE_SWORDS_DANCE})
    specs_dmg = hp1 - b1.live_stats.HP
    none_dmg = hp2 - b2.live_stats.HP
    assert specs_dmg > none_dmg
    assert abs(specs_dmg * 2 - none_dmg * 3) <= 3


def test_choice_scarf_boosts_effective_speed_by_50_percent():
    a = _mk("A", item=Item.CHOICE_SCARF)
    base_speed = a.effective_stat(Stats.SPEED)
    side = SideState(team=[a])
    boosted = effective_speed(a, side, FieldState())
    assert boosted == base_speed * 3 // 2


def test_focus_sash_survives_one_hit_when_at_full_hp():
    glass_cannon = _mk(
        "Glass",
        base_stats=BaseStats(HP=100, ATTACK=200, DEFENCE=100, SP_ATTACK=100, SP_DEFENCE=100, SPEED=200),
    )
    fragile = _mk(
        "Fragile",
        base_stats=BaseStats(HP=1, ATTACK=1, DEFENCE=1, SP_ATTACK=1, SP_DEFENCE=1, SPEED=1),
        item=Item.FOCUS_SASH,
    )
    state = _battle([glass_cannon], [fragile])
    step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
    assert fragile.live_stats.HP == 1
    assert fragile.item is Item.NONE


def test_focus_sash_does_not_trigger_when_not_at_full_hp():
    glass_cannon = _mk(
        "Glass",
        base_stats=BaseStats(HP=100, ATTACK=200, DEFENCE=100, SP_ATTACK=100, SP_DEFENCE=100, SPEED=200),
    )
    fragile = _mk(
        "Fragile",
        base_stats=BaseStats(HP=10, ATTACK=1, DEFENCE=1, SP_ATTACK=1, SP_DEFENCE=1, SPEED=1),
        item=Item.FOCUS_SASH,
    )
    max_hp = fragile.stat_totals.HP
    fragile.live_stats.HP = max_hp - 1
    state = _battle([glass_cannon], [fragile])
    step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
    assert fragile.is_fainted()
    assert fragile.item is Item.FOCUS_SASH


def test_heavy_duty_boots_skips_entry_hazards():
    a, a2 = _mk("A"), _mk("A2", item=Item.HEAVY_DUTY_BOOTS)
    state = _battle([a, a2], [_mk("B")])
    state.sides[0].hazards[Hazards.STEALTH_ROCK] = 1
    state.sides[0].hazards[Hazards.SPIKES] = 3
    switch = Action(action=ActionType.SWITCH_OUT, switch_in=a2)
    step(state, {0: switch, 1: USE_SWORDS_DANCE})
    assert a2.live_stats.HP == a2.stat_totals.HP


def test_air_balloon_grants_ground_immunity():
    earthquake = get_move("Earthquake")
    attacker = _mk(
        "A",
        moves=MoveSet(earthquake, TACKLE, EMBER, SWORDS_DANCE),
        base_stats=BaseStats(HP=100, ATTACK=200, DEFENCE=100, SP_ATTACK=100, SP_DEFENCE=100, SPEED=200),
    )
    defender = _mk("B", item=Item.AIR_BALLOON)
    state = _battle([attacker], [defender])
    use_eq = Action(action=ActionType.USE_MOVE, target=Target.ALL_ADJACENT, move=MoveSlot.FIRST)
    step(state, {0: use_eq, 1: USE_SWORDS_DANCE})
    assert defender.live_stats.HP == defender.stat_totals.HP
    assert defender.item is Item.AIR_BALLOON


def test_air_balloon_pops_on_non_ground_hit():
    attacker = _mk(
        "A",
        base_stats=BaseStats(HP=100, ATTACK=200, DEFENCE=100, SP_ATTACK=100, SP_DEFENCE=100, SPEED=200),
    )
    defender = _mk("B", item=Item.AIR_BALLOON)
    state = _battle([attacker], [defender])
    step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
    assert defender.item is Item.NONE


def test_rocky_helmet_chips_contact_attacker():
    attacker = _mk("A")
    defender = _mk("B", item=Item.ROCKY_HELMET)
    state = _battle([attacker], [defender])
    hp_before = attacker.live_stats.HP
    step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
    expected = max(1, attacker.stat_totals.HP // 6)
    assert hp_before - attacker.live_stats.HP == expected


def test_rocky_helmet_does_not_fire_on_non_contact():
    attacker = _mk("A", types=(Type.FIRE, None))
    defender = _mk("B", item=Item.ROCKY_HELMET)
    state = _battle([attacker], [defender])
    hp_before = attacker.live_stats.HP
    step(state, {0: USE_EMBER, 1: USE_SWORDS_DANCE})
    assert hp_before == attacker.live_stats.HP


def test_leftovers_heals_each_turn():
    a = _mk("A", item=Item.LEFTOVERS)
    b = _mk("B")
    state = _battle([a], [b])
    max_hp = a.stat_totals.HP
    a.live_stats.HP = max_hp - 50
    step(state, {0: USE_SWORDS_DANCE, 1: USE_SWORDS_DANCE})
    expected = max(1, max_hp // 16)
    assert (max_hp - 50) + expected == a.live_stats.HP


def test_black_sludge_heals_poison_types():
    a = _mk("A", types=(Type.POISON, None), item=Item.BLACK_SLUDGE)
    b = _mk("B")
    state = _battle([a], [b])
    max_hp = a.stat_totals.HP
    a.live_stats.HP = max_hp - 50
    step(state, {0: USE_SWORDS_DANCE, 1: USE_SWORDS_DANCE})
    expected = max(1, max_hp // 16)
    assert (max_hp - 50) + expected == a.live_stats.HP


def test_black_sludge_hurts_non_poison_types():
    a = _mk("A", types=(Type.NORMAL, None), item=Item.BLACK_SLUDGE)
    b = _mk("B")
    state = _battle([a], [b])
    max_hp = a.stat_totals.HP
    hp_before = a.live_stats.HP
    step(state, {0: USE_SWORDS_DANCE, 1: USE_SWORDS_DANCE})
    expected = max(1, max_hp // 16)
    assert hp_before - a.live_stats.HP == expected


def test_expert_belt_boosts_super_effective_only():
    fighting = get_move("Close Combat")
    belt = _mk("A", item=Item.EXPERT_BELT, moves=MoveSet(fighting, TACKLE, EMBER, SWORDS_DANCE))
    plain = _mk("P", moves=MoveSet(fighting, TACKLE, EMBER, SWORDS_DANCE))
    use_cc = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.FIRST)

    rock_a, rock_b = _mk("R1", types=(Type.ROCK, None)), _mk("R2", types=(Type.ROCK, None))
    s1, s2 = _battle([belt], [rock_a]), _battle([plain], [rock_b])
    step(s1, {0: use_cc, 1: USE_SWORDS_DANCE})
    step(s2, {0: use_cc, 1: USE_SWORDS_DANCE})
    assert rock_b.stat_totals.HP - rock_b.live_stats.HP < rock_a.stat_totals.HP - rock_a.live_stats.HP

    n1, n2 = _mk("N1"), _mk("N2")
    s3, s4 = _battle([_mk("A", item=Item.EXPERT_BELT)], [n1]), _battle([_mk("P")], [n2])
    step(s3, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
    step(s4, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
    assert n1.stat_totals.HP - n1.live_stats.HP == n2.stat_totals.HP - n2.live_stats.HP  # neutral hit: no boost


def test_assault_vest_reduces_special_damage_only():
    vest_phys, vest_spec = _mk("V1", item=Item.ASSAULT_VEST), _mk("V2", item=Item.ASSAULT_VEST)
    bare_phys, bare_spec = _mk("B1"), _mk("B2")
    fire = (Type.FIRE, None)
    s1 = _battle([_mk("A", types=fire)], [vest_phys])
    s2 = _battle([_mk("A", types=fire)], [bare_phys])
    s3 = _battle([_mk("A", types=fire)], [vest_spec])
    s4 = _battle([_mk("A", types=fire)], [bare_spec])
    step(s1, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
    step(s2, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
    step(s3, {0: USE_EMBER, 1: USE_SWORDS_DANCE})
    step(s4, {0: USE_EMBER, 1: USE_SWORDS_DANCE})

    def dmg(mon: Pokemon) -> int:
        return mon.stat_totals.HP - mon.live_stats.HP

    assert dmg(vest_phys) == dmg(bare_phys)  # physical untouched
    assert dmg(vest_spec) < dmg(bare_spec)  # special reduced


def test_chople_berry_halves_one_super_effective_fighting_hit():
    fighting = get_move("Close Combat")
    attacker = _mk("A", moves=MoveSet(fighting, TACKLE, EMBER, SWORDS_DANCE))
    berry_holder = _mk("B", types=(Type.NORMAL, None), item=Item.CHOPLE_BERRY)
    bare = _mk("C", types=(Type.NORMAL, None))
    use_cc = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.FIRST)
    s1 = _battle([attacker], [berry_holder])
    s2 = _battle([_mk("A2", moves=MoveSet(fighting, TACKLE, EMBER, SWORDS_DANCE))], [bare])
    step(s1, {0: use_cc, 1: USE_SWORDS_DANCE})
    step(s2, {0: use_cc, 1: USE_SWORDS_DANCE})
    assert berry_holder.stat_totals.HP - berry_holder.live_stats.HP < bare.stat_totals.HP - bare.live_stats.HP
    assert berry_holder.item is Item.NONE  # consumed


# -- New gen-9 item coverage -------------------------------------------------------

from battle_sim.models.log_events import MultiHitSummary, StatDropBlockedByItem
from battle_sim.utils import Ability, Terrain, Weather

BULLET_SEED = get_move("Bullet Seed")
BITE = get_move("Bite")
EARTHQUAKE = get_move("Earthquake")
REFLECT = get_move("Reflect")
RAIN_DANCE = get_move("Rain Dance")
SPORE = get_move("Spore")
CLOSE_COMBAT = get_move("Close Combat")
SPLASH = get_move("Splash")
ROCK_SMASH = get_move("Rock Smash")
USE_FIRST = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.FIRST)
USE_FIRST_SELF = Action(action=ActionType.USE_MOVE, target=Target.SELF, move=MoveSlot.FIRST)
USE_FIRST_SIDE = Action(action=ActionType.USE_MOVE, target=Target.USER_SIDE, move=MoveSlot.FIRST)
USE_FIRST_FIELD = Action(action=ActionType.USE_MOVE, target=Target.FIELD, move=MoveSlot.FIRST)
IDLE_MOVES = MoveSet(SPLASH, TACKLE, EMBER, SWORDS_DANCE)
USE_SPLASH = Action(action=ActionType.USE_MOVE, target=Target.SELF, move=MoveSlot.FIRST)


def _field_battle(side0: list[Pokemon], side1: list[Pokemon], field: FieldState, seed: int = 0) -> BattleState:
    return BattleState(sides=(SideState(team=side0), SideState(team=side1)), rng=RNG(seed=seed), field=field)


def test_loaded_dice_rolls_at_least_four_hits():
    moves = MoveSet(BULLET_SEED, TACKLE, EMBER, SWORDS_DANCE)
    state = _battle([_mk("A", item=Item.LOADED_DICE, moves=moves)], [_mk("B")])
    log = step(state, {0: USE_FIRST, 1: USE_SWORDS_DANCE})
    summary = next(entry for entry in log if isinstance(entry, MultiHitSummary))
    assert summary.hits >= 4


def test_light_clay_extends_screens_to_eight_turns():
    moves = MoveSet(REFLECT, TACKLE, EMBER, SWORDS_DANCE)
    state = _battle([_mk("A", item=Item.LIGHT_CLAY, moves=moves)], [_mk("B")])
    step(state, {0: USE_FIRST_SIDE, 1: USE_SWORDS_DANCE})
    assert state.sides[0].screens[Hazards.REFLECT] == 7  # 8 minus this turn's tick


def test_damp_rock_extends_rain_to_eight_turns():
    moves = MoveSet(RAIN_DANCE, TACKLE, EMBER, SWORDS_DANCE)
    state = _battle([_mk("A", item=Item.DAMP_ROCK, moves=moves)], [_mk("B")])
    step(state, {0: USE_FIRST_FIELD, 1: USE_SWORDS_DANCE})
    assert state.field.weather is Weather.RAIN
    assert state.field.weather_turns_left == 7  # 8 minus this turn's tick

    state = _battle([_mk("A", moves=MoveSet(RAIN_DANCE, TACKLE, EMBER, SWORDS_DANCE))], [_mk("B")])
    step(state, {0: USE_FIRST_FIELD, 1: USE_SWORDS_DANCE})
    assert state.field.weather_turns_left == 4


def test_terrain_extender_stretches_ability_terrain():
    from battle_sim.utils import Ability

    state = _battle([_mk("A", ability=Ability.ELECTRIC_SURGE, item=Item.TERRAIN_EXTENDER)], [_mk("B")])
    step(state, {0: USE_SWORDS_DANCE, 1: USE_SWORDS_DANCE})
    assert state.field.terrain is Terrain.ELECTRIC
    assert state.field.terrain_turns_left == 7  # 8 minus this turn's tick


def test_sitrus_berry_heals_when_dropping_to_half():
    holder = _mk("B", item=Item.SITRUS_BERRY)
    holder.live_stats.HP = holder.stat_totals.HP // 2 + 1
    state = _battle([_mk("A")], [holder])
    step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
    assert holder.item is Item.NONE
    assert holder.item_consumed


def test_lum_berry_cures_the_status_that_lands():
    moves = MoveSet(SPORE, TACKLE, EMBER, SWORDS_DANCE)
    holder = _mk("B", item=Item.LUM_BERRY)
    state = _battle([_mk("A", moves=moves)], [holder])
    step(state, {0: USE_FIRST, 1: USE_SWORDS_DANCE})
    assert holder.status is Status.NONE
    assert holder.item is Item.NONE


def test_toxic_orb_badly_poisons_its_holder():
    holder = _mk("A", item=Item.TOXIC_ORB)
    state = _battle([holder], [_mk("B")])
    step(state, {0: USE_SWORDS_DANCE, 1: USE_SWORDS_DANCE})
    assert holder.status is Status.TOXIC


def test_flame_orb_burns_its_holder():
    holder = _mk("A", item=Item.FLAME_ORB)
    state = _battle([holder], [_mk("B")])
    step(state, {0: USE_SWORDS_DANCE, 1: USE_SWORDS_DANCE})
    assert holder.status is Status.BURN


def test_black_glasses_boost_dark_moves():
    def dmg(item: Item) -> int:
        attacker = _mk("A", item=item, moves=MoveSet(BITE, TACKLE, EMBER, SWORDS_DANCE))
        defender = _mk("B")
        state = _battle([attacker], [defender])
        hp = defender.live_stats.HP
        step(state, {0: USE_FIRST, 1: USE_SWORDS_DANCE})
        return hp - defender.live_stats.HP

    assert dmg(Item.BLACK_GLASSES) > dmg(Item.NONE)


def test_eviolite_bolsters_only_unevolved_holders():
    def dmg(item: Item, fully_evolved: bool) -> int:
        defender = _mk("B", item=item)
        defender.fully_evolved = fully_evolved
        state = _battle([_mk("A")], [defender])
        hp = defender.live_stats.HP
        step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
        return hp - defender.live_stats.HP

    assert dmg(Item.EVIOLITE, fully_evolved=False) < dmg(Item.NONE, fully_evolved=False)
    assert dmg(Item.EVIOLITE, fully_evolved=True) == dmg(Item.NONE, fully_evolved=True)


def test_weakness_policy_fires_on_super_effective_hits():
    holder = _mk("B", types=(Type.GRASS, None), item=Item.WEAKNESS_POLICY, moves=IDLE_MOVES)
    state = _battle([_mk("A", types=(Type.FIRE, None))], [holder])
    step(state, {0: USE_EMBER, 1: USE_SPLASH})
    assert holder.stat_stages.ATTACK == 2
    assert holder.stat_stages.SP_ATTACK == 2
    assert holder.item is Item.NONE


def test_shuca_berry_halves_one_super_effective_ground_hit():
    def dmg(item: Item) -> tuple[int, Item]:
        defender = _mk("B", types=(Type.ELECTRIC, None), item=item)
        state = _battle([_mk("A", moves=MoveSet(EARTHQUAKE, TACKLE, EMBER, SWORDS_DANCE))], [defender])
        hp = defender.live_stats.HP
        step(
            state,
            {
                0: Action(action=ActionType.USE_MOVE, target=Target.ALL_ADJACENT, move=MoveSlot.FIRST),
                1: USE_SWORDS_DANCE,
            },
        )
        return hp - defender.live_stats.HP, defender.item

    shielded, item_after = dmg(Item.SHUCA_BERRY)
    plain, _ = dmg(Item.NONE)
    assert 0 < shielded < plain
    assert item_after is Item.NONE


def test_clear_amulet_blocks_intimidate():
    holder = _mk("B", item=Item.CLEAR_AMULET, moves=IDLE_MOVES)
    state = _battle([_mk("A", ability=Ability.INTIMIDATE)], [holder])
    log = step(state, {0: USE_SWORDS_DANCE, 1: USE_SPLASH})
    assert holder.stat_stages.ATTACK == 0
    assert any(isinstance(entry, StatDropBlockedByItem) for entry in log)


def test_white_herb_restores_self_inflicted_drops():
    moves = MoveSet(CLOSE_COMBAT, TACKLE, EMBER, SWORDS_DANCE)
    holder = _mk("A", item=Item.WHITE_HERB, moves=moves)
    state = _battle([holder], [_mk("B")])
    step(state, {0: USE_FIRST, 1: USE_SWORDS_DANCE})
    assert holder.stat_stages.DEFENCE == 0
    assert holder.stat_stages.SP_DEFENCE == 0
    assert holder.item is Item.NONE


def test_grassy_seed_pops_when_the_terrain_comes_up():
    holder = _mk("B", item=Item.GRASSY_SEED, moves=IDLE_MOVES)
    state = _battle([_mk("A", ability=Ability.GRASSY_SURGE)], [holder])
    step(state, {0: USE_SWORDS_DANCE, 1: USE_SPLASH})
    assert holder.stat_stages.DEFENCE == 1
    assert holder.item is Item.NONE


def test_covert_cloak_blocks_secondary_effects():
    moves = MoveSet(ROCK_SMASH, TACKLE, EMBER, SWORDS_DANCE)
    cloaked = _mk("B", item=Item.COVERT_CLOAK, moves=IDLE_MOVES)
    state = _battle([_mk("A", ability=Ability.SERENE_GRACE, moves=moves)], [cloaked])
    step(state, {0: USE_FIRST, 1: USE_SPLASH})
    assert cloaked.stat_stages.DEFENCE == 0


def test_wellspring_mask_boosts_its_ogerpon():
    def dmg(name: str) -> int:
        attacker = _mk("A", item=Item.WELLSPRING_MASK)
        attacker.name = name
        defender = _mk("B")
        state = _battle([attacker], [defender])
        hp = defender.live_stats.HP
        step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
        return hp - defender.live_stats.HP

    assert dmg("Ogerpon-Wellspring") > dmg("TestMon")
