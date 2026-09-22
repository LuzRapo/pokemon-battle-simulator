"""Coverage for PS code-driven move behaviours: power formulas, coded effects, and riders."""

from battle_sim.database.loader import get_move
from battle_sim.engine import apply_forced_switch, legal_actions, step
from battle_sim.engine.power import ROLLING_LOCK_TURNS
from battle_sim.maths.rng import RNG
from battle_sim.mechanics.battle import BattleState, FieldState, SideState
from battle_sim.models.actions import Action, ActionType
from battle_sim.models.log_events import CantAct, ChargingUp, MoveBounced
from battle_sim.models.moves import MoveSet, MoveSlot
from battle_sim.models.pokemon import Pokemon
from battle_sim.models.stats import BaseStats, EVs, IVs
from battle_sim.utils import Ability, ExtraStatus, Hazards, Item, Nature, Status, Target, Type, Weather

TACKLE = get_move("Tackle")
EMBER = get_move("Ember")
QUICK_ATTACK = get_move("Quick Attack")
SWORDS_DANCE = get_move("Swords Dance")
SPLASH = get_move("Splash")
USE_FIRST = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.FIRST)
USE_FIRST_SELF = Action(action=ActionType.USE_MOVE, target=Target.SELF, move=MoveSlot.FIRST)
USE_FIRST_SIDE = Action(action=ActionType.USE_MOVE, target=Target.OPPONENT_SIDE, move=MoveSlot.FIRST)
USE_FIRST_FIELD = Action(action=ActionType.USE_MOVE, target=Target.FIELD, move=MoveSlot.FIRST)
USE_TACKLE_2 = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.SECOND)
USE_SPLASH_4 = Action(action=ActionType.USE_MOVE, target=Target.SELF, move=MoveSlot.FOURTH)
IDLE = Action(action=ActionType.USE_MOVE, target=Target.SELF, move=MoveSlot.FOURTH)


def _mk(
    nickname: str = "X",
    types: tuple[Type, Type | None] = (Type.NORMAL, None),
    ability: Ability = Ability.NONE,
    item: Item = Item.NONE,
    base_stats: BaseStats | None = None,
    first: str | None = None,
    weight_kg: float = 100.0,
    status: Status = Status.NONE,
) -> Pokemon:
    if base_stats is None:
        base_stats = BaseStats(HP=100, ATTACK=100, DEFENCE=100, SP_ATTACK=100, SP_DEFENCE=100, SPEED=100)
    first_move = get_move(first) if first is not None else TACKLE
    return Pokemon(
        name="TestMon",
        nickname=nickname,
        level=50,
        base_stats=base_stats,
        effort_values=EVs(),
        individual_values=IVs(),
        types=types,
        moves=MoveSet(first_move, TACKLE, EMBER, SPLASH),
        nature=Nature.HARDY,
        item=item,
        ability=ability,
        weight_kg=weight_kg,
        status=status,
    )


def _battle(side0: list[Pokemon], side1: list[Pokemon], seed: int = 0, field: FieldState | None = None) -> BattleState:
    return BattleState(
        sides=(SideState(team=side0), SideState(team=side1)), rng=RNG(seed=seed), field=field or FieldState()
    )


def _duel(attacker: Pokemon, defender: Pokemon, action: Action = USE_FIRST, field: FieldState | None = None) -> int:
    state = _battle([attacker], [defender], field=field)
    hp = defender.live_stats.HP
    step(state, {0: action, 1: USE_SPLASH_4})
    return hp - defender.live_stats.HP


def test_low_kick_scales_with_target_weight():
    light = _duel(_mk(first="Low Kick"), _mk(weight_kg=5.0))
    heavy = _duel(_mk(first="Low Kick"), _mk(weight_kg=250.0))
    assert heavy > light > 0


def test_heavy_slam_scales_with_weight_ratio():
    crushing = _duel(_mk(first="Heavy Slam", weight_kg=500.0), _mk(weight_kg=50.0))
    even = _duel(_mk(first="Heavy Slam", weight_kg=100.0), _mk(weight_kg=100.0))
    assert crushing > even > 0


def test_stored_power_scales_with_boosts():
    flat = _duel(_mk(first="Stored Power"), _mk())
    boosted_user = _mk(first="Stored Power")
    boosted_user.stat_stages.SPEED = 6
    boosted = _duel(boosted_user, _mk())
    assert boosted > flat > 0


_SLOW_STATS = BaseStats(HP=100, ATTACK=100, DEFENCE=100, SP_ATTACK=100, SP_DEFENCE=100, SPEED=10)
_FAST_STATS = BaseStats(HP=100, ATTACK=100, DEFENCE=100, SP_ATTACK=100, SP_DEFENCE=100, SPEED=300)


def test_gyro_ball_hits_harder_the_slower_the_user_is_next_to_its_target():
    against_fast = _duel(_mk(first="Gyro Ball", base_stats=_SLOW_STATS), _mk(base_stats=_FAST_STATS))
    against_slow = _duel(_mk(first="Gyro Ball", base_stats=_SLOW_STATS), _mk(base_stats=_SLOW_STATS))
    assert against_fast > against_slow > 0


def test_knock_off_removes_the_item_and_hits_harder_for_it():
    bare = _duel(_mk(first="Knock Off"), _mk())
    holder = _mk(item=Item.LEFTOVERS)
    loaded = _duel(_mk(first="Knock Off"), holder)
    assert loaded > bare
    assert holder.item is Item.NONE


def test_sucker_punch_fails_against_status_moves():
    state = _battle([_mk(first="Sucker Punch")], [_mk()])
    hp = state.sides[1].active_pokemon.live_stats.HP
    step(state, {0: USE_FIRST, 1: USE_SPLASH_4})  # target is set up to Splash
    assert hp == state.sides[1].active_pokemon.live_stats.HP


def test_sucker_punch_hits_an_attacking_target():
    defender = _mk()
    state = _battle([_mk(first="Sucker Punch")], [defender])
    hp = defender.live_stats.HP
    step(state, {0: USE_FIRST, 1: USE_TACKLE_2})
    assert hp > defender.live_stats.HP


def test_fake_out_only_works_on_the_switch_in_turn():
    attacker = _mk(first="Fake Out")
    defender = _mk()
    state = _battle([attacker], [defender])

    hp = defender.live_stats.HP
    step(state, {0: USE_FIRST, 1: USE_SPLASH_4})
    assert hp > defender.live_stats.HP  # works the turn it switches in

    hp = defender.live_stats.HP
    step(state, {0: USE_FIRST, 1: USE_SPLASH_4})
    assert hp == defender.live_stats.HP  # every turn after, it fails


def test_first_impression_only_works_on_the_switch_in_turn():
    attacker = _mk(first="First Impression")
    defender = _mk()
    state = _battle([attacker], [defender])

    hp = defender.live_stats.HP
    step(state, {0: USE_FIRST, 1: USE_SPLASH_4})
    assert hp > defender.live_stats.HP  # works the turn it switches in

    hp = defender.live_stats.HP
    step(state, {0: USE_FIRST, 1: USE_SPLASH_4})
    assert hp == defender.live_stats.HP  # every turn after, it fails


def test_first_impression_works_again_after_switching_back_in():
    attacker = _mk(first="First Impression")
    bench = _mk("Bench")
    defender = _mk()
    state = _battle([attacker, bench], [defender])

    step(state, {0: USE_FIRST, 1: USE_SPLASH_4})  # spend the switch-in turn
    step(state, {0: Action(action=ActionType.SWITCH_OUT, switch_in=bench), 1: USE_SPLASH_4})
    step(state, {0: Action(action=ActionType.SWITCH_OUT, switch_in=attacker), 1: USE_SPLASH_4})  # fresh stint

    hp = defender.live_stats.HP
    step(state, {0: USE_FIRST, 1: USE_SPLASH_4})
    assert hp > defender.live_stats.HP  # works again on the new switch-in


def test_dream_eater_fails_unless_the_target_is_asleep():
    awake = _mk()
    state = _battle([_mk(first="Dream Eater")], [awake])
    hp = awake.live_stats.HP
    step(state, {0: USE_FIRST, 1: USE_SPLASH_4})
    assert hp == awake.live_stats.HP


def test_dream_eater_drains_from_a_sleeping_target():
    asleep = _mk(status=Status.SLEEP)
    asleep.status_turns = 3  # stays asleep even if it acts first this turn
    attacker = _mk(first="Dream Eater")
    attacker.live_stats.HP -= 20
    state = _battle([attacker], [asleep])
    hp = asleep.live_stats.HP
    hp_before_drain = attacker.live_stats.HP
    step(state, {0: USE_FIRST, 1: USE_SPLASH_4})
    assert hp > asleep.live_stats.HP
    assert hp_before_drain < attacker.live_stats.HP


def test_body_press_attacks_with_defence():
    tank_body = BaseStats(HP=100, ATTACK=10, DEFENCE=200, SP_ATTACK=100, SP_DEFENCE=100, SPEED=100)
    frail_body = BaseStats(HP=100, ATTACK=10, DEFENCE=20, SP_ATTACK=100, SP_DEFENCE=100, SPEED=100)
    tanky = _duel(_mk(first="Body Press", base_stats=tank_body), _mk())
    frail = _duel(_mk(first="Body Press", base_stats=frail_body), _mk())
    assert tanky > frail


def test_foul_play_uses_the_targets_attack():
    muscle_body = BaseStats(HP=100, ATTACK=250, DEFENCE=100, SP_ATTACK=100, SP_DEFENCE=100, SPEED=100)
    weak_body = BaseStats(HP=100, ATTACK=10, DEFENCE=100, SP_ATTACK=100, SP_DEFENCE=100, SPEED=100)
    vs_muscle = _duel(_mk(first="Foul Play"), _mk(base_stats=muscle_body))
    vs_weak = _duel(_mk(first="Foul Play"), _mk(base_stats=weak_body))
    assert vs_muscle > vs_weak


def test_facade_doubles_when_statused():
    healthy = _duel(_mk(first="Facade"), _mk())
    burned = _duel(_mk(first="Facade", status=Status.BURN), _mk())
    assert burned > healthy  # doubled AND the burn halving is ignored


def test_acrobatics_doubles_without_an_item():
    unburdened = _duel(_mk(first="Acrobatics"), _mk())
    laden = _duel(_mk(first="Acrobatics", item=Item.LEFTOVERS), _mk())
    assert unburdened > laden


def test_freeze_dry_is_super_effective_on_water():
    soaked = _duel(_mk(first="Freeze-Dry"), _mk(types=(Type.WATER, None)))
    neutral = _duel(_mk(first="Freeze-Dry"), _mk(types=(Type.NORMAL, None)))
    assert soaked > neutral


def test_rest_fully_heals_into_a_two_turn_sleep():
    sleeper = _mk(first="Rest")
    sleeper.live_stats.HP = 10
    state = _battle([sleeper], [_mk()])
    step(state, {0: USE_FIRST_SELF, 1: USE_SPLASH_4})
    assert sleeper.live_stats.HP == sleeper.stat_totals.HP
    assert sleeper.status is Status.SLEEP


def test_morning_sun_heals_more_in_sun():
    basking = _mk(first="Morning Sun")
    basking.live_stats.HP = 1
    state = _battle([basking], [_mk()], field=FieldState(weather=Weather.SUN, weather_turns_left=8))
    step(state, {0: USE_FIRST_SELF, 1: USE_SPLASH_4})
    assert basking.live_stats.HP == 1 + basking.stat_totals.HP * 2 // 3


def test_pain_split_averages_hp():
    hurt = _mk(first="Pain Split")
    hurt.live_stats.HP = 20
    healthy = _mk()
    state = _battle([hurt], [healthy])
    step(state, {0: USE_FIRST, 1: USE_SPLASH_4})
    average = (20 + healthy.stat_totals.HP) // 2
    assert average == hurt.live_stats.HP
    assert average == healthy.live_stats.HP


def test_strength_sap_heals_by_the_targets_attack_and_drops_it():
    sapper = _mk(first="Strength Sap")
    sapper.live_stats.HP = 10
    target = _mk()
    state = _battle([sapper], [target])
    step(state, {0: USE_FIRST, 1: USE_SPLASH_4})
    assert sapper.live_stats.HP > 10
    assert target.stat_stages.ATTACK == -1


def test_belly_drum_pays_half_for_maximum_attack():
    drummer = _mk(first="Belly Drum")
    state = _battle([drummer], [_mk()])
    step(state, {0: USE_FIRST_SELF, 1: USE_SPLASH_4})
    assert drummer.stat_stages.ATTACK == 6
    assert drummer.live_stats.HP == drummer.stat_totals.HP - drummer.stat_totals.HP // 2


def test_haze_resets_both_actives():
    hazer = _mk(first="Haze")
    hazer.stat_stages.ATTACK = -3
    opponent = _mk()
    opponent.stat_stages.SP_ATTACK = 4
    state = _battle([hazer], [opponent])
    step(state, {0: USE_FIRST_FIELD, 1: USE_SPLASH_4})
    assert hazer.stat_stages.ATTACK == 0
    assert opponent.stat_stages.SP_ATTACK == 0


def test_court_change_swaps_side_conditions():
    state = _battle([_mk(first="Court Change")], [_mk()])
    state.sides[0].hazards[Hazards.STEALTH_ROCK] = 1
    state.sides[1].screens[Hazards.REFLECT] = 3
    step(state, {0: USE_FIRST_FIELD, 1: USE_SPLASH_4})
    assert Hazards.STEALTH_ROCK in state.sides[1].hazards
    assert Hazards.REFLECT in state.sides[0].screens


def test_wish_heals_at_the_end_of_the_next_turn():
    wisher = _mk(first="Wish")
    wisher.live_stats.HP = 10
    state = _battle([wisher], [_mk()])
    step(state, {0: USE_FIRST_SELF, 1: USE_SPLASH_4})
    assert wisher.live_stats.HP == 10
    step(state, {0: USE_SPLASH_4, 1: USE_SPLASH_4})
    assert wisher.live_stats.HP == 10 + wisher.stat_totals.HP // 2


def test_healing_wish_faints_the_user_and_restores_the_replacement():
    wisher, partner = _mk(first="Healing Wish"), _mk("P", status=Status.BURN)
    partner.live_stats.HP = 30
    state = _battle([wisher, partner], [_mk()])
    step(state, {0: USE_FIRST_SELF, 1: USE_SPLASH_4})
    assert wisher.is_fainted()
    apply_forced_switch(state, 0, Action(action=ActionType.SWITCH_OUT, switch_in=partner))
    assert partner.live_stats.HP == partner.stat_totals.HP
    assert partner.status is Status.NONE


def test_future_sight_lands_two_turns_after_it_is_used_not_immediately():
    defender = _mk()
    state = _battle([_mk(first="Future Sight")], [defender])
    hp = defender.live_stats.HP

    step(state, {0: USE_FIRST, 1: USE_SPLASH_4})
    assert hp == defender.live_stats.HP  # nothing on the turn it is used

    step(state, {0: USE_SPLASH_4, 1: USE_SPLASH_4})
    assert hp == defender.live_stats.HP  # nor the turn after that

    step(state, {0: USE_SPLASH_4, 1: USE_SPLASH_4})
    assert hp > defender.live_stats.HP  # lands at the end of the second turn after use


def test_future_sight_hits_whoever_is_in_that_slot_when_it_lands_even_a_switched_in_replacement():
    replacement = _mk("R")
    state = _battle([_mk(first="Future Sight")], [_mk(), replacement])
    step(state, {0: USE_FIRST, 1: USE_SPLASH_4})
    step(state, {0: USE_SPLASH_4, 1: Action(action=ActionType.SWITCH_OUT, switch_in=replacement)})
    hp = replacement.live_stats.HP
    step(state, {0: USE_SPLASH_4, 1: USE_SPLASH_4})
    assert hp > replacement.live_stats.HP


def test_a_second_future_sight_fails_while_one_is_already_pending():
    attacker = _mk(first="Future Sight")
    state = _battle([attacker], [_mk()])
    step(state, {0: USE_FIRST, 1: USE_SPLASH_4})
    turns_after_first_use = state.sides[1].future_sight_turns
    step(state, {0: USE_FIRST, 1: USE_SPLASH_4})  # a second attempt while one is already queued
    # the retry must not have restarted the countdown — it only ticks down by the residual phase
    assert state.sides[1].future_sight_turns == turns_after_first_use - 1


def test_future_sight_queues_despite_a_currently_immune_target_and_checks_type_on_landing():
    """The type match is only known for certain once it actually lands: the target could switch."""
    dark_mon = _mk(types=(Type.DARK, None))
    state = _battle([_mk(first="Future Sight")], [dark_mon])
    step(state, {0: USE_FIRST, 1: USE_SPLASH_4})
    assert state.sides[1].future_sight_turns > 0  # queued rather than failing outright

    step(state, {0: USE_SPLASH_4, 1: USE_SPLASH_4})
    hp = dark_mon.live_stats.HP
    step(state, {0: USE_SPLASH_4, 1: USE_SPLASH_4})
    assert hp == dark_mon.live_stats.HP  # Dark is immune to Psychic, so it does nothing once it lands


def test_curse_by_a_ghost_afflicts_the_target():
    ghost = _mk(first="Curse", types=(Type.GHOST, None))
    victim = _mk()
    state = _battle([ghost], [victim])
    hp = victim.live_stats.HP
    step(state, {0: USE_FIRST, 1: USE_SPLASH_4})
    assert ExtraStatus.CURSE in victim.volatiles
    assert ghost.live_stats.HP == ghost.stat_totals.HP - ghost.stat_totals.HP // 2
    assert hp - victim.stat_totals.HP // 4 == victim.live_stats.HP  # the residual already bit


def test_curse_by_a_non_ghost_trades_speed_for_bulk():
    snorlax = _mk(first="Curse")
    state = _battle([snorlax], [_mk()])
    step(state, {0: USE_FIRST, 1: USE_SPLASH_4})
    assert snorlax.stat_stages.ATTACK == 1
    assert snorlax.stat_stages.DEFENCE == 1
    assert snorlax.stat_stages.SPEED == -1


def test_perish_song_fells_both_actives_after_three_turns():
    singer = _mk(first="Perish Song")
    other = _mk()
    state = _battle([singer], [other])
    step(state, {0: USE_FIRST_FIELD, 1: USE_SPLASH_4})
    step(state, {0: USE_SPLASH_4, 1: USE_SPLASH_4})
    step(state, {0: USE_SPLASH_4, 1: USE_SPLASH_4})
    assert not singer.is_fainted() and not other.is_fainted()
    step(state, {0: USE_SPLASH_4, 1: USE_SPLASH_4})
    assert singer.is_fainted() and other.is_fainted()


def test_destiny_bond_takes_the_attacker_down():
    bonded = _mk(first="Destiny Bond")
    bonded.live_stats.HP = 1
    nuke_body = BaseStats(HP=100, ATTACK=250, DEFENCE=100, SP_ATTACK=100, SP_DEFENCE=100, SPEED=10)
    attacker = _mk(base_stats=nuke_body)
    state = _battle([bonded], [attacker])
    step(state, {0: USE_FIRST_SELF, 1: USE_TACKLE_2})  # bond goes up first (faster), then the KO lands
    assert bonded.is_fainted()
    assert attacker.is_fainted()


def test_endure_survives_a_lethal_hit():
    survivor = _mk(first="Endure")
    survivor.live_stats.HP = 5
    nuke_body = BaseStats(HP=100, ATTACK=250, DEFENCE=100, SP_ATTACK=100, SP_DEFENCE=100, SPEED=10)
    state = _battle([survivor], [_mk(base_stats=nuke_body)])
    step(state, {0: USE_FIRST_SELF, 1: USE_TACKLE_2})
    assert survivor.live_stats.HP == 1


def test_salt_cure_chips_until_switch():
    salter = _mk(first="Salt Cure")
    victim = _mk()
    state = _battle([salter], [victim])
    step(state, {0: USE_FIRST, 1: USE_SPLASH_4})
    assert ExtraStatus.SALT_CURE in victim.volatiles
    hp_after_turn = victim.live_stats.HP
    step(state, {0: USE_SPLASH_4, 1: USE_SPLASH_4})
    assert hp_after_turn - victim.stat_totals.HP // 8 == victim.live_stats.HP


def test_ruination_halves_the_targets_hp():
    state = _battle([_mk(first="Ruination")], [_mk()])
    victim = state.sides[1].active_pokemon
    step(state, {0: USE_FIRST, 1: USE_SPLASH_4})
    assert victim.live_stats.HP == victim.stat_totals.HP - victim.stat_totals.HP // 2


def test_counter_returns_double_the_physical_hit():
    countering = _mk(
        first="Counter", base_stats=BaseStats(HP=200, ATTACK=100, DEFENCE=100, SP_ATTACK=100, SP_DEFENCE=100, SPEED=10)
    )
    attacker = _mk()
    state = _battle([countering], [attacker])
    step(state, {0: USE_FIRST, 1: USE_TACKLE_2})
    taken = countering.stat_totals.HP - countering.live_stats.HP
    returned = attacker.stat_totals.HP - attacker.live_stats.HP
    assert taken > 0
    assert returned == 2 * taken


def test_counter_fails_against_a_counter_proof_target():
    """Its own strength cannot be turned back on it — see `Pokemon.counter_proof`."""
    countering = _mk(
        first="Counter", base_stats=BaseStats(HP=200, ATTACK=100, DEFENCE=100, SP_ATTACK=100, SP_DEFENCE=100, SPEED=10)
    )
    attacker = _mk()
    attacker.counter_proof = True
    state = _battle([countering], [attacker])
    step(state, {0: USE_FIRST, 1: USE_TACKLE_2})
    assert countering.stat_totals.HP - countering.live_stats.HP > 0, "it still took the hit"
    assert attacker.live_stats.HP == attacker.stat_totals.HP, "and nothing came back"


def test_final_gambit_deals_the_users_hp_and_faints_it():
    gambler = _mk(
        first="Final Gambit",
        base_stats=BaseStats(HP=120, ATTACK=100, DEFENCE=100, SP_ATTACK=100, SP_DEFENCE=100, SPEED=200),
    )
    victim = _mk(base_stats=BaseStats(HP=250, ATTACK=100, DEFENCE=100, SP_ATTACK=100, SP_DEFENCE=100, SPEED=100))
    state = _battle([gambler], [victim])
    user_hp = gambler.live_stats.HP
    step(state, {0: USE_FIRST, 1: USE_SPLASH_4})
    assert gambler.is_fainted()
    assert user_hp == victim.stat_totals.HP - victim.live_stats.HP


def test_explosion_faints_the_user():
    bomber = _mk(first="Explosion")
    state = _battle([bomber], [_mk()])
    step(state, {0: USE_FIRST, 1: USE_SPLASH_4})
    assert bomber.is_fainted()


def test_whirlwind_drags_in_a_replacement():
    blower = _mk(first="Whirlwind")
    lead, bench = _mk("Lead"), _mk("Bench")
    state = _battle([blower], [lead, bench])
    step(state, {0: USE_FIRST, 1: USE_SPLASH_4})
    assert state.sides[1].active_pokemon is bench


def test_shed_tail_passes_a_substitute_to_the_replacement():
    shedder, partner = _mk(first="Shed Tail"), _mk("P")
    state = _battle([shedder, partner], [_mk()])
    step(state, {0: USE_FIRST_SELF, 1: USE_SPLASH_4})
    assert state.sides[0].needs_switch
    assert shedder.stat_totals.HP - max(1, shedder.stat_totals.HP // 2) == shedder.live_stats.HP
    apply_forced_switch(state, 0, Action(action=ActionType.SWITCH_OUT, switch_in=partner))
    assert ExtraStatus.SUBSTITUTE in partner.volatiles


def test_solar_beam_charges_for_a_turn():
    beamer = _mk(first="Solar Beam")
    victim = _mk()
    state = _battle([beamer], [victim])
    log = step(state, {0: USE_FIRST, 1: USE_SPLASH_4})
    assert any(isinstance(entry, ChargingUp) for entry in log)
    assert victim.live_stats.HP == victim.stat_totals.HP
    assert [a.move for a in legal_actions(state, 0)] == [MoveSlot.FIRST]  # committed
    step(state, {0: USE_FIRST, 1: USE_SPLASH_4})
    assert victim.live_stats.HP < victim.stat_totals.HP


def test_solar_beam_fires_immediately_in_sun_or_with_power_herb():
    sunny = FieldState(weather=Weather.SUN, weather_turns_left=8)
    assert _duel(_mk(first="Solar Beam"), _mk(), field=sunny) > 0
    herbed = _mk(first="Solar Beam", item=Item.POWER_HERB)
    assert _duel(herbed, _mk()) > 0
    assert herbed.item is Item.NONE


def test_hyper_beam_forces_a_recharge_turn():
    beamer = _mk(first="Hyper Beam")
    state = _battle(
        [beamer], [_mk(base_stats=BaseStats(HP=250, ATTACK=100, DEFENCE=200, SP_ATTACK=100, SP_DEFENCE=200, SPEED=10))]
    )
    step(state, {0: USE_FIRST, 1: USE_SPLASH_4})
    log = step(state, {0: USE_TACKLE_2, 1: USE_SPLASH_4})
    assert any(isinstance(entry, CantAct) and entry.reason == "recharge" for entry in log)


def test_sleep_talk_acts_through_sleep():
    victim = _mk()
    hp = victim.live_stats.HP
    for _ in range(10):  # the called move is random; Tackle/Ember both damage, Splash does not
        talker = _mk(first="Sleep Talk", status=Status.SLEEP)
        talker.status_turns = 5
        state = _battle([talker], [victim])
        step(state, {0: USE_FIRST_SELF, 1: USE_SPLASH_4})
        if hp > victim.live_stats.HP:
            return
    raise AssertionError("Sleep Talk never landed a damaging move across 10 attempts")


def test_transform_copies_the_targets_battle_form():
    ditto = _mk(first="Transform")
    target = _mk(
        types=(Type.DRAGON, None),
        base_stats=BaseStats(HP=100, ATTACK=170, DEFENCE=100, SP_ATTACK=100, SP_DEFENCE=100, SPEED=100),
    )
    state = _battle([ditto], [target])
    step(state, {0: USE_FIRST, 1: USE_SPLASH_4})
    assert ditto.types == (Type.DRAGON, None)
    assert ditto.base_stats.ATTACK == 170
    assert all(pp == 5 for pp in ditto.pp.values())


def test_magic_bounce_reflects_hazards_onto_the_thrower():
    rocker = _mk(first="Stealth Rock")
    bouncer = _mk(ability=Ability.MAGIC_BOUNCE)
    state = _battle([rocker], [bouncer])
    log = step(state, {0: USE_FIRST_SIDE, 1: USE_SPLASH_4})
    assert any(isinstance(entry, MoveBounced) for entry in log)
    assert Hazards.STEALTH_ROCK in state.sides[0].hazards
    assert Hazards.STEALTH_ROCK not in state.sides[1].hazards


def test_mold_breaker_hits_through_levitate():
    quaker = _mk(first="Earthquake", ability=Ability.MOLD_BREAKER)
    floater = _mk(ability=Ability.LEVITATE)
    state = _battle([quaker], [floater])
    step(
        state, {0: Action(action=ActionType.USE_MOVE, target=Target.ALL_ADJACENT, move=MoveSlot.FIRST), 1: USE_SPLASH_4}
    )
    assert floater.live_stats.HP < floater.stat_totals.HP


def test_neutralizing_gas_suppresses_handler_abilities():
    suppressed = _duel(_mk(ability=Ability.NEUTRALIZING_GAS), _mk(ability=Ability.MULTISCALE))
    baseline = _duel(_mk(), _mk())
    assert suppressed == baseline  # Multiscale never halved the hit


def test_air_lock_suppresses_sandstorm_chip():
    chilled = _mk(ability=Ability.AIR_LOCK)
    other = _mk()
    state = _battle([chilled], [other], field=FieldState(weather=Weather.SANDSTORM, weather_turns_left=8))
    hp = other.live_stats.HP
    step(state, {0: USE_SPLASH_4, 1: USE_SPLASH_4})
    assert hp == other.live_stats.HP


def test_shadow_tag_traps_non_ghosts():
    trapped, partner = _mk("T"), _mk("P")
    state = _battle([_mk(ability=Ability.SHADOW_TAG)], [trapped, partner])
    assert all(a.action is ActionType.USE_MOVE for a in legal_actions(state, 1))
    ghost, ghost_partner = _mk("G", types=(Type.GHOST, None)), _mk("P2")
    state = _battle([_mk(ability=Ability.SHADOW_TAG)], [ghost, ghost_partner])
    assert any(a.action is ActionType.SWITCH_OUT for a in legal_actions(state, 1))


def test_sheer_force_boosts_but_strips_secondaries():
    forced = _duel(_mk(first="Rock Smash", ability=Ability.SHEER_FORCE), _mk())
    plain = _duel(_mk(first="Rock Smash"), _mk())
    assert forced > plain
    victim = _mk()
    state = _battle(
        [
            _mk(
                first="Rock Smash",
                ability=Ability.SHEER_FORCE,
                base_stats=BaseStats(HP=100, ATTACK=10, DEFENCE=100, SP_ATTACK=100, SP_DEFENCE=100, SPEED=100),
            )
        ],
        [victim],
    )
    for _ in range(5):
        step(state, {0: USE_FIRST, 1: USE_SPLASH_4})
    assert victim.stat_stages.DEFENCE == 0  # the 50% drop never fires


def test_eject_button_pulls_the_holder():
    holder, partner = _mk("H", item=Item.EJECT_BUTTON), _mk("P")
    state = _battle([_mk()], [holder, partner])
    step(state, {0: USE_FIRST, 1: USE_SPLASH_4})
    assert state.sides[1].needs_switch
    assert holder.item is Item.NONE


def test_red_card_drags_the_attacker_out():
    holder = _mk("H", item=Item.RED_CARD)
    attacker, bench = _mk("A"), _mk("B")
    state = _battle([attacker, bench], [holder])
    step(state, {0: USE_FIRST, 1: USE_SPLASH_4})
    assert state.sides[0].active_pokemon is bench
    assert holder.item is Item.NONE


def test_mirror_armor_reflects_intimidate():
    mirrored = _mk(ability=Ability.MIRROR_ARMOR)
    intimidator = _mk(ability=Ability.INTIMIDATE)
    state = _battle([intimidator], [mirrored])
    step(state, {0: USE_SPLASH_4, 1: USE_SPLASH_4})
    assert mirrored.stat_stages.ATTACK == 0
    assert intimidator.stat_stages.ATTACK == -1


def test_guard_dog_turns_intimidate_into_a_boost():
    dog = _mk(ability=Ability.GUARD_DOG)
    state = _battle([_mk(ability=Ability.INTIMIDATE)], [dog])
    step(state, {0: USE_SPLASH_4, 1: USE_SPLASH_4})
    assert dog.stat_stages.ATTACK == 1


def test_scrappy_hits_ghosts_with_normal_moves():
    assert _duel(_mk(ability=Ability.SCRAPPY), _mk(types=(Type.GHOST, None))) > 0
    assert _duel(_mk(), _mk(types=(Type.GHOST, None))) == 0


def test_trick_swaps_items():
    trickster = _mk(first="Trick", item=Item.CHOICE_BAND)
    mark = _mk(item=Item.LEFTOVERS)
    state = _battle([trickster], [mark])
    step(state, {0: USE_FIRST, 1: USE_SPLASH_4})
    assert trickster.item is Item.LEFTOVERS
    assert mark.item is Item.CHOICE_BAND


def test_zero_to_hero_transforms_on_switch_out():
    from battle_sim.database.loader import get_species

    palafin = _mk("Fin", ability=Ability.ZERO_TO_HERO)
    palafin.name = "Palafin"
    partner = _mk("P")
    state = _battle([palafin, partner], [_mk()])
    step(state, {0: Action(action=ActionType.SWITCH_OUT, switch_in=partner), 1: USE_SPLASH_4})
    assert palafin.name == "Palafin-Hero"
    assert palafin.base_stats == get_species("Palafin-Hero").base_stats


def _rolling_duel() -> tuple[Pokemon, Pokemon, BattleState]:
    """A Rollout user against a wall too bulky to faint, so a whole run fits in one battle."""
    attacker = _mk(first="Rollout")
    defender = _mk(base_stats=BaseStats(HP=255, ATTACK=100, DEFENCE=255, SP_ATTACK=100, SP_DEFENCE=100, SPEED=1))
    return attacker, defender, _battle([attacker], [defender])


def _roll(turns: int) -> list[int]:
    """Damage from each of `turns` consecutive Rollouts, the target topped up between them."""
    attacker, defender, state = _rolling_duel()
    dealt = []
    for _ in range(turns):
        defender.live_stats.HP = defender.stat_totals.HP
        step(state, {0: USE_FIRST, 1: USE_SPLASH_4})
        dealt.append(defender.stat_totals.HP - defender.live_stats.HP)
    return dealt


def test_rollout_doubles_with_every_connected_hit():
    """30 -> 60 -> 120 -> 240 -> 480: the ramp is the entire point of the move."""
    dealt = _roll(4)
    assert all(later > earlier for earlier, later in zip(dealt, dealt[1:], strict=False)), dealt
    assert dealt[3] >= 6 * dealt[0], f"four hits should be roughly eight times the first: {dealt}"


def test_rollout_commits_the_user_until_the_run_ends():
    attacker, _, state = _rolling_duel()
    step(state, {0: USE_FIRST, 1: USE_SPLASH_4})
    assert ExtraStatus.LOCKED_MOVE in attacker.volatiles
    assert [a.move for a in legal_actions(state, 0)] == [MoveSlot.FIRST]  # no switching out of a rollout


def test_a_finished_rollout_leaves_no_fatigue_and_starts_over():
    """Unlike Outrage, a Rollout that runs its course carries no confusion — and the ramp resets."""
    attacker, defender, state = _rolling_duel()
    for _ in range(ROLLING_LOCK_TURNS):
        defender.live_stats.HP = defender.stat_totals.HP
        step(state, {0: USE_FIRST, 1: USE_SPLASH_4})
    assert ExtraStatus.CONFUSION not in attacker.volatiles
    assert ExtraStatus.LOCKED_MOVE not in attacker.volatiles
    assert attacker.rolling_hits == 0


def test_a_different_move_resets_the_rollout_ramp():
    attacker, _, state = _rolling_duel()
    step(state, {0: USE_FIRST, 1: USE_SPLASH_4})
    attacker.volatiles.pop(ExtraStatus.LOCKED_MOVE)  # let it choose freely again
    attacker.locked_slot = None
    step(state, {0: USE_TACKLE_2, 1: USE_SPLASH_4})
    assert attacker.rolling_hits == 0
