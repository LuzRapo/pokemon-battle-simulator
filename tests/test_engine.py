import pytest

from battle_sim.database.loader import get_move
from battle_sim.engine import apply_forced_switch, step
from battle_sim.maths.rng import RNG
from battle_sim.mechanics.battle import BattleState, FieldState, SideState
from battle_sim.mechanics.events import Event, EventContext, HandlerResult, Payload
from battle_sim.mechanics.log import BattleLog
from battle_sim.models.actions import Action, ActionType
from battle_sim.models.log_events import (
    BattleEnded,
    DamageDealt,
    DisableApplied,
    DisabledBlocked,
    DoesNotAffect,
    Effectiveness,
    Fainted,
    LeechSeedSap,
    MoveFailed,
    MoveMissed,
    MoveUsed,
    MultiHitSummary,
    NoEffect,
    Protected,
    ResidualDamage,
    StatStageChanged,
    StatusInflicted,
    SubstituteAlready,
    SubstituteTooWeak,
    Switched,
    TailwindSet,
    TauntBlocked,
)
from battle_sim.models.moves import MoveSet, MoveSlot
from battle_sim.models.pokemon import Pokemon
from battle_sim.models.stats import BaseStats, EVs, IVs
from battle_sim.models.type_matchups import TypePair
from battle_sim.utils import (
    ExtraStatus,
    Hazards,
    Item,
    Nature,
    Outcome,
    PseudoWeather,
    Stats,
    Status,
    Target,
    Terrain,
    Type,
    Weather,
)

TACKLE = get_move("Tackle")
QUICK_ATTACK = get_move("Quick Attack")
EMBER = get_move("Ember")
SWORDS_DANCE = get_move("Swords Dance")
WILL_O_WISP = get_move("Will-O-Wisp")
ROCK_SLIDE = get_move("Rock Slide")

USE_TACKLE = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.FIRST)
USE_QUICK_ATTACK = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.SECOND)
USE_EMBER = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.THIRD)
USE_SWORDS_DANCE = Action(action=ActionType.USE_MOVE, target=Target.SELF, move=MoveSlot.FOURTH)


def _mk(
    nickname: str = "X",
    types: TypePair = (Type.NORMAL, None),
    level: int = 50,
    base_stats: BaseStats | None = None,
    moves: MoveSet | None = None,
    status: Status = Status.NONE,
    spe_stage: int = 0,
    nature: Nature = Nature.HARDY,
    item: Item = Item.NONE,
) -> Pokemon:
    if base_stats is None:
        base_stats = BaseStats(HP=100, ATTACK=100, DEFENCE=100, SP_ATTACK=100, SP_DEFENCE=100, SPEED=100)
    if moves is None:
        moves = MoveSet(TACKLE, QUICK_ATTACK, EMBER, SWORDS_DANCE)
    pokemon = Pokemon(
        name="TestMon",
        nickname=nickname,
        level=level,
        base_stats=base_stats,
        effort_values=EVs(),
        individual_values=IVs(),
        types=types,
        moves=moves,
        nature=nature,
        status=status,
        item=item,
    )
    pokemon.stat_stages.SPEED = spe_stage
    return pokemon


def _battle(
    side0_team: list[Pokemon], side1_team: list[Pokemon], seed: int = 0, field: FieldState | None = None
) -> BattleState:
    return BattleState(
        sides=(SideState(team=side0_team), SideState(team=side1_team)),
        rng=RNG(seed=seed),
        field=field or FieldState(),
    )


def test_phazed_out_pokemon_takes_its_queued_move_with_it():
    """Double-phaze turns: the dragged-in replacement must not execute the departed mon's action."""
    whirlwind, roar = get_move("Whirlwind"), get_move("Roar")
    victim = _mk("V", moves=MoveSet(whirlwind, TACKLE, EMBER, SWORDS_DANCE))
    teammate = _mk("V2")
    phazer = _mk("P", spe_stage=2, moves=MoveSet(roar, TACKLE, EMBER, SWORDS_DANCE))
    state = _battle([victim, teammate], [phazer])
    use_whirlwind = Action(action=ActionType.USE_MOVE, target=whirlwind.target, move=MoveSlot.FIRST)
    use_roar = Action(action=ActionType.USE_MOVE, target=roar.target, move=MoveSlot.FIRST)
    step(state, {0: use_whirlwind, 1: use_roar})
    assert state.sides[0].active_pokemon is teammate  # the faster Roar landed before the victim moved
    assert all(teammate.pp[slot] == move.pp for slot in MoveSlot if (move := teammate.moves[slot]) is not None)
    assert phazer.live_stats.HP == phazer.stat_totals.HP  # nobody executed the victim's queued slot


def test_turn_counter_advances():
    state = _battle([_mk("A")], [_mk("B")])
    assert state.turn == 0
    step(state, {0: USE_TACKLE, 1: USE_TACKLE})
    assert state.turn == 1
    step(state, {0: USE_TACKLE, 1: USE_TACKLE})
    assert state.turn == 2


def test_damage_move_reduces_target_hp():
    a, b = _mk("A"), _mk("B")
    state = _battle([a], [b])
    hp_before = b.live_stats.HP
    step(state, {0: USE_TACKLE, 1: USE_TACKLE})
    assert hp_before > b.live_stats.HP


def test_faster_pokemon_acts_first():
    fast = _mk("Fast", spe_stage=2)
    slow = _mk("Slow", base_stats=BaseStats(HP=1, ATTACK=200, DEFENCE=1, SP_ATTACK=1, SP_DEFENCE=1, SPEED=100))
    state = _battle([fast], [slow])
    step(state, {0: USE_TACKLE, 1: USE_TACKLE})
    assert slow.is_fainted()
    assert state.outcome is Outcome.P1_WIN


def test_ko_sets_outcome_p1_wins():
    attacker = _mk(
        "Strong", base_stats=BaseStats(HP=100, ATTACK=200, DEFENCE=100, SP_ATTACK=100, SP_DEFENCE=100, SPEED=200)
    )
    defender = _mk("Weak", base_stats=BaseStats(HP=1, ATTACK=10, DEFENCE=1, SP_ATTACK=10, SP_DEFENCE=1, SPEED=10))
    state = _battle([attacker], [defender])
    step(state, {0: USE_TACKLE, 1: USE_TACKLE})
    assert defender.is_fainted()
    assert state.outcome is Outcome.P1_WIN


def test_ko_sets_outcome_p2_wins():
    weak = _mk("Weak", base_stats=BaseStats(HP=1, ATTACK=10, DEFENCE=1, SP_ATTACK=10, SP_DEFENCE=1, SPEED=10))
    strong = _mk(
        "Strong", base_stats=BaseStats(HP=100, ATTACK=200, DEFENCE=100, SP_ATTACK=100, SP_DEFENCE=100, SPEED=200)
    )
    state = _battle([weak], [strong])
    step(state, {0: USE_TACKLE, 1: USE_TACKLE})
    assert state.outcome is Outcome.P2_WIN


def test_switch_changes_active():
    a, a2 = _mk("A"), _mk("A2")
    b = _mk("B")
    state = _battle([a, a2], [b])
    switch_action = Action(action=ActionType.SWITCH_OUT, switch_in=a2)
    step(state, {0: switch_action, 1: USE_TACKLE})
    assert state.sides[0].active_pokemon is a2


def test_switch_resets_stat_stages():
    a, a2 = _mk("A"), _mk("A2")
    b = _mk("B")
    a.stat_stages.ATTACK = 4
    state = _battle([a, a2], [b])
    switch_action = Action(action=ActionType.SWITCH_OUT, switch_in=a2)
    step(state, {0: switch_action, 1: USE_TACKLE})
    assert a.stat_stages.ATTACK == 0


def test_switch_clears_volatiles():
    a, a2 = _mk("A"), _mk("A2")
    b = _mk("B")
    a.volatiles[ExtraStatus.CONFUSION] = 3
    state = _battle([a, a2], [b])
    switch_action = Action(action=ActionType.SWITCH_OUT, switch_in=a2)
    step(state, {0: switch_action, 1: USE_TACKLE})
    assert a.volatiles == {}


def test_status_move_inflicts_burn():
    a = _mk("A", moves=MoveSet(WILL_O_WISP, TACKLE, EMBER, SWORDS_DANCE))
    b = _mk("B")
    state = _battle([a], [b], seed=1)
    will_o_wisp = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.FIRST)
    step(state, {0: will_o_wisp, 1: USE_TACKLE})
    assert b.status is Status.BURN


def test_stat_change_move_raises_attack():
    a = _mk("A")
    b = _mk("B")
    state = _battle([a], [b])
    step(state, {0: USE_SWORDS_DANCE, 1: USE_TACKLE})
    assert a.stat_stages.ATTACK == 2


def test_burn_damages_attacker_at_turn_end():
    a = _mk("A", status=Status.BURN)
    b = _mk("B")
    state = _battle([a], [b])
    max_hp = a.stat_totals.HP
    hp_before = a.live_stats.HP
    step(state, {0: USE_SWORDS_DANCE, 1: USE_TACKLE})
    burn_chip = max(1, max_hp // 16)
    assert hp_before - a.live_stats.HP >= burn_chip


def test_poison_damages_attacker_at_turn_end():
    a = _mk("A", status=Status.POISON)
    b = _mk("B")
    state = _battle([a], [b])
    max_hp = a.stat_totals.HP
    hp_before = a.live_stats.HP
    step(state, {0: USE_SWORDS_DANCE, 1: USE_SWORDS_DANCE})
    assert hp_before - a.live_stats.HP == max(1, max_hp // 8)


def test_weather_counts_down_and_expires():
    a, b = _mk("A"), _mk("B")
    state = _battle([a], [b], field=FieldState(weather=Weather.RAIN, weather_turns_left=2))
    step(state, {0: USE_SWORDS_DANCE, 1: USE_SWORDS_DANCE})
    assert state.field.weather is Weather.RAIN
    assert state.field.weather_turns_left == 1
    step(state, {0: USE_SWORDS_DANCE, 1: USE_SWORDS_DANCE})
    # read through a widened local: mypy's narrowing from the assert above can't see step() mutate the field
    weather_after: Weather = state.field.weather
    assert weather_after is Weather.NONE
    assert state.field.weather_turns_left == 0


def test_terrain_counts_down():
    a, b = _mk("A"), _mk("B")
    state = _battle([a], [b], field=FieldState(terrain=Terrain.GRASSY, terrain_turns_left=3))
    step(state, {0: USE_SWORDS_DANCE, 1: USE_SWORDS_DANCE})
    assert state.field.terrain_turns_left == 2


def test_toxic_damages_increases_each_turn():
    a = _mk("A", status=Status.TOXIC)
    b = _mk("B")
    state = _battle([a], [b])
    max_hp = a.stat_totals.HP
    hp_start = a.live_stats.HP

    step(state, {0: USE_SWORDS_DANCE, 1: USE_SWORDS_DANCE})
    first_tick = hp_start - a.live_stats.HP
    assert first_tick == max(1, max_hp * 1 // 16)

    hp_after_1 = a.live_stats.HP
    step(state, {0: USE_SWORDS_DANCE, 1: USE_SWORDS_DANCE})
    second_tick = hp_after_1 - a.live_stats.HP
    assert second_tick == max(1, max_hp * 2 // 16)


def test_toxic_counter_resets_on_switch_out():
    a = _mk("A", status=Status.TOXIC)
    a2 = _mk("A2")
    a.status_turns = 5
    state = _battle([a, a2], [_mk("B")])
    switch_action = Action(action=ActionType.SWITCH_OUT, switch_in=a2)
    step(state, {0: switch_action, 1: USE_SWORDS_DANCE})
    assert a.status_turns == 0


def test_sleep_initial_counter_is_one_to_three():
    from battle_sim.engine.status_apply import _apply_status
    from battle_sim.models.moves import InflictStatusEffect

    durations: set[int] = set()
    for seed in range(100):
        defender = _mk("B")
        rng = RNG(seed=seed)
        _apply_status(
            InflictStatusEffect(status=Status.SLEEP, probability=1.0),
            defender,
            1,
            rng,
            BattleLog(),
        )
        durations.add(defender.status_turns)
    assert durations == {1, 2, 3}


def test_sleep_pokemon_wakes_when_counter_hits_zero():
    sleeper = _mk("S", status=Status.SLEEP)
    sleeper.status_turns = 1
    state = _battle([sleeper], [_mk("B")])
    step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
    assert sleeper.status is Status.NONE
    assert sleeper.status_turns == 0


def test_sleep_pokemon_stays_asleep_when_counter_positive():
    sleeper = _mk("S", status=Status.SLEEP)
    sleeper.status_turns = 3
    state = _battle([sleeper], [_mk("B")])
    step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
    assert sleeper.status is Status.SLEEP
    assert sleeper.status_turns == 2


def test_fire_type_cannot_be_burned():
    fire_def = _mk("FireDef", types=(Type.FIRE, None))
    state = _battle([_mk("A", moves=MoveSet(WILL_O_WISP, TACKLE, EMBER, SWORDS_DANCE), spe_stage=6)], [fire_def])
    will_o_wisp = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.FIRST)
    step(state, {0: will_o_wisp, 1: USE_SWORDS_DANCE})
    assert fire_def.status is Status.NONE


def test_electric_type_cannot_be_paralyzed():
    thunder_wave = get_move("Thunder Wave")
    attacker = _mk("A", moves=MoveSet(thunder_wave, TACKLE, EMBER, SWORDS_DANCE), spe_stage=6)
    electric_def = _mk("E", types=(Type.ELECTRIC, None))
    state = _battle([attacker], [electric_def])
    use_tw = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.FIRST)
    step(state, {0: use_tw, 1: USE_SWORDS_DANCE})
    assert electric_def.status is Status.NONE


def test_poison_type_cannot_be_poisoned():
    toxic_move = get_move("Toxic")
    attacker = _mk("A", moves=MoveSet(toxic_move, TACKLE, EMBER, SWORDS_DANCE), spe_stage=6)
    poison_def = _mk("P", types=(Type.POISON, None))
    state = _battle([attacker], [poison_def])
    use_toxic = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.FIRST)
    step(state, {0: use_toxic, 1: USE_SWORDS_DANCE})
    assert poison_def.status is Status.NONE


def test_steel_type_cannot_be_poisoned():
    toxic_move = get_move("Toxic")
    attacker = _mk("A", moves=MoveSet(toxic_move, TACKLE, EMBER, SWORDS_DANCE), spe_stage=6)
    steel_def = _mk("S", types=(Type.STEEL, None))
    state = _battle([attacker], [steel_def])
    use_toxic = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.FIRST)
    step(state, {0: use_toxic, 1: USE_SWORDS_DANCE})
    assert steel_def.status is Status.NONE


def test_ice_type_cannot_be_frozen():
    ice_beam = get_move("Ice Beam")
    attacker = _mk("A", moves=MoveSet(ice_beam, TACKLE, EMBER, SWORDS_DANCE), spe_stage=6)
    ice_def = _mk("I", types=(Type.ICE, None))
    state = _battle([attacker], [ice_def])
    for seed in range(50):
        fresh_attacker = _mk("A", moves=MoveSet(ice_beam, TACKLE, EMBER, SWORDS_DANCE), spe_stage=6)
        fresh_def = _mk("I", types=(Type.ICE, None))
        state = _battle([fresh_attacker], [fresh_def], seed=seed)
        use_ice = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.FIRST)
        step(state, {0: use_ice, 1: USE_SWORDS_DANCE})
        assert fresh_def.status is not Status.FREEZE, f"Ice-type got frozen at seed {seed}"


def test_sandstorm_chips_non_immune_types():
    a = _mk("A", types=(Type.NORMAL, None))
    b = _mk("B", types=(Type.NORMAL, None))
    state = _battle([a], [b], field=FieldState(weather=Weather.SANDSTORM, weather_turns_left=4))
    max_hp = a.stat_totals.HP
    step(state, {0: USE_SWORDS_DANCE, 1: USE_SWORDS_DANCE})
    expected = max(1, max_hp // 16)
    assert max_hp - a.live_stats.HP == expected
    assert max_hp - b.live_stats.HP == expected


def test_pivot_move_sets_needs_switch_flag():
    u_turn = get_move("U-turn")
    attacker = _mk("A", moves=MoveSet(u_turn, TACKLE, EMBER, SWORDS_DANCE), spe_stage=6)
    attacker2 = _mk("A2")
    state = _battle([attacker, attacker2], [_mk("B")])
    use_u_turn = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.FIRST)
    step(state, {0: use_u_turn, 1: USE_SWORDS_DANCE})
    assert state.sides[0].needs_switch is True


def test_pivot_move_does_not_set_flag_if_no_bench():
    u_turn = get_move("U-turn")
    attacker = _mk("A", moves=MoveSet(u_turn, TACKLE, EMBER, SWORDS_DANCE), spe_stage=6)
    state = _battle([attacker], [_mk("B")])
    use_u_turn = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.FIRST)
    step(state, {0: use_u_turn, 1: USE_SWORDS_DANCE})
    assert state.sides[0].needs_switch is False


def test_apply_forced_switch_clears_needs_switch_flag():
    attacker = _mk("A")
    attacker2 = _mk("A2")
    state = _battle([attacker, attacker2], [_mk("B")])
    state.sides[0].needs_switch = True
    apply_forced_switch(state, 0, Action(action=ActionType.SWITCH_OUT, switch_in=attacker2))
    assert state.sides[0].needs_switch is False


def test_stealth_rock_set_via_move():
    stealth_rock = get_move("Stealth Rock")
    attacker = _mk("A", moves=MoveSet(stealth_rock, TACKLE, EMBER, SWORDS_DANCE), spe_stage=6)
    state = _battle([attacker], [_mk("B")])
    use_sr = Action(action=ActionType.USE_MOVE, target=Target.OPPONENT_SIDE, move=MoveSlot.FIRST)
    step(state, {0: use_sr, 1: USE_SWORDS_DANCE})
    assert Hazards.STEALTH_ROCK in state.sides[1].hazards


def test_stealth_rock_damages_incoming_pokemon_by_rock_effectiveness():
    a, a2 = _mk("A"), _mk("A2", types=(Type.NORMAL, None))
    b = _mk("B")
    state = _battle([a, a2], [b])
    state.sides[0].hazards[Hazards.STEALTH_ROCK] = 1
    switch = Action(action=ActionType.SWITCH_OUT, switch_in=a2)
    step(state, {0: switch, 1: USE_SWORDS_DANCE})
    max_hp = a2.stat_totals.HP
    assert max_hp > a2.live_stats.HP
    assert max_hp - a2.live_stats.HP == max(1, max_hp // 8)


def test_stealth_rock_quadruple_damage_on_bug_flying():
    a, a2 = _mk("A"), _mk("A2", types=(Type.BUG, Type.FLYING))
    state = _battle([a, a2], [_mk("B")])
    state.sides[0].hazards[Hazards.STEALTH_ROCK] = 1
    switch = Action(action=ActionType.SWITCH_OUT, switch_in=a2)
    step(state, {0: switch, 1: USE_SWORDS_DANCE})
    max_hp = a2.stat_totals.HP
    assert max_hp - a2.live_stats.HP == max(1, max_hp * 4 // 8)


def test_spikes_stack_up_to_three_layers():
    spikes = get_move("Spikes")
    attacker = _mk("A", moves=MoveSet(spikes, TACKLE, EMBER, SWORDS_DANCE), spe_stage=6)
    state = _battle([attacker], [_mk("B")])
    use_spikes = Action(action=ActionType.USE_MOVE, target=Target.OPPONENT_SIDE, move=MoveSlot.FIRST)
    step(state, {0: use_spikes, 1: USE_SWORDS_DANCE})
    step(state, {0: use_spikes, 1: USE_SWORDS_DANCE})
    step(state, {0: use_spikes, 1: USE_SWORDS_DANCE})
    assert state.sides[1].hazards[Hazards.SPIKES] == 3
    log = step(state, {0: use_spikes, 1: USE_SWORDS_DANCE})
    assert any(isinstance(entry, MoveFailed) for entry in log)
    assert state.sides[1].hazards[Hazards.SPIKES] == 3


def test_spikes_damage_scales_with_layers():
    a = _mk("A")
    b, b2 = _mk("B"), _mk("B2")
    state = _battle([a], [b, b2])
    state.sides[1].hazards[Hazards.SPIKES] = 2
    switch = Action(action=ActionType.SWITCH_OUT, switch_in=b2)
    step(state, {0: USE_SWORDS_DANCE, 1: switch})
    max_hp = b2.stat_totals.HP
    assert max_hp - b2.live_stats.HP == max(1, max_hp // 6)


def test_spikes_do_not_affect_flying_types():
    a = _mk("A")
    b, b2 = _mk("B"), _mk("B2", types=(Type.NORMAL, Type.FLYING))
    state = _battle([a], [b, b2])
    state.sides[1].hazards[Hazards.SPIKES] = 3
    switch = Action(action=ActionType.SWITCH_OUT, switch_in=b2)
    step(state, {0: USE_SWORDS_DANCE, 1: switch})
    assert b2.live_stats.HP == b2.stat_totals.HP


def test_toxic_spikes_one_layer_inflicts_poison():
    a = _mk("A")
    b, b2 = _mk("B"), _mk("B2")
    state = _battle([a], [b, b2])
    state.sides[1].hazards[Hazards.TOXIC_SPIKES] = 1
    switch = Action(action=ActionType.SWITCH_OUT, switch_in=b2)
    step(state, {0: USE_SWORDS_DANCE, 1: switch})
    assert b2.status is Status.POISON


def test_toxic_spikes_two_layers_inflicts_toxic():
    a = _mk("A")
    b, b2 = _mk("B"), _mk("B2")
    state = _battle([a], [b, b2])
    state.sides[1].hazards[Hazards.TOXIC_SPIKES] = 2
    switch = Action(action=ActionType.SWITCH_OUT, switch_in=b2)
    step(state, {0: USE_SWORDS_DANCE, 1: switch})
    assert b2.status is Status.TOXIC


def test_poison_type_absorbs_toxic_spikes():
    a = _mk("A")
    b, b2 = _mk("B"), _mk("B2", types=(Type.POISON, None))
    state = _battle([a], [b, b2])
    state.sides[1].hazards[Hazards.TOXIC_SPIKES] = 2
    switch = Action(action=ActionType.SWITCH_OUT, switch_in=b2)
    step(state, {0: USE_SWORDS_DANCE, 1: switch})
    assert Hazards.TOXIC_SPIKES not in state.sides[1].hazards
    assert b2.status is Status.NONE


def test_steel_type_immune_to_toxic_spikes_status():
    a = _mk("A")
    b, b2 = _mk("B"), _mk("B2", types=(Type.STEEL, None))
    state = _battle([a], [b, b2])
    state.sides[1].hazards[Hazards.TOXIC_SPIKES] = 2
    switch = Action(action=ActionType.SWITCH_OUT, switch_in=b2)
    step(state, {0: USE_SWORDS_DANCE, 1: switch})
    assert b2.status is Status.NONE
    assert state.sides[1].hazards[Hazards.TOXIC_SPIKES] == 2


def test_sticky_web_drops_speed_of_grounded_incoming():
    a = _mk("A")
    b, b2 = _mk("B"), _mk("B2")
    state = _battle([a], [b, b2])
    state.sides[1].hazards[Hazards.STICKY_WEB] = 1
    switch = Action(action=ActionType.SWITCH_OUT, switch_in=b2)
    step(state, {0: USE_SWORDS_DANCE, 1: switch})
    assert b2.stat_stages.SPEED == -1


def test_sticky_web_does_not_affect_flying():
    a = _mk("A")
    b, b2 = _mk("B"), _mk("B2", types=(Type.NORMAL, Type.FLYING))
    state = _battle([a], [b, b2])
    state.sides[1].hazards[Hazards.STICKY_WEB] = 1
    switch = Action(action=ActionType.SWITCH_OUT, switch_in=b2)
    step(state, {0: USE_SWORDS_DANCE, 1: switch})
    assert b2.stat_stages.SPEED == 0


def test_sandstorm_does_not_chip_rock_ground_steel():
    rock = _mk("Rock", types=(Type.ROCK, None))
    ground = _mk("Ground", types=(Type.GROUND, None))
    state = _battle([rock], [ground], field=FieldState(weather=Weather.SANDSTORM, weather_turns_left=4))
    step(state, {0: USE_SWORDS_DANCE, 1: USE_SWORDS_DANCE})
    assert rock.live_stats.HP == rock.stat_totals.HP
    assert ground.live_stats.HP == ground.stat_totals.HP


def test_pseudo_weather_counts_down():
    a, b = _mk("A"), _mk("B")
    state = _battle([a], [b], field=FieldState(pseudo_weather={PseudoWeather.TRICK_ROOM: 2}))
    step(state, {0: USE_SWORDS_DANCE, 1: USE_SWORDS_DANCE})
    assert state.field.pseudo_weather[PseudoWeather.TRICK_ROOM] == 1
    step(state, {0: USE_SWORDS_DANCE, 1: USE_SWORDS_DANCE})
    assert PseudoWeather.TRICK_ROOM not in state.field.pseudo_weather


def test_screens_count_down_and_expire():
    a, b = _mk("A"), _mk("B")
    side0 = SideState(team=[a], screens={Hazards.REFLECT: 2})
    side1 = SideState(team=[b])
    state = BattleState(sides=(side0, side1), rng=RNG(seed=0))
    step(state, {0: USE_SWORDS_DANCE, 1: USE_SWORDS_DANCE})
    assert state.sides[0].screens[Hazards.REFLECT] == 1
    step(state, {0: USE_SWORDS_DANCE, 1: USE_SWORDS_DANCE})
    assert Hazards.REFLECT not in state.sides[0].screens


def test_tailwind_counts_down():
    a, b = _mk("A"), _mk("B")
    side0 = SideState(team=[a], tailwind_turns=3)
    side1 = SideState(team=[b])
    state = BattleState(sides=(side0, side1), rng=RNG(seed=0))
    step(state, {0: USE_SWORDS_DANCE, 1: USE_SWORDS_DANCE})
    assert state.sides[0].tailwind_turns == 2


def test_fainted_pokemon_skips_move_action():
    a = _mk("A")
    b = _mk("B")
    a.live_stats.HP = 0
    state = _battle([a], [b])
    hp_before = b.live_stats.HP
    step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
    assert hp_before == b.live_stats.HP
    assert state.outcome is Outcome.P2_WIN


def test_fainted_pokemon_can_switch_out():
    a = _mk("A")
    a2 = _mk("A2")
    b = _mk("B")
    state = _battle([a, a2], [b])
    state.sides[0].active_pokemon.live_stats.HP = 0
    switch_action = Action(action=ActionType.SWITCH_OUT, switch_in=a2)
    step(state, {0: switch_action, 1: USE_SWORDS_DANCE})
    assert state.sides[0].active_pokemon is a2
    assert not state.sides[0].active_pokemon.is_fainted()
    assert state.outcome is None


def test_apply_forced_switch_swaps_active_without_advancing_turn():
    a = _mk("A")
    a2 = _mk("A2")
    state = _battle([a, a2], [_mk("B")])
    state.sides[0].active_pokemon.live_stats.HP = 0
    turn_before = state.turn
    apply_forced_switch(state, 0, Action(action=ActionType.SWITCH_OUT, switch_in=a2))
    assert state.sides[0].active_pokemon is a2
    assert state.turn == turn_before


def test_apply_forced_switch_rejects_non_switch_action():
    state = _battle([_mk("A")], [_mk("B")])
    with pytest.raises(ValueError, match="SWITCH_OUT"):
        apply_forced_switch(state, 0, USE_TACKLE)


def test_multi_hit_logs_each_hit_and_total_count():
    bullet_seed = get_move("Bullet Seed")
    attacker = _mk("A", moves=MoveSet(bullet_seed, TACKLE, EMBER, SWORDS_DANCE))
    bulky = _mk(
        "Bulky",
        base_stats=BaseStats(HP=200, ATTACK=1, DEFENCE=200, SP_ATTACK=1, SP_DEFENCE=200, SPEED=1),
    )
    state = _battle([attacker], [bulky], seed=0)
    use_bs = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.FIRST)
    log = step(state, {0: use_bs, 1: USE_SWORDS_DANCE})
    damage_entries = [entry for entry in log if isinstance(entry, DamageDealt)]
    summary = next(entry for entry in log if isinstance(entry, MultiHitSummary))
    assert 2 <= len(damage_entries) <= 5
    assert summary.hits == len(damage_entries)


def test_multi_hit_stops_early_when_defender_faints():
    bullet_seed = get_move("Bullet Seed")
    attacker = _mk(
        "Strong",
        base_stats=BaseStats(HP=100, ATTACK=200, DEFENCE=100, SP_ATTACK=100, SP_DEFENCE=100, SPEED=200),
        moves=MoveSet(bullet_seed, TACKLE, EMBER, SWORDS_DANCE),
    )
    fragile = _mk("Fragile", base_stats=BaseStats(HP=1, ATTACK=1, DEFENCE=1, SP_ATTACK=1, SP_DEFENCE=1, SPEED=1))
    state = _battle([attacker], [fragile])
    use_bs = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.FIRST)
    log = step(state, {0: use_bs, 1: USE_SWORDS_DANCE})
    damage_entries = [entry for entry in log if isinstance(entry, DamageDealt)]
    assert len(damage_entries) == 1
    assert MultiHitSummary(hits=1) in log.entries


def test_single_hit_move_does_not_log_hit_count():
    state = _battle([_mk("A")], [_mk("B")])
    log = step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
    assert not any(isinstance(entry, MultiHitSummary) for entry in log)


def test_apply_forced_switch_does_not_apply_residuals():
    a = _mk("A", status=Status.BURN)
    a2 = _mk("A2")
    state = _battle([a, a2], [_mk("B")])
    state.sides[0].active_pokemon.live_stats.HP = 0
    bench_hp_before = a2.live_stats.HP
    apply_forced_switch(state, 0, Action(action=ActionType.SWITCH_OUT, switch_in=a2))
    assert bench_hp_before == a2.live_stats.HP


def test_flinch_clears_at_turn_end():
    a, b = _mk("A"), _mk("B")
    a.volatiles[ExtraStatus.FLINCH] = 1
    state = _battle([a], [b])
    step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
    assert ExtraStatus.FLINCH not in a.volatiles


def test_flinch_prevents_acting():
    a, b = _mk("A"), _mk("B")
    a.volatiles[ExtraStatus.FLINCH] = 1
    state = _battle([a], [b])
    hp_before = b.live_stats.HP
    step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
    assert hp_before == b.live_stats.HP


def test_paralysis_full_paralysis_occurs():
    full_paralysis_count = 0
    trials = 400
    for seed in range(trials):
        a = _mk("A", status=Status.PARALYSIS)
        b = _mk("B", base_stats=BaseStats(HP=1000, ATTACK=1, DEFENCE=1000, SP_ATTACK=1, SP_DEFENCE=1000, SPEED=1))
        state = _battle([a], [b], seed=seed)
        hp_before = b.live_stats.HP
        step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
        if hp_before == b.live_stats.HP:
            full_paralysis_count += 1
    rate = full_paralysis_count / trials
    assert 0.15 < rate < 0.40, f"Full-paralysis rate {rate} not near 1/4"


def test_simultaneous_fainted_results_in_draw():
    state = _battle([_mk("A")], [_mk("B")])
    state.sides[0].active_pokemon.live_stats.HP = 0
    state.sides[1].active_pokemon.live_stats.HP = 0
    step(state, {0: USE_SWORDS_DANCE, 1: USE_SWORDS_DANCE})
    assert state.outcome is Outcome.DRAW


def test_switch_resolves_before_move():
    a = _mk("A", base_stats=BaseStats(HP=1, ATTACK=1, DEFENCE=1, SP_ATTACK=1, SP_DEFENCE=1, SPEED=1))
    a2 = _mk("A2", base_stats=BaseStats(HP=200, ATTACK=1, DEFENCE=200, SP_ATTACK=1, SP_DEFENCE=200, SPEED=1))
    b = _mk("B", base_stats=BaseStats(HP=100, ATTACK=200, DEFENCE=100, SP_ATTACK=100, SP_DEFENCE=100, SPEED=200))
    state = _battle([a, a2], [b])
    switch_action = Action(action=ActionType.SWITCH_OUT, switch_in=a2)
    step(state, {0: switch_action, 1: USE_TACKLE})
    assert state.sides[0].active_pokemon is a2
    assert a.live_stats.HP > 0
    assert a2.live_stats.HP < a2.stat_totals.HP


def test_engine_does_not_advance_after_outcome():
    a = _mk("A", base_stats=BaseStats(HP=1, ATTACK=1, DEFENCE=1, SP_ATTACK=1, SP_DEFENCE=1, SPEED=1))
    b = _mk("B", base_stats=BaseStats(HP=100, ATTACK=200, DEFENCE=100, SP_ATTACK=100, SP_DEFENCE=100, SPEED=100))
    state = _battle([a], [b])
    step(state, {0: USE_TACKLE, 1: USE_TACKLE})
    assert state.outcome is Outcome.P2_WIN
    turn_after_outcome = state.turn
    step(state, {0: USE_TACKLE, 1: USE_TACKLE})
    assert state.outcome is Outcome.P2_WIN
    assert state.turn == turn_after_outcome + 1


def test_inaccurate_move_misses_sometimes():
    miss_count = 0
    will_o_wisp = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.FIRST)
    trials = 200
    for seed in range(trials):
        fresh_a = _mk("A", moves=MoveSet(WILL_O_WISP, TACKLE, EMBER, SWORDS_DANCE))
        fresh_b = _mk("B")
        state = _battle([fresh_a], [fresh_b], seed=seed)
        step(state, {0: will_o_wisp, 1: USE_SWORDS_DANCE})
        if fresh_b.status is Status.NONE:
            miss_count += 1
    miss_rate = miss_count / trials
    assert 0.05 < miss_rate < 0.30, f"Will-O-Wisp miss rate {miss_rate} not near 0.15"


def test_resisted_move_still_does_some_damage():
    fire_attacker = _mk("Fire", types=(Type.FIRE, None))
    fire_def = _mk("FireDef", types=(Type.FIRE, None))
    state = _battle([fire_attacker], [fire_def])
    hp_before = fire_def.live_stats.HP
    step(state, {0: USE_EMBER, 1: USE_SWORDS_DANCE})
    assert hp_before > fire_def.live_stats.HP


def test_immune_target_takes_no_damage():
    normal_attacker = _mk("Normal", types=(Type.NORMAL, None))
    ghost_defender = _mk("Ghost", types=(Type.GHOST, None))
    state = _battle([normal_attacker], [ghost_defender])
    hp_before = ghost_defender.live_stats.HP
    step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
    assert hp_before == ghost_defender.live_stats.HP


def test_engine_emits_turn_lifecycle_through_bus():
    a, b = _mk("A"), _mk("B")
    state = _battle([a], [b])
    captured: list[Event] = []

    class Capture:
        name = "Capture"

    def make_handler(event: Event):
        def handler(context: EventContext, payload: Payload) -> HandlerResult:
            captured.append(event)
            return HandlerResult()

        return handler

    owner = Capture()
    for event in (Event.ON_TURN_START, Event.ON_BEFORE_ACTION, Event.ON_AFTER_ACTION, Event.ON_TURN_END):
        state.bus.on(event, make_handler(event), owner=owner)

    step(state, {0: USE_TACKLE, 1: USE_TACKLE})

    assert captured.count(Event.ON_TURN_START) == 1
    assert captured.count(Event.ON_BEFORE_ACTION) == 2
    assert captured.count(Event.ON_AFTER_ACTION) == 2
    assert captured.count(Event.ON_TURN_END) == 1


def test_battle_bus_persists_across_turns():
    state = _battle([_mk("A")], [_mk("B")])
    seen: list[Event] = []

    def handler(context: EventContext, payload: Payload) -> HandlerResult:
        seen.append(Event.ON_TURN_START)
        return HandlerResult()

    state.bus.on(Event.ON_TURN_START, handler)
    step(state, {0: USE_TACKLE, 1: USE_TACKLE})
    step(state, {0: USE_TACKLE, 1: USE_TACKLE})
    assert seen.count(Event.ON_TURN_START) == 2


@pytest.mark.parametrize("seed", range(5))
def test_multi_turn_battle_terminates(seed):
    a = _mk("A", base_stats=BaseStats(HP=80, ATTACK=120, DEFENCE=80, SP_ATTACK=80, SP_DEFENCE=80, SPEED=100))
    b = _mk("B", base_stats=BaseStats(HP=80, ATTACK=120, DEFENCE=80, SP_ATTACK=80, SP_DEFENCE=80, SPEED=100))
    state = _battle([a], [b], seed=seed)
    for _ in range(50):
        if state.outcome is not None:
            break
        step(state, {0: USE_TACKLE, 1: USE_TACKLE})
    assert state.outcome is not None
    assert state.turn < 50


def test_speed_used_at_action_time_not_decision_time():
    fast_a = _mk("A", base_stats=BaseStats(HP=200, ATTACK=50, DEFENCE=200, SP_ATTACK=50, SP_DEFENCE=200, SPEED=200))
    slow_b = _mk("B", base_stats=BaseStats(HP=100, ATTACK=50, DEFENCE=100, SP_ATTACK=50, SP_DEFENCE=100, SPEED=50))
    a_speed = fast_a.effective_stat(Stats.SPEED)
    b_speed = slow_b.effective_stat(Stats.SPEED)
    assert a_speed > b_speed
    state = _battle([fast_a], [slow_b])
    step(state, {0: USE_TACKLE, 1: USE_TACKLE})
    assert slow_b.live_stats.HP < slow_b.stat_totals.HP


def test_step_returns_log_with_used_and_damage_entries():
    state = _battle([_mk("A")], [_mk("B")])
    log = step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
    assert MoveUsed(side=0, pokemon="A", move="Tackle") in log.entries
    assert any(isinstance(entry, DamageDealt) for entry in log)


def test_log_reports_stage_change_with_signed_delta():
    state = _battle([_mk("A")], [_mk("B")])
    log = step(state, {0: USE_SWORDS_DANCE, 1: USE_SWORDS_DANCE})
    assert any(isinstance(entry, StatStageChanged) and entry.stat is Stats.ATTACK and entry.delta == 2 for entry in log)


def test_log_reports_cap_when_stage_cannot_rise_further():
    a = _mk("A")
    a.stat_stages.ATTACK = 6
    state = _battle([a], [_mk("B")])
    log = step(state, {0: USE_SWORDS_DANCE, 1: USE_SWORDS_DANCE})
    assert any(isinstance(entry, StatStageChanged) and entry.delta == 0 and entry.requested > 0 for entry in log)


def test_log_reports_status_application():
    a = _mk("A", moves=MoveSet(WILL_O_WISP, TACKLE, EMBER, SWORDS_DANCE))
    b = _mk("B")
    state = _battle([a], [b], seed=1)
    will_o_wisp = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.FIRST)
    log = step(state, {0: will_o_wisp, 1: USE_SWORDS_DANCE})
    assert any(isinstance(entry, StatusInflicted) and entry.status is Status.BURN for entry in log)


def test_log_reports_faint():
    weak = _mk("Weak", base_stats=BaseStats(HP=1, ATTACK=10, DEFENCE=1, SP_ATTACK=10, SP_DEFENCE=1, SPEED=10))
    strong = _mk(
        "Strong", base_stats=BaseStats(HP=100, ATTACK=200, DEFENCE=100, SP_ATTACK=100, SP_DEFENCE=100, SPEED=200)
    )
    state = _battle([strong], [weak])
    log = step(state, {0: USE_TACKLE, 1: USE_TACKLE})
    assert any(isinstance(entry, Fainted) for entry in log)
    assert BattleEnded(outcome=Outcome.P1_WIN) in log.entries


def test_log_reports_immune_target():
    normal_attacker = _mk("Normal", types=(Type.NORMAL, None))
    ghost_defender = _mk("Ghost", types=(Type.GHOST, None))
    state = _battle([normal_attacker], [ghost_defender])
    log = step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
    assert any(isinstance(entry, NoEffect) for entry in log)


def test_log_reports_type_effectiveness_messages():
    fire_attacker = _mk("Fire", types=(Type.FIRE, None))
    grass_defender = _mk("Grass", types=(Type.GRASS, None))
    state = _battle([fire_attacker], [grass_defender])
    log = step(state, {0: USE_EMBER, 1: USE_SWORDS_DANCE})
    assert Effectiveness(level="super") in log.entries


def test_log_reports_residual_burn_damage():
    a = _mk("A", status=Status.BURN)
    state = _battle([a], [_mk("B")])
    log = step(state, {0: USE_SWORDS_DANCE, 1: USE_SWORDS_DANCE})
    assert any(isinstance(entry, ResidualDamage) and entry.source == "burn" for entry in log)


def test_log_reports_missed_move():
    miss_log_count = 0
    will_o_wisp = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.FIRST)
    trials = 50
    for seed in range(trials):
        fresh_a = _mk("A", moves=MoveSet(WILL_O_WISP, TACKLE, EMBER, SWORDS_DANCE))
        fresh_b = _mk("B")
        state = _battle([fresh_a], [fresh_b], seed=seed)
        log = step(state, {0: will_o_wisp, 1: USE_SWORDS_DANCE})
        if any(isinstance(entry, MoveMissed) for entry in log):
            miss_log_count += 1
    assert miss_log_count >= 3, f"Expected some misses across {trials} seeds, got {miss_log_count}"


def test_log_reports_failed_move_with_no_effects():
    celebrate = get_move("Celebrate")
    a = _mk("A", moves=MoveSet(celebrate, TACKLE, EMBER, SWORDS_DANCE))
    use_celebrate = Action(action=ActionType.USE_MOVE, target=Target.SELF, move=MoveSlot.FIRST)
    state = _battle([a], [_mk("B")])
    log = step(state, {0: use_celebrate, 1: USE_SWORDS_DANCE})
    assert MoveUsed(side=0, pokemon="A", move="Celebrate") in log.entries
    assert any(isinstance(entry, MoveFailed) for entry in log)


def test_apply_forced_switch_returns_log():
    a = _mk("A")
    a2 = _mk("A2")
    state = _battle([a, a2], [_mk("B")])
    state.sides[0].active_pokemon.live_stats.HP = 0
    log = apply_forced_switch(state, 0, Action(action=ActionType.SWITCH_OUT, switch_in=a2))
    assert any(isinstance(entry, Switched) and entry.sent_out == "A2" for entry in log)


def test_fixed_damage_level_deals_user_level_hp():
    seismic_toss = get_move("Seismic Toss")
    attacker = _mk("A", level=42, moves=MoveSet(seismic_toss, TACKLE, EMBER, SWORDS_DANCE))
    defender = _mk("B")
    state = _battle([attacker], [defender])
    use_st = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.FIRST)
    hp_before = defender.live_stats.HP
    step(state, {0: use_st, 1: USE_SWORDS_DANCE})
    assert hp_before - defender.live_stats.HP == 42


def test_fixed_damage_set_amount():
    sonic_boom = get_move("Sonic Boom")
    attacker = _mk("A", moves=MoveSet(sonic_boom, TACKLE, EMBER, SWORDS_DANCE))
    defender = _mk("B")
    state = _battle([attacker], [defender])
    use_sb = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.FIRST)
    hp_before = defender.live_stats.HP
    step(state, {0: use_sb, 1: USE_SWORDS_DANCE})
    assert hp_before - defender.live_stats.HP == 20


def test_fixed_damage_respects_type_immunity():
    seismic_toss = get_move("Seismic Toss")
    attacker = _mk("A", moves=MoveSet(seismic_toss, TACKLE, EMBER, SWORDS_DANCE))
    defender = _mk("B", types=(Type.GHOST, None))
    state = _battle([attacker], [defender])
    use_st = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.FIRST)
    log = step(state, {0: use_st, 1: USE_SWORDS_DANCE})
    assert defender.live_stats.HP == defender.stat_totals.HP
    assert any(isinstance(entry, NoEffect) for entry in log)


def test_weather_setup_move_sets_weather():
    rain_dance = get_move("Rain Dance")
    a = _mk("A", moves=MoveSet(rain_dance, TACKLE, EMBER, SWORDS_DANCE))
    state = _battle([a], [_mk("B")])
    use_rd = Action(action=ActionType.USE_MOVE, target=Target.FIELD, move=MoveSlot.FIRST)
    step(state, {0: use_rd, 1: USE_SWORDS_DANCE})
    assert state.field.weather is Weather.RAIN
    assert state.field.weather_turns_left == 4


def test_terrain_setup_move_sets_terrain():
    grassy = get_move("Grassy Terrain")
    a = _mk("A", moves=MoveSet(grassy, TACKLE, EMBER, SWORDS_DANCE))
    state = _battle([a], [_mk("B")])
    use_g = Action(action=ActionType.USE_MOVE, target=Target.FIELD, move=MoveSlot.FIRST)
    step(state, {0: use_g, 1: USE_SWORDS_DANCE})
    assert state.field.terrain is Terrain.GRASSY


def test_pseudo_weather_setup_move():
    trick_room = get_move("Trick Room")
    a = _mk("A", moves=MoveSet(trick_room, TACKLE, EMBER, SWORDS_DANCE))
    state = _battle([a], [_mk("B")])
    use_tr = Action(action=ActionType.USE_MOVE, target=Target.FIELD, move=MoveSlot.FIRST)
    step(state, {0: use_tr, 1: USE_SWORDS_DANCE})
    assert PseudoWeather.TRICK_ROOM in state.field.pseudo_weather


def test_setup_move_still_applies_when_opponent_is_fainted():
    trick_room = get_move("Trick Room")
    a = _mk("A", moves=MoveSet(trick_room, TACKLE, EMBER, SWORDS_DANCE))
    b = _mk("B")
    state = _battle([a], [b])
    state.sides[1].active_pokemon.live_stats.HP = 0
    use_tr = Action(action=ActionType.USE_MOVE, target=Target.FIELD, move=MoveSlot.FIRST)
    step(state, {0: use_tr, 1: USE_SWORDS_DANCE})
    assert PseudoWeather.TRICK_ROOM in state.field.pseudo_weather


def test_confuse_ray_applies_confusion_volatile():
    confuse_ray = get_move("Confuse Ray")
    a = _mk("A", moves=MoveSet(confuse_ray, TACKLE, EMBER, SWORDS_DANCE))
    b = _mk("B")
    state = _battle([a], [b], seed=2)
    use_cr = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.FIRST)
    step(state, {0: use_cr, 1: USE_SWORDS_DANCE})
    assert ExtraStatus.CONFUSION in b.volatiles
    counter = b.volatiles[ExtraStatus.CONFUSION]
    assert 2 <= counter <= 5


def test_confusion_self_hit_distribution_about_one_third():
    confuse_ray = get_move("Confuse Ray")
    use_cr = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.FIRST)
    self_hits = 0
    trials = 0
    for seed in range(2000):
        a = _mk("A", moves=MoveSet(confuse_ray, TACKLE, EMBER, SWORDS_DANCE))
        b = _mk("B")
        state = _battle([a], [b], seed=seed)
        step(state, {0: use_cr, 1: USE_SWORDS_DANCE})
        if ExtraStatus.CONFUSION not in b.volatiles:
            continue
        b.volatiles[ExtraStatus.CONFUSION] = 5
        hp_before = b.live_stats.HP
        step(state, {0: use_cr, 1: USE_TACKLE})
        if hp_before > b.live_stats.HP:
            self_hits += 1
        trials += 1
    rate = self_hits / trials
    assert 0.25 < rate < 0.42, f"Confusion self-hit rate {rate} outside expected ~1/3 over {trials} trials"


def test_confusion_clears_after_counter_expires():
    a = _mk("A")
    b = _mk("B")
    b.volatiles[ExtraStatus.CONFUSION] = 1
    state = _battle([a], [b], seed=0)
    step(state, {0: USE_SWORDS_DANCE, 1: USE_TACKLE})
    assert ExtraStatus.CONFUSION not in b.volatiles


def test_taunt_blocks_status_moves():
    taunt = get_move("Taunt")
    a = _mk("A", moves=MoveSet(taunt, TACKLE, EMBER, SWORDS_DANCE), spe_stage=2)
    b = _mk("B")
    state = _battle([a], [b])
    use_taunt = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.FIRST)
    log = step(state, {0: use_taunt, 1: USE_SWORDS_DANCE})
    assert ExtraStatus.TAUNT in b.volatiles
    assert b.stat_stages.ATTACK == 0
    assert TauntBlocked(side=1, pokemon="B", move="Swords Dance") in log.entries


def test_taunt_allows_damage_moves():
    a = _mk("A")
    b = _mk("B")
    b.volatiles[ExtraStatus.TAUNT] = 3
    state = _battle([a], [b])
    hp_before_a = a.live_stats.HP
    step(state, {0: USE_SWORDS_DANCE, 1: USE_TACKLE})
    assert hp_before_a > a.live_stats.HP


def test_taunt_counter_ticks_down_and_clears():
    a = _mk("A")
    b = _mk("B")
    b.volatiles[ExtraStatus.TAUNT] = 1
    state = _battle([a], [b])
    step(state, {0: USE_SWORDS_DANCE, 1: USE_SWORDS_DANCE})
    assert ExtraStatus.TAUNT not in b.volatiles


def test_tailwind_sets_side_condition_and_doubles_speed():
    tailwind = get_move("Tailwind")
    a = _mk("A", moves=MoveSet(tailwind, TACKLE, EMBER, SWORDS_DANCE))
    state = _battle([a], [_mk("B")])
    use_tailwind = Action(action=ActionType.USE_MOVE, target=Target.USER_SIDE, move=MoveSlot.FIRST)
    log = step(state, {0: use_tailwind, 1: USE_SWORDS_DANCE})
    assert TailwindSet(side=0) in log.entries
    assert state.sides[0].tailwind_turns == 3  # set to 4, end-of-turn tick consumed one
    from battle_sim.mechanics.priority import effective_speed

    boosted = effective_speed(a, state.sides[0], state.field)
    assert boosted == a.effective_stat(Stats.SPEED) * 2


def test_tailwind_fails_while_already_active():
    tailwind = get_move("Tailwind")
    a = _mk("A", moves=MoveSet(tailwind, TACKLE, EMBER, SWORDS_DANCE))
    state = _battle([a], [_mk("B")])
    use_tailwind = Action(action=ActionType.USE_MOVE, target=Target.USER_SIDE, move=MoveSlot.FIRST)
    step(state, {0: use_tailwind, 1: USE_SWORDS_DANCE})
    log = step(state, {0: use_tailwind, 1: USE_SWORDS_DANCE})
    assert any(isinstance(entry, MoveFailed) for entry in log)
    assert state.sides[0].tailwind_turns == 2


def test_sand_attack_lowers_accuracy_stage():
    sand_attack = get_move("Sand Attack")
    a = _mk("A", moves=MoveSet(sand_attack, TACKLE, EMBER, SWORDS_DANCE))
    b = _mk("B")
    state = _battle([a], [b])
    use_sand_attack = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.FIRST)
    step(state, {0: use_sand_attack, 1: USE_SWORDS_DANCE})
    assert b.stat_stages.ACCURACY == -1


def test_double_team_raises_evasion_stage():
    double_team = get_move("Double Team")
    a = _mk("A", moves=MoveSet(double_team, TACKLE, EMBER, SWORDS_DANCE))
    state = _battle([a], [_mk("B")])
    use_double_team = Action(action=ActionType.USE_MOVE, target=Target.SELF, move=MoveSlot.FIRST)
    step(state, {0: use_double_team, 1: USE_SWORDS_DANCE})
    assert a.stat_stages.EVASION == 1


def test_accuracy_boosted_past_100_percent_always_hits():
    hydro_pump = get_move("Hydro Pump")
    a = _mk("A", moves=MoveSet(hydro_pump, TACKLE, EMBER, SWORDS_DANCE))
    b = _mk("B")
    a.stat_stages.ACCURACY = 6
    state = _battle([a], [b])
    use_hydro_pump = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.FIRST)
    step(state, {0: use_hydro_pump, 1: USE_SWORDS_DANCE})
    assert b.live_stats.HP < b.stat_totals.HP


def test_leech_seed_saps_target_and_heals_user():
    leech_seed = get_move("Leech Seed")
    a = _mk("A", moves=MoveSet(leech_seed, TACKLE, EMBER, SWORDS_DANCE))
    b = _mk("B")
    state = _battle([a], [b])
    a.live_stats.HP = a.stat_totals.HP // 2
    use_leech_seed = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.FIRST)
    log = step(state, {0: use_leech_seed, 1: USE_SWORDS_DANCE})
    sap = max(1, b.stat_totals.HP // 8)
    assert ExtraStatus.LEECH_SEED in b.volatiles
    assert sap == b.stat_totals.HP - b.live_stats.HP
    assert a.stat_totals.HP // 2 + sap == a.live_stats.HP
    assert any(isinstance(entry, LeechSeedSap) for entry in log)


def test_leech_seed_does_not_affect_grass_types():
    leech_seed = get_move("Leech Seed")
    a = _mk("A", moves=MoveSet(leech_seed, TACKLE, EMBER, SWORDS_DANCE))
    grass = _mk("G", types=(Type.GRASS, None))
    state = _battle([a], [grass])
    use_leech_seed = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.FIRST)
    log = step(state, {0: use_leech_seed, 1: USE_SWORDS_DANCE})
    assert ExtraStatus.LEECH_SEED not in grass.volatiles
    assert any(isinstance(entry, DoesNotAffect) for entry in log)


def test_nightmare_fails_on_awake_target():
    nightmare = get_move("Nightmare")
    a = _mk("A", moves=MoveSet(nightmare, TACKLE, EMBER, SWORDS_DANCE))
    b = _mk("B")
    state = _battle([a], [b])
    use_nightmare = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.FIRST)
    log = step(state, {0: use_nightmare, 1: USE_SWORDS_DANCE})
    assert ExtraStatus.NIGHTMARE not in b.volatiles
    assert any(isinstance(entry, MoveFailed) for entry in log)


def test_nightmare_damages_sleeping_target_and_ends_on_wake():
    nightmare = get_move("Nightmare")
    a = _mk("A", moves=MoveSet(nightmare, TACKLE, EMBER, SWORDS_DANCE))
    sleeper = _mk("S", status=Status.SLEEP)
    sleeper.status_turns = 5
    state = _battle([a], [sleeper])
    use_nightmare = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.FIRST)
    step(state, {0: use_nightmare, 1: USE_SWORDS_DANCE})
    assert ExtraStatus.NIGHTMARE in sleeper.volatiles
    assert max(1, sleeper.stat_totals.HP // 4) == sleeper.stat_totals.HP - sleeper.live_stats.HP
    sleeper.status = Status.NONE
    sleeper.status_turns = 0
    step(state, {0: USE_SWORDS_DANCE, 1: USE_SWORDS_DANCE})
    assert ExtraStatus.NIGHTMARE not in sleeper.volatiles


def test_residual_order_items_recover_before_status_chip():
    # Showdown order: Leftovers (order 5) fires before burn damage (order 10),
    # so a full-HP burned holder nets a 1/16 loss each turn.
    burned = _mk("A", status=Status.BURN, item=Item.LEFTOVERS)
    state = _battle([burned], [_mk("B")])
    step(state, {0: USE_SWORDS_DANCE, 1: USE_SWORDS_DANCE})
    assert max(1, burned.stat_totals.HP // 16) == burned.stat_totals.HP - burned.live_stats.HP


PROTECT = get_move("Protect")
SUBSTITUTE = get_move("Substitute")
USE_PROTECT = Action(action=ActionType.USE_MOVE, target=Target.SELF, move=MoveSlot.FIRST)
USE_SUBSTITUTE = Action(action=ActionType.USE_MOVE, target=Target.SELF, move=MoveSlot.FIRST)


def _mk_with(move, nickname: str = "X", **kwargs) -> Pokemon:
    return _mk(nickname, moves=MoveSet(move, TACKLE, EMBER, SWORDS_DANCE), **kwargs)


def test_protect_blocks_damaging_move():
    a = _mk_with(PROTECT, "A")
    b = _mk("B")
    state = _battle([a], [b])
    log = step(state, {0: USE_PROTECT, 1: USE_TACKLE})
    assert a.live_stats.HP == a.stat_totals.HP
    assert any(isinstance(entry, Protected) for entry in log)


def test_protect_blocks_status_move():
    a = _mk_with(PROTECT, "A")
    b = _mk_with(WILL_O_WISP, "B")
    state = _battle([a], [b])
    use_wow = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.FIRST)
    step(state, {0: USE_PROTECT, 1: use_wow})
    assert a.status is Status.NONE


def test_protect_does_not_block_hazard_setting():
    stealth_rock = get_move("Stealth Rock")
    a = _mk_with(PROTECT, "A")
    b = _mk_with(stealth_rock, "B")
    state = _battle([a], [b])
    use_rocks = Action(action=ActionType.USE_MOVE, target=Target.OPPONENT_SIDE, move=MoveSlot.FIRST)
    step(state, {0: USE_PROTECT, 1: use_rocks})
    assert Hazards.STEALTH_ROCK in state.sides[0].hazards


def test_protect_volatile_clears_at_end_of_turn():
    a = _mk_with(PROTECT, "A")
    state = _battle([a], [_mk("B")])
    step(state, {0: USE_PROTECT, 1: USE_SWORDS_DANCE})
    assert ExtraStatus.PROTECT not in a.volatiles


def test_consecutive_protect_fails_about_two_thirds_of_the_time():
    failures = 0
    trials = 60
    for seed in range(trials):
        a = _mk_with(PROTECT, "A")
        state = _battle([a], [_mk("B")], seed=seed)
        step(state, {0: USE_PROTECT, 1: USE_SWORDS_DANCE})
        log = step(state, {0: USE_PROTECT, 1: USE_SWORDS_DANCE})
        if any(isinstance(entry, MoveFailed) for entry in log):
            failures += 1
    assert 0.5 < failures / trials < 0.85  # expected 2/3


def test_protect_streak_resets_after_other_move():
    a = _mk_with(PROTECT, "A")
    state = _battle([a], [_mk("B")])
    step(state, {0: USE_PROTECT, 1: USE_SWORDS_DANCE})
    step(state, {0: USE_SWORDS_DANCE, 1: USE_SWORDS_DANCE})
    assert a.protect_streak == 0
    log = step(state, {0: USE_PROTECT, 1: USE_SWORDS_DANCE})  # streak 0 -> always succeeds
    assert not any(isinstance(entry, MoveFailed) for entry in log)


def test_substitute_costs_quarter_hp_and_stores_it():
    a = _mk_with(SUBSTITUTE, "A")
    state = _battle([a], [_mk("B")])
    step(state, {0: USE_SUBSTITUTE, 1: USE_SWORDS_DANCE})
    cost = a.stat_totals.HP // 4
    assert a.stat_totals.HP - cost == a.live_stats.HP
    assert a.volatiles[ExtraStatus.SUBSTITUTE] == cost


def test_substitute_fails_when_too_weak():
    a = _mk_with(SUBSTITUTE, "A")
    a.live_stats.HP = a.stat_totals.HP // 4
    state = _battle([a], [_mk("B")])
    log = step(state, {0: USE_SUBSTITUTE, 1: USE_SWORDS_DANCE})
    assert ExtraStatus.SUBSTITUTE not in a.volatiles
    assert SubstituteTooWeak() in log.entries


def test_substitute_fails_when_already_up():
    a = _mk_with(SUBSTITUTE, "A")
    state = _battle([a], [_mk("B")])
    step(state, {0: USE_SUBSTITUTE, 1: USE_SWORDS_DANCE})
    log = step(state, {0: USE_SUBSTITUTE, 1: USE_SWORDS_DANCE})
    assert any(isinstance(entry, SubstituteAlready) for entry in log)
    assert a.live_stats.HP == a.stat_totals.HP - a.stat_totals.HP // 4  # paid only once


def test_substitute_absorbs_damage_and_breaks():
    a = _mk_with(SUBSTITUTE, "A")
    strong = _mk("S", base_stats=BaseStats(HP=100, ATTACK=200, DEFENCE=100, SP_ATTACK=100, SP_DEFENCE=100, SPEED=1))
    state = _battle([a], [strong])
    step(state, {0: USE_SUBSTITUTE, 1: USE_TACKLE})  # sub goes up first (A faster), tackle breaks it
    assert ExtraStatus.SUBSTITUTE not in a.volatiles
    assert a.live_stats.HP == a.stat_totals.HP - a.stat_totals.HP // 4  # mon untouched beyond the cost


def test_substitute_blocks_status_and_stat_drops():
    a = _mk_with(SUBSTITUTE, "A", spe_stage=2)  # sub must go up before the burn attempt
    b = _mk_with(WILL_O_WISP, "B")
    state = _battle([a], [b])
    use_wow = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.FIRST)
    step(state, {0: USE_SUBSTITUTE, 1: use_wow})
    assert a.status is Status.NONE


def test_sound_moves_bypass_substitute():
    growl = get_move("Growl")
    a = _mk_with(SUBSTITUTE, "A")
    b = _mk_with(growl, "B")
    state = _battle([a], [b])
    use_growl = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.FIRST)
    step(state, {0: USE_SUBSTITUTE, 1: use_growl})
    assert a.stat_stages.ATTACK == -1


def test_recoil_counts_damage_dealt_to_substitute():
    brave_bird = get_move("Brave Bird")
    a = _mk_with(SUBSTITUTE, "A")
    attacker = _mk_with(
        brave_bird, "S", base_stats=BaseStats(HP=100, ATTACK=10, DEFENCE=100, SP_ATTACK=100, SP_DEFENCE=100, SPEED=1)
    )
    state = _battle([a], [attacker])
    use_bb = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.FIRST)
    step(state, {0: USE_SUBSTITUTE, 1: use_bb})
    assert attacker.live_stats.HP < attacker.stat_totals.HP  # took recoil despite hitting the sub


def test_recover_heals_half_max_hp():
    recover = get_move("Recover")
    a = _mk_with(recover, "A")
    a.live_stats.HP = 1
    state = _battle([a], [_mk("B")])
    use_recover = Action(action=ActionType.USE_MOVE, target=Target.SELF, move=MoveSlot.FIRST)
    step(state, {0: use_recover, 1: USE_SWORDS_DANCE})
    assert a.live_stats.HP == 1 + a.stat_totals.HP // 2


def test_recover_fails_at_full_hp():
    recover = get_move("Recover")
    a = _mk_with(recover, "A")
    state = _battle([a], [_mk("B")])
    use_recover = Action(action=ActionType.USE_MOVE, target=Target.SELF, move=MoveSlot.FIRST)
    log = step(state, {0: use_recover, 1: USE_SWORDS_DANCE})
    assert any(isinstance(entry, MoveFailed) for entry in log)


def test_pp_decrements_on_use():
    a = _mk("A")
    state = _battle([a], [_mk("B")])
    starting_pp = a.pp[MoveSlot.FOURTH]
    step(state, {0: USE_SWORDS_DANCE, 1: USE_SWORDS_DANCE})
    assert a.pp[MoveSlot.FOURTH] == starting_pp - 1


def test_struggle_when_all_pp_exhausted():
    a = _mk("A")
    a.pp = dict.fromkeys(a.pp, 0)
    b = _mk("B", types=(Type.GHOST, None))  # Struggle is typeless: hits Ghosts anyway
    state = _battle([a], [b])
    log = step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
    assert MoveUsed(side=0, pokemon="A", move="Struggle") in log.entries
    assert b.live_stats.HP < b.stat_totals.HP
    assert max(1, a.stat_totals.HP // 4) == a.stat_totals.HP - a.live_stats.HP  # struggle recoil


def test_rapid_spin_clears_own_hazards_and_leech_seed():
    rapid_spin = get_move("Rapid Spin")
    a = _mk_with(rapid_spin, "A")
    a.volatiles[ExtraStatus.LEECH_SEED] = 1
    state = _battle([a], [_mk("B")])
    state.sides[0].hazards[Hazards.STEALTH_ROCK] = 1
    state.sides[0].hazards[Hazards.SPIKES] = 2
    state.sides[1].hazards[Hazards.STEALTH_ROCK] = 1
    use_spin = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.FIRST)
    step(state, {0: use_spin, 1: USE_SWORDS_DANCE})
    assert state.sides[0].hazards == {}
    assert state.sides[1].hazards == {Hazards.STEALTH_ROCK: 1}  # opponent's side untouched
    assert ExtraStatus.LEECH_SEED not in a.volatiles
    assert a.stat_stages.SPEED == 1  # the data-driven self speed boost


def test_rapid_spin_blocked_by_ghost_leaves_hazards():
    rapid_spin = get_move("Rapid Spin")
    a = _mk_with(rapid_spin, "A")
    ghost = _mk("G", types=(Type.GHOST, None))
    state = _battle([a], [ghost])
    state.sides[0].hazards[Hazards.STEALTH_ROCK] = 1
    use_spin = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.FIRST)
    step(state, {0: use_spin, 1: USE_SWORDS_DANCE})
    assert state.sides[0].hazards == {Hazards.STEALTH_ROCK: 1}


def test_defog_clears_both_sides_screens_and_terrain():
    defog = get_move("Defog")
    a = _mk_with(defog, "A")
    b = _mk("B")
    state = _battle([a], [b], field=FieldState(terrain=Terrain.GRASSY, terrain_turns_left=5))
    state.sides[0].hazards[Hazards.SPIKES] = 1
    state.sides[1].hazards[Hazards.STEALTH_ROCK] = 1
    state.sides[1].screens[Hazards.REFLECT] = 3
    use_defog = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.FIRST)
    step(state, {0: use_defog, 1: USE_SWORDS_DANCE})
    assert state.sides[0].hazards == {}
    assert state.sides[1].hazards == {}
    assert state.sides[1].screens == {}
    assert state.field.terrain is Terrain.NONE
    assert b.stat_stages.EVASION == -1


def test_focus_energy_raises_crit_rate_to_half():
    crits = 0
    trials = 120
    pumped = _mk("A")
    pumped.volatiles[ExtraStatus.FOCUS_ENERGY] = 1
    defender = _mk("B")
    side = SideState(team=[defender])
    from battle_sim.maths.damage import calculate_damage

    crit_damage = calculate_damage(
        pumped, defender, TACKLE, FieldState(), side, rng=RNG(seed=0), is_crit=True, random_roll=100
    )
    for seed in range(trials):
        rolled = calculate_damage(pumped, defender, TACKLE, FieldState(), side, rng=RNG(seed=seed), random_roll=100)
        if rolled == crit_damage:
            crits += 1
    assert 0.35 < crits / trials < 0.65  # +2 crit stages -> 1/2


def test_encore_forces_last_move():
    encore = get_move("Encore")
    a = _mk_with(encore, "A", spe_stage=2)
    b = _mk("B")
    state = _battle([a], [b])
    step(state, {0: USE_SWORDS_DANCE, 1: USE_SWORDS_DANCE})  # b's last move: Swords Dance (+2)
    use_encore = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.FIRST)
    log = step(state, {0: use_encore, 1: USE_TACKLE})  # b picks Tackle, encore forces Swords Dance
    assert MoveUsed(side=1, pokemon="B", move="Swords Dance") in log.entries
    assert b.stat_stages.ATTACK == 4


def test_encore_fails_before_target_has_moved():
    encore = get_move("Encore")
    a = _mk_with(encore, "A", spe_stage=2)
    b = _mk("B")
    state = _battle([a], [b])
    use_encore = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.FIRST)
    log = step(state, {0: use_encore, 1: USE_SWORDS_DANCE})
    assert any(isinstance(entry, MoveFailed) for entry in log)
    assert ExtraStatus.ENCORE not in b.volatiles


def test_disable_blocks_last_move():
    disable = get_move("Disable")
    a = _mk_with(disable, "A", spe_stage=2)
    b = _mk("B")
    state = _battle([a], [b])
    step(state, {0: USE_SWORDS_DANCE, 1: USE_TACKLE})  # b's last move: Tackle
    use_disable = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.FIRST)
    log = step(state, {0: use_disable, 1: USE_TACKLE})
    assert any(isinstance(entry, DisableApplied) for entry in log)
    assert b.disabled_slot is MoveSlot.FIRST
    hp_before = a.live_stats.HP
    log = step(state, {0: USE_SWORDS_DANCE, 1: USE_TACKLE})
    assert any(isinstance(entry, DisabledBlocked) for entry in log)
    assert hp_before == a.live_stats.HP  # the disabled Tackle never landed


def test_yawn_puts_target_to_sleep_after_one_turn():
    yawn = get_move("Yawn")
    a = _mk_with(yawn, "A", spe_stage=2)
    b = _mk("B")
    state = _battle([a], [b])
    use_yawn = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.FIRST)
    step(state, {0: use_yawn, 1: USE_SWORDS_DANCE})
    assert ExtraStatus.YAWN in b.volatiles
    assert b.status is Status.NONE
    step(state, {0: USE_SWORDS_DANCE, 1: USE_SWORDS_DANCE})
    status_after: Status = b.status  # widened read: mypy's narrowing can't see step() mutate it
    assert status_after is Status.SLEEP
    assert ExtraStatus.YAWN not in b.volatiles


def test_yawn_fails_on_statused_target():
    yawn = get_move("Yawn")
    a = _mk_with(yawn, "A", spe_stage=2)
    b = _mk("B", status=Status.BURN)
    state = _battle([a], [b])
    use_yawn = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.FIRST)
    log = step(state, {0: use_yawn, 1: USE_SWORDS_DANCE})
    assert ExtraStatus.YAWN not in b.volatiles
    assert any(isinstance(entry, MoveFailed) for entry in log)


def test_outrage_locks_in_then_confuses_from_fatigue():
    outrage = get_move("Outrage")
    a = _mk_with(outrage, "A")
    b = _mk("B", base_stats=BaseStats(HP=255, ATTACK=1, DEFENCE=255, SP_ATTACK=1, SP_DEFENCE=255, SPEED=1))
    state = _battle([a], [b], seed=3)
    use_outrage = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.FIRST)
    starting_pp = a.pp[MoveSlot.FIRST]

    step(state, {0: use_outrage, 1: USE_SWORDS_DANCE})
    assert ExtraStatus.LOCKED_MOVE in a.volatiles
    assert a.pp[MoveSlot.FIRST] == starting_pp - 1

    outrage_uses = 1
    for _ in range(3):
        if ExtraStatus.LOCKED_MOVE not in a.volatiles:
            break
        log = step(state, {0: USE_SWORDS_DANCE, 1: USE_SWORDS_DANCE})  # choice ignored: rampage forces Outrage
        if MoveUsed(side=0, pokemon="A", move="Outrage") in log.entries:
            outrage_uses += 1

    assert 2 <= outrage_uses <= 3
    assert ExtraStatus.LOCKED_MOVE not in a.volatiles
    assert ExtraStatus.CONFUSION in a.volatiles  # fatigue
    assert a.pp[MoveSlot.FIRST] == starting_pp - 1  # the rampage spent PP only once


def test_choice_locked_into_empty_move_struggles():
    a = _mk("A", item=Item.CHOICE_BAND)
    b = _mk("B", base_stats=BaseStats(HP=255, ATTACK=1, DEFENCE=255, SP_ATTACK=1, SP_DEFENCE=255, SPEED=1))
    state = _battle([a], [b])
    a.pp[MoveSlot.FIRST] = 1
    step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})  # locks into Tackle, spends its last PP
    assert a.pp[MoveSlot.FIRST] == 0
    log = step(state, {0: USE_TACKLE, 1: USE_SWORDS_DANCE})
    assert MoveUsed(side=0, pokemon="A", move="Struggle") in log.entries


def test_forced_switch_into_lethal_hazards_ends_the_battle():
    a = _mk("A", base_stats=BaseStats(HP=200, ATTACK=1, DEFENCE=200, SP_ATTACK=1, SP_DEFENCE=200, SPEED=1))
    b1 = _mk("B1", types=(Type.FIRE, Type.FLYING))  # 4x weak to Stealth Rock
    b2 = _mk("B2", types=(Type.FIRE, Type.FLYING))
    state = _battle([a], [b1, b2])
    state.sides[1].hazards[Hazards.STEALTH_ROCK] = 1
    b1.live_stats.HP = 0
    b2.live_stats.HP = 1  # rocks will finish it on entry
    log = apply_forced_switch(state, 1, Action(action=ActionType.SWITCH_OUT, switch_in=b2))
    assert b2.is_fainted()
    assert state.outcome is Outcome.P1_WIN
    assert BattleEnded(outcome=Outcome.P1_WIN) in log.entries
