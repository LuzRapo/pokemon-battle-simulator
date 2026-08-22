from battle_sim.database.loader import get_move
from battle_sim.engine import apply_forced_switch, step
from battle_sim.maths.rng import RNG
from battle_sim.mechanics.battle import BattleState, SideState
from battle_sim.models.actions import Action, ActionType
from battle_sim.models.moves import MoveSet, MoveSlot
from battle_sim.models.pokemon import Pokemon
from battle_sim.models.stats import BaseStats, EVs, IVs
from battle_sim.utils import Ability, Hazards, Item, Nature, Status, Target, Type, Weather

TACKLE = get_move("Tackle")
EMBER = get_move("Ember")
QUICK_ATTACK = get_move("Quick Attack")
SWORDS_DANCE = get_move("Swords Dance")
USE_TACKLE = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.FIRST)
USE_QUICK_ATTACK = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.SECOND)
USE_EMBER = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.THIRD)
USE_SWORDS_DANCE = Action(action=ActionType.USE_MOVE, target=Target.SELF, move=MoveSlot.FOURTH)


def _mk(
    nickname: str,
    types: tuple[Type, Type | None] = (Type.NORMAL, None),
    ability: Ability = Ability.NONE,
    item: Item = Item.NONE,
    base_stats: BaseStats | None = None,
    moves: MoveSet | None = None,
    status: Status = Status.NONE,
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


def _battle(s0: list[Pokemon], s1: list[Pokemon], seed: int = 0) -> BattleState:
    return BattleState(sides=(SideState(team=s0), SideState(team=s1)), rng=RNG(seed=seed))


def test_heavy_duty_boots_holder_ignores_stealth_rock_and_spikes():
    a = _mk("A")
    b, b2 = _mk("B"), _mk("B2", item=Item.HEAVY_DUTY_BOOTS)
    state = _battle([a], [b, b2])
    state.sides[1].hazards[Hazards.STEALTH_ROCK] = 1
    state.sides[1].hazards[Hazards.SPIKES] = 3
    switch = Action(action=ActionType.SWITCH_OUT, switch_in=b2)
    step(state, {0: USE_SWORDS_DANCE, 1: switch})
    assert b2.live_stats.HP == b2.stat_totals.HP


def test_drought_switch_in_sets_sun_and_boosts_own_fire_attack():
    starter = _mk("Starter")
    fire_droughter = _mk("Drought", types=(Type.FIRE, None), ability=Ability.DROUGHT)
    target = _mk("T", types=(Type.NORMAL, None))
    state = _battle([starter, fire_droughter], [target])
    switch = Action(action=ActionType.SWITCH_OUT, switch_in=fire_droughter)
    step(state, {0: switch, 1: USE_SWORDS_DANCE})
    assert state.field.weather is Weather.SUN

    hp_before = target.live_stats.HP
    step(state, {0: USE_EMBER, 1: USE_SWORDS_DANCE})
    sun_damage = hp_before - target.live_stats.HP

    clear_fire = _mk("Fire", types=(Type.FIRE, None))
    clear_target = _mk("T", types=(Type.NORMAL, None))
    state_clear = _battle([clear_fire], [clear_target])
    hp_before_clear = clear_target.live_stats.HP
    step(state_clear, {0: USE_EMBER, 1: USE_SWORDS_DANCE})
    clear_damage = hp_before_clear - clear_target.live_stats.HP

    assert sun_damage > clear_damage


def test_toxic_counter_ramps_over_three_turns_and_resets_on_switch():
    a = _mk("A", status=Status.TOXIC)
    a2 = _mk("A2")
    b = _mk("B")
    state = _battle([a, a2], [b])
    max_hp = a.stat_totals.HP
    losses: list[int] = []
    for _ in range(3):
        before = a.live_stats.HP
        step(state, {0: USE_SWORDS_DANCE, 1: USE_SWORDS_DANCE})
        losses.append(before - a.live_stats.HP)
    assert losses[0] == max(1, max_hp * 1 // 16)
    assert losses[1] == max(1, max_hp * 2 // 16)
    assert losses[2] == max(1, max_hp * 3 // 16)

    switch = Action(action=ActionType.SWITCH_OUT, switch_in=a2)
    step(state, {0: switch, 1: USE_SWORDS_DANCE})
    assert a.status_turns == 0


def test_pivot_move_triggers_forced_switch_and_next_actor_acts_normally():
    u_turn = get_move("U-turn")
    pivot = _mk("Pivot", moves=MoveSet(u_turn, TACKLE, EMBER, SWORDS_DANCE))
    backup = _mk("Backup")
    opp = _mk("Opp")
    state = _battle([pivot, backup], [opp])
    pivot.stat_stages.SPEED = 6
    use_u_turn = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.FIRST)
    step(state, {0: use_u_turn, 1: USE_TACKLE})
    assert state.sides[0].needs_switch is True
    apply_forced_switch(state, 0, Action(action=ActionType.SWITCH_OUT, switch_in=backup))
    assert state.sides[0].needs_switch is False
    assert state.sides[0].active_pokemon is backup


def test_magic_guard_immune_to_sandstorm_chip():
    magic = _mk("Magic", ability=Ability.MAGIC_GUARD)
    sand_setter = _mk("Sand", ability=Ability.SAND_STREAM)
    starter = _mk("Starter")
    state = _battle([starter, sand_setter], [magic])
    switch = Action(action=ActionType.SWITCH_OUT, switch_in=sand_setter)
    step(state, {0: switch, 1: USE_SWORDS_DANCE})
    assert state.field.weather is Weather.SANDSTORM
    assert magic.live_stats.HP == magic.stat_totals.HP


def test_full_battle_runs_to_completion():
    p1_team = [
        _mk("A1", ability=Ability.INTIMIDATE),
        _mk(
            "A2",
            base_stats=BaseStats(HP=80, ATTACK=120, DEFENCE=80, SP_ATTACK=80, SP_DEFENCE=80, SPEED=100),
            item=Item.CHOICE_BAND,
        ),
    ]
    p2_team = [
        _mk("B1", item=Item.FOCUS_SASH),
        _mk("B2", ability=Ability.SPEED_BOOST),
    ]
    state = _battle(p1_team, p2_team, seed=5)

    turn_count = 0
    while state.outcome is None and turn_count < 50:
        for i in (0, 1):
            if state.sides[i].needs_switch or state.sides[i].active_pokemon.is_fainted():
                bench = [
                    p for j, p in enumerate(state.sides[i].team) if j != state.sides[i].active[0] and not p.is_fainted()
                ]
                if bench:
                    apply_forced_switch(state, i, Action(action=ActionType.SWITCH_OUT, switch_in=bench[0]))
        if state.outcome is not None:
            break
        step(state, {0: USE_TACKLE, 1: USE_TACKLE})
        turn_count += 1
    assert state.outcome is not None
