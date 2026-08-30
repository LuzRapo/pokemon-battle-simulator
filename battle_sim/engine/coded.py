"""Engine implementations for CodedEffect move behaviours (PS onHit code, not move data)."""

from battle_sim.mechanics.battle import BattleState, effective_weather
from battle_sim.mechanics.effects import rewire_active
from battle_sim.mechanics.log import BattleLog
from battle_sim.mechanics.stages import apply_stage_changes
from battle_sim.models.log_events import (
    AbilitiesSwapped,
    AllStatsReset,
    CourtChanged,
    FutureAttackQueued,
    Healed,
    ItemRemoved,
    ItemsSwapped,
    MoveFailed,
    Revived,
    StatusCleared,
    StatusInflicted,
    VolatileInflicted,
    WishMade,
)
from battle_sim.models.moves import CodedEffect, CodedMoveKind, Move
from battle_sim.models.pokemon import Pokemon
from battle_sim.utils import Ability, ExtraStatus, Item, Stats, Status, Type, Weather

_SUB_BLOCKED_KINDS = frozenset(
    {CodedMoveKind.PAIN_SPLIT, CodedMoveKind.STRENGTH_SAP, CodedMoveKind.KNOCK_OFF_ITEM, CodedMoveKind.TRICK}
)


def _apply_coded(  # noqa: C901 — a flat dispatch over every coded move kind
    effect: CodedEffect,
    move: Move,
    attacker: Pokemon,
    attacker_side_index: int,
    defender: Pokemon,
    state: BattleState,
    log: BattleLog,
    behind_substitute: bool,
) -> None:
    if behind_substitute and effect.kind in _SUB_BLOCKED_KINDS:
        return
    match effect.kind:
        case CodedMoveKind.REST:
            _rest(attacker, attacker_side_index, log)
        case CodedMoveKind.WEATHER_HEAL:
            _weather_heal(attacker, attacker_side_index, state, log)
        case CodedMoveKind.PAIN_SPLIT:
            _pain_split(attacker, attacker_side_index, defender, log)
        case CodedMoveKind.STRENGTH_SAP:
            _strength_sap(attacker, attacker_side_index, defender, log)
        case CodedMoveKind.BELLY_DRUM:
            _belly_drum(attacker, attacker_side_index, log)
        case CodedMoveKind.HAZE:
            for side in state.sides:
                side.active_pokemon.reset_stat_stages()
            log.add(AllStatsReset())
        case CodedMoveKind.COURT_CHANGE:
            _court_change(state, log)
        case CodedMoveKind.WISH:
            side = state.sides[attacker_side_index]
            if side.wish_turns == 0:
                side.wish_pending = max(1, attacker.stat_totals.HP // 2)
                side.wish_turns = 2
                log.add(WishMade(side=attacker_side_index, pokemon=attacker.nickname))
        case CodedMoveKind.HEALING_WISH:
            state.sides[attacker_side_index].healing_wish_pending = True
            log.add(WishMade(side=attacker_side_index, pokemon=attacker.nickname))
        case CodedMoveKind.CURE_SELF:
            if attacker.status is not Status.NONE:
                attacker.status = Status.NONE
                attacker.status_turns = 0
                log.add(StatusCleared(side=attacker_side_index, pokemon=attacker.nickname, clearance="refreshed"))
        case CodedMoveKind.TIDY_UP:
            _tidy_up(state, log)
        case CodedMoveKind.CURSE:
            _curse(attacker, attacker_side_index, defender, state, log)
        case CodedMoveKind.PERISH_SONG:
            for side_index, side in enumerate(state.sides):
                active = side.active_pokemon
                if not active.is_fainted() and ExtraStatus.PERISH not in active.volatiles:
                    active.volatiles[ExtraStatus.PERISH] = 4
                    log.add(VolatileInflicted(side=side_index, pokemon=active.nickname, volatile=ExtraStatus.PERISH))
        case CodedMoveKind.REVIVAL_BLESSING:
            _revival_blessing(attacker_side_index, state, log)
        case CodedMoveKind.TRANSFORM:
            from battle_sim.engine.transform import transform_into  # lazy: transform pulls in effect rewiring

            transform_into(attacker, attacker_side_index, defender, state, log)
        case CodedMoveKind.SHED_TAIL:
            _shed_tail(attacker, attacker_side_index, state, log)
        case CodedMoveKind.KNOCK_OFF_ITEM:
            _knock_off_item(attacker, attacker_side_index, defender, state, log)
        case CodedMoveKind.TRICK:
            _trick(attacker, attacker_side_index, defender, state, log)
        case CodedMoveKind.SKILL_SWAP:
            _skill_swap(attacker, attacker_side_index, defender, state, log)
        case CodedMoveKind.FUTURE_SIGHT:
            _future_sight(move, attacker, attacker_side_index, state, log)


def _rest(attacker: Pokemon, side_index: int, log: BattleLog) -> None:
    if attacker.live_stats.HP == attacker.stat_totals.HP:
        log.add(MoveFailed())
        return
    attacker.status = Status.SLEEP
    attacker.status_turns = 3  # two full turns asleep
    healed = attacker.apply_healing(attacker.stat_totals.HP)
    log.add(StatusInflicted(side=side_index, pokemon=attacker.nickname, status=Status.SLEEP))
    log.add(Healed(side=side_index, pokemon=attacker.nickname, amount=healed))


def _weather_heal(attacker: Pokemon, side_index: int, state: BattleState, log: BattleLog) -> None:
    weather = effective_weather(state)
    if weather in (Weather.SUN, Weather.HARSH_SUN):
        fraction = (2, 3)
    elif weather is Weather.NONE:
        fraction = (1, 2)
    else:
        fraction = (1, 4)
    healed = attacker.apply_healing(max(1, attacker.stat_totals.HP * fraction[0] // fraction[1]))
    if healed > 0:
        log.add(Healed(side=side_index, pokemon=attacker.nickname, amount=healed))


def _pain_split(attacker: Pokemon, attacker_side_index: int, defender: Pokemon, log: BattleLog) -> None:
    average = (attacker.live_stats.HP + defender.live_stats.HP) // 2
    for pokemon, side_index in ((attacker, attacker_side_index), (defender, 1 - attacker_side_index)):
        delta = average - pokemon.live_stats.HP
        if delta > 0:
            pokemon.apply_healing(delta)
            log.add(Healed(side=side_index, pokemon=pokemon.nickname, amount=delta))
        else:
            pokemon.apply_damage(-delta)


def _strength_sap(attacker: Pokemon, attacker_side_index: int, defender: Pokemon, log: BattleLog) -> None:
    if defender.stat_stages.ATTACK <= -6:
        log.add(MoveFailed())
        return
    healed = attacker.apply_healing(max(1, defender.effective_stat(Stats.ATTACK)))
    if healed > 0:
        log.add(Healed(side=attacker_side_index, pokemon=attacker.nickname, amount=healed))
    apply_stage_changes(
        defender,
        1 - attacker_side_index,
        {Stats.ATTACK: -1},
        log,
        inflicted_by_opponent=True,
        source="move",
        inflictor=attacker,
    )


def _belly_drum(attacker: Pokemon, side_index: int, log: BattleLog) -> None:
    cost = attacker.stat_totals.HP // 2
    if cost >= attacker.live_stats.HP or attacker.stat_stages.ATTACK >= 6:
        log.add(MoveFailed())
        return
    attacker.apply_damage(cost)
    apply_stage_changes(attacker, side_index, {Stats.ATTACK: 12}, log, inflicted_by_opponent=False, source="move")


def _court_change(state: BattleState, log: BattleLog) -> None:
    first, second = state.sides
    first.hazards, second.hazards = second.hazards, first.hazards
    first.screens, second.screens = second.screens, first.screens
    first.tailwind_turns, second.tailwind_turns = second.tailwind_turns, first.tailwind_turns
    log.add(CourtChanged())


def _tidy_up(state: BattleState, log: BattleLog) -> None:
    from battle_sim.engine.field_apply import _clear_hazards  # lazy: field_apply imports upward through items

    for side_index, side in enumerate(state.sides):
        _clear_hazards(side, side_index, log)
        side.active_pokemon.volatiles.pop(ExtraStatus.SUBSTITUTE, None)


def _curse(attacker: Pokemon, attacker_side_index: int, defender: Pokemon, state: BattleState, log: BattleLog) -> None:
    if Type.GHOST in attacker.types:
        if ExtraStatus.CURSE in defender.volatiles or defender.is_fainted():
            log.add(MoveFailed())
            return
        attacker.apply_damage(attacker.stat_totals.HP // 2)
        defender.volatiles[ExtraStatus.CURSE] = 1
        log.add(VolatileInflicted(side=1 - attacker_side_index, pokemon=defender.nickname, volatile=ExtraStatus.CURSE))
        return
    apply_stage_changes(
        attacker,
        attacker_side_index,
        {Stats.ATTACK: 1, Stats.DEFENCE: 1, Stats.SPEED: -1},
        log,
        inflicted_by_opponent=False,
        source="move",
    )


def _shed_tail(attacker: Pokemon, side_index: int, state: BattleState, log: BattleLog) -> None:
    side = state.sides[side_index]
    cost = max(1, attacker.stat_totals.HP // 2)
    has_healthy_bench = any(i != side.active[0] and not p.is_fainted() for i, p in enumerate(side.team))
    if cost >= attacker.live_stats.HP or not has_healthy_bench:
        log.add(MoveFailed())
        return
    attacker.apply_damage(cost)
    side.pending_substitute = attacker.stat_totals.HP // 4  # the passed substitute has normal substitute HP
    side.needs_switch = True
    log.add(VolatileInflicted(side=side_index, pokemon=attacker.nickname, volatile=ExtraStatus.SUBSTITUTE))


def _revival_blessing(side_index: int, state: BattleState, log: BattleLog) -> None:
    side = state.sides[side_index]
    fallen = next((member for member in side.team if member.is_fainted()), None)
    if fallen is None:
        log.add(MoveFailed())
        return
    fallen.apply_healing(max(1, fallen.stat_totals.HP // 2))
    log.add(Revived(side=side_index, pokemon=fallen.nickname))


def _knock_off_item(
    attacker: Pokemon, attacker_side_index: int, defender: Pokemon, state: BattleState, log: BattleLog
) -> None:
    if defender.is_fainted() or defender.item is Item.NONE:
        return
    removed = defender.item
    defender.consume_item()
    rewire_active(state.bus, state.effects, defender)
    log.add(ItemRemoved(side=1 - attacker_side_index, pokemon=defender.nickname, item=removed))


def _trick(attacker: Pokemon, attacker_side_index: int, defender: Pokemon, state: BattleState, log: BattleLog) -> None:
    if attacker.item is Item.NONE and defender.item is Item.NONE:
        log.add(MoveFailed())
        return
    attacker.item, defender.item = defender.item, attacker.item
    rewire_active(state.bus, state.effects, attacker)
    rewire_active(state.bus, state.effects, defender)
    log.add(ItemsSwapped(side=attacker_side_index, pokemon=attacker.nickname))


def _skill_swap(
    attacker: Pokemon, attacker_side_index: int, defender: Pokemon, state: BattleState, log: BattleLog
) -> None:
    if attacker.ability is Ability.NONE and defender.ability is Ability.NONE:
        log.add(MoveFailed())
        return
    attacker.ability, defender.ability = defender.ability, attacker.ability
    rewire_active(state.bus, state.effects, attacker)
    rewire_active(state.bus, state.effects, defender)
    log.add(AbilitiesSwapped(side=attacker_side_index, pokemon=attacker.nickname))


def _future_sight(move: Move, attacker: Pokemon, attacker_side_index: int, state: BattleState, log: BattleLog) -> None:
    """Queue the hit against whichever position was targeted; engine.residuals lands it two turns on.

    Only one can be pending against a position at a time (PS: a second use just fails), so this is a
    no-op — and so adds nothing to the log, which is what makes `_execute_move` report the failure —
    when one is already queued.
    """
    target_side = state.sides[1 - attacker_side_index]
    if target_side.future_sight_turns > 0:
        return
    target_side.future_sight_attacker = attacker
    target_side.future_sight_move = move
    target_side.future_sight_turns = 3
    log.add(FutureAttackQueued(side=attacker_side_index, pokemon=attacker.nickname, move=move.name))
