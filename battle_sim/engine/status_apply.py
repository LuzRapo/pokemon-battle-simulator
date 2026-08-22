from battle_sim.maths.rng import RNG
from battle_sim.mechanics.battle import FieldState
from battle_sim.mechanics.log import BattleLog
from battle_sim.mechanics.stages import apply_stage_changes
from battle_sim.models.log_events import (
    DisableApplied,
    DoesNotAffect,
    StatusAlready,
    StatusCleared,
    StatusInflicted,
    SubstituteAlready,
    SubstituteTooWeak,
    VolatileInflicted,
)
from battle_sim.models.moves import InflictStatusEffect, StatStageChangeEffect
from battle_sim.models.pokemon import Pokemon
from battle_sim.utils import Ability, ExtraStatus, Item, Stats, Status, Type, Weather

_STATUS_TYPE_IMMUNITY: dict[Status, frozenset[Type]] = {
    Status.BURN: frozenset({Type.FIRE}),
    Status.PARALYSIS: frozenset({Type.ELECTRIC}),
    Status.FREEZE: frozenset({Type.ICE}),
    Status.POISON: frozenset({Type.POISON, Type.STEEL}),
    Status.TOXIC: frozenset({Type.POISON, Type.STEEL}),
}
_ALL_STATUSES = frozenset(Status) - {Status.NONE}
_SYNCHRONIZE_STATUSES = frozenset({Status.BURN, Status.PARALYSIS, Status.POISON, Status.TOXIC})
_STATUS_ABILITY_IMMUNITY: dict[Ability, frozenset[Status]] = {
    Ability.PURIFYING_SALT: _ALL_STATUSES,
    Ability.WATER_BUBBLE: frozenset({Status.BURN}),
    Ability.THERMAL_EXCHANGE: frozenset({Status.BURN}),
    Ability.LIMBER: frozenset({Status.PARALYSIS}),
    Ability.INSOMNIA: frozenset({Status.SLEEP}),
    Ability.VITAL_SPIRIT: frozenset({Status.SLEEP}),
    Ability.WATER_VEIL: frozenset({Status.BURN}),
    Ability.MAGMA_ARMOR: frozenset({Status.FREEZE}),
    Ability.IMMUNITY: frozenset({Status.POISON, Status.TOXIC}),
}
_STATUS_CURE_ITEMS: dict[Item, frozenset[Status]] = {
    Item.LUM_BERRY: _ALL_STATUSES,
    Item.CHESTO_BERRY: frozenset({Status.SLEEP}),
}
_VOLATILE_TYPE_IMMUNITY: dict[ExtraStatus, frozenset[Type]] = {
    ExtraStatus.LEECH_SEED: frozenset({Type.GRASS}),
}
_VOLATILE_ABILITY_IMMUNITY: dict[ExtraStatus, frozenset[Ability]] = {
    ExtraStatus.FLINCH: frozenset({Ability.INNER_FOCUS}),
    ExtraStatus.CONFUSION: frozenset({Ability.OWN_TEMPO}),
}
_VOLATILE_INITIAL_DURATIONS: dict[ExtraStatus, tuple[int, int]] = {
    ExtraStatus.CONFUSION: (2, 6),  # random.randrange(2, 6) -> 2..5 turns
    ExtraStatus.TAUNT: (3, 4),  # fixed at 3
    ExtraStatus.FLINCH: (1, 2),  # cleared at end of turn anyway
    ExtraStatus.YAWN: (2, 3),  # fixed at 2: drowsy through this turn, asleep at the end of the next
}


def _apply_status(
    effect: InflictStatusEffect,
    target: Pokemon,
    target_index: int,
    rng: RNG,
    log: BattleLog,
    inflictor: Pokemon | None = None,
    field: FieldState | None = None,
) -> None:
    if not rng.roll_chance(effect.probability):
        return
    if isinstance(effect.status, Status):
        _apply_main_status(effect.status, target, target_index, rng, log, inflictor, field)
    elif effect.status is ExtraStatus.SUBSTITUTE:
        _make_substitute(target, target_index, log)
    elif effect.status is ExtraStatus.LOCKED_MOVE:
        if ExtraStatus.LOCKED_MOVE not in target.volatiles:
            target.volatiles[ExtraStatus.LOCKED_MOVE] = rng.random_integer(2, 4)  # 2-3 total uses (PS)
            target.locked_slot = target.last_move_slot
    elif effect.status is ExtraStatus.ENCORE:
        _start_encore(target, target_index, log)
    elif effect.status is ExtraStatus.DISABLE:
        _start_disable(target, target_index, log)
    elif effect.status not in target.volatiles:
        _apply_volatile(effect.status, target, target_index, rng, log)


def _reflect_synchronize(
    status: Status,
    target: Pokemon,
    target_index: int,
    inflictor: Pokemon | None,
    rng: RNG,
    log: BattleLog,
    field: FieldState | None,
) -> None:
    if inflictor is None or target.ability is not Ability.SYNCHRONIZE:
        return
    if status not in _SYNCHRONIZE_STATUSES or inflictor.status is not Status.NONE:
        return
    # inflictor=None: the mirrored status doesn't itself re-trigger Synchronize or Poison
    # Puppeteer — it respects the inflictor's own type/ability immunities, nothing more.
    _apply_main_status(status, inflictor, 1 - target_index, rng, log, inflictor=None, field=field)


def _apply_main_status(
    status: Status,
    target: Pokemon,
    target_index: int,
    rng: RNG,
    log: BattleLog,
    inflictor: Pokemon | None = None,
    field: FieldState | None = None,
) -> None:
    corrosive = inflictor is not None and inflictor.ability is Ability.CORROSION
    immune_types = _STATUS_TYPE_IMMUNITY.get(status, frozenset())
    if status in (Status.POISON, Status.TOXIC) and corrosive:
        immune_types = frozenset()  # Corrosion poisons Steel and Poison types
    if immune_types and immune_types.intersection(t for t in target.types if t is not None):
        return
    if status in _STATUS_ABILITY_IMMUNITY.get(target.ability, frozenset()):
        log.add(DoesNotAffect(side=target_index, pokemon=target.nickname))
        return
    in_sun = field is not None and field.weather in (Weather.SUN, Weather.HARSH_SUN)
    if target.ability is Ability.LEAF_GUARD and in_sun:
        log.add(DoesNotAffect(side=target_index, pokemon=target.nickname))
        return
    if target.status is not Status.NONE:
        log.add(StatusAlready(side=target_index, pokemon=target.nickname, status=target.status))
        return
    target.status = status
    if status is Status.SLEEP:
        target.status_turns = rng.random_integer(1, 4)
    elif status is Status.TOXIC:
        target.status_turns = 0
    log.add(StatusInflicted(side=target_index, pokemon=target.nickname, status=status))
    _reflect_synchronize(status, target, target_index, inflictor, rng, log, field)
    puppeteered = (
        status in (Status.POISON, Status.TOXIC)
        and inflictor is not None
        and inflictor.ability is Ability.POISON_PUPPETEER
        and ExtraStatus.CONFUSION not in target.volatiles
    )
    if puppeteered:
        _apply_volatile(ExtraStatus.CONFUSION, target, target_index, rng, log)
    if status in _STATUS_CURE_ITEMS.get(target.item, frozenset()):
        target.consume_item()
        target.status = Status.NONE
        target.status_turns = 0
        log.add(StatusCleared(side=target_index, pokemon=target.nickname, clearance="berry"))


def _apply_volatile(volatile: ExtraStatus, target: Pokemon, target_index: int, rng: RNG, log: BattleLog) -> None:
    immune_types = _VOLATILE_TYPE_IMMUNITY.get(volatile, frozenset())
    if immune_types.intersection(t for t in target.types if t is not None):
        log.add(DoesNotAffect(side=target_index, pokemon=target.nickname))
        return
    if target.ability in _VOLATILE_ABILITY_IMMUNITY.get(volatile, frozenset()):
        return
    if volatile is ExtraStatus.NIGHTMARE and target.status is not Status.SLEEP:
        return  # Nightmare only takes hold on a sleeping target; the move then logs "But it failed!"
    if volatile is ExtraStatus.YAWN and target.status is not Status.NONE:
        return  # Yawn fails against an already-statused target
    target.volatiles[volatile] = _initial_volatile_duration(volatile, rng)
    log.add(VolatileInflicted(side=target_index, pokemon=target.nickname, volatile=volatile))
    if volatile is ExtraStatus.FLINCH and target.ability is Ability.STEADFAST:
        apply_stage_changes(
            target, target_index, {Stats.SPEED: 1}, log, inflicted_by_opponent=False, source="steadfast"
        )
    _mental_herb_cure(target, target_index, volatile, log)


def _start_encore(target: Pokemon, target_index: int, log: BattleLog) -> None:
    if target.last_move_slot is None or ExtraStatus.ENCORE in target.volatiles:
        return  # nothing to encore; the move then logs "But it failed!"
    target.encored_slot = target.last_move_slot
    target.volatiles[ExtraStatus.ENCORE] = 3
    log.add(VolatileInflicted(side=target_index, pokemon=target.nickname, volatile=ExtraStatus.ENCORE))
    if _mental_herb_cure(target, target_index, ExtraStatus.ENCORE, log):
        target.encored_slot = None


def _start_disable(target: Pokemon, target_index: int, log: BattleLog) -> None:
    if target.last_move_slot is None or target.disabled_slot is not None:
        return  # nothing to disable (or one already is); the move then logs "But it failed!"
    disabled_move = target.moves[target.last_move_slot]
    assert disabled_move is not None  # a used slot is a filled slot
    target.disabled_slot = target.last_move_slot
    target.volatiles[ExtraStatus.DISABLE] = 5  # PS condition duration; four usable turns after the tick
    log.add(DisableApplied(side=target_index, pokemon=target.nickname, move=disabled_move.name))
    if _mental_herb_cure(target, target_index, ExtraStatus.DISABLE, log):
        target.disabled_slot = None


_MENTAL_HERB_CURES = frozenset({ExtraStatus.TAUNT, ExtraStatus.ENCORE, ExtraStatus.DISABLE})


def _mental_herb_cure(target: Pokemon, target_index: int, volatile: ExtraStatus, log: BattleLog) -> bool:
    if volatile not in _MENTAL_HERB_CURES or target.item is not Item.MENTAL_HERB:
        return False
    target.consume_item()
    target.volatiles.pop(volatile, None)
    log.add(StatusCleared(side=target_index, pokemon=target.nickname, clearance="berry"))
    return True


def _make_substitute(target: Pokemon, target_index: int, log: BattleLog) -> None:
    cost = target.stat_totals.HP // 4
    if ExtraStatus.SUBSTITUTE in target.volatiles:
        log.add(SubstituteAlready(side=target_index, pokemon=target.nickname))
        return
    if cost >= target.live_stats.HP:
        log.add(SubstituteTooWeak())
        return
    target.apply_damage(cost)
    target.volatiles[ExtraStatus.SUBSTITUTE] = cost
    log.add(VolatileInflicted(side=target_index, pokemon=target.nickname, volatile=ExtraStatus.SUBSTITUTE))


def _initial_volatile_duration(volatile: ExtraStatus, rng: RNG) -> int:
    span = _VOLATILE_INITIAL_DURATIONS.get(volatile)
    if span is None:
        return 1
    low, high = span
    return low if low == high - 1 else rng.random_integer(low, high)


def _apply_stage_change(
    effect: StatStageChangeEffect,
    attacker: Pokemon,
    attacker_side_index: int,
    defender: Pokemon,
    rng: RNG,
    log: BattleLog,
) -> None:
    if not rng.roll_chance(effect.probability):
        return
    target_is_self = effect.target == "SELF"
    target = attacker if target_is_self else defender
    target_index = attacker_side_index if target_is_self else 1 - attacker_side_index
    apply_stage_changes(
        target, target_index, effect.stages, log, inflicted_by_opponent=not target_is_self, inflictor=attacker
    )
    raises = {stat: change for stat, change in effect.stages.items() if change > 0}
    if target_is_self and raises and defender.item is Item.MIRROR_HERB and not defender.is_fainted():
        defender.consume_item()
        apply_stage_changes(defender, 1 - attacker_side_index, raises, log, inflicted_by_opponent=False, source="seed")
