"""Stat-stage application and the interactions every stage change shares:
Contrary inversion, drop blockers (Clear Body, Full Metal Body, White Smoke, Clear Amulet),
single-stat drop blockers (Keen Eye, Hyper Cutter, Big Pecks), drop reflection (Mirror Armor),
retaliation (Defiant, Competitive, Guard Dog, Adrenaline Orb), Eject Pack arming, and White
Herb restoration.
"""

from battle_sim.mechanics.log import BattleLog
from battle_sim.models.log_events import (
    StatChangeSource,
    StatDropBlocked,
    StatDropBlockedByItem,
    StatStageChanged,
    WhiteHerbRestored,
)
from battle_sim.models.pokemon import Pokemon
from battle_sim.utils import Ability, Item, Stats

type _Retaliation = tuple[Stats, StatChangeSource]

_DROP_BLOCKING_ABILITIES = (Ability.CLEAR_BODY, Ability.FULL_METAL_BODY, Ability.WHITE_SMOKE)
_SINGLE_STAT_DROP_BLOCKERS: dict[Ability, Stats] = {
    Ability.KEEN_EYE: Stats.ACCURACY,
    Ability.HYPER_CUTTER: Stats.ATTACK,
    Ability.BIG_PECKS: Stats.DEFENCE,
}
_DROP_RETALIATION: dict[Ability, _Retaliation] = {
    Ability.DEFIANT: (Stats.ATTACK, "defiant"),
    Ability.COMPETITIVE: (Stats.SP_ATTACK, "competitive"),
}
_STAGED_STATS = tuple(s for s in Stats if s is not Stats.HP)


def apply_stage_changes(
    target: Pokemon,
    target_index: int,
    stages: dict[Stats, int],
    log: BattleLog,
    inflicted_by_opponent: bool,
    source: StatChangeSource = "move",
    inflictor: Pokemon | None = None,
    reflected: bool = False,
) -> None:
    if target.ability is Ability.CONTRARY:
        stages = {stat: -change for stat, change in stages.items()}
    elif target.ability is Ability.SIMPLE:
        stages = {stat: 2 * change for stat, change in stages.items()}  # both ways, drops included
    has_drop = any(change < 0 for change in stages.values())
    if inflicted_by_opponent and has_drop:
        surviving = _intercept_drops(target, target_index, stages, log, source, inflictor, reflected)
        if surviving is None:
            return
        stages = surviving
    dropped = 0
    for stat, requested in stages.items():
        delta = target.change_stat_stage(stat, requested)
        log.add(
            StatStageChanged(
                side=target_index, pokemon=target.nickname, stat=stat, delta=delta, requested=requested, source=source
            )
        )
        if delta < 0 and inflicted_by_opponent:
            dropped += 1
    if dropped:
        _retaliate_drops(target, target_index, dropped, log)
        if source == "intimidate" and target.item is Item.ADRENALINE_ORB:
            target.consume_item()
            apply_stage_changes(target, target_index, {Stats.SPEED: 1}, log, inflicted_by_opponent=False, source="seed")
        if target.item is Item.EJECT_PACK:
            target.eject_pending = True  # the turn loop pulls the holder once the action resolves
    _white_herb_restore(target, target_index, log)


def _intercept_drops(
    target: Pokemon,
    target_index: int,
    stages: dict[Stats, int],
    log: BattleLog,
    source: StatChangeSource,
    inflictor: Pokemon | None,
    reflected: bool,
) -> dict[Stats, int] | None:
    """None: a blocker or reflector consumed the whole opponent-inflicted change.

    A dict: what still applies — the original `stages`, or (Keen Eye/Hyper Cutter/Big Pecks)
    that same dict with just their one protected stat's drop stripped out, since those only
    guard a single stat and a move can drop several at once (e.g. Tickle: Attack and Defense).
    """
    if source == "intimidate" and target.ability is Ability.GUARD_DOG:
        log.add(StatDropBlocked(side=target_index, pokemon=target.nickname, ability=Ability.GUARD_DOG))
        apply_stage_changes(target, target_index, {Stats.ATTACK: 1}, log, inflicted_by_opponent=False)
        return None
    if target.ability in _DROP_BLOCKING_ABILITIES:
        log.add(StatDropBlocked(side=target_index, pokemon=target.nickname, ability=target.ability))
        return None
    if target.item is Item.CLEAR_AMULET:
        log.add(StatDropBlockedByItem(side=target_index, pokemon=target.nickname, item=Item.CLEAR_AMULET))
        return None
    if target.ability is Ability.MIRROR_ARMOR and inflictor is not None and not reflected:
        log.add(StatDropBlocked(side=target_index, pokemon=target.nickname, ability=Ability.MIRROR_ARMOR))
        drops = {stat: change for stat, change in stages.items() if change < 0}
        apply_stage_changes(
            inflictor, 1 - target_index, drops, log, inflicted_by_opponent=True, source=source, reflected=True
        )
        return None
    protected = _SINGLE_STAT_DROP_BLOCKERS.get(target.ability)
    if protected is not None and stages.get(protected, 0) < 0:
        log.add(StatDropBlocked(side=target_index, pokemon=target.nickname, ability=target.ability))
        return {stat: change for stat, change in stages.items() if stat is not protected}
    return stages


def _retaliate_drops(target: Pokemon, target_index: int, dropped: int, log: BattleLog) -> None:
    retaliation = _DROP_RETALIATION.get(target.ability)
    if retaliation is None:
        return
    stat, retaliation_source = retaliation
    requested = 2 * dropped  # +2 per stat lowered (PS)
    delta = target.change_stat_stage(stat, requested)
    if delta > 0:
        log.add(
            StatStageChanged(
                side=target_index,
                pokemon=target.nickname,
                stat=stat,
                delta=delta,
                requested=requested,
                source=retaliation_source,
            )
        )


def _white_herb_restore(target: Pokemon, target_index: int, log: BattleLog) -> None:
    if target.item is not Item.WHITE_HERB:
        return
    lowered = [stat for stat in _STAGED_STATS if target.stat_stages[stat] < 0]
    if not lowered:
        return
    for stat in lowered:
        target.stat_stages[stat] = 0
    target.consume_item()
    log.add(WhiteHerbRestored(side=target_index, pokemon=target.nickname))
