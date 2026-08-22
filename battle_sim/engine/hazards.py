from battle_sim.mechanics.battle import BattleState, SideState
from battle_sim.mechanics.events import Event, EventContext
from battle_sim.mechanics.log import BattleLog
from battle_sim.models.log_events import Fainted, HazardAbsorbed, HazardDamage, HazardStatus, StatStageChanged
from battle_sim.models.pokemon import Pokemon
from battle_sim.models.type_matchups import type_effectiveness
from battle_sim.utils import Hazards, Stats, Status, Type

_SPIKES_FRACTION: dict[int, int] = {1: 8, 2: 6, 3: 4}


def _apply_entry_hazards(state: BattleState, incoming: Pokemon, side_index: int, log: BattleLog) -> None:
    side = state.sides[side_index]
    ctx = EventContext(rng=state.rng, battle=state, actor=incoming, log=log)
    payload = state.bus.emit(Event.ON_ENTRY_HAZARD, ctx, {"side_index": side_index})
    if payload.get("hazards_blocked", False):
        return
    grounded = incoming.is_grounded()

    if _stealth_rock(side, incoming, side_index, log):
        return
    if grounded and _spikes(side, incoming, side_index, log):
        return
    if grounded and Hazards.TOXIC_SPIKES in side.hazards:
        _toxic_spikes(side, incoming, side_index, log)
    if grounded and Hazards.STICKY_WEB in side.hazards:
        _sticky_web(incoming, side_index, log)


def _stealth_rock(side: SideState, incoming: Pokemon, side_index: int, log: BattleLog) -> bool:
    """True if the incoming Pokemon fainted."""
    if Hazards.STEALTH_ROCK not in side.hazards:
        return False
    rock_effectiveness = type_effectiveness(Type.ROCK, incoming.types)
    if rock_effectiveness <= 0:
        return False
    chip = max(1, int(incoming.stat_totals.HP * rock_effectiveness // 8))
    dealt = incoming.apply_damage(chip)
    log.add(HazardDamage(side=side_index, pokemon=incoming.nickname, hazard=Hazards.STEALTH_ROCK, amount=dealt))
    if incoming.is_fainted():
        log.add(Fainted(side=side_index, pokemon=incoming.nickname))
        return True
    return False


def _spikes(side: SideState, incoming: Pokemon, side_index: int, log: BattleLog) -> bool:
    """True if the incoming Pokemon fainted."""
    layers = side.hazards.get(Hazards.SPIKES, 0)
    if layers <= 0:
        return False
    denom = _SPIKES_FRACTION[layers]
    dealt = incoming.apply_damage(max(1, incoming.stat_totals.HP // denom))
    log.add(HazardDamage(side=side_index, pokemon=incoming.nickname, hazard=Hazards.SPIKES, amount=dealt))
    if incoming.is_fainted():
        log.add(Fainted(side=side_index, pokemon=incoming.nickname))
        return True
    return False


def _toxic_spikes(side: SideState, incoming: Pokemon, side_index: int, log: BattleLog) -> None:
    if Type.POISON in incoming.types:
        del side.hazards[Hazards.TOXIC_SPIKES]
        log.add(HazardAbsorbed(side=side_index, pokemon=incoming.nickname))
    elif Type.STEEL not in incoming.types and incoming.status is Status.NONE:
        if side.hazards[Hazards.TOXIC_SPIKES] >= 2:
            incoming.status = Status.TOXIC
            incoming.status_turns = 0
            log.add(HazardStatus(side=side_index, pokemon=incoming.nickname, status=Status.TOXIC))
        else:
            incoming.status = Status.POISON
            log.add(HazardStatus(side=side_index, pokemon=incoming.nickname, status=Status.POISON))


def _sticky_web(incoming: Pokemon, side_index: int, log: BattleLog) -> None:
    delta = incoming.change_stat_stage(Stats.SPEED, -1)
    if delta < 0:
        log.add(
            StatStageChanged(
                side=side_index,
                pokemon=incoming.nickname,
                stat=Stats.SPEED,
                delta=delta,
                requested=-1,
                source="sticky_web",
            )
        )
