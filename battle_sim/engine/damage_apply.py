from battle_sim.engine.power import effective_power, payload_overrides
from battle_sim.formes import MIMIKYU_BUSTED, disguise_broken, swap_forme
from battle_sim.maths.damage import calculate_hit, immunity_bypass, move_effectiveness
from battle_sim.maths.rng import RNG
from battle_sim.mechanics.battle import BattleState, SideState, effective_weather
from battle_sim.mechanics.events import Event, EventContext
from battle_sim.mechanics.log import BattleLog
from battle_sim.models.log_events import (
    CriticalHit,
    DamageDealt,
    DrainBackfired,
    Drained,
    Effectiveness,
    Fainted,
    ItemRestored,
    ItemsSwapped,
    LastStand,
    MultiHitSummary,
    NineLivesRestored,
    NoEffect,
    RecoilDamage,
    ScreenFaded,
    StatChangesSwept,
    StatusCleared,
    SubstituteBroke,
    SubstituteTookHit,
    SurvivedAtOneHp,
)
from battle_sim.models.moves import DamageEffect, FixedDamageEffect, Move
from battle_sim.models.pokemon import NINE_LIVES, Pokemon
from battle_sim.models.stats import STAGED_STATS
from battle_sim.models.type_matchups import type_effectiveness
from battle_sim.utils import Ability, Category, ExtraStatus, Item, Status, Type


def _apply_damage(
    effect: DamageEffect,
    move: Move,
    attacker: Pokemon,
    attacker_side_index: int,
    defender: Pokemon,
    defender_side: SideState,
    state: BattleState,
    log: BattleLog,
    behind_substitute: bool,
) -> None:
    defender_index = 1 - attacker_side_index
    # Delta Stream can make a super-effective hit neutral, so the message and Weakness Policy must agree.
    type_mult = move_effectiveness(move, attacker, defender, state.field.weather)
    if _stopped_before_any_hit(state, type_mult, defender, defender_index, log):
        return

    is_multi_hit = effect.multi_hit is not None
    hits_planned = _planned_hits(effect, attacker, state.rng)
    _log_effectiveness(type_mult, log)

    ctx = EventContext(rng=state.rng, battle=state, log=log, actor=attacker, defender=defender, move=move)
    hit_payload_base = {
        "move_type": move.type,
        "category": effect.category,
        "contact": (
            effect.contact
            and attacker.item is not Item.PROTECTIVE_PADS
            and attacker.ability is not Ability.LONG_REACH  # Decidueye never actually touches anything
            and not (attacker.item is Item.PUNCHING_GLOVE and move.punching)
        ),
        "attacker_index": attacker_side_index,
        "defender_index": defender_index,
        "behind_substitute": behind_substitute,
        "power_override": effective_power(move, effect, attacker, defender, state),
        "weather_suppressed": effective_weather(state) is not state.field.weather,
        **payload_overrides(move, attacker, defender),
    }

    hits_landed = 0
    total_dealt = 0
    # A crit is announced immediately before the `DamageDealt` it explains, and nowhere else.
    critical = False
    for _ in range(hits_planned):
        modifiers = state.bus.emit(Event.ON_DAMAGE_CALC, ctx, dict(hit_payload_base))
        hit = calculate_hit(attacker, defender, move, state.field, defender_side, rng=state.rng, modifiers=modifiers)
        critical = hit.is_crit
        if behind_substitute and ExtraStatus.SUBSTITUTE in defender.volatiles:
            # The substitute soaks the hit: no survival clamps, no on-hit item reactions,
            # but recoil/drain later still count the absorbed damage (PS gen 5+).
            total_dealt += _damage_substitute(defender, defender_index, hit.amount, log)
            defender.times_hit += 1  # Rage Fist counts hits taken behind a substitute (PS)
            hits_landed += 1
            continue
        before = state.bus.emit(Event.ON_BEFORE_HIT, ctx, {**hit_payload_base, "damage": hit.amount})
        dealt = _land_hit(before["damage"], effect, defender, defender_index, log)
        total_dealt += dealt
        hits_landed += 1
        if is_multi_hit:
            _announce_crit(hit.is_crit, log)
            log.add(DamageDealt(side=defender_index, pokemon=defender.nickname, amount=dealt))
        state.bus.emit(Event.ON_AFTER_HIT, ctx, {**hit_payload_base, "dealt": dealt})
        if defender.is_fainted():
            break

    if is_multi_hit:
        log.add(MultiHitSummary(hits=hits_landed))
    elif total_dealt > 0:
        _announce_crit(critical, log)
        log.add(DamageDealt(side=defender_index, pokemon=defender.nickname, amount=total_dealt))

    _thaw_on_hit(move, defender, defender_index, behind_substitute, log)
    _log_revival(state, defender, defender_index, log)
    if defender.is_fainted():
        log.add(Fainted(side=defender_index, pokemon=defender.nickname))
        state.bus.emit(Event.ON_FAINT, ctx, dict(hit_payload_base))
        if ExtraStatus.DESTINY_BOND in defender.volatiles and not attacker.is_fainted():
            attacker.apply_damage(attacker.live_stats.HP)
            log.add(Fainted(side=attacker_side_index, pokemon=attacker.nickname))

    _apply_recoil_and_drain(effect, attacker, attacker_side_index, defender, total_dealt, log)
    state.bus.emit(Event.ON_ACTION_RESOLVE, ctx, {**hit_payload_base, "total_dealt": total_dealt})


def _announce_crit(is_crit: bool, log: BattleLog) -> None:
    if is_crit:
        log.add(CriticalHit())


def _thaw_on_hit(move: Move, defender: Pokemon, defender_index: int, behind_substitute: bool, log: BattleLog) -> None:
    """Fire melts ice off whatever it hits, and so do the handful of moves that say they do."""
    soaked = behind_substitute and ExtraStatus.SUBSTITUTE in defender.volatiles
    if defender.status is not Status.FREEZE or defender.is_fainted() or soaked:
        return
    if not (move.thaws_target or (move.type is Type.FIRE and move.category is not Category.STATUS)):
        return
    defender.status = Status.NONE
    defender.status_turns = 0
    log.add(StatusCleared(side=defender_index, pokemon=defender.nickname, clearance="thawed"))


def _stopped_before_any_hit(
    state: BattleState, type_mult: float, defender: Pokemon, defender_index: int, log: BattleLog
) -> bool:
    """Whether the move ends before any hit is rolled: nothing to touch, or a disguise in the way."""
    if type_mult == 0:
        log.add(NoEffect(side=defender_index, pokemon=defender.nickname))
        return True
    return _disguise_absorbs(state, defender, defender_index, log)


def _disguise_absorbs(state: BattleState, defender: Pokemon, defender_index: int, log: BattleLog) -> bool:
    """Whether Mimikyu's disguise takes this attack instead of Mimikyu."""
    if defender.ability is not Ability.DISGUISE or disguise_broken(defender):
        return False
    swap_forme(state.bus, state.effects, defender, MIMIKYU_BUSTED, defender_index, log)
    return True


def _land_hit(damage: int, effect: DamageEffect, defender: Pokemon, defender_index: int, log: BattleLog) -> int:
    if ExtraStatus.ENDURE in defender.volatiles and damage >= defender.live_stats.HP:
        damage = defender.live_stats.HP - 1
        log.add(SurvivedAtOneHp(side=defender_index, pokemon=defender.nickname, cause="endure"))
    dealt = defender.apply_damage(damage)
    if dealt > 0:
        defender.times_hit += 1
        defender.last_hit_taken = dealt
        defender.last_hit_category = effect.category
    return dealt


def _apply_recoil_and_drain(
    effect: DamageEffect,
    attacker: Pokemon,
    attacker_side_index: int,
    defender: Pokemon,
    total_dealt: int,
    log: BattleLog,
) -> None:
    # Move recoil/drain are move mechanics, not item/ability effects — they stay core.
    if effect.struggle_recoil:  # unconditional: not prevented by Magic Guard or Rock Head (PS)
        recoil = max(1, attacker.stat_totals.HP // 4)
        attacker.apply_damage(recoil)
        log.add(RecoilDamage(side=attacker_side_index, pokemon=attacker.nickname, amount=recoil))
        return
    magic_guard = attacker.ability is Ability.MAGIC_GUARD
    if (
        effect.recoil_percent is not None
        and total_dealt > 0
        and not magic_guard
        and attacker.ability is not Ability.ROCK_HEAD
    ):
        recoil = max(1, int(total_dealt * effect.recoil_percent))
        attacker.apply_damage(recoil)
        log.add(RecoilDamage(side=attacker_side_index, pokemon=attacker.nickname, amount=recoil))
    if effect.drain_percent is not None and total_dealt > 0:
        heal = max(1, int(total_dealt * effect.drain_percent))
        if defender.ability is Ability.LIQUID_OOZE:
            attacker.apply_damage(heal)
            log.add(
                DrainBackfired(
                    side=attacker_side_index,
                    pokemon=attacker.nickname,
                    ability=Ability.LIQUID_OOZE,
                    amount=heal,
                )
            )
        else:
            attacker.apply_healing(heal)
            log.add(Drained(side=attacker_side_index, pokemon=attacker.nickname, amount=heal))


def _damage_substitute(defender: Pokemon, defender_index: int, damage: int, log: BattleLog) -> int:
    """Returns the damage absorbed (counts toward recoil/drain)."""
    sub_hp = defender.volatiles[ExtraStatus.SUBSTITUTE]
    if damage >= sub_hp:
        del defender.volatiles[ExtraStatus.SUBSTITUTE]
        log.add(SubstituteBroke(side=defender_index, pokemon=defender.nickname))
        return sub_hp
    defender.volatiles[ExtraStatus.SUBSTITUTE] = sub_hp - damage
    log.add(SubstituteTookHit(side=defender_index, pokemon=defender.nickname))
    return damage


def _planned_hits(effect: DamageEffect, attacker: Pokemon, rng: RNG) -> int:
    if effect.multi_hit is None:
        return 1
    low, high = effect.multi_hit
    if low == high:
        return low
    if attacker.ability is Ability.SKILL_LINK:
        return high
    if attacker.item is Item.LOADED_DICE and high - low >= 2:
        return rng.random_integer(high - 1, high + 1)  # a 2-5-hit move rolls 4 or 5 (PS)
    return rng.random_integer(low, high + 1)


def _log_effectiveness(type_mult: float, log: BattleLog) -> None:
    if type_mult >= 2:
        log.add(Effectiveness(level="super"))
    elif type_mult <= 0.5:
        log.add(Effectiveness(level="resisted"))


def _apply_fixed_damage(
    effect: FixedDamageEffect,
    move: Move,
    attacker: Pokemon,
    defender: Pokemon,
    defender_index: int,
    log: BattleLog,
    behind_substitute: bool,
    state: BattleState,
) -> None:
    type_mult = type_effectiveness(move.type, defender.types, immunity_bypass=immunity_bypass(defender))
    if type_mult == 0:
        log.add(NoEffect(side=defender_index, pokemon=defender.nickname))
        return

    fixed = _fixed_amount(effect, attacker, defender)
    if fixed is None:
        return  # the condition wasn't met; the empty-log fallback reports "But it failed!"
    damage = fixed

    if behind_substitute and ExtraStatus.SUBSTITUTE in defender.volatiles:
        _damage_substitute(defender, defender_index, damage, log)
        return

    dealt = defender.apply_damage(damage)
    if dealt > 0:
        defender.times_hit += 1
        defender.last_hit_taken = dealt
        defender.last_hit_category = effect_category(effect)
        log.add(DamageDealt(side=defender_index, pokemon=defender.nickname, amount=dealt))
    _log_revival(state, defender, defender_index, log)
    if defender.is_fainted():
        log.add(Fainted(side=defender_index, pokemon=defender.nickname))


def _log_revival(state: BattleState, defender: Pokemon, defender_index: int, log: BattleLog) -> None:
    """Announce a life being spent."""
    if defender.just_stood:
        defender.just_stood = False
        _the_butler_yields(state, defender, defender_index, log)
        return
    if not defender.just_revived:
        return
    defender.just_revived = False
    log.add(
        NineLivesRestored(
            side=defender_index,
            pokemon=defender.nickname,
            healed=defender.stat_totals.HP,
            remaining=NINE_LIVES - defender.lives_used,
        )
    )
    # Rebind the restored item's handlers, which `_revive` cannot reach from the model.
    taken_back = defender.restored_item
    if taken_back is not Item.NONE:
        from battle_sim.mechanics.effects import rewire_active  # local, as everywhere else: cycles

        log.add(ItemRestored(side=defender_index, pokemon=defender.nickname, item=taken_back))
        defender.restored_item = Item.NONE
        rewire_active(state.bus, state.effects, defender)
    _hand_back_the_trade(state, defender, defender_index, taken_back, log)
    _sweep_the_opposing_board(state, defender_index, log)


def concede_if_stood(state: BattleState, log: BattleLog) -> None:
    """Announce a last stand that was made somewhere `_log_revival` is not watching."""
    for side_index, side in enumerate(state.sides):
        for pokemon in side.team:
            if pokemon.just_stood:
                pokemon.just_stood = False
                _the_butler_yields(state, pokemon, side_index, log)


def _the_butler_yields(state: BattleState, defender: Pokemon, defender_index: int, log: BattleLog) -> None:
    """The ninth life is spent, the tenth blow has landed, and he is still on his feet at 1 HP."""
    from battle_sim.utils import Outcome, Terrain, Weather

    log.add(LastStand(side=defender_index, pokemon=defender.nickname))
    state.field.weather = Weather.NONE
    state.field.weather_turns_left = 0
    state.field.terrain = Terrain.NONE
    state.field.terrain_turns_left = 0
    state.field.pseudo_weather.clear()
    for side in state.sides:
        side.hazards.clear()
        side.screens.clear()
        side.tailwind_turns = 0
        side.active_pokemon.reset_stat_stages()
    state.outcome = Outcome.P1_WIN if defender_index == 1 else Outcome.P2_WIN


def _hand_back_the_trade(
    state: BattleState, revived: Pokemon, revived_index: int, taken_back: Item, log: BattleLog
) -> None:
    """A Trick taken back is a trade reversed, so the other half goes home too."""
    owed = revived.owed_item
    revived.owed_item = Item.NONE
    if taken_back is Item.NONE:
        return
    facing = state.sides[1 - revived_index].active_pokemon
    if facing.is_fainted() or facing.item is not taken_back:
        return
    from battle_sim.mechanics.effects import rewire_active

    facing.item = owed
    rewire_active(state.bus, state.effects, facing)
    log.add(ItemsSwapped(side=revived_index, pokemon=revived.nickname))


def _sweep_the_opposing_board(state: BattleState, revived_index: int, log: BattleLog) -> None:
    """When he rises, the opposing side's stat stages and screens are cleared."""
    opposing = state.sides[1 - revived_index]
    facing = opposing.active_pokemon
    if any(facing.stat_stages[stat] for stat in STAGED_STATS):
        facing.reset_stat_stages()
        log.add(StatChangesSwept(side=1 - revived_index, pokemon=facing.nickname))
    # One line per screen, the way Brick Break and Defog announce them.
    for screen in list(opposing.screens):
        del opposing.screens[screen]
        log.add(ScreenFaded(side=1 - revived_index, screen=screen))


def _fixed_amount(effect: FixedDamageEffect, attacker: Pokemon, defender: Pokemon) -> int | None:
    match effect.amount_formula:
        case "LEVEL":
            return attacker.level
        case "SET":
            assert effect.set_amount is not None  # the loader always sets an amount for SET-formula moves
            return effect.set_amount
        case "HALF_TARGET_HP":
            return max(1, defender.live_stats.HP // 2)
        case "ENDEAVOR":
            difference = defender.live_stats.HP - attacker.live_stats.HP
            return difference if difference > 0 else None
        case "COUNTER":
            valid = attacker.last_hit_category is Category.PHYSICAL and attacker.last_hit_taken > 0
            return 2 * attacker.last_hit_taken if valid else None
        case "MIRROR_COAT":
            valid = attacker.last_hit_category is Category.SPECIAL and attacker.last_hit_taken > 0
            return 2 * attacker.last_hit_taken if valid else None
        case "USER_HP":
            return attacker.live_stats.HP
        case "TARGET_HP":
            return defender.live_stats.HP  # the OHKO moves, which their own 30% accuracy already gates
        case "PSYWAVE":
            # Really the user's level times a roll between 0.5 and 1.5.
            return attacker.level


def effect_category(effect: FixedDamageEffect) -> Category:
    """Counter-style bookkeeping for fixed hits: Counter is physical, Mirror Coat special."""
    return Category.SPECIAL if effect.amount_formula == "MIRROR_COAT" else Category.PHYSICAL
