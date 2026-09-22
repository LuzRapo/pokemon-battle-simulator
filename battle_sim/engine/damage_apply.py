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
from battle_sim.models.stats import StatStages
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
    # Weather-aware: under Delta Stream a hit that would read "super effective" is only neutral,
    # so the message and the Weakness Policy trigger have to agree with the damage.
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
    # A critical hit is announced immediately before the `DamageDealt` it explains, and nowhere else,
    # so the two are strictly paired however the damage is reported. A multi-hit move reports each
    # hit separately and so announces each crit where it happened; everything else reports one total
    # after the loop, which for a single hit is the same hit, and this carries its verdict there.
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
    """Fire melts ice off whatever it hits, and so do the handful of moves that say they do.

    Gen 6 onward every damaging Fire move thaws its target, which is why freeze is a status you get
    out of rather than one you sit in — without this a frozen Pokemon could be hit by Fire Blast all
    day and stay solid. `thaws_target` is the short list that does it off-type: Scald, Steam Eruption
    and Matcha Gotcha. A substitute takes the hit instead, so nothing reaches the Pokemon to thaw.
    """
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
    """Whether the move is over before a single hit is rolled: nothing it can touch, or a disguise
    standing in the way. Either way no damage, no recoil and no on-hit reactions follow."""
    if type_mult == 0:
        log.add(NoEffect(side=defender_index, pokemon=defender.nickname))
        return True
    return _disguise_absorbs(state, defender, defender_index, log)


def _disguise_absorbs(state: BattleState, defender: Pokemon, defender_index: int, log: BattleLog) -> bool:
    """Whether Mimikyu's disguise takes this attack instead of Mimikyu.

    It absorbs one attacking move whole — every strike of a multi-hit one included, which is how Gen 7
    plays it — and busts. No damage means no recoil, no drain and no on-hit item reaction; in Gen 7 it
    costs Mimikyu nothing else, since the 1/8 chip on breaking is a Gen 8 addition.
    """
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
            log.add(RecoilDamage(side=attacker_side_index, pokemon=attacker.nickname, amount=heal))
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


def _apply_fixed_damage(  # noqa: PLR0913
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
    """Announce a life being spent. Set in `Pokemon._adjust_hp`, which is the only place that can
    see every damage source, and read here, which is the only place that has the log.

    KNOWN DEFECT, and it is this log line only -- the mechanic underneath is sound. A battle can
    print more revivals than the nine that exist, out of order ("8 8 7 7 6 5 4 3 2 1 6 0"), because
    a search clone's revival reaches a real battle log. Measured: twelve lines written into real
    logs while the live Pokemon had spent exactly three lives, its counter running 1, 2, 3 as it
    should. `_revive` caps at nine per Pokemon, so no one ever gets more than nine -- only the
    telling of it is wrong. Filtering on side membership does not help: the clone is a legitimate
    member of the clone's own state, which is what gets passed -- which is why `state` is used here
    only to reach the bus, and never to decide whether this revival is one worth announcing.
    """
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
    # `_revive` put the item back on him but cannot wire it up: it lives in the model, with no bus
    # to reach. Knocking it off unbound its handlers, so without this he would hold a dead Leftovers.
    # Read before it is cleared -- the trade-reversal below needs to know what came back, and clearing
    # it first left the other half of a Trick sitting on the opponent as a second copy.
    taken_back = defender.restored_item
    if taken_back is not Item.NONE:
        from battle_sim.mechanics.effects import rewire_active  # local, as everywhere else: cycles

        log.add(ItemRestored(side=defender_index, pokemon=defender.nickname, item=taken_back))
        defender.restored_item = Item.NONE
        rewire_active(state.bus, state.effects, defender)
    _hand_back_the_trade(state, defender, defender_index, taken_back, log)
    _sweep_the_opposing_board(state, defender_index, log)


def _the_butler_yields(state: BattleState, defender: Pokemon, defender_index: int, log: BattleLog) -> None:
    """The ninth life is spent, the tenth blow has landed, and he is still on his feet at 1 HP.

    Everything wearing anybody down goes with it — the weather, the terrain, the hazards, the
    screens, every stat stage on the board. He is not fighting on, so nothing that was there to help
    him fight has any business remaining, and the trainer who got him here should not have to pick
    their way out through somebody else's sandstorm.

    Then the battle is over, and won by whoever did it: this is a concession, not a stalemate. The
    engine says so rather than leaving it to a caller, because a battle with a Pokemon standing at
    1 HP that will not fight looks exactly like a battle still in progress.
    """
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
    """A Trick taken back is a trade reversed, so the other half goes home too.

    Only against whoever is actually standing there holding his item. If they have switched away the
    trick stands — reaching onto the bench to swap an item back would be a mechanic of its own, and
    handing his item back while a copy walks around on their bench would mint a second one.

    He may be holding nothing by now, having eaten it (see `_greedy_gourmand`). Then they get nothing
    back, which is the whole of what tricking the butler is worth.
    """
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
    """Whatever the side facing him had built up goes when he rises: their stages, and their screens.

    Without it the nine lives are not a gauntlet so much as nine turns to stand still in: stack
    evasion and he cannot touch you however many times he gets up, and the same is true of a Swords
    Dance sweep set up once and paid off nine times. Each life starts the exchange over.

    Stages go both ways rather than only the boosts, so a Close Combat's own Defence drop clears with
    the rest -- the board is reset, not confiscated. His own drops go too (see
    `Pokemon.clear_stat_drops`) while his boosts stay, which is the asymmetry the ability is *for*:
    it is his, not the field's.

    Screens go for the same reason, and they are the more durable half of the problem. A stage boost
    belongs to whoever is standing there and leaves when they do; Reflect and Light Screen sit on the
    *side* for five turns and outlive any number of his lives, halving everything he lands for the
    whole stretch. That is precisely the standing-still this sweep exists to stop.
    """
    opposing = state.sides[1 - revived_index]
    facing = opposing.active_pokemon
    if any(getattr(facing.stat_stages, name) for name in StatStages.model_fields):
        facing.reset_stat_stages()
        log.add(StatChangesSwept(side=1 - revived_index, pokemon=facing.nickname))
    # One line per screen, the way Brick Break and Defog announce them, rather than a single line
    # standing in for up to three.
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
            # Really the user's level times a roll between 0.5 and 1.5. Taken at its mean, since
            # `_fixed_amount` is handed no RNG and the spread is not worth threading one through for.
            return attacker.level


def effect_category(effect: FixedDamageEffect) -> Category:
    """Counter-style bookkeeping for fixed hits: Counter is physical, Mirror Coat special."""
    return Category.SPECIAL if effect.amount_formula == "MIRROR_COAT" else Category.PHYSICAL
