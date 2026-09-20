from dataclasses import replace

from battle_sim.database.loader import get_move
from battle_sim.engine.coded import _apply_coded
from battle_sim.engine.damage_apply import _apply_damage, _apply_fixed_damage
from battle_sim.engine.field_apply import _apply_field_effect, _apply_remove_hazards, _apply_side_condition
from battle_sim.engine.power import (
    ESCALATING_MOVES,
    ROLLING_LOCK_TURNS,
    ROLLING_MOVES,
    coded_move_fails,
    move_type_override,
)
from battle_sim.engine.status_apply import _apply_stage_change, _apply_status
from battle_sim.engine.switching import _force_random_switch
from battle_sim.formes import stance_forme, swap_forme
from battle_sim.maths.damage import move_effectiveness
from battle_sim.maths.rng import RNG
from battle_sim.mechanics.battle import BattleState, FieldState, SideState, effective_weather
from battle_sim.mechanics.events import Event, EventContext
from battle_sim.mechanics.log import BattleLog
from battle_sim.mechanics.stages import apply_stage_changes
from battle_sim.models.actions import Action, ActionType
from battle_sim.models.log_events import (
    CantAct,
    ChargingUp,
    ConfusionSelfHit,
    DisabledBlocked,
    DoesNotAffect,
    Fainted,
    Healed,
    ItemDevoured,
    MoveBounced,
    MoveFailed,
    MoveMissed,
    MoveUsed,
    NoEffect,
    PpRestored,
    Protected,
    RecoilDamage,
    ScreenFaded,
    SelfSwitchPending,
    StatusCleared,
    TauntBlocked,
)
from battle_sim.models.moves import (
    CodedEffect,
    DamageEffect,
    FixedDamageEffect,
    HealEffect,
    InflictStatusEffect,
    Move,
    MoveEffect,
    MoveSlot,
    PseudoWeatherEffect,
    RemoveHazardsEffect,
    SideConditionEffect,
    StatStageChangeEffect,
    TerrainEffect,
    WeatherEffect,
)
from battle_sim.models.pokemon import Pokemon
from battle_sim.utils import Ability, Category, ExtraStatus, Item, Stats, Status, Target, Type, Weather
from battle_sim.zmoves import z_move_for

_DEFENDER_FACING_TARGETS = frozenset({Target.SINGLE_OPPONENT, Target.ALL_ADJACENT_ENEMIES, Target.ALL_ADJACENT})
# Same effect, two names: an opponent holding either simply cannot be hit by a priority move.
_PRIORITY_BLOCKERS = frozenset({Ability.DAZZLING, Ability.QUEENLY_MAJESTY})
# The semi-invulnerable charges, and the moves that reach them anyway. Membership here is what makes
# a charge semi-invulnerable at all — an ordinary two-turn move like Solar Beam is not listed, so its
# user stays hittable while it charges.
_UP_IN_THE_AIR = frozenset({"Gust", "Twister", "Thunder", "Hurricane", "Sky Uppercut", "Smack Down", "Thousand Arrows"})
_REACHES_THROUGH: dict[str, frozenset[str]] = {
    "Fly": _UP_IN_THE_AIR,
    "Bounce": _UP_IN_THE_AIR,
    "Sky Drop": _UP_IN_THE_AIR,
    "Dig": frozenset({"Earthquake", "Magnitude", "Fissure"}),
    "Dive": frozenset({"Surf", "Whirlpool"}),
    "Phantom Force": frozenset(),
    "Shadow Force": frozenset(),
}
_CHOICE_ITEMS = frozenset({Item.CHOICE_BAND, Item.CHOICE_SPECS, Item.CHOICE_SCARF})
_STRUGGLE = get_move("Struggle")
# Half his maximum. It costs him the turn, so it has to be worth a turn — and half is what makes
# handing him anything a real mistake rather than a small one.
GREEDY_GOURMAND_DIVISOR = 2
# These carry a real DamageEffect for engine.residuals to apply two turns on (see CodedMoveKind.
# FUTURE_SIGHT); the effect must not also land immediately on the turn the move is used.
_DELAYED_DAMAGE_MOVES = frozenset({"Future Sight", "Doom Desire"})
# The one move whose healing has a cost: the bird comes down to do it. See `_apply_heal`.
ROOST = "Roost"


def _execute_move(  # noqa: C901 — the move-flow gate ladder reads top-to-bottom in priority order; that IS the contract
    state: BattleState, side_index: int, action: Action, log: BattleLog
) -> None:
    side = state.sides[side_index]
    opponent_side = state.sides[1 - side_index]
    attacker = side.active_pokemon
    defender = opponent_side.active_pokemon
    defender_index = 1 - side_index

    chosen = attacker.moves[action.move] if action.move is not None else None
    sleep_talking = chosen is not None and chosen.name == "Sleep Talk" and attacker.status is Status.SLEEP
    defrosting = chosen is not None and chosen.defrosts_user and attacker.status is Status.FREEZE
    if not _can_act(attacker, side_index, state.rng, log, sleep_talking, defrosting):
        attacker.protect_streak = 0  # a skipped turn breaks the consecutive-Protect chain
        return
    if _greedy_gourmand(attacker, side_index, state, log):
        attacker.protect_streak = 0  # his turn went on dinner, not on a move
        return
    attacker.volatiles.pop(ExtraStatus.DESTINY_BOND, None)  # the bond lasts until the user's next action

    if (
        attacker.item in _CHOICE_ITEMS
        and attacker.choice_locked_move is not None
        and action.move is not attacker.choice_locked_move
    ):
        action = Action(action=ActionType.USE_MOVE, target=action.target, move=attacker.choice_locked_move)
    if (
        ExtraStatus.ENCORE in attacker.volatiles
        and attacker.encored_slot is not None
        and action.move is not attacker.encored_slot
    ):
        action = Action(action=ActionType.USE_MOVE, target=action.target, move=attacker.encored_slot)
    rampaging = ExtraStatus.LOCKED_MOVE in attacker.volatiles and attacker.locked_slot is not None
    if rampaging and action.move is not attacker.locked_slot:
        action = Action(action=ActionType.USE_MOVE, target=action.target, move=attacker.locked_slot)
    releasing_charge = ExtraStatus.CHARGING in attacker.volatiles and attacker.charging_slot is not None
    if releasing_charge and action.move is not attacker.charging_slot:
        action = Action(action=ActionType.USE_MOVE, target=action.target, move=attacker.charging_slot)
    slot = action.move
    assert slot is not None  # Action's validator guarantees a move slot for USE_MOVE
    move = attacker.moves[slot]
    assert move is not None  # a USE_MOVE action must point at a filled slot

    if slot is attacker.disabled_slot:
        log.add(DisabledBlocked(side=side_index, pokemon=attacker.nickname, move=move.name))
        return

    if rampaging or releasing_charge:
        pass  # a rampage/charge continues the original use: no further PP is spent
    elif attacker.pp[slot] == 0:
        # Struggle is legal when constrained (choice/encore lock) into an empty move,
        # or when every move is exhausted; anything else is an illegal action.
        constrained = slot in (attacker.choice_locked_move, attacker.encored_slot)
        assert constrained or all(remaining == 0 for remaining in attacker.pp.values())
        move = _STRUGGLE
    else:
        cost = 2 if defender.ability is Ability.PRESSURE and move.target in _DEFENDER_FACING_TARGETS else 1
        attacker.pp[slot] = max(0, attacker.pp[slot] - cost)
        if attacker.pp[slot] == 0 and attacker.item is Item.LEPPA_BERRY:
            attacker.consume_item()
            attacker.pp[slot] = min(10, move.pp)
            log.add(PpRestored(side=side_index, pokemon=attacker.nickname, move=move.name))

    unleashed_as: str | None = None
    if action.z_move and not side.has_used_z_move:
        # Spent after the base move's PP is paid: a Z-move costs the slot it upgrades, not extra.
        upgraded = z_move_for(attacker.item, move)
        if upgraded is not None:
            # The slot keeps its own move for logging: `BattleObserver` treats every MoveUsed name as
            # a move that fills a slot, and a Z-move name fills none — believing one would make the
            # opponent's set unbuildable. Using the Z-move still reveals the base move, which is the
            # true inference anyway. `unleashed_as` carries the Z-move's own name through to whichever
            # MoveUsed below actually fires, so it reads as one move becoming its Z-move, not two.
            unleashed_as = upgraded.name
            move = replace(upgraded, name=move.name)
            side.has_used_z_move = True

    if ExtraStatus.TAUNT in attacker.volatiles and move.category is Category.STATUS:
        log.add(TauntBlocked(side=side_index, pokemon=attacker.nickname, move=move.name))
        return

    if (
        attacker.ability is Ability.PRANKSTER
        and move.category is Category.STATUS
        and move.target in _DEFENDER_FACING_TARGETS
        and Type.DARK in defender.types
    ):
        log.add(MoveUsed(side=side_index, pokemon=attacker.nickname, move=move.name, unleashed_as=unleashed_as))
        log.add(DoesNotAffect(side=defender_index, pokemon=defender.nickname))
        return

    log.add(MoveUsed(side=side_index, pokemon=attacker.nickname, move=move.name, unleashed_as=unleashed_as))
    # Drawn before the move resolves, so Blade forme's Attack is what the hit is calculated from —
    # which is the whole of Aegislash, and why it was rated as a wall that cannot hurt anything.
    stance = stance_forme(attacker, move)
    if stance is not None:
        swap_forme(state.bus, state.effects, attacker, stance, side_index, log)
    attacker.last_move_slot = slot
    if move.name not in ESCALATING_MOVES:
        attacker.rolling_hits = 0  # any other move ends the run, so the next one starts from base
    if attacker.item in _CHOICE_ITEMS and attacker.choice_locked_move is None:
        attacker.choice_locked_move = action.move

    if sleep_talking:
        sleep_choice = _sleep_talk_choice(attacker, state.rng)
        if sleep_choice is None:
            log.add(MoveFailed())
            return
        move = sleep_choice
        log.add(MoveUsed(side=side_index, pokemon=attacker.nickname, move=move.name))

    override_type = move_type_override(move, attacker, state)
    if override_type is not None and move.type is not override_type:
        move = replace(move, type=override_type)

    if move.charge and not releasing_charge and not _skips_charge_turn(move, attacker, state, side_index, log):
        attacker.volatiles[ExtraStatus.CHARGING] = 1
        attacker.charging_slot = slot
        log.add(ChargingUp(side=side_index, pokemon=attacker.nickname, move=move.name))
        return
    if releasing_charge:
        attacker.volatiles.pop(ExtraStatus.CHARGING, None)
        attacker.charging_slot = None

    if not _stall_check(move, attacker, state.rng):
        log.add(MoveFailed())
        return

    if coded_move_fails(move, attacker, defender, state):
        log.add(MoveFailed())
        return

    if move.priority > 0 and defender.ability in _PRIORITY_BLOCKERS and move.target in _DEFENDER_FACING_TARGETS:
        log.add(DoesNotAffect(side=defender_index, pokemon=defender.nickname))
        return

    if move.reflectable and defender.ability is Ability.MAGIC_BOUNCE:
        log.add(MoveBounced(side=defender_index, pokemon=defender.nickname))
        attacker, defender = defender, attacker
        side_index, defender_index = defender_index, side_index
        side, opponent_side = opponent_side, side

    if move.protectable and move.target in _DEFENDER_FACING_TARGETS and ExtraStatus.PROTECT in defender.volatiles:
        log.add(Protected(side=defender_index, pokemon=defender.nickname))
        _break_rolling(attacker)
        _apply_crash_damage(move, attacker, side_index, log)
        return

    if move.target in _DEFENDER_FACING_TARGETS and _out_of_reach(defender, move):
        log.add(MoveMissed())
        _break_rolling(attacker)
        _apply_crash_damage(move, attacker, side_index, log)
        return

    if not _accuracy_check(move, attacker, defender, state.field, state.rng):
        log.add(MoveMissed())
        _break_rolling(attacker)
        _apply_crash_damage(move, attacker, side_index, log)
        return

    if not move.effects and not move.force_switch:
        log.add(MoveFailed())
        return

    damaging = any(isinstance(e, (DamageEffect, FixedDamageEffect)) for e in move.effects)
    if damaging or move.target in _DEFENDER_FACING_TARGETS:
        # Status moves face absorption/immunity too: Thunder Wave into Volt Absorb, anything into Good as Gold.
        ctx = EventContext(rng=state.rng, battle=state, log=log, actor=attacker, defender=defender, move=move)
        payload = state.bus.emit(
            Event.ON_BEFORE_MOVE,
            ctx,
            {"move_type": move.type, "category": move.category, "defender_index": defender_index},
        )
        if payload.get("absorbed", False):
            _apply_crash_damage(move, attacker, side_index, log)
            return
    if (
        damaging
        and move.name not in _DELAYED_DAMAGE_MOVES  # effectiveness is checked fresh when it actually lands
        and move_effectiveness(move, attacker, defender) == 0
    ):
        log.add(NoEffect(side=defender_index, pokemon=defender.nickname))
        _apply_crash_damage(move, attacker, side_index, log)
        return

    if move.name in _SCREEN_BREAKERS:
        for screen in list(opponent_side.screens):
            del opponent_side.screens[screen]
            log.add(ScreenFaded(side=defender_index, screen=screen))

    # Substitute presence at move start governs the whole move: a hit that breaks
    # the substitute still has its secondaries blocked (PS behaviour).
    behind_substitute = (
        ExtraStatus.SUBSTITUTE in defender.volatiles
        and not move.bypass_substitute
        and attacker.ability is not Ability.INFILTRATOR
    )
    # A disguise takes the whole move, not merely its damage. Decided here rather than inside the
    # damage effect for the same reason the substitute is: by the time the flinch is reached the
    # disguise has already busted, so a Mimikyu that was never actually hit would flinch anyway.
    # Showdown does this by returning 0 from Disguise's `onTryHit`, which skips secondaries too.
    disguised = _disguise_intercepts(defender, move)
    log_before = len(log)
    for effect in move.effects:
        if move.name in _DELAYED_DAMAGE_MOVES and isinstance(effect, DamageEffect):
            continue  # the CodedEffect alongside it queues this for engine.residuals instead
        if defender.is_fainted() and _targets_defender(effect, move):
            continue  # self/side effects (boosts, hazard clearing) still apply after a KO
        if disguised and _targets_defender(effect, move) and not isinstance(effect, DamageEffect):
            continue  # the damage effect still runs — absorbing it is what breaks the disguise
        _apply_effect(effect, move, attacker, side_index, defender, opponent_side, state, log, behind_substitute)

    if move.force_switch and not defender.is_fainted():
        _force_random_switch(state, defender_index, log)

    if len(log) == log_before:
        log.add(MoveFailed())
        _apply_crash_damage(move, attacker, side_index, log)
        return

    if move.name in ROLLING_MOVES:
        _continue_rolling(attacker, slot)
    elif move.name in ESCALATING_MOVES:
        attacker.rolling_hits += 1  # Fury Cutter counts its run without being locked into it
    if move.recharges:
        attacker.volatiles[ExtraStatus.MUST_RECHARGE] = 1
    if move.self_destructs and not attacker.is_fainted():
        attacker.apply_damage(attacker.live_stats.HP)
        log.add(Fainted(side=side_index, pokemon=attacker.nickname))

    has_healthy_bench = any(i != side.active[0] and not p.is_fainted() for i, p in enumerate(side.team))
    if move.self_switch and not attacker.is_fainted() and has_healthy_bench:
        side.needs_switch = True
        log.add(SelfSwitchPending(side=side_index, pokemon=attacker.nickname))


def _apply_crash_damage(move: Move, attacker: Pokemon, side_index: int, log: BattleLog) -> None:
    """(High) Jump Kick: half the user's own max HP, rounded down, whenever the attack does not
    land — blocked by Protect/Detect, a miss, absorbed by an ability, or any other failure — same as
    every indirect source of damage, Magic Guard blocks it."""
    if not move.has_crash_damage or attacker.ability is Ability.MAGIC_GUARD:
        return
    crash = max(1, attacker.stat_totals.HP // 2)
    dealt = attacker.apply_damage(crash)
    log.add(RecoilDamage(side=side_index, pokemon=attacker.nickname, amount=dealt))
    if attacker.is_fainted():
        log.add(Fainted(side=side_index, pokemon=attacker.nickname))


def _sleep_talk_choice(attacker: Pokemon, rng: RNG) -> Move | None:
    options = [m for m in attacker.known_moves() if m.name != "Sleep Talk" and not m.charge]
    if not options:
        return None
    return options[rng.random_integer(0, len(options))]


_CHARGE_TURN_BOOSTS: dict[str, Stats] = {"Meteor Beam": Stats.SP_ATTACK, "Electro Shot": Stats.SP_ATTACK}
_SUN_SKIP_CHARGE = frozenset({"Solar Beam", "Solar Blade"})
_SCREEN_BREAKERS = frozenset({"Psychic Fangs", "Raging Bull", "Brick Break"})


def _skips_charge_turn(move: Move, attacker: Pokemon, state: BattleState, side_index: int, log: BattleLog) -> bool:
    boosted_stat = _CHARGE_TURN_BOOSTS.get(move.name)
    if boosted_stat is not None:
        apply_stage_changes(attacker, side_index, {boosted_stat: 1}, log, inflicted_by_opponent=False, source="move")
    if move.name in _SUN_SKIP_CHARGE and effective_weather(state) in (Weather.SUN, Weather.HARSH_SUN):
        return True
    if attacker.item is Item.POWER_HERB:
        attacker.consume_item()
        return True
    return False


def _disguise_intercepts(defender: Pokemon, move: Move) -> bool:
    """Whether an intact disguise is about to take this move on the defender's behalf.

    Asked before the move resolves, so callers can tell a move that was absorbed from one that
    landed — `damage_apply._disguise_absorbs` does the absorbing itself, and by then the forme has
    already changed and the question can no longer be answered.

    Damaging moves only, which is the Gen 7 rule: a status move goes straight through a disguise.
    """
    from battle_sim.formes import disguise_broken

    if defender.ability is not Ability.DISGUISE or disguise_broken(defender) or move.target is Target.SELF:
        return False
    return any(isinstance(effect, (DamageEffect, FixedDamageEffect)) for effect in move.effects)


def _targets_defender(effect: MoveEffect, move: Move) -> bool:
    if isinstance(effect, (DamageEffect, FixedDamageEffect)):
        return True
    if isinstance(effect, InflictStatusEffect):
        return move.target is not Target.SELF
    if isinstance(effect, StatStageChangeEffect):
        return effect.target == "TARGET"
    return False


def _apply_status_routed(
    effect: InflictStatusEffect,
    move: Move,
    attacker: Pokemon,
    attacker_side_index: int,
    defender: Pokemon,
    state: BattleState,
    log: BattleLog,
    behind_substitute: bool,
) -> None:
    teams = (state.sides[0].team, state.sides[1].team)
    if move.target is Target.SELF or effect.to_self:
        _apply_status(effect, attacker, attacker_side_index, state.rng, log, teams, field=state.field)
    elif not behind_substitute:
        _apply_status(
            effect, defender, 1 - attacker_side_index, state.rng, log, teams, inflictor=attacker, field=state.field
        )


def _apply_heal(effect: HealEffect, move: Move, attacker: Pokemon, attacker_side_index: int, log: BattleLog) -> None:
    healed = attacker.apply_healing(max(1, int(attacker.stat_totals.HP * effect.fraction)))
    if healed > 0:
        log.add(Healed(side=attacker_side_index, pokemon=attacker.nickname, amount=healed))
    if move.name == ROOST:
        # The bird comes down to heal. Its Flying type is ignored for the rest of the turn, which is
        # the whole cost of the move: a roosting Zapdos is Electric alone, so Earthquake lands on it
        # and Ice Beam stops being doubly effective. `Pokemon.battle_types` reads the volatile;
        # `_tick_volatiles` drops it at the end of the turn.
        attacker.volatiles[ExtraStatus.ROOSTED] = 1


def _continue_rolling(attacker: Pokemon, slot: MoveSlot) -> None:
    """Bank a connected Rollout and commit the user to the run.

    The lock is what pays for the doubling: the games give no way out of a Rollout once it is
    rolling, and `legal_actions` reads `locked_slot` to enforce exactly that.
    """
    attacker.rolling_hits += 1
    if ExtraStatus.LOCKED_MOVE not in attacker.volatiles:
        attacker.volatiles[ExtraStatus.LOCKED_MOVE] = ROLLING_LOCK_TURNS
        attacker.locked_slot = slot


def _out_of_reach(defender: Pokemon, move: Move) -> bool:
    """Whether the defender is mid-charge somewhere this move cannot follow.

    Fly, Dig and the rest spend their charge turn out of reach entirely, which is most of why they
    are worth a slot at all — without it they were two-turn moves that paid the cost and got none of
    the protection. A handful of moves reach anyway, and each of those is the well-known answer to
    the move it beats (Earthquake on a Dig, Surf on a Dive), so they are worth carrying.
    """
    if ExtraStatus.CHARGING not in defender.volatiles or defender.charging_slot is None:
        return False
    charging = defender.moves[defender.charging_slot]
    if charging is None or charging.name not in _REACHES_THROUGH:
        return False  # an ordinary charge — a Solar Beam's user stands there in plain sight
    return move.name not in _REACHES_THROUGH[charging.name]


def _break_rolling(attacker: Pokemon) -> None:
    """A miss or a Protect ends the run, power and commitment together."""
    attacker.rolling_hits = 0
    if attacker.locked_slot is not None and _is_rolling_slot(attacker):
        del attacker.volatiles[ExtraStatus.LOCKED_MOVE]
        attacker.locked_slot = None


def _is_rolling_slot(attacker: Pokemon) -> bool:
    if attacker.locked_slot is None or ExtraStatus.LOCKED_MOVE not in attacker.volatiles:
        return False
    locked = attacker.moves[attacker.locked_slot]
    return locked is not None and locked.name in ROLLING_MOVES


def _greedy_gourmand(attacker: Pokemon, side_index: int, state: BattleState, log: BattleLog) -> bool:
    """Trick something onto the butler and he spends his turn eating it. True if he did.

    Not a move and not a choice he can be talked out of — the turn an item is forced into his paws
    is the turn it goes, and it costs him the action he had planned. `mechanics.priority` sees the
    Trick coming and sorts him after it, so the trade has always landed by the time this is asked:
    he moves second that turn by construction, however fast he is.

    Only an item somebody handed him. What he brought to the field is his own and stays where it is,
    or he would spend every battle devouring his own Monocle.
    """
    eaten = attacker.tricked_item
    if attacker.ability is not Ability.NINE_LIVES or eaten is Item.NONE:
        return False
    attacker.tricked_item = Item.NONE
    if attacker.item is not eaten:
        return False  # already gone — knocked off, or tricked away again before he could get to it
    from battle_sim.mechanics.effects import rewire_active  # lazy: avoids an engine->mechanics cycle

    attacker.item = Item.NONE
    attacker.item_consumed = True
    rewire_active(state.bus, state.effects, attacker)
    healed = attacker.apply_healing(max(1, attacker.stat_totals.HP // GREEDY_GOURMAND_DIVISOR))
    log.add(ItemDevoured(side=side_index, pokemon=attacker.nickname, item=eaten))
    if healed > 0:
        log.add(Healed(side=side_index, pokemon=attacker.nickname, amount=healed))
    return True


def _can_act(  # noqa: PLR0913 — one argument per gate; they are independent and all belong here
    pokemon: Pokemon,
    side_index: int,
    rng: RNG,
    log: BattleLog,
    sleep_talking: bool = False,
    defrosting: bool = False,
) -> bool:
    if ExtraStatus.MUST_RECHARGE in pokemon.volatiles:
        del pokemon.volatiles[ExtraStatus.MUST_RECHARGE]
        log.add(CantAct(side=side_index, pokemon=pokemon.nickname, reason="recharge"))
        return False
    if pokemon.ability is Ability.TRUANT and not _truant_allows_acting(pokemon, side_index, log):
        return False
    if not _status_allows_acting(pokemon, side_index, rng, log, sleep_talking, defrosting):
        return False
    if ExtraStatus.FLINCH in pokemon.volatiles:
        log.add(CantAct(side=side_index, pokemon=pokemon.nickname, reason="flinch"))
        return False
    if ExtraStatus.CONFUSION in pokemon.volatiles and not _confusion_allows_acting(pokemon, side_index, rng, log):
        return False
    if pokemon.status is Status.PARALYSIS and rng.roll_chance(0.25):
        log.add(CantAct(side=side_index, pokemon=pokemon.nickname, reason="paralysis"))
        return False
    return True


def _truant_allows_acting(pokemon: Pokemon, side_index: int, log: BattleLog) -> bool:
    """Truant alternates: loaf, then act, then loaf again. `volatiles.clear()` on switch-out means a
    fresh stint always starts able to act, exactly as it does when Slaking is first sent out."""
    if ExtraStatus.LOAFING in pokemon.volatiles:
        del pokemon.volatiles[ExtraStatus.LOAFING]
        log.add(CantAct(side=side_index, pokemon=pokemon.nickname, reason="loafing"))
        return False
    pokemon.volatiles[ExtraStatus.LOAFING] = 1
    return True


def _status_allows_acting(
    pokemon: Pokemon, side_index: int, rng: RNG, log: BattleLog, sleep_talking: bool, defrosting: bool = False
) -> bool:
    if pokemon.status is Status.FREEZE:
        # Freeze has no minimum duration, unlike sleep: it is an independent 20% roll at every move
        # attempt, so thawing on the very turn it was inflicted is legal and simply unlucky for the
        # freezer. What is *not* a roll is a defrosting move — Flame Wheel, Sacred Fire, Scald and
        # the rest melt their own user free and go off anyway.
        if not defrosting and not rng.roll_chance(0.2):
            log.add(CantAct(side=side_index, pokemon=pokemon.nickname, reason="frozen"))
            return False
        pokemon.status = Status.NONE
        log.add(StatusCleared(side=side_index, pokemon=pokemon.nickname, clearance="thawed"))
    if pokemon.status is Status.SLEEP:
        pokemon.status_turns -= 1
        if pokemon.status_turns <= 0:
            pokemon.status = Status.NONE
            pokemon.status_turns = 0
            log.add(StatusCleared(side=side_index, pokemon=pokemon.nickname, clearance="woke"))
        elif not sleep_talking:  # Sleep Talk acts through the sleep
            log.add(CantAct(side=side_index, pokemon=pokemon.nickname, reason="asleep"))
            return False
    return True


def _confusion_allows_acting(pokemon: Pokemon, side_index: int, rng: RNG, log: BattleLog) -> bool:
    pokemon.volatiles[ExtraStatus.CONFUSION] -= 1
    if pokemon.volatiles[ExtraStatus.CONFUSION] <= 0:
        del pokemon.volatiles[ExtraStatus.CONFUSION]
        log.add(StatusCleared(side=side_index, pokemon=pokemon.nickname, clearance="confusion_ended"))
    elif rng.roll_chance(1 / 3):
        damage = _confusion_self_damage(pokemon)
        pokemon.apply_damage(damage)
        log.add(ConfusionSelfHit(side=side_index, pokemon=pokemon.nickname, amount=damage))
        return False
    else:
        log.add(CantAct(side=side_index, pokemon=pokemon.nickname, reason="confused"))
    return True


def _confusion_self_damage(pokemon: Pokemon) -> int:
    attack = pokemon.effective_stat(Stats.ATTACK)
    defense = pokemon.effective_stat(Stats.DEFENCE)
    base = ((2 * pokemon.level) // 5 + 2) * 40 * attack // defense // 50 + 2
    return max(1, base)


def _stall_check(move: Move, attacker: Pokemon, rng: RNG) -> bool:
    """Consecutive Protect-likes fail with odds 1 - 1/3^n (PS caps the counter at 729 = 3^6)."""
    if not move.stalling:
        attacker.protect_streak = 0
        return True
    if attacker.protect_streak > 0 and not rng.roll_chance(3.0 ** -min(attacker.protect_streak, 6)):
        attacker.protect_streak = 0
        return False
    attacker.protect_streak += 1
    return True


# Weather that makes the move an automatic hit, and weather that instead halves its accuracy —
# Thunder and Hurricane share both (rain guarantees a hit, sun halves it); Blizzard only ever gets
# the guarantee, never the penalty, and only from hail.
_WEATHER_ALWAYS_HITS: dict[str, frozenset[Weather]] = {
    "Thunder": frozenset({Weather.RAIN, Weather.HEAVY_RAIN}),
    "Hurricane": frozenset({Weather.RAIN, Weather.HEAVY_RAIN}),
    "Blizzard": frozenset({Weather.SNOW}),
}
_WEATHER_HALVES_ACCURACY: dict[str, frozenset[Weather]] = {
    "Thunder": frozenset({Weather.SUN, Weather.HARSH_SUN}),
    "Hurricane": frozenset({Weather.SUN, Weather.HARSH_SUN}),
}
_GUARANTEED = 1e6  # pushes any nonzero base accuracy past the final min(1.0, ...) clamp


def _weather_accuracy_multiplier(move: Move, weather: Weather) -> float:
    """What weather alone does to `move`'s accuracy multiplier: huge (so the final clamp guarantees
    a hit) for Thunder/Hurricane's rain and Blizzard's hail, 0.5 for Thunder/Hurricane's sun, else
    no effect. Folded into the ordinary multiplier chain rather than an early return, so a move that
    both always-hits in this weather and would otherwise have missed still reads as one accuracy
    roll, not a special case."""
    if weather in _WEATHER_ALWAYS_HITS.get(move.name, frozenset()):
        return _GUARANTEED
    if weather in _WEATHER_HALVES_ACCURACY.get(move.name, frozenset()):
        return 0.5
    return 1.0


def _accuracy_check(move: Move, attacker: Pokemon, defender: Pokemon, field: FieldState, rng: RNG) -> bool:
    if move.accuracy_probability is None:
        return True
    if Ability.NO_GUARD in (attacker.ability, defender.ability):
        return True
    net_stage = max(-6, min(6, attacker.stat_stages.ACCURACY - defender.stat_stages.EVASION))
    multiplier = (3 + net_stage) / 3 if net_stage >= 0 else 3 / (3 - net_stage)
    multiplier *= _weather_accuracy_multiplier(move, field.weather)
    if defender.ability is Ability.SAND_VEIL and field.weather is Weather.SANDSTORM:
        multiplier *= 0.8
    if defender.ability is Ability.SNOW_CLOAK and field.weather is Weather.SNOW:
        multiplier *= 0.8
    if defender.ability is Ability.TANGLED_FEET and ExtraStatus.CONFUSION in defender.volatiles:
        multiplier *= 0.8
    if attacker.ability is Ability.COMPOUND_EYES:
        multiplier *= 1.3
    if attacker.ability is Ability.HUSTLE and move.category is Category.PHYSICAL:
        multiplier *= 0.8
    if attacker.ability is Ability.VICTORY_STAR:
        multiplier *= 1.1
    if attacker.item is Item.WIDE_LENS:
        multiplier *= 1.1
    return rng.roll_chance(min(1.0, move.accuracy_probability * multiplier))  # boosted past 100% always hits


def _apply_effect(
    effect: MoveEffect,
    move: Move,
    attacker: Pokemon,
    attacker_side_index: int,
    defender: Pokemon,
    defender_side: SideState,
    state: BattleState,
    log: BattleLog,
    behind_substitute: bool,
) -> None:
    if isinstance(effect, DamageEffect):
        _apply_damage(
            effect, move, attacker, attacker_side_index, defender, defender_side, state, log, behind_substitute
        )
    elif isinstance(effect, FixedDamageEffect):
        _apply_fixed_damage(effect, move, attacker, defender, 1 - attacker_side_index, log, behind_substitute, state)
    elif isinstance(effect, HealEffect):
        _apply_heal(effect, move, attacker, attacker_side_index, log)
    elif isinstance(effect, (InflictStatusEffect, StatStageChangeEffect)):
        _apply_status_or_stages(effect, move, attacker, attacker_side_index, defender, state, log, behind_substitute)
    elif isinstance(effect, RemoveHazardsEffect):
        _apply_remove_hazards(effect, attacker, attacker_side_index, state, log)
    elif isinstance(effect, (WeatherEffect, TerrainEffect, PseudoWeatherEffect)):
        _apply_field_effect(effect, attacker, state, log)
    elif isinstance(effect, SideConditionEffect):
        _apply_side_condition(effect, move, attacker, attacker_side_index, state, log)
    elif isinstance(effect, CodedEffect):
        _apply_coded(effect, move, attacker, attacker_side_index, defender, state, log, behind_substitute)


def _apply_status_or_stages(
    effect: InflictStatusEffect | StatStageChangeEffect,
    move: Move,
    attacker: Pokemon,
    attacker_side_index: int,
    defender: Pokemon,
    state: BattleState,
    log: BattleLog,
    behind_substitute: bool,
) -> None:
    if isinstance(effect, InflictStatusEffect):
        tuned_status = _tuned_status_secondary(effect, move, attacker, defender)
        if tuned_status is not None:
            _apply_status_routed(
                tuned_status, move, attacker, attacker_side_index, defender, state, log, behind_substitute
            )
        return
    tuned_stages = _tuned_stage_secondary(effect, attacker, defender)
    if tuned_stages is not None and (tuned_stages.target == "SELF" or not behind_substitute):
        _apply_stage_change(tuned_stages, attacker, attacker_side_index, defender, state.rng, log)


def _blocks_secondaries(defender: Pokemon) -> bool:
    return defender.ability is Ability.SHIELD_DUST or defender.item is Item.COVERT_CLOAK


def _tuned_status_secondary(
    effect: InflictStatusEffect, move: Move, attacker: Pokemon, defender: Pokemon
) -> InflictStatusEffect | None:
    if not effect.is_secondary:
        return effect
    if attacker.ability is Ability.SHEER_FORCE:
        return None  # the secondary is traded for Sheer Force's power boost
    if not effect.to_self and move.target is not Target.SELF and _blocks_secondaries(defender):
        return None
    if attacker.ability is Ability.SERENE_GRACE:
        return replace(effect, probability=min(1.0, effect.probability * 2))
    return effect


def _tuned_stage_secondary(
    effect: StatStageChangeEffect, attacker: Pokemon, defender: Pokemon
) -> StatStageChangeEffect | None:
    if not effect.is_secondary:
        return effect
    if attacker.ability is Ability.SHEER_FORCE:
        return None  # the secondary is traded for Sheer Force's power boost
    if effect.target == "TARGET" and _blocks_secondaries(defender):
        return None
    if attacker.ability is Ability.SERENE_GRACE:
        return replace(effect, probability=min(1.0, effect.probability * 2))
    return effect
