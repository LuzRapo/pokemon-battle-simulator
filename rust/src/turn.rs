//! One turn, in the order the Python resolves it.
//!
//! The order matters twice over. Once because it decides the battle, and once because **it decides
//! where the tape is**: every draw either engine makes has to be the same draw in the same place,
//! or the two immediately start reading each other's randomness. The sequence within a turn is
//!
//!   1. one tie-break probability per side, in `order_actions`
//!   2. then, per move that resolves: the accuracy roll (only if the move has an accuracy), the
//!      crit roll, and the damage roll — in that order, and only on the paths that reach them
//!
//! Every early return in the Python happens *before* its draw would have been taken, which is why
//! a miss costs one draw and a no-effect costs none. Porting that faithfully is most of the work.
//!
//! Restricted on purpose, for now: damaging moves and switches, no abilities, no items, no
//! volatiles, no residuals. Anything outside that raises `Unsupported` rather than guessing, so a
//! scenario that wanders out of the ported subset fails loudly instead of diverging quietly.

use crate::battle::{Pokemon, Side, State, Status};
use crate::abilities::{apply_damage_calc, Calc};
use crate::damage::{calculate_hit, Payload, Rolls};
use crate::hooks::{on_after_hit, on_switch_in, Hit};
use crate::data::{Database, Effect, Move};
use crate::log::{Event, Log};
use crate::tape::Tape;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    Move { slot: usize },
    Switch { to: usize },
}

/// Why this engine stopped, and the distinction the whole project rests on.
///
/// `Unported` means the scenario asked for something not written yet — a fair, expected answer
/// while most of the game is still missing, and one the harness may skip over. `Diverged` means
/// the two engines have already parted: the tape it is reading was recorded by a Python that made
/// different draws in a different order. That is never skippable. Collapsing the two was how five
/// real disagreements sat in a green test run labelled "outside the ported slice".
#[derive(Debug)]
pub enum Refusal {
    Unported(String),
    Diverged(String),
}

impl Refusal {
    pub fn reason(&self) -> &str {
        match self {
            Refusal::Unported(why) | Refusal::Diverged(why) => why,
        }
    }

    /// The process exit code, which is the only channel the Python harness reads this on.
    pub fn exit_code(&self) -> i32 {
        match self {
            Refusal::Unported(_) => 2,
            Refusal::Diverged(_) => 3,
        }
    }
}

/// What a side's action counts as for ordering: switches resolve before moves, as in
/// `_CATEGORY_ORDER`.
fn category_of(action: &Action) -> i32 {
    match action {
        Action::Switch { .. } => 0,
        Action::Move { .. } => 1,
    }
}

/// The export writes the bracket by name; these are the values of the Python's `PriorityLevel`,
/// taken from the enum rather than guessed — the first version of this invented plausible names
/// like "QUICK" and panicked on contact with the real data.
fn priority_of(the_move: &Move) -> Result<i32, Refusal> {
    Ok(match the_move.priority.as_str() {
        "HELPING_HAND" => 5,
        "PROTECT" => 4,
        "FAKE_OUT" => 3,
        "E_SPEED" => 2,
        "QUICK_ATTACK" => 1,
        "NORMAL" => 0,
        "VITAL_THROW" => -1,
        "FOCUS_PUNCH" => -3,
        "AVALANCHE" => -4,
        "COUNTER" => -5,
        "ROAR" => -6,
        "TRICK_ROOM" => -7,
        other => return Err(Refusal::Unported(format!("unmapped priority bracket {other:?}"))),
    })
}

/// Sorts exactly as `_sort_key` does: category, then priority (descending), then speed
/// (descending), then the tie-break draw. Rust sorts ascending, so speed and priority are negated
/// the same way the Python negates speed.
fn order_actions(state: &State, actions: &[Action; 2], db: &Database, tape: &mut Tape) -> Result<Vec<usize>, Refusal> {
    // Drawn for both sides before anything resolves, in side order — the Python builds this dict
    // by comprehension over `actions`, which is insertion-ordered 0 then 1.
    let tie_breakers = [tape.probability()?, tape.probability()?];
    let mut keys: Vec<(i32, i32, i32, f64, usize)> = Vec::new();
    for side in 0..2 {
        let actor = state.sides[side].active_pokemon();
        let speed = effective_speed(actor, &state.sides[side]);
        let priority = match &actions[side] {
            Action::Switch { .. } => 0,
            Action::Move { slot } => priority_of(move_in_slot(actor, *slot, db)?)?,
        };
        keys.push((category_of(&actions[side]), -priority, -speed, tie_breakers[side], side));
    }
    keys.sort_by(|a, b| a.partial_cmp(b).expect("no NaNs in a sort key"));
    Ok(keys.into_iter().map(|k| k.4).collect())
}

/// `priority.effective_speed`, as much of it as is ported.
///
/// Paralysis halves it, and that halving decides who moves first — which is not a detail. A
/// paralysed Kangaskhan that this engine let move first took a paralysis check off the tape at the
/// moment the Python was rolling accuracy for somebody else, and the two engines never recovered.
/// Everything else in the Python's version is an ability, an item or a field effect, none of them
/// ported, and each will have to be added here as it lands.
fn effective_speed(pokemon: &Pokemon, side: &Side) -> i32 {
    let mut speed = pokemon.effective("SPEED");
    if pokemon.status == Status::Paralysis {
        speed /= 2;
    }
    if side.tailwind_turns > 0 {
        speed *= 2;
    }
    std::cmp::max(1, speed)
}

fn move_in_slot<'a>(actor: &Pokemon, slot: usize, db: &'a Database) -> Result<&'a Move, Refusal> {
    let name = actor
        .moves
        .get(slot)
        .ok_or_else(|| Refusal::Unported(format!("{} has no move in slot {slot}", actor.nickname)))?;
    db.move_named(name)
        .ok_or_else(|| Refusal::Unported(format!("unknown move {name:?}")))
}

/// Everything this engine has not learned yet. A scenario containing one of these is refused up
/// front rather than played wrongly — the whole point of the differential work is that silence is
/// the one unacceptable failure mode.
/// Moves the Python special-cases by name that this engine has nonetheless implemented, and which
/// the `coded_moves` net must therefore stop refusing.
///
/// Deliberately short and explicit. That net is why silent wrong answers have been rare; a move
/// only comes off it once its Python behaviour has been read, ported, and agreed about across a
/// sweep.
pub const PORTED_CODED_MOVES: [&str; 1] = ["Struggle"];

/// The volatiles this engine knows. Everything else in `ExtraStatus` still makes a move unplayable.
///
/// These two are most of what volatiles actually are in practice: 32 moves can flinch and 18 can
/// confuse, against one apiece for Leech Seed, Taunt, Encore and the rest.
pub const PORTED_VOLATILES: [&str; 2] = ["FLINCH", "CONFUSION"];

/// Everything this engine has implemented, gathered from the modules that implement it so a name
/// cannot be claimed in one place and missing from the other.
///
/// A Pokemon carrying live behaviour absent from these is refused, not played with part of its
/// rules missing. 220 abilities and 110 items are live in the Python; these say how far along the
/// port is, and a name joins one only once it has been agreed across a sweep.
pub fn ported_abilities() -> Vec<&'static str> {
    let mut all: Vec<&str> = crate::abilities::PORTED.to_vec();
    all.extend(crate::hooks::PORTED_ABILITIES);
    all.sort_unstable();
    all
}

pub fn ported_items() -> Vec<&'static str> {
    let mut all: Vec<&str> = crate::items::PORTED.to_vec();
    all.extend(crate::hooks::PORTED_ITEMS);
    all.sort_unstable();
    all
}

/// Why this Pokemon cannot be played, if it cannot.
///
/// Reading an ability off a Pokemon and doing nothing with it is the exact failure this project is
/// built to catch — a wrong answer delivered in silence. Until Intimidate is written here, a
/// scenario containing one stops the run.
pub fn unsupported_pokemon(pokemon: &Pokemon, db: &Database) -> Option<String> {
    if db.live_abilities.contains(&pokemon.ability) && !ported_abilities().contains(&pokemon.ability.as_str()) {
        return Some(format!("{} has {}, which is not ported", pokemon.nickname, pokemon.ability));
    }
    if db.live_items.contains(&pokemon.item) && !ported_items().contains(&pokemon.item.as_str()) {
        return Some(format!("{} is holding {}, which is not ported", pokemon.nickname, pokemon.item));
    }
    None
}

pub fn unsupported_reason(the_move: &Move, db: &Database) -> Option<String> {
    if db.coded_moves.contains(&the_move.name) && !PORTED_CODED_MOVES.contains(&the_move.name.as_str()) {
        return Some(format!("{} is special-cased by name in the Python engine", the_move.name));
    }
    if the_move.self_switch
        || the_move.healing
        || the_move.force_switch
        || the_move.recharges
        || the_move.charge
        || the_move.self_destructs
    {
        return Some(format!("{} does something to its user or the field that is not ported", the_move.name));
    }
    if the_move.effects.is_empty() {
        return Some(format!("{} has no modelled effect", the_move.name));
    }
    for effect in &the_move.effects {
        match effect {
            Effect::DamageEffect { power: None, .. } => {
                return Some(format!("{}'s power is computed at use time", the_move.name))
            }
            Effect::DamageEffect { multi_hit: Some(_), .. } => {
                return Some(format!("{} hits more than once", the_move.name))
            }
            Effect::DamageEffect { drain_percent: Some(_), .. }
            | Effect::DamageEffect { recoil_percent: Some(_), .. } => {
                return Some(format!("{} drains or recoils", the_move.name))
            }
            Effect::InflictStatusEffect { status, .. }
                if Status::parse(status).is_none() && !PORTED_VOLATILES.contains(&status.as_str()) =>
            {
                return Some(format!("{} inflicts {status}, which is a volatile", the_move.name))
            }
            _ => {}
        }
    }
    None
}

pub fn step(
    state: &mut State,
    actions: [Action; 2],
    db: &Database,
    tape: &mut Tape,
) -> Result<Log, Refusal> {
    let mut log = Log::new();
    for side in 0..2 {
        state.sides[side].acted_this_turn = false;
    }
    // `_send_out_leads`: the leads' switch-in abilities fire before the turn is ordered, in
    // descending speed, so the slower weather-setter's weather is the one that stands. It takes no
    // draws, but it does log, and the entries belong at the very top of turn zero.
    if state.turn == 0 {
        let mut leads = [0usize, 1];
        leads.sort_by_key(|side| {
            std::cmp::Reverse(effective_speed(state.sides[*side].active_pokemon(), &state.sides[*side]))
        });
        for side in leads {
            on_switch_in(state, side, &mut log);
        }
    }
    let order = order_actions(state, &actions, db, tape)?;
    for side in order {
        if state.outcome.is_some() {
            break;
        }
        // A fainted Pokemon cannot move, but its side must still send out a replacement — and
        // that switch is the first thing to resolve. Skipping the side outright left the Rust
        // engine a Pokemon behind for the rest of the battle.
        if state.sides[side].active_pokemon().fainted() && matches!(actions[side], Action::Move { .. }) {
            continue;
        }
        match &actions[side] {
            Action::Switch { to } => {
                let sent_out = state.sides[side].team[*to].nickname.clone();
                let withdrew = switch_out(&mut state.sides[side], *to);
                state.register_active(side);
                state.sides[side].active_mut().just_switched_in = true;
                log.push(Event::Switched { side: side as i32, withdrew, sent_out });
                // `_execute_switch` logs the swap and then emits, so an Intimidate lands after the
                // line announcing who arrived. A Pokemon sent out already fainted is unregistered
                // instead of announced, which cannot happen here — a switch target is never one.
                on_switch_in(state, side, &mut log);
            }
            Action::Move { slot } => {
                // Whether the *chosen* move melts its own user free, decided before anything is
                // rolled: Flame Wheel, Sacred Fire and Scald thaw and go off anyway, with no 20%
                // check taken. The Python reads the chosen move here too, so a Struggle
                // substitution later does not change it.
                let defrosting = {
                    let actor = state.sides[side].active_pokemon();
                    actor.status == Status::Freeze && move_in_slot(actor, *slot, db)?.defrosts_user
                };
                if can_act(state, side, defrosting, tape, &mut log)? {
                    resolve_move(state, side, *slot, db, tape, &mut log)?
                }
            }
        }
        // Set after the action, not before: Analytic asks whether the *other* side has already
        // moved, and a side that has just finished moving is exactly what that means.
        state.sides[side].acted_this_turn = true;
        let was_decided = state.outcome.is_some();
        state.update_outcome();
        // `_update_outcome` announces the result the moment it is decided, once.
        if !was_decided {
            if let Some(outcome) = state.outcome {
                log.push(Event::BattleEnded { outcome: outcome.name().to_string() });
            }
        }
    }
    if state.outcome.is_none() {
        residuals(state, &mut log);
        state.update_outcome();
        if let Some(outcome) = state.outcome {
            log.push(Event::BattleEnded { outcome: outcome.name().to_string() });
        }
    }
    // `_tick_turns_active`, which runs at turn end whether or not the battle is over: the flag's
    // rule is "every turn-end except the one I entered on", so it clears here rather than when its
    // owner acted. Clearing it after the action instead cost Stakeout its whole effect, since the
    // Pokemon it punishes is the one that came in *this* turn and has not moved yet.
    for side in 0..2 {
        if !state.sides[side].active_pokemon().fainted() {
            state.sides[side].active_mut().just_switched_in = false;
        }
    }
    state.turn += 1;
    Ok(log)
}

/// `_confusion_allows_acting`: tick the counter, then a third of the time hurt yourself instead.
///
/// The odd branch is the last one, and it is the Python's: on the two-thirds where the Pokemon does
/// *not* hit itself, a `CantAct` entry is logged with reason "confused" and then the move goes off
/// anyway. Reproduced as written — this engine's job is to agree with that one.
fn confusion_allows_acting(state: &mut State, side: usize, tape: &mut Tape, log: &mut Log) -> Result<bool, Refusal> {
    let pokemon = state.sides[side].active_mut();
    let left = pokemon.volatiles.get("CONFUSION").copied().unwrap_or(0) - 1;
    pokemon.volatiles.insert("CONFUSION".to_string(), left);
    let nickname = pokemon.nickname.clone();
    if left <= 0 {
        state.sides[side].active_mut().volatiles.remove("CONFUSION");
        log.push(Event::StatusCleared {
            side: side as i32,
            pokemon: nickname,
            clearance: "confusion_ended".into(),
        });
        return Ok(true);
    }
    if tape.probability()? < 1.0 / 3.0 {
        let pokemon = state.sides[side].active_mut();
        // Its own Attack against its own Defence, 40 base power, typeless and never a crit.
        let attack = pokemon.effective("ATTACK");
        let defence = pokemon.effective("DEFENCE");
        let amount = std::cmp::max(1, ((2 * pokemon.level) / 5 + 2) * 40 * attack / defence / 50 + 2);
        pokemon.take_damage(amount);
        log.push(Event::ConfusionSelfHit { side: side as i32, pokemon: nickname, amount });
        return Ok(false);
    }
    log.push(Event::CantAct { side: side as i32, pokemon: nickname, reason: "confused".into() });
    Ok(true)
}

/// Withdraw whatever is active and send out `to`, returning the outgoing nickname.
///
/// The resets are `switching._execute_switch`'s, and they are the point of the function: a Pokemon
/// that leaves the field drops its stat stages, its volatiles and — the easy one to miss — its
/// toxic counter, so it comes back poisoned but counting from zero again. Leaving the stages on was
/// how a Rust battle reached -6 Special Attack against a Python that had long since reset to 0.
///
/// The rest of `_execute_switch` clears fields this engine does not have yet (choice lock, encore,
/// charging slot). They arrive with the volatiles milestone; until then there is nothing to clear.
fn switch_out(side: &mut Side, to: usize) -> String {
    let outgoing = side.active_mut();
    for value in outgoing.stages.values_mut() {
        *value = 0;
    }
    outgoing.volatiles.clear();
    if outgoing.status == Status::Toxic {
        outgoing.status_turns = 0;
    }
    let withdrew = outgoing.nickname.clone();
    side.active = to;
    withdrew
}

/// End-of-turn status chip, from `residuals._status_chip`. Burn is a sixteenth, poison an eighth,
/// and toxic climbs by a sixteenth a turn — counting up *before* it bites, which is why a fresh
/// toxic takes a sixteenth rather than nothing.
///
/// Sides are ticked in order, which is the order the Python emits `ON_RESIDUAL` for them.
fn residuals(state: &mut State, log: &mut Log) {
    for side in 0..2 {
        if state.sides[side].active_pokemon().fainted() {
            continue;
        }
        status_chip(state, side, log);
        // `_tick_volatiles`, which runs for the active whatever its status was — including one
        // that just fainted to the chip above. A flinch lasts exactly the turn it was inflicted.
        state.sides[side].active_mut().volatiles.remove("FLINCH");
    }
}

fn status_chip(state: &mut State, side: usize, log: &mut Log) {
    let active = state.sides[side].active_mut();
    let (source, amount) = match active.status {
        Status::Burn => ("burn", (active.totals.hp / 16).max(1)),
        Status::Poison => ("poison", (active.totals.hp / 8).max(1)),
        Status::Toxic => {
            active.status_turns += 1;
            ("toxic", (active.totals.hp * active.status_turns / 16).max(1))
        }
        _ => return,
    };
    let dealt = active.take_damage(amount);
    let nickname = active.nickname.clone();
    let fainted = active.fainted();
    log.push(Event::ResidualDamage {
        side: side as i32,
        pokemon: nickname.clone(),
        source: source.into(),
        amount: dealt,
    });
    if fainted {
        log.push(Event::Fainted { side: side as i32, pokemon: nickname });
    }
}

/// `_can_act`, minus the volatiles and abilities that are not ported. The order is the Python's
/// and so is where each draw falls: freeze rolls to thaw, sleep counts down without drawing, and
/// paralysis rolls last.
fn can_act(
    state: &mut State,
    side: usize,
    defrosting: bool,
    tape: &mut Tape,
    log: &mut Log,
) -> Result<bool, Refusal> {
    let (status, nickname) = {
        let active = state.sides[side].active_pokemon();
        (active.status, active.nickname.clone())
    };
    if status == Status::Freeze {
        // No draw at all when the move defrosts its user — which is the point. Rolling the check
        // anyway thawed the right Pokemon for the wrong reason and left the tape one draw short
        // for the rest of the battle.
        if !defrosting && tape.probability()? >= 0.2 {
            log.push(Event::CantAct { side: side as i32, pokemon: nickname, reason: "frozen".into() });
            return Ok(false);
        }
        state.sides[side].active_mut().status = Status::None;
        log.push(Event::StatusCleared {
            side: side as i32,
            pokemon: nickname.clone(),
            clearance: "thawed".into(),
        });
    }
    if state.sides[side].active_pokemon().status == Status::Sleep {
        let active = state.sides[side].active_mut();
        active.status_turns -= 1;
        if active.status_turns <= 0 {
            active.status = Status::None;
            active.status_turns = 0;
            log.push(Event::StatusCleared {
                side: side as i32,
                pokemon: nickname.clone(),
                clearance: "woke".into(),
            });
        } else {
            log.push(Event::CantAct { side: side as i32, pokemon: nickname, reason: "asleep".into() });
            return Ok(false);
        }
    }
    // Flinch, then confusion, then paralysis — the Python's order, and therefore the order the
    // draws come off the tape. Confusion rolls before paralysis does, which matters on any turn
    // where both could fire.
    if state.sides[side].active_pokemon().volatiles.contains_key("FLINCH") {
        log.push(Event::CantAct { side: side as i32, pokemon: nickname, reason: "flinch".into() });
        return Ok(false);
    }
    if state.sides[side].active_pokemon().volatiles.contains_key("CONFUSION")
        && !confusion_allows_acting(state, side, tape, log)?
    {
        return Ok(false);
    }
    if state.sides[side].active_pokemon().status == Status::Paralysis && tape.probability()? < 0.25 {
        let nickname = state.sides[side].active_pokemon().nickname.clone();
        log.push(Event::CantAct { side: side as i32, pokemon: nickname, reason: "paralysis".into() });
        return Ok(false);
    }
    Ok(true)
}

/// `_targets_defender`: which effects a knocked-out target stops taking.
///
/// A status effect is aimed at the defender unless the *move* targets its user — the effect's own
/// `to_self` flag is not what decides this in the Python, and using it here would let Hypnosis-like
/// self-targeting cases drift. A stat change is aimed at whoever the effect names.
fn targets_defender(effect: &Effect, the_move: &Move) -> bool {
    match effect {
        Effect::DamageEffect { .. } => true,
        Effect::InflictStatusEffect { .. } => the_move.target != "SELF",
        Effect::StatStageChangeEffect { target, .. } => target == "TARGET",
        Effect::Unmodelled => false,
    }
}

fn resolve_move(
    state: &mut State,
    side: usize,
    slot: usize,
    db: &Database,
    tape: &mut Tape,
    log: &mut Log,
) -> Result<(), Refusal> {
    let other = 1 - side;
    let chosen = {
        let actor = state.sides[side].active_pokemon();
        move_in_slot(actor, slot, db)?.clone()
    };
    // An empty slot is Struggle, and Struggle costs nothing — there is nothing left to spend. The
    // Python substitutes here rather than at choice time, so the recorded action still names the
    // move that was picked. Missing this only showed up past turn 60, once the PP had run out.
    let empty = state.sides[side].active_pokemon().pp.get(crate::battle::SLOT_NAMES[slot]) == Some(&0);
    let the_move = if empty {
        db.move_named("Struggle")
            .ok_or_else(|| Refusal::Unported("the database has no Struggle".into()))?
            .clone()
    } else {
        chosen
    };
    if let Some(why) = unsupported_reason(&the_move, db) {
        return Err(Refusal::Unported(why));
    }
    // Spent before anything resolves, as `_spend_pp` does it: a move that misses still costs its
    // point, which is why this is here rather than after the hit lands.
    if !empty {
        if let Some(left) = state.sides[side].active_mut().pp.get_mut(crate::battle::SLOT_NAMES[slot]) {
            *left = (*left - 1).max(0);
        }
    }
    log.push(Event::MoveUsed {
        side: side as i32,
        pokemon: state.sides[side].active_pokemon().nickname.clone(),
        the_move: the_move.name.clone(),
        unleashed_as: None,
    });

    // Accuracy first, and only when the move has one — `_accuracy_check` returns True without
    // drawing when `accuracy_probability` is None, which is how a never-missing move leaves the
    // tape untouched.
    if let Some(accuracy) = the_move.accuracy_probability {
        let attacker_stage = state.sides[side].active_pokemon().stage("ACCURACY");
        let defender_stage = state.sides[other].active_pokemon().stage("EVASION");
        let net = (attacker_stage - defender_stage).clamp(-6, 6);
        let multiplier = if net >= 0 { (3 + net) as f64 / 3.0 } else { 3.0 / (3 - net) as f64 };
        if tape.probability()? >= (accuracy * multiplier).min(1.0) {
            log.push(Event::MoveMissed);
            crash_damage(state, side, &the_move, log);
            return Ok(());
        }
    }

    let effectiveness = if the_move.typeless {
        1.0
    } else {
        db.effectiveness(&the_move.move_type, &state.sides[other].active_pokemon().types)
    };
    // The immunity gate is for *damaging* moves only, exactly as the Python writes it. Charge is
    // Electric and targets its user, so a Ground-type across the field does not stop it boosting —
    // and gating it here anyway skipped the boost's probability draw, which put every later draw
    // in the battle one place out.
    let damaging = the_move.effects.iter().any(|e| matches!(e, Effect::DamageEffect { .. }));
    if damaging && effectiveness == 0.0 {
        log.push(Event::NoEffect {
            side: other as i32,
            pokemon: state.sides[other].active_pokemon().nickname.clone(),
        });
        crash_damage(state, side, &the_move, log);
        return Ok(());
    }

    let log_before = log.entries.len();

    // Effects resolve in the order the move lists them, which is the order `_apply_effect` is
    // called in and therefore the order their draws come off the tape.
    for effect in &the_move.effects {
        // A knocked-out target takes no more of the move — not the burn from Steam Eruption, not
        // the speed drop from Icy Wind. What still lands is anything aimed elsewhere: the user's
        // own boost, a hazard, a side effect. Skipping the effect has to skip its probability draw
        // too, which is how this was found: the Python stopped after the faint and this engine
        // rolled on, so every draw from there wasread out of another turn.
        if state.sides[other].active_pokemon().fainted() && targets_defender(effect, &the_move) {
            continue;
        }
        match effect {
            Effect::DamageEffect { .. } => {
                log.effectiveness(effectiveness);
                apply_damage(state, side, &the_move, db, tape, log)?;
            }
            Effect::InflictStatusEffect { status, probability, to_self, .. } => {
                apply_status(state, side, status, *probability, *to_self, tape, log)?;
            }
            Effect::StatStageChangeEffect { stages, probability, target, .. } => {
                apply_stages(state, side, stages, *probability, target, tape, log)?;
            }
            Effect::Unmodelled => return Err(Refusal::Unported(format!("{} has an unmodelled effect", the_move.name))),
        }
    }

    // Nothing at all happened: every effect was skipped, most often because the target had already
    // been knocked out by the other side this turn. The Python decides this by whether the log grew
    // rather than by inspecting the move, so this does too.
    if log.entries.len() == log_before {
        log.push(Event::MoveFailed);
        crash_damage(state, side, &the_move, log);
    }
    Ok(())
}

/// (High) Jump Kick and friends: half the user's own max HP whenever the attack does not land.
///
/// It applies to a miss, to an immunity, and to a move that simply did nothing — every way of
/// failing, which is why the Python calls it from six places and why it is a separate function
/// here rather than inlined into the miss path.
fn crash_damage(state: &mut State, side: usize, the_move: &Move, log: &mut Log) {
    if !the_move.has_crash_damage {
        return;
    }
    let attacker = state.sides[side].active_mut();
    let amount = std::cmp::max(1, attacker.totals.hp / 2);
    let dealt = attacker.take_damage(amount);
    let nickname = attacker.nickname.clone();
    let fainted = attacker.fainted();
    log.push(Event::RecoilDamage { side: side as i32, pokemon: nickname.clone(), amount: dealt });
    if fainted {
        log.push(Event::Fainted { side: side as i32, pokemon: nickname });
    }
}

/// `_thaw_on_hit`: the hit itself melting a frozen defender.
fn thaw_on_hit(state: &mut State, defender_side: usize, the_move: &Move, log: &mut Log) {
    let defender = state.sides[defender_side].active_pokemon();
    if defender.status != Status::Freeze || defender.fainted() {
        return;
    }
    if !(the_move.thaws_target || (the_move.move_type == "FIRE" && the_move.category != "STATUS")) {
        return;
    }
    let defender = state.sides[defender_side].active_mut();
    defender.status = Status::None;
    defender.status_turns = 0;
    let nickname = defender.nickname.clone();
    log.push(Event::StatusCleared {
        side: defender_side as i32,
        pokemon: nickname,
        clearance: "thawed".into(),
    });
}

/// Everything the abilities and items want to say about this hit, gathered before the formula runs.
///
/// The Python emits `ON_DAMAGE_CALC` with a base payload and lets handlers fill it in; this builds
/// the same base and walks the same handlers in the same order. `contact` is a property of the hit
/// rather than of the move, which is why it is computed here and passed along.
/// The category and contact flag of the move's first damage effect — the two facts both the
/// damage payload and the after-hit handlers ask about.
fn hit_shape(the_move: &Move) -> (&str, bool) {
    the_move
        .effects
        .iter()
        .find_map(|e| match e {
            Effect::DamageEffect { category, contact, .. } => Some((category.as_str(), *contact)),
            _ => None,
        })
        .unwrap_or(("STATUS", false))
}

fn collect_damage_payload(
    state: &State,
    side: usize,
    the_move: &Move,
    db: &Database,
) -> (Payload, Option<crate::items::Consumed>) {
    let other = 1 - side;
    let attacker = state.sides[side].active_pokemon();
    let defender = state.sides[other].active_pokemon();
    let (category, contact) = hit_shape(the_move);
    let calc = Calc {
        move_type: &the_move.move_type,
        category,
        // Protective Pads, Long Reach and a Punching Glove all clear the contact flag. All three
        // are still refused, so none of them can be on the field; they belong here when they land.
        contact,
        the_move,
        attacker,
        defender,
        fallen_on_attacker_side: state.sides[side].team.iter().filter(|p| p.fainted()).count(),
        defender_side_acted: state.sides[other].acted_this_turn,
        weather: &state.field.weather,
        db,
    };
    let mut payload = Payload::new();
    let consumed = apply_damage_calc(state, side, &calc, &mut payload);
    (payload, consumed)
}

fn apply_damage(
    state: &mut State,
    side: usize,
    the_move: &Move,
    db: &Database,
    tape: &mut Tape,
    log: &mut Log,
) -> Result<(), Refusal> {
    let other = 1 - side;
    // Drawn here, in the Python's order: the crit first, then the damage roll. Both are certain
    // to be consumed by the time the formula is entered — see `Rolls`.
    let rolls = Rolls { crit: tape.probability()?, damage: tape.integer(85, 101)? };
    let (payload, eaten) = collect_damage_payload(state, side, the_move, db);
    // The berry is spent whether or not the hit goes on to kill, exactly where the Python's
    // handler spends it: during the calculation, before the damage lands.
    if let Some(consumed) = eaten {
        let holder = state.sides[consumed.side].active_mut();
        crate::items::consume(holder, consumed.side, &consumed.item, log);
    }
    let hit = {
        let (mine, theirs) = state.sides.split_at(1);
        let (attacker, defender_side) = if side == 0 {
            (mine[0].active_pokemon(), &theirs[0])
        } else {
            (theirs[0].active_pokemon(), &mine[0])
        };
        calculate_hit(
            attacker,
            defender_side.active_pokemon(),
            the_move,
            &state.field,
            defender_side,
            db,
            rolls,
            &payload,
        )
    };
    let dealt = state.sides[other].active_mut().take_damage(hit.amount);
    // Inside the per-hit loop in the Python, which for a single hit means *before* the crit and
    // damage entries that get logged after it. So Rough Skin's chip is announced before the damage
    // that caused it, and a berry is eaten before the number that made it ripen is printed.
    let (category, contact) = hit_shape(the_move);
    let shape = Hit { attacker_side: side, move_type: &the_move.move_type, category, contact, dealt };
    on_after_hit(state, &shape, db, tape, log)?;
    if hit.is_crit {
        log.push(Event::CriticalHit);
    }
    log.push(Event::DamageDealt {
        side: other as i32,
        pokemon: state.sides[other].active_pokemon().nickname.clone(),
        amount: dealt,
    });
    // Between the damage and the faint, in that order, as `_thaw_on_hit` sits between them: a
    // frozen defender that takes a Fire move — or one of the three off-type thawers — is free
    // again, and then takes its turn normally instead of rolling the 20% check.
    thaw_on_hit(state, other, the_move, log);
    if state.sides[other].active_pokemon().fainted() {
        log.push(Event::Fainted {
            side: other as i32,
            pokemon: state.sides[other].active_pokemon().nickname.clone(),
        });
    }
    // Struggle's quarter, which the Python applies unconditionally — neither Magic Guard nor Rock
    // Head stops it — and logs as the amount it asked for rather than the amount that landed.
    if the_move.effects.iter().any(|e| matches!(e, Effect::DamageEffect { struggle_recoil: true, .. })) {
        let attacker = state.sides[side].active_mut();
        let recoil = std::cmp::max(1, attacker.totals.hp / 4);
        attacker.take_damage(recoil);
        let nickname = attacker.nickname.clone();
        log.push(Event::RecoilDamage { side: side as i32, pokemon: nickname, amount: recoil });
    }
    Ok(())
}

/// `_apply_status`, which draws its probability *first and always* — even at 1.0, which is why a
/// guaranteed status still costs a place on the tape.
fn apply_status(
    state: &mut State,
    side: usize,
    status_name: &str,
    probability: f64,
    to_self: bool,
    tape: &mut Tape,
    log: &mut Log,
) -> Result<(), Refusal> {
    if tape.probability()? >= probability {
        return Ok(());
    }
    let target_side = if to_self { side } else { 1 - side };
    match Status::parse(status_name) {
        Some(status) => apply_main_status(state, target_side, status, tape, log),
        None if PORTED_VOLATILES.contains(&status_name) => {
            apply_volatile(state, target_side, status_name, tape, log)
        }
        None => Err(Refusal::Unported(format!("{status_name} is a volatile, which is not ported"))),
    }
}

/// `_apply_volatile`, for the two this engine knows.
///
/// A volatile already present is not re-applied and — this is the part that matters — takes no
/// duration draw either, because the Python's `elif effect.status not in target.volatiles` skips
/// the whole branch. Inner Focus and Own Tempo would block these outright; both are still refused.
fn apply_volatile(
    state: &mut State,
    target_side: usize,
    volatile: &str,
    tape: &mut Tape,
    log: &mut Log,
) -> Result<(), Refusal> {
    if state.sides[target_side].active_pokemon().volatiles.contains_key(volatile) {
        return Ok(());
    }
    // `_initial_volatile_duration`: a span whose ends are adjacent is a constant and costs no draw,
    // which is why a flinch never touches the tape and a confusion always does.
    let turns = match volatile {
        "CONFUSION" => tape.integer(2, 6)?,
        _ => 1,
    };
    let pokemon = state.sides[target_side].active_mut();
    pokemon.volatiles.insert(volatile.to_string(), turns);
    let nickname = pokemon.nickname.clone();
    log.push(Event::VolatileInflicted {
        side: target_side as i32,
        pokemon: nickname,
        volatile: volatile.to_string(),
    });
    Ok(())
}

/// `_apply_main_status`: everything after whatever roll decided the status should be attempted.
///
/// Shared, because a move's secondary and an ability like Static reach it by different routes and
/// must land identically once they get there. The sleep-clause check the Python makes here is a
/// no-op for this format — `_CLAUSED_STATUSES` is deliberately empty for Anything Goes — so it is
/// not reproduced; if a format ever wants one, it belongs right here.
pub fn apply_main_status(
    state: &mut State,
    target_side: usize,
    status: Status,
    tape: &mut Tape,
    log: &mut Log,
) -> Result<(), Refusal> {
    {
        let target = state.sides[target_side].active_pokemon();
        // Type immunity is silent in the Python; an ability immunity announces itself, and those
        // abilities are not ported, so only the silent half exists here.
        if target
            .types
            .iter()
            .flatten()
            .any(|t| status.immune_types().contains(&t.as_str()))
        {
            return Ok(());
        }
        if target.status != Status::None {
            let already = target.status.name().to_string();
            log.push(Event::StatusAlready {
                side: target_side as i32,
                pokemon: target.nickname.clone(),
                status: already,
            });
            return Ok(());
        }
    }
    // Sleep rolls its duration as it lands; toxic starts its counter at zero. The status is set
    // before the draw in the Python, which does not matter here but is why the order reads oddly.
    let turns = match status {
        Status::Sleep => tape.integer(2, 5)?,
        _ => 0,
    };
    let target = state.sides[target_side].active_mut();
    target.status = status;
    target.status_turns = turns;
    let nickname = target.nickname.clone();
    log.push(Event::StatusInflicted {
        side: target_side as i32,
        pokemon: nickname,
        status: status.name().to_string(),
    });
    Ok(())
}

fn apply_stages(
    state: &mut State,
    side: usize,
    stages: &[(String, i32)],
    probability: f64,
    target: &str,
    tape: &mut Tape,
    log: &mut Log,
) -> Result<(), Refusal> {
    if tape.probability()? >= probability {
        return Ok(());
    }
    let target_side = if target == "SELF" { side } else { 1 - side };
    apply_stage_changes(state, target_side, stages, "move", log);
    Ok(())
}

/// `mechanics.stages.apply_stage_changes`, for the cases this engine can reach.
///
/// One entry is logged per stat *whether or not the stage moved* — something already at +6 still
/// reports a requested +1 with a delta of 0, and the comparator would notice its absence. What is
/// missing is everything gated on an ability or item that is still refused: Contrary and Simple
/// rewriting the request, Clear Body and friends intercepting an opponent's drop, Defiant
/// retaliating, a White Herb undoing it.
pub fn apply_stage_changes(state: &mut State, target_side: usize, stages: &[(String, i32)], source: &str, log: &mut Log) {
    let pokemon = state.sides[target_side].active_mut();
    let nickname = pokemon.nickname.clone();
    for (stat, requested) in stages {
        let before = pokemon.stage(stat);
        let after = (before + requested).clamp(-6, 6);
        pokemon.stages.insert(stat.clone(), after);
        log.push(Event::StatStageChanged {
            side: target_side as i32,
            pokemon: nickname.clone(),
            stat: stat.clone(),
            delta: after - before,
            requested: *requested,
            source: source.into(),
        });
    }
}

/// Status is on the struct but nothing sets it yet; kept so the digest comparison has a field to
/// disagree about the moment statuses are ported.
pub fn _status_is_modelled(status: Status) -> bool {
    matches!(status, Status::None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn switches_resolve_before_moves() {
        assert!(category_of(&Action::Switch { to: 1 }) < category_of(&Action::Move { slot: 0 }));
    }

    #[test]
    fn priority_brackets_map_to_the_pythons_numbers() {
        let mut the_move = Move { priority: "QUICK_ATTACK".into(), ..Move::default() };
        assert_eq!(priority_of(&the_move).unwrap(), 1);
        the_move.priority = "NORMAL".into();
        assert_eq!(priority_of(&the_move).unwrap(), 0);
        the_move.priority = "TRICK_ROOM".into();
        assert_eq!(priority_of(&the_move).unwrap(), -7);
    }

    #[test]
    fn an_unknown_priority_bracket_is_refused_rather_than_guessed() {
        // The first version of this table invented plausible names and mapped everything else to
        // zero. A bracket nobody has mapped has to stop the run, not quietly become normal speed.
        let the_move = Move { priority: "NOT_A_BRACKET".into(), ..Move::default() };
        assert!(matches!(priority_of(&the_move), Err(Refusal::Unported(_))));
    }
}
