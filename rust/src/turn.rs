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

use crate::battle::{Pokemon, State, Status};
use crate::damage::{calculate_hit, Rolls};
use crate::data::{Database, Move};
use crate::log::{Event, Log};
use crate::tape::Tape;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    Move { slot: usize },
    Switch { to: usize },
}

#[derive(Debug)]
pub struct Unsupported(pub String);

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
fn priority_of(the_move: &Move) -> Result<i32, Unsupported> {
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
        other => return Err(Unsupported(format!("unmapped priority bracket {other:?}"))),
    })
}

/// Sorts exactly as `_sort_key` does: category, then priority (descending), then speed
/// (descending), then the tie-break draw. Rust sorts ascending, so speed and priority are negated
/// the same way the Python negates speed.
fn order_actions(state: &State, actions: &[Action; 2], db: &Database, tape: &mut Tape) -> Result<Vec<usize>, Unsupported> {
    // Drawn for both sides before anything resolves, in side order — the Python builds this dict
    // by comprehension over `actions`, which is insertion-ordered 0 then 1.
    let tie_breakers = [tape.probability()?, tape.probability()?];
    let mut keys: Vec<(i32, i32, i32, f64, usize)> = Vec::new();
    for side in 0..2 {
        let actor = state.sides[side].active_pokemon();
        let (priority, speed) = match &actions[side] {
            Action::Switch { .. } => (0, actor.effective("SPEED")),
            Action::Move { slot } => {
                let the_move = move_in_slot(actor, *slot, db)?;
                (priority_of(the_move)?, actor.effective("SPEED"))
            }
        };
        keys.push((category_of(&actions[side]), -priority, -speed, tie_breakers[side], side));
    }
    keys.sort_by(|a, b| a.partial_cmp(b).expect("no NaNs in a sort key"));
    Ok(keys.into_iter().map(|k| k.4).collect())
}

fn move_in_slot<'a>(actor: &Pokemon, slot: usize, db: &'a Database) -> Result<&'a Move, Unsupported> {
    let name = actor
        .moves
        .get(slot)
        .ok_or_else(|| Unsupported(format!("{} has no move in slot {slot}", actor.nickname)))?;
    db.move_named(name)
        .ok_or_else(|| Unsupported(format!("unknown move {name:?}")))
}

/// Everything this engine has not learned yet. A scenario containing one of these is refused up
/// front rather than played wrongly — the whole point of the differential work is that silence is
/// the one unacceptable failure mode.
pub fn unsupported_reason(the_move: &Move, db: &Database) -> Option<String> {
    if db.special_power_moves.contains(&the_move.name) {
        return Some(format!("{}'s power is computed from the board, not read from the data", the_move.name));
    }
    if !crate::damage::is_damaging(the_move) {
        return Some(format!("{} is not a plain damaging move", the_move.name));
    }
    if the_move.effects.len() > 1 {
        return Some(format!("{} carries a secondary effect", the_move.name));
    }
    if the_move.self_switch || the_move.healing {
        return Some(format!("{} switches or heals", the_move.name));
    }
    None
}

pub fn step(
    state: &mut State,
    actions: [Action; 2],
    db: &Database,
    tape: &mut Tape,
) -> Result<Log, Unsupported> {
    let mut log = Log::new();
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
                let withdrew = state.sides[side].active_pokemon().nickname.clone();
                let sent_out = state.sides[side].team[*to].nickname.clone();
                state.sides[side].active = *to;
                log.push(Event::Switched { side: side as i32, withdrew, sent_out });
            }
            Action::Move { slot } => resolve_move(state, side, *slot, db, tape, &mut log)?,
        }
        let was_decided = state.outcome.is_some();
        state.update_outcome();
        // `_update_outcome` announces the result the moment it is decided, once.
        if !was_decided {
            if let Some(outcome) = state.outcome {
                log.push(Event::BattleEnded { outcome: outcome.name().to_string() });
            }
        }
    }
    state.turn += 1;
    Ok(log)
}

fn resolve_move(
    state: &mut State,
    side: usize,
    slot: usize,
    db: &Database,
    tape: &mut Tape,
    log: &mut Log,
) -> Result<(), Unsupported> {
    let other = 1 - side;
    let the_move = {
        let actor = state.sides[side].active_pokemon();
        move_in_slot(actor, slot, db)?.clone()
    };
    if let Some(why) = unsupported_reason(&the_move, db) {
        return Err(Unsupported(why));
    }
    // Spent before anything resolves, as `_spend_pp` does it: a move that misses still costs its
    // point, which is why this is here rather than after the hit lands.
    if let Some(left) = state.sides[side].active_mut().pp.get_mut(crate::battle::SLOT_NAMES[slot]) {
        *left = (*left - 1).max(0);
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
            return Ok(());
        }
    }

    let effectiveness = if the_move.typeless {
        1.0
    } else {
        db.effectiveness(&the_move.move_type, &state.sides[other].active_pokemon().types)
    };
    if effectiveness == 0.0 {
        log.push(Event::NoEffect {
            side: other as i32,
            pokemon: state.sides[other].active_pokemon().nickname.clone(),
        });
        return Ok(());
    }
    log.effectiveness(effectiveness);

    // Drawn here, in the Python's order: the crit first, then the damage roll. Both are certain
    // to be consumed by the time the formula is entered — see `Rolls`.
    let rolls = Rolls { crit: tape.probability()?, damage: tape.integer(85, 101)? };
    let hit = {
        let (mine, theirs) = state.sides.split_at(1);
        let (attacker, defender_side) = if side == 0 {
            (mine[0].active_pokemon(), &theirs[0])
        } else {
            (theirs[0].active_pokemon(), &mine[0])
        };
        let defender = defender_side.active_pokemon();
        calculate_hit(attacker, defender, &the_move, &state.field, defender_side, db, rolls)
    };
    if hit.is_crit {
        log.push(Event::CriticalHit);
    }
    let dealt = state.sides[other].active_mut().take_damage(hit.amount);
    log.push(Event::DamageDealt {
        side: other as i32,
        pokemon: state.sides[other].active_pokemon().nickname.clone(),
        amount: dealt,
    });
    if state.sides[other].active_pokemon().fainted() {
        log.push(Event::Fainted {
            side: other as i32,
            pokemon: state.sides[other].active_pokemon().nickname.clone(),
        });
    }
    Ok(())
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
        let mut quick = Move {
            name: "Quick Attack".into(),
            move_type: "NORMAL".into(),
            category: "PHYSICAL".into(),
            accuracy_probability: Some(1.0),
            priority: "QUICK".into(),
            pp: 30,
            target: "SINGLE_OPPONENT".into(),
            effects: vec![],
            protectable: true,
            healing: false,
            typeless: false,
            self_switch: false,
        };
        assert_eq!(priority_of(&quick), 1);
        quick.priority = "NORMAL".into();
        assert_eq!(priority_of(&quick), 0);
    }
}
