//! `battle_sim/engine/choices.py`: the actions a side may submit, and nothing it may not.
//!
//! A port, line for line — legality is a rule of the game the two engines have to agree on, and
//! the differential tests compare this list against the Python's at every decision a battle asks
//! for. Actions are numbered in one fixed space so a policy network can mask them:
//!
//!   0..=3   use the move in slot FIRST..FOURTH
//!   4..=7   the same slot as a Z-move
//!   8..=13  switch to (or lead with) team member 0..=5
//!
//! Mega Evolution is not an action: both engines evolve at the first opportunity.

use crate::battle::{Pokemon, State, SLOT_NAMES};
use crate::data::Database;

pub const MOVE: u8 = 0;
pub const Z_MOVE: u8 = 4;
pub const SWITCH: u8 = 8;
pub const ACTION_SPACE: usize = 14;

/// Items whose entire effect is on legality, so porting `legal_actions` ported them: the Python
/// reads Shed Shell nowhere but `_trapped`.
pub const PORTED_LEGALITY_ONLY_ITEMS: [&str; 1] = ["SHED_SHELL"];

/// `legal_actions`: moves first, then switches, each in slot/team order.
pub fn legal_actions(state: &State, side: usize, db: &Database) -> Vec<u8> {
    let own = &state.sides[side];
    let active = own.active_pokemon();
    if active.fainted() || own.needs_switch {
        return switch_actions(state, side);
    }
    // A rampage forbids switching, and a charge is committed to its release.
    if active.volatiles.contains_key("LOCKED_MOVE") {
        if let Some(slot) = active.locked_slot {
            return vec![MOVE + slot as u8];
        }
    }
    if active.volatiles.contains_key("CHARGING") {
        if let Some(slot) = active.charging_slot {
            return vec![MOVE + slot as u8];
        }
    }
    let mut actions = move_actions(state, side, db);
    if !trapped(active, state.sides[1 - side].active_pokemon()) {
        actions.extend(switch_actions(state, side));
    }
    actions
}

/// `_trapped`: Shadow Tag, Arena Trap, Magnet Pull, and being wrapped. A Ghost (by its plain types,
/// as the Python reads them) or a Shed Shell holder is free to leave whatever holds it.
fn trapped(active: &Pokemon, opponent: &Pokemon) -> bool {
    let has_type = |kind: &str| active.types.iter().flatten().any(|t| t == kind);
    if opponent.fainted() || has_type("GHOST") || active.item == "SHED_SHELL" {
        return false;
    }
    if active.volatiles.contains_key("PARTIALLY_TRAPPED") {
        return true;
    }
    match opponent.ability.as_str() {
        "SHADOW_TAG" => active.ability != "SHADOW_TAG",
        "ARENA_TRAP" => crate::field::is_grounded(active),
        "MAGNET_PULL" => has_type("STEEL"),
        _ => false,
    }
}

/// `_move_actions`.
fn move_actions(state: &State, side: usize, db: &Database) -> Vec<u8> {
    let active = state.sides[side].active_pokemon();
    let forced = active.choice_locked_move.or(active.encored_slot);
    let slots: Vec<usize> = match forced {
        Some(slot) => vec![slot],
        None => (0..active.moves.len()).collect(),
    };
    let pp = |slot: usize| active.pp.get(SLOT_NAMES[slot]).copied().unwrap_or(0);
    let usable: Vec<usize> = slots
        .iter()
        .copied()
        .filter(|&slot| pp(slot) > 0 && active.disabled_slot != Some(slot) && !taunt_blocked(active, slot, db))
        .collect();
    if !usable.is_empty() {
        return usable.into_iter().flat_map(|slot| slot_actions(state, side, slot, db)).collect();
    }
    // Constrained into an empty move, or everything spent: the engine substitutes Struggle.
    if forced.is_some() || active.pp.values().all(|left| *left == 0) {
        return vec![MOVE + forced.unwrap_or(slots[0]) as u8];
    }
    // Only Disabled or Taunt-blocked moves remain: accepted, and the turn is wasted.
    slots.into_iter().filter(|&slot| pp(slot) > 0).map(|slot| MOVE + slot as u8).collect()
}

fn taunt_blocked(active: &Pokemon, slot: usize, db: &Database) -> bool {
    active.volatiles.contains_key("TAUNT")
        && db.move_named(&active.moves[slot]).is_some_and(|m| m.category == "STATUS")
}

/// `_slot_actions`: the plain use of a slot, plus its Z-move if the held crystal upgrades it.
fn slot_actions(state: &State, side: usize, slot: usize, db: &Database) -> Vec<u8> {
    let plain = MOVE + slot as u8;
    if state.sides[side].has_used_z_move {
        return vec![plain];
    }
    let active = state.sides[side].active_pokemon();
    let upgraded = db
        .move_named(&active.moves[slot])
        .and_then(|base| crate::zmoves::z_move_for(&active.item, base, db));
    match upgraded {
        Some(_) => vec![plain, Z_MOVE + slot as u8],
        None => vec![plain],
    }
}

/// `_switch_actions`: every healthy benched Pokemon, in team order.
fn switch_actions(state: &State, side: usize) -> Vec<u8> {
    let own = &state.sides[side];
    (0..own.team.len())
        .filter(|&index| index != own.active && !own.team[index].fainted())
        .map(|index| SWITCH + index as u8)
        .collect()
}

/// `differential.name_action`: one action as the stable string both engines record it by.
pub fn name_action(state: &State, side: usize, action: u8) -> String {
    let own = &state.sides[side];
    if action >= SWITCH {
        return format!("switch:{}", own.team[(action - SWITCH) as usize].nickname);
    }
    let (prefix, slot) = if action >= Z_MOVE { ("zmove", action - Z_MOVE) } else { ("move", action - MOVE) };
    let slot = slot as usize;
    format!("{prefix}:{}:{}", SLOT_NAMES[slot], own.active_pokemon().moves[slot])
}
