//! Mid-battle species swaps: Mega Evolution, Primal Reversion, and Ultra Burst.
//!
//! Mirrors `battle_sim/formes.py`, but carries none of its own table-building logic across —
//! `_forme_by_base_and_item`'s filtering (`_is_transformed_forme`, `_is_playable`, `_source_forme`)
//! runs once, in Python, at export time, and what lands here (`Database::mega_formes`,
//! `Database::ultra_burst_formes`, `Database::move_gated_formes`) is already the answer, not the
//! rule. The one thing Python's own export cannot know is whether *this* engine has ported a given
//! forme's ability yet, so `playable` adds that one extra filter on top — the same "held back
//! rather than half-implemented" principle `_is_playable` uses, aimed at this engine's own,
//! currently narrower, ported set rather than Python's.

use crate::battle::{Pokemon, State};
use crate::data::Database;
use crate::log::{Event, Log};
use crate::turn::Action;

/// Whether `forme` can actually be fielded by this engine today: its own regular ability has to be
/// one this engine dispatches, or the swap would hand out an ability nothing here ever reads —
/// exactly the silent-gap failure this whole port is built to catch, just one turn later than the
/// usual case since the ability was not on the Pokemon's *sheet*, only reached by playing it out.
fn playable(forme_name: &str, db: &Database) -> bool {
    let Some(species) = db.species_named(forme_name) else { return false };
    match species.regular_ability.as_deref() {
        Some(ability) => crate::turn::ported_abilities().contains(ability),
        None => false,
    }
}

/// Whether this (species, item) pairing Ultra Bursts rather than Mega Evolving — checked first,
/// and kept apart from `has_mega_evolved` since a side may do both in one battle.
pub fn ultra_bursts(species_name: &str, item: &str, db: &Database) -> bool {
    let key = Database::normalize_id(species_name);
    db.ultra_burst_formes.iter().any(|f| f.base_species == key && f.item == item)
}

/// The forme this species reaches right now, or `None` if nothing about it transforms — either
/// because no table names this pairing at all, or because it does and `playable` refuses it.
///
/// Held item first (Ultra Burst, then the generic Mega/Primal table), then the move-gated pairs —
/// `known_moves` may be left empty by a caller that only cares about the item half, since every
/// Gen 7 move-gated forme is Rayquaza's.
pub fn mega_forme(species_name: &str, item: &str, known_moves: &[String], db: &Database) -> Option<String> {
    let key = Database::normalize_id(species_name);
    if let Some(entry) = db.ultra_burst_formes.iter().find(|f| f.base_species == key && f.item == item) {
        return playable(&entry.forme, db).then(|| entry.forme.clone());
    }
    if let Some(entry) = db.mega_formes.iter().find(|f| f.base_species == key && f.item == item) {
        return playable(&entry.forme, db).then(|| entry.forme.clone());
    }
    // Holding a Z-Crystal locks Rayquaza out of Mega Evolution — the whole reason a Z-move
    // Rayquaza is a different Pokemon from a Mega one rather than strictly worse. Every Z-Crystal
    // in this dex, generic or signature, ends in `_Z` (the same test `Database::is_fused_to` already
    // relies on), so this needs no Z-move-specific lookup to answer.
    if item.ends_with("_Z") {
        return None;
    }
    known_moves.iter().find_map(|known| {
        let move_key = Database::normalize_id(known);
        let entry = db.move_gated_formes.iter().find(|f| f.base_species == key && f.move_name == move_key)?;
        playable(&entry.forme, db).then(|| entry.forme.clone())
    })
}

/// Whether this Pokemon's held item or moveset *would* trigger a forme change per the game's own
/// rules, ignoring `playable` entirely. Used only by `turn::unsupported_pokemon`, which needs to
/// tell "this pairing does nothing" (fine, plays on unmodified) apart from "this pairing reaches a
/// forme this engine cannot yet field" (refuse the scenario, rather than silently leaving a
/// Pokemon that should have transformed sitting in its base forme all battle).
pub fn wants_to_transform(species_name: &str, item: &str, known_moves: &[String], db: &Database) -> bool {
    let key = Database::normalize_id(species_name);
    db.ultra_burst_formes.iter().any(|f| f.base_species == key && f.item == item)
        || db.mega_formes.iter().any(|f| f.base_species == key && f.item == item)
        || (!item.ends_with("_Z")
            && known_moves.iter().any(|known| {
                let move_key = Database::normalize_id(known);
                db.move_gated_formes.iter().any(|f| f.base_species == key && f.move_name == move_key)
            }))
}

/// Swap a live Pokemon onto another forme's species data, in place — name, base stats, types,
/// ability and weight change; moves, PP, item, nickname, level, EVs/IVs/nature do not. Current HP
/// keeps its *fraction* of the maximum rather than carrying over as a raw number: a no-op for every
/// Gen 6/7 mega or primal pair, which all share a base HP stat with their source forme (`playable`
/// checked that already), and the only reason it is a fraction at all rather than a straight copy.
pub fn apply_forme(pokemon: &mut Pokemon, forme_name: &str, db: &Database) {
    let species = db.species_named(forme_name).expect("mega_forme/ultra_bursts only name formes that exist");
    let fraction = pokemon.hp as f64 / pokemon.totals.hp as f64;
    pokemon.species_name = species.name.clone();
    pokemon.base_stats = species.base_stats;
    pokemon.types = species.types.clone();
    pokemon.ability = species.regular_ability.clone().expect("playable() already checked this forme has one");
    pokemon.weight_kg = species.weight_kg;
    pokemon.recompute_totals();
    pokemon.hp = if fraction > 0.0 { (fraction * pokemon.totals.hp as f64).round().max(1.0) as i32 } else { 0 };
}

/// `_resolve_mega_evolution`: Mega Evolve (or Primal Revert, or Ultra Burst) any active Pokemon
/// holding — or, for Rayquaza, knowing — its match, before a single move is ordered.
///
/// Placed before `order_actions` deliberately: `effective_speed` reads `totals` at sort time, so
/// the new Speed decides this turn's order, exactly as Gen 7 does. A side that chose to switch does
/// not transform; the two are mutually exclusive in one turn. A Pokemon megas at its first
/// opportunity rather than the player choosing when — the same simplification the Python makes, for
/// the same reason (a Mega variant of every move would double the branching factor at every search
/// node for a choice that is almost always right immediately).
#[allow(clippy::needless_range_loop)]
pub fn resolve_forme_changes(state: &mut State, actions: &[Action; 2], db: &Database, log: &mut Log) {
    for side in 0..2 {
        if !matches!(actions[side], Action::Move { .. }) || state.sides[side].active_pokemon().fainted() {
            continue;
        }
        let species_name = state.sides[side].active_pokemon().species_name.clone();
        let item = state.sides[side].active_pokemon().item.clone();
        let bursts = ultra_bursts(&species_name, &item, db);
        if if bursts { state.sides[side].has_ultra_bursted } else { state.sides[side].has_mega_evolved } {
            continue;
        }
        let known_moves = state.sides[side].active_pokemon().moves.clone();
        let Some(forme) = mega_forme(&species_name, &item, &known_moves, db) else { continue };
        apply_forme(state.sides[side].active_mut(), &forme, db);
        // `rewire_active`: the new forme's ability replaces the old one's handlers, which re-register
        // behind everyone else's at their priority.
        state.register_active(side);
        if bursts {
            state.sides[side].has_ultra_bursted = true;
        } else {
            state.sides[side].has_mega_evolved = true;
        }
        let nickname = state.sides[side].active_pokemon().nickname.clone();
        log.push(Event::FormeChanged { side: side as i32, pokemon: nickname, forme });
        crate::hooks::apply_weather_from_ability(state, side, log);
    }
}
