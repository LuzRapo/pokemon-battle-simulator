//! `ON_SWITCH_IN` and `ON_AFTER_HIT`: the two events where abilities and items *do* something
//! rather than adjust a number.
//!
//! Separate from `abilities.rs` because the shape of the problem is different. A damage-calc
//! handler reads the board and pushes a modifier; these ones deal damage, inflict statuses, move
//! stat stages and eat berries, which means they need the state by mutable reference and they can
//! consume randomness. Both facts constrain the order they run in far more tightly, and both make
//! a mistake here visible on the tape rather than merely in an arithmetic result.
//!
//! Dispatch order is the bus's: every ability (priority 2000) before any item (1000), and within
//! each, registration order. `ON_AFTER_HIT` fires *before* the `CriticalHit` and `DamageDealt`
//! entries for a single-hit move — the Python emits it inside the per-hit loop and logs the summary
//! after — so Rough Skin's chip is announced before the damage that caused it.

use crate::battle::{State, Status};
use crate::data::Database;
use crate::log::{Event, Log};
use crate::tape::Tape;
use crate::turn::{apply_main_status, apply_stage_changes, Refusal};

/// Abilities implemented here, on top of the damage-calc ones.
pub const PORTED_ABILITIES: [&str; 17] = [
    "AFTERMATH",
    "BERSERK",
    "DAUNTLESS_SHIELD",
    "DOWNLOAD",
    "EFFECT_SPORE",
    "FLAME_BODY",
    "INTIMIDATE",
    "INTREPID_SWORD",
    "IRON_BARBS",
    "JUSTIFIED",
    "POISON_POINT",
    "POISON_TOUCH",
    "ROUGH_SKIN",
    "STAMINA",
    "STATIC",
    "TOXIC_CHAIN",
    "WEAK_ARMOR",
];

/// Items implemented here, on top of the damage-calc ones.
pub const PORTED_ITEMS: [&str; 5] =
    ["ROCKY_HELMET", "SITRUS_BERRY", "STARF_BERRY", "WEAKNESS_POLICY", "WIKI_BERRY"];

/// `_SPORE_STATUSES`, and the draw that picks from it.
///
/// The Python indexes with `random_integer(0, len - 1)`, and `random_integer` is exclusive of its
/// upper bound, so the third entry can never come up: Effect Spore never puts anything to sleep in
/// this engine. Reproduced exactly, bound and all — agreeing with the reference is the job.
const SPORE_STATUSES: [Status; 3] = [Status::Poison, Status::Paralysis, Status::Sleep];
/// `_STARF_STATS`, indexed with `random_integer(0, 5)` — exclusive, so all five are reachable.
const STARF_STATS: [&str; 5] = ["ATTACK", "DEFENCE", "SP_ATTACK", "SP_DEFENCE", "SPEED"];

/// What a hit was, for the handlers that ask.
pub struct Hit<'a> {
    pub attacker_side: usize,
    pub move_type: &'a str,
    pub category: &'a str,
    pub contact: bool,
    /// How much this hit actually took off, which Berserk and Weakness Policy both read.
    pub dealt: i32,
}

/// Walk both actives' abilities, then both actives' items.
pub fn on_after_hit(
    state: &mut State,
    hit: &Hit,
    db: &Database,
    tape: &mut Tape,
    log: &mut Log,
) -> Result<(), Refusal> {
    let defender_side = 1 - hit.attacker_side;
    let mut order = [hit.attacker_side, defender_side];
    order.sort_by_key(|side| state.sides[*side].active_pokemon().registered_at);
    for side in order {
        ability_after_hit(state, side, hit, tape, log)?;
    }
    for side in order {
        item_after_hit(state, side, hit, db, tape, log)?;
    }
    Ok(())
}

/// Chip damage from an ability, as `AbilityChipDamage`.
fn chip(state: &mut State, side: usize, divisor: i32, ability: &str, log: &mut Log) {
    if crate::inline::ignores_indirect_damage(state.sides[side].active_pokemon()) {
        return;
    }
    let victim = state.sides[side].active_mut();
    if victim.fainted() {
        return;
    }
    let amount = std::cmp::max(1, victim.totals.hp / divisor);
    let dealt = victim.take_damage(amount);
    let nickname = victim.nickname.clone();
    log.push(Event::AbilityChipDamage {
        side: side as i32,
        pokemon: nickname,
        ability: ability.to_string(),
        amount: dealt,
    });
}

// The draw stays inside the arm rather than folding into the match guard. A guard is evaluated
// while Rust is still deciding which arm applies, and a draw taken there is a draw taken during a
// decision — in the one part of this engine where *when* a number comes off the tape is the whole
// contract, that is not a trade worth making for a tidier shape.
#[allow(clippy::collapsible_match)]
fn ability_after_hit(
    state: &mut State,
    side: usize,
    hit: &Hit,
    tape: &mut Tape,
    log: &mut Log,
) -> Result<(), Refusal> {
    let attacker_side = hit.attacker_side;
    let is_actor = side == attacker_side;
    let other = 1 - side;
    let ability = state.sides[side].active_pokemon().ability.clone();

    // The defender's answers to being hit.
    if !is_actor {
        let fainted = state.sides[side].active_pokemon().fainted();
        let attacker_up = !state.sides[other].active_pokemon().fainted();
        match ability.as_str() {
            "ROUGH_SKIN" | "IRON_BARBS" if hit.contact && attacker_up => {
                // Logged under Rough Skin's name whichever of the two it was, as the Python does.
                chip(state, other, 8, "ROUGH_SKIN", log);
            }
            "AFTERMATH" if hit.contact && fainted && attacker_up => {
                chip(state, other, 4, "AFTERMATH", log);
            }
            "FLAME_BODY" | "STATIC" | "POISON_POINT" if hit.contact && attacker_up => {
                if tape.probability()? < 0.3 {
                    let status = match ability.as_str() {
                        "FLAME_BODY" => Status::Burn,
                        "STATIC" => Status::Paralysis,
                        _ => Status::Poison,
                    };
                    apply_main_status(state, other, status, tape, log)?;
                }
            }
            "EFFECT_SPORE" if hit.contact && attacker_up => {
                if tape.probability()? < 0.3 {
                    let picked = tape.integer(0, SPORE_STATUSES.len() as i32 - 1)? as usize;
                    apply_main_status(state, other, SPORE_STATUSES[picked], tape, log)?;
                }
            }
            "WEAK_ARMOR" if hit.category == "PHYSICAL" && !fainted => {
                let stages = [("DEFENCE".to_string(), -1), ("SPEED".to_string(), 2)];
                apply_stage_changes(state, side, &stages, "weak_armor", log);
            }
            "STAMINA" if !fainted => {
                apply_stage_changes(state, side, &[("DEFENCE".to_string(), 1)], "stamina", log);
            }
            "JUSTIFIED" if !fainted && hit.move_type == "DARK" => {
                apply_stage_changes(state, side, &[("ATTACK".to_string(), 1)], "justified", log);
            }
            "BERSERK" if !fainted => {
                // Only the hit that crosses the half mark, which is why it needs `dealt` rather
                // than just the resulting HP: at or below half now, above it a moment ago.
                let pokemon = state.sides[side].active_pokemon();
                let (max_hp, live) = (pokemon.totals.hp, pokemon.hp);
                if hit.dealt > 0 && 2 * live <= max_hp && max_hp < 2 * (live + hit.dealt) {
                    apply_stage_changes(state, side, &[("SP_ATTACK".to_string(), 1)], "berserk", log);
                }
            }
            _ => {}
        }
        return Ok(());
    }

    // The attacker's own on-hit effects.
    let defender_up = !state.sides[other].active_pokemon().fainted();
    match ability.as_str() {
        "POISON_TOUCH" if hit.contact && defender_up => {
            if tape.probability()? < 0.3 {
                apply_main_status(state, other, Status::Poison, tape, log)?;
            }
        }
        "TOXIC_CHAIN" if defender_up => {
            if tape.probability()? < 0.3 {
                apply_main_status(state, other, Status::Toxic, tape, log)?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn item_after_hit(
    state: &mut State,
    side: usize,
    hit: &Hit,
    db: &Database,
    tape: &mut Tape,
    log: &mut Log,
) -> Result<(), Refusal> {
    if side == hit.attacker_side {
        return Ok(()); // nothing ported here fires for the attacker
    }
    let other = 1 - side;
    let item = state.sides[side].active_pokemon().item.clone();
    let holder = state.sides[side].active_pokemon();
    if holder.fainted() {
        // Sitrus, Wiki, Starf and the Policy all check this; Rocky Helmet does not, so it is
        // handled before the guard rather than after.
        if item == "ROCKY_HELMET" && hit.contact && !state.sides[other].active_pokemon().fainted() {
            helmet(state, other, log);
        }
        return Ok(());
    }
    let (max_hp, live) = (holder.totals.hp, holder.hp);
    match item.as_str() {
        "ROCKY_HELMET" if hit.contact && !state.sides[other].active_pokemon().fainted() => helmet(state, other, log),
        "SITRUS_BERRY" if 2 * live <= max_hp => heal(state, side, 4, "SITRUS_BERRY", log),
        "WIKI_BERRY" if 4 * live <= max_hp => heal(state, side, 3, "WIKI_BERRY", log),
        "STARF_BERRY" if 4 * live <= max_hp => {
            consume(state, side);
            let picked = tape.integer(0, STARF_STATS.len() as i32)? as usize;
            let stages = [(STARF_STATS[picked].to_string(), 2)];
            // Logged as "seed", not "starf_berry" — the Python's own choice of source.
            apply_stage_changes(state, side, &stages, "seed", log);
        }
        "WEAKNESS_POLICY" if hit.dealt > 0 => {
            let types = state.sides[side].active_pokemon().types.clone();
            if db.effectiveness(hit.move_type, &types) >= 2.0 {
                consume(state, side);
                let stages = [("ATTACK".to_string(), 2), ("SP_ATTACK".to_string(), 2)];
                apply_stage_changes(state, side, &stages, "weakness_policy", log);
            }
        }
        _ => {}
    }
    Ok(())
}

fn helmet(state: &mut State, attacker_side: usize, log: &mut Log) {
    let attacker = state.sides[attacker_side].active_mut();
    let amount = std::cmp::max(1, attacker.totals.hp / 6);
    attacker.take_damage(amount);
    let nickname = attacker.nickname.clone();
    // The *requested* recoil, not what landed — the Python logs the number it asked for.
    log.push(Event::ItemChipDamage {
        side: attacker_side as i32,
        pokemon: nickname,
        item: "ROCKY_HELMET".to_string(),
        amount,
    });
}

fn heal(state: &mut State, side: usize, divisor: i32, item: &str, log: &mut Log) {
    consume(state, side);
    let pokemon = state.sides[side].active_mut();
    let amount = std::cmp::max(1, pokemon.totals.hp / divisor);
    let before = pokemon.hp;
    pokemon.hp = std::cmp::min(pokemon.totals.hp, pokemon.hp + amount);
    if pokemon.hp > before {
        let nickname = pokemon.nickname.clone();
        log.push(Event::ItemHealed { side: side as i32, pokemon: nickname, item: item.to_string() });
    }
}

fn consume(state: &mut State, side: usize) {
    let pokemon = state.sides[side].active_mut();
    pokemon.item = "NONE".to_string();
    pokemon.item_consumed = true;
}

/// `ON_SWITCH_IN`, for whoever has just arrived on this side.
#[allow(clippy::collapsible_match)] // same reason: a draw belongs in the arm, not in the guard
pub fn on_switch_in(state: &mut State, side: usize, log: &mut Log) {
    let other = 1 - side;
    let ability = state.sides[side].active_pokemon().ability.clone();
    match ability.as_str() {
        "INTIMIDATE" => {
            if !state.sides[other].active_pokemon().fainted() {
                crate::turn::apply_stage_changes_from(
                    state,
                    other,
                    &[("ATTACK".to_string(), -1)],
                    "intimidate",
                    true,
                    log,
                );
            }
        }
        "DOWNLOAD" => {
            let opponent = state.sides[other].active_pokemon();
            if !opponent.fainted() {
                // The opponent's *totals*, before stages: the higher defence decides which side of
                // its wall to attack, and a stage change does not move that judgement.
                let stat = if opponent.totals.defence >= opponent.totals.sp_defence {
                    "SP_ATTACK"
                } else {
                    "ATTACK"
                };
                apply_stage_changes(state, side, &[(stat.to_string(), 1)], "download", log);
            }
        }
        _ if crate::inline::weather_from_ability(&ability).is_some() => {
            // Nothing happens if this weather is already blowing — and nothing is logged either.
            let weather = crate::inline::weather_from_ability(&ability).expect("just checked");
            if state.field.weather != weather {
                state.field.weather = weather.to_string();
                // A Weather Rock would make it eight; those items are still refused.
                state.field.weather_turns_left = 5;
                let nickname = state.sides[side].active_pokemon().nickname.clone();
                log.push(Event::WeatherSetByAbility {
                    side: side as i32,
                    pokemon: nickname,
                    ability: ability.clone(),
                });
            }
        }
        _ if crate::inline::terrain_from_ability(&ability).is_some() => {
            let terrain = crate::inline::terrain_from_ability(&ability).expect("just checked");
            if state.field.terrain != terrain {
                state.field.terrain = terrain.to_string();
                state.field.terrain_turns_left = 5;
                let nickname = state.sides[side].active_pokemon().nickname.clone();
                log.push(Event::TerrainSetByAbility {
                    side: side as i32,
                    pokemon: nickname,
                    ability: ability.clone(),
                });
            }
        }
        "DAUNTLESS_SHIELD" | "INTREPID_SWORD" => {
            // Once per battle per Pokemon, not once per switch-in.
            if !state.sides[side].active_pokemon().switch_in_boost_used {
                state.sides[side].active_mut().switch_in_boost_used = true;
                let stat = if ability == "DAUNTLESS_SHIELD" { "DEFENCE" } else { "ATTACK" };
                let source = if ability == "DAUNTLESS_SHIELD" { "dauntless_shield" } else { "intrepid_sword" };
                apply_stage_changes(state, side, &[(stat.to_string(), 1)], source, log);
            }
        }
        _ => {}
    }
}
