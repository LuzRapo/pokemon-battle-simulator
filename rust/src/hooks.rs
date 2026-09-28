//! `ON_SWITCH_IN`, `ON_AFTER_HIT`, `ON_BEFORE_MOVE` and `ON_FAINT`: the events where abilities and
//! items *do* something rather than adjust a number.
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

use crate::battle::{Pokemon, State, Status};
use crate::data::{Database, Effect, Move};
use crate::log::{Event, Log};
use crate::tape::Tape;
use crate::turn::{apply_main_status_from, apply_stage_changes, start_disable, Refusal, DEFENDER_FACING};

/// Abilities implemented here, at `ON_BEFORE_MOVE` — a move that could reach the defender (a
/// damaging one, or one that targets the opponent directly, exactly the Python's own `damaging or
/// move.target in _DEFENDER_FACING_TARGETS` gate), before the effectiveness/immunity check. None
/// of the handlers below draw from the tape — every one of them is a deterministic function of the
/// defending ability and the move's type or sound flag.
pub const PORTED_BEFORE_MOVE_ABILITIES: [&str; 13] = [
    "EARTH_EATER",
    "FLASH_FIRE",
    "LEVITATE",
    "LIGHTNING_ROD",
    "MOTOR_DRIVE",
    "SAP_SIPPER",
    "SOUNDPROOF",
    "STORM_DRAIN",
    "VOLT_ABSORB",
    "WATER_ABSORB",
    "WELL_BAKED_BODY",
    "WIND_RIDER",
    "WONDER_GUARD",
];

/// `True` if the move was cancelled here — the caller skips the effectiveness gate and the hit
/// entirely, the same as a Python handler returning `HandlerResult(cancel=True, ...)`.
pub fn ability_before_move(state: &mut State, side: usize, the_move: &Move, db: &Database, log: &mut Log) -> bool {
    let other = 1 - side;
    let damaging = the_move
        .effects
        .iter()
        .any(|e| matches!(e, Effect::DamageEffect { .. } | Effect::FixedDamageEffect { .. }));
    // `if not move.effects and not move.force_switch: MoveFailed; return` — the Python's own check
    // one line above where it emits `ON_BEFORE_MOVE`, so a data-only move (Electrify's own effect
    // is unmodelled and its list is empty) never reaches an absorber at all, even one of a matching
    // type. Rust's own equivalent is the generic `log_before == log.entries.len()` fallback in
    // `resolve_move`, reached further down and unaffected by returning `false` here.
    if the_move.effects.is_empty() && !the_move.force_switch {
        return false;
    }
    if !damaging && !DEFENDER_FACING.contains(&the_move.target.as_str()) {
        return false;
    }
    let ability = state.sides[other].active_pokemon().ability.clone();
    let move_type = the_move.move_type.as_str();
    match ability.as_str() {
        "VOLT_ABSORB" if move_type == "ELECTRIC" => absorb_heal(state, other, "VOLT_ABSORB", log),
        "WATER_ABSORB" if move_type == "WATER" => absorb_heal(state, other, "WATER_ABSORB", log),
        "EARTH_EATER" if move_type == "GROUND" => absorb_heal(state, other, "EARTH_EATER", log),
        "MOTOR_DRIVE" if move_type == "ELECTRIC" => absorb_boost(state, other, "SPEED", 1, "motor_drive", log),
        "LIGHTNING_ROD" if move_type == "ELECTRIC" => {
            absorb_boost(state, other, "SP_ATTACK", 1, "lightning_rod", log)
        }
        "STORM_DRAIN" if move_type == "WATER" => absorb_boost(state, other, "SP_ATTACK", 1, "storm_drain", log),
        "SAP_SIPPER" if move_type == "GRASS" => absorb_boost(state, other, "ATTACK", 1, "sap_sipper", log),
        "WELL_BAKED_BODY" if move_type == "FIRE" => {
            absorb_boost(state, other, "DEFENCE", 2, "well_baked_body", log)
        }
        "FLASH_FIRE" if move_type == "FIRE" => {
            let pokemon = state.sides[other].active_mut();
            let nickname = pokemon.nickname.clone();
            if pokemon.flash_fire_active {
                log.push(Event::FlashFireAbsorbed { side: other as i32, pokemon: nickname });
            } else {
                pokemon.flash_fire_active = true;
                log.push(Event::FlashFireActivated { side: other as i32, pokemon: nickname });
            }
            true
        }
        "LEVITATE" if move_type == "GROUND" => {
            let nickname = state.sides[other].active_pokemon().nickname.clone();
            log.push(Event::AvoidedWithLevitate { side: other as i32, pokemon: nickname });
            true
        }
        "SOUNDPROOF" if the_move.sound => {
            let nickname = state.sides[other].active_pokemon().nickname.clone();
            log.push(Event::DoesNotAffect { side: other as i32, pokemon: nickname });
            true
        }
        "WONDER_GUARD" => {
            // `type_effectiveness(payload["move_type"], pokemon.types)`: the *plain* type chart
            // against the defender's raw types — not `effective_bypass`/`battle_types`, which is
            // what every other effectiveness read in this file uses. Matched exactly rather than
            // corrected: since `ON_BEFORE_MOVE` fires for a defender-facing status move too, a
            // status move whose type happens not to be super effective gets blocked here as well,
            // which is not how Wonder Guard behaves in the real games — a quirk of this simplified
            // read, reproduced per invariant 3, not fixed.
            let types = state.sides[other].active_pokemon().types.clone();
            if db.effectiveness(move_type, &types) < 2.0 {
                let nickname = state.sides[other].active_pokemon().nickname.clone();
                log.push(Event::DoesNotAffect { side: other as i32, pokemon: nickname });
                true
            } else {
                false
            }
        }
        "WIND_RIDER" if the_move.wind => {
            // The log line's own `source` says "justified", not "wind_rider" — a copy-paste slip
            // in the Python (`_bind_wind_rider` borrows `_bind_hit_reaction_boost`'s pattern by
            // hand and keeps its source string), reproduced rather than corrected per invariant 3.
            absorb_boost(state, other, "ATTACK", 1, "justified", log)
        }
        _ => false,
    }
}

/// Volt Absorb / Water Absorb / Earth Eater: a quarter heal, or `AbsorbBlocked` at full HP —
/// `_bind_type_absorb`'s own `if healed > 0 ... else ...`, not the silent-when-zero convention
/// `heal_by` uses for the residual healers.
fn absorb_heal(state: &mut State, side: usize, ability: &str, log: &mut Log) -> bool {
    let pokemon = state.sides[side].active_mut();
    let amount = std::cmp::max(1, pokemon.totals.hp / 4);
    let before = pokemon.hp;
    pokemon.hp = std::cmp::min(pokemon.totals.hp, pokemon.hp + amount);
    let healed = pokemon.hp - before;
    let nickname = pokemon.nickname.clone();
    if healed > 0 {
        log.push(Event::AbsorbHealed { side: side as i32, pokemon: nickname, ability: ability.to_string(), amount: healed });
    } else {
        log.push(Event::AbsorbBlocked { side: side as i32, pokemon: nickname, ability: ability.to_string() });
    }
    true
}

/// Motor Drive / Lightning Rod / Storm Drain / Sap Sipper / Well Baked Body: the raw
/// `change_stat_stage`, logged only when the stage actually moved — same shape as Speed Boost's
/// residual bump, and deliberately not `apply_stage_changes`, which (correctly, for the callers
/// that want it) logs unconditionally. Contrary and Simple never enter into it: the ability
/// boosting itself here is the only ability this Pokemon has.
fn absorb_boost(state: &mut State, side: usize, stat: &str, amount: i32, source: &str, log: &mut Log) -> bool {
    let pokemon = state.sides[side].active_mut();
    let before = pokemon.stage(stat);
    let after = (before + amount).clamp(-6, 6);
    pokemon.stages.insert(stat.to_string(), after);
    if after > before {
        let nickname = pokemon.nickname.clone();
        log.push(Event::StatStageChanged {
            side: side as i32,
            pokemon: nickname,
            stat: stat.to_string(),
            delta: after - before,
            requested: amount,
            source: source.to_string(),
        });
    }
    true
}

/// Abilities implemented here, at `ON_FAINT` — emitted from exactly one place in the Python
/// (`_apply_damage`, right after the defender's own `Fainted` line, before the Destiny Bond
/// retaliation check and before recoil/drain), not from every place a Pokemon can reach zero HP.
/// Residual damage, recoil, confusion and fixed-damage moves never reach it, so a KO from any of
/// those does not trigger these — matched here by calling this from the same single site, not from
/// wherever `fainted()` happens to be checked.
pub const PORTED_ON_FAINT_ABILITIES: [&str; 3] = ["BEAST_BOOST", "MOXIE", "SOUL_HEART"];

/// `attacker_side` is the Pokemon whose hit just fainted its target — `context.actor is pokemon` in
/// the Python, which is why this reads the *attacker's* ability, not the one that just fainted.
pub fn ability_on_faint(state: &mut State, attacker_side: usize, log: &mut Log) {
    if state.sides[attacker_side].active_pokemon().fainted() {
        return;
    }
    match state.sides[attacker_side].active_pokemon().ability.as_str() {
        "MOXIE" => {
            absorb_boost(state, attacker_side, "ATTACK", 1, "moxie", log);
        }
        "SOUL_HEART" => {
            absorb_boost(state, attacker_side, "SP_ATTACK", 1, "soul_heart", log);
        }
        "BEAST_BOOST" => {
            let totals = state.sides[attacker_side].active_pokemon().totals;
            // `max(_BEAST_BOOST_STATS, key=...)`: Python's `max` keeps the first element seen on a
            // tie, so this has to as well — a plain `Iterator::max_by_key` would keep the last.
            let candidates =
                [("ATTACK", totals.attack), ("DEFENCE", totals.defence), ("SP_ATTACK", totals.sp_attack), (
                    "SP_DEFENCE",
                    totals.sp_defence,
                ), ("SPEED", totals.speed)];
            let mut best = candidates[0];
            for candidate in &candidates[1..] {
                if candidate.1 > best.1 {
                    best = *candidate;
                }
            }
            absorb_boost(state, attacker_side, best.0, 1, "beast_boost", log);
        }
        _ => {}
    }
}

/// Abilities implemented here, at `ON_SWITCH_OUT` — emitted once, from `turn::switch_out`, and only
/// when the outgoing Pokemon did not faint (a fainted switch is a replacement, not a choice — the
/// Python's own comment on the event says "not emitted for fainted switches"). Called before any of
/// `withdraw`'s own resets, so both handlers below see the outgoing Pokemon exactly as it stood the
/// moment it left: its stat stages, its status, its volatiles all still in place.
pub const PORTED_ON_SWITCH_OUT_ABILITIES: [&str; 2] = ["NATURAL_CURE", "REGENERATOR"];

pub fn ability_on_switch_out(state: &mut State, side: usize, log: &mut Log) {
    match state.sides[side].active_pokemon().ability.as_str() {
        "REGENERATOR" => heal_by(state, side, 3, Healer::Ability("REGENERATOR"), log),
        "NATURAL_CURE" if state.sides[side].active_pokemon().status != Status::None => {
            clear_status(state, side, "natural_cure", log);
        }
        _ => {}
    }
}

/// Protosynthesis / Quark Drive: the same `_paradox_evaluate` runs from three separate events —
/// `ON_SWITCH_IN`, `ON_TURN_START` and `ON_RESIDUAL` (at `ResidualOrder.PARADOX`, right after the
/// field's own duration tick and before the weather chip) — with no event-specific behaviour of
/// its own, which is why one function is called from all three sites rather than three copies of
/// it. The 1.3x damage contribution lives in `abilities::handle`; the 1.5x Speed contribution lives
/// in `effective_speed`; this function only ever decides *which* stat, if any, is boosted.
pub const PORTED_PARADOX_ABILITIES: [&str; 2] = ["PROTOSYNTHESIS", "QUARK_DRIVE"];
const PARADOX_STATS: [&str; 5] = ["ATTACK", "DEFENCE", "SP_ATTACK", "SP_DEFENCE", "SPEED"];

/// `_best_stat`: `max(_PARADOX_STATS, key=pokemon.effective_stat)` — keeps the first equal element
/// on a tie, matched here the same way Beast Boost's own tie-break is.
fn best_effective_stat(pokemon: &Pokemon) -> &'static str {
    let mut best = PARADOX_STATS[0];
    let mut best_value = pokemon.effective(best);
    for &stat in &PARADOX_STATS[1..] {
        let value = pokemon.effective(stat);
        if value > best_value {
            best = stat;
            best_value = value;
        }
    }
    best
}

fn activate_paradox(state: &mut State, side: usize, ability: &str, from_booster: bool, log: &mut Log) {
    let best = best_effective_stat(state.sides[side].active_pokemon());
    let pokemon = state.sides[side].active_mut();
    pokemon.paradox_boost = Some(best.to_string());
    pokemon.paradox_from_booster = from_booster;
    let nickname = pokemon.nickname.clone();
    log.push(Event::ParadoxActivated {
        side: side as i32,
        pokemon: nickname,
        ability: ability.to_string(),
        stat: best.to_string(),
        from_booster,
    });
}

pub fn evaluate_paradox(state: &mut State, side: usize, log: &mut Log) {
    let pokemon = state.sides[side].active_pokemon();
    let ability = pokemon.ability.clone();
    let energized = match ability.as_str() {
        "PROTOSYNTHESIS" => matches!(state.field.weather.as_str(), "SUN" | "HARSH_SUN"),
        "QUARK_DRIVE" => state.field.terrain == "ELECTRIC",
        _ => return,
    };
    if pokemon.fainted() {
        return;
    }
    if pokemon.paradox_boost.is_some() {
        if energized || pokemon.paradox_from_booster {
            return;
        }
        // The condition ended; a held Booster Energy may still take over below, in the same call.
        state.sides[side].active_mut().paradox_boost = None;
    }
    if energized {
        activate_paradox(state, side, &ability, false, log);
    } else if state.sides[side].active_pokemon().item == "BOOSTER_ENERGY" {
        let pokemon = state.sides[side].active_mut();
        pokemon.last_consumed_item = pokemon.item.clone();
        pokemon.item = "NONE".to_string();
        pokemon.item_consumed = true;
        activate_paradox(state, side, &ability, true, log);
    }
}

/// `ON_TURN_START`: emitted once per turn, before actions are ordered — before `order_actions`
/// reads `effective_speed`, which is the entire reason this exists rather than leaving Paradox
/// abilities to `ON_SWITCH_IN` and `ON_RESIDUAL` alone. Visited in `registered_at` order, the same
/// rule `abilities::apply_damage_calc` sorts by, since both sides register at the same bus
/// priority.
pub fn ability_on_turn_start(state: &mut State, log: &mut Log) {
    let mut order = [0usize, 1];
    order.sort_by_key(|side| state.sides[*side].active_pokemon().registered_at);
    for side in order {
        if !state.sides[side].active_pokemon().fainted() {
            evaluate_paradox(state, side, log);
        }
    }
}

/// Abilities implemented here, on top of the damage-calc ones.
pub const PORTED_ABILITIES: [&str; 20] = [
    "AFTERMATH",
    "BERSERK",
    "CURSED_BODY",
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
    "THERMAL_EXCHANGE",
    "TOXIC_CHAIN",
    "TOXIC_DEBRIS",
    "WEAK_ARMOR",
];

/// Items implemented here, on top of the damage-calc ones.
pub const PORTED_ITEMS: [&str; 13] = [
    "BOOSTER_ENERGY",
    "EJECT_BUTTON",
    "ELECTRIC_SEED",
    "GRASSY_SEED",
    "LIFE_ORB",
    "MISTY_SEED",
    "PSYCHIC_SEED",
    "RED_CARD",
    "ROCKY_HELMET",
    "SITRUS_BERRY",
    "STARF_BERRY",
    "WEAKNESS_POLICY",
    "WIKI_BERRY",
];

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
/// Solar Power's own chip, at `ResidualOrder.WEATHER` (9000) — the same band as the sandstorm
/// chip itself, which is why this is called alongside `field::weather_residual` in `residuals()`
/// rather than from `residual_before_status` down at `WEATHER_ABILITY` (8500) with Ice Body and
/// Dry Skin's own weather reactions.
pub fn solar_power_chip(state: &mut State, side: usize, log: &mut Log) {
    if state.sides[side].active_pokemon().ability == "SOLAR_POWER"
        && matches!(effective_weather(state).as_str(), "SUN" | "HARSH_SUN")
    {
        chip(state, side, 8, "SOLAR_POWER", log);
    }
}

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
                    apply_main_status_from(state, other, status, None, None, tape, log)?;
                }
            }
            "EFFECT_SPORE" if hit.contact && attacker_up => {
                if tape.probability()? < 0.3 {
                    let picked = tape.integer(0, SPORE_STATUSES.len() as i32 - 1)? as usize;
                    apply_main_status_from(state, other, SPORE_STATUSES[picked], None, None, tape, log)?;
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
            "THERMAL_EXCHANGE" if !fainted && hit.move_type == "FIRE" => {
                apply_stage_changes(state, side, &[("ATTACK".to_string(), 1)], "thermal_exchange", log);
            }
            // Toxic Debris: no fainted guard in the Python at all (unlike every stat-bump reaction
            // above it) — a holder that faints on the very hit that triggers this still scatters
            // the spikes. Reproduced as written, not brought in line with its neighbours.
            "TOXIC_DEBRIS" if hit.category == "PHYSICAL" => {
                let layers = state.sides[other].hazards.get("TOXIC_SPIKES").unwrap_or(0);
                if layers < 2 {
                    state.sides[other].hazards.insert("TOXIC_SPIKES".to_string(), layers + 1);
                    log.push(Event::HazardSet { side: other as i32, hazard: "TOXIC_SPIKES".to_string() });
                }
            }
            "CURSED_BODY" if hit.dealt > 0 && attacker_up => {
                let attacker = state.sides[other].active_pokemon();
                if attacker.disabled_slot.is_none()
                    && attacker.last_move_slot.is_some()
                    && tape.probability()? < 0.3
                {
                    start_disable(state, other, log);
                }
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
                apply_main_status_from(state, other, Status::Poison, None, None, tape, log)?;
            }
        }
        "TOXIC_CHAIN" if defender_up => {
            if tape.probability()? < 0.3 {
                apply_main_status_from(state, other, Status::Toxic, None, None, tape, log)?;
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
            let types = state.sides[side].active_pokemon().battle_types();
            if db.effectiveness(hit.move_type, &types) >= 2.0 {
                consume(state, side);
                let stages = [("ATTACK".to_string(), 2), ("SP_ATTACK".to_string(), 2)];
                apply_stage_changes(state, side, &stages, "weakness_policy", log);
            }
        }
        // Eject Button: announced and consumed the instant a hit lands, but the actual switch waits
        // — `side.needs_switch` only gets read once the whole action has finished resolving, so a
        // single-hit move's own `DamageDealt` summary still logs under the Pokemon that is, for the
        // moment, still standing. `turn::resolve_pending_switches` performs the switch itself, once
        // per completed action.
        "EJECT_BUTTON" if hit.dealt > 0 => {
            let active = state.sides[side].active;
            let has_bench =
                (0..state.sides[side].team.len()).any(|index| index != active && !state.sides[side].team[index].fainted());
            if has_bench {
                consume(state, side);
                let nickname = state.sides[side].active_pokemon().nickname.clone();
                log.push(Event::SelfSwitchPending { side: side as i32, pokemon: nickname });
                state.sides[side].needs_switch = true;
            }
        }
        // Red Card: drags the *attacker* out instead, at random — the same draw Whirlwind/Roar/
        // Dragon Tail already take, reused rather than reimplemented.
        "RED_CARD" if hit.dealt > 0 && !state.sides[other].active_pokemon().fainted() => {
            consume(state, side);
            crate::turn::force_random_switch(state, other, db, tape, log)?;
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
    pokemon.last_consumed_item = pokemon.item.clone();
    pokemon.item = "NONE".to_string();
    pokemon.item_consumed = true;
}

/// Life Orb's own `ON_ACTION_RESOLVE`: a tenth of its holder's max HP, once for the whole move
/// (against the summed multi-hit total, not per blow), skipped on a miss, a fainted holder, or
/// Magic Guard — the same guard every other indirect-damage site already asks.
pub fn life_orb_recoil(state: &mut State, side: usize, total_dealt: i32, log: &mut Log) {
    if state.sides[side].active_pokemon().item != "LIFE_ORB" {
        return;
    }
    if total_dealt <= 0
        || state.sides[side].active_pokemon().fainted()
        || crate::inline::ignores_indirect_damage(state.sides[side].active_pokemon())
    {
        return;
    }
    let pokemon = state.sides[side].active_mut();
    let chip = std::cmp::max(1, pokemon.totals.hp / 10);
    pokemon.take_damage(chip);
    let nickname = pokemon.nickname.clone();
    log.push(Event::ItemChipDamage { side: side as i32, pokemon: nickname, item: "LIFE_ORB".to_string(), amount: chip });
}

/// `ON_SWITCH_IN`, for whoever has just arrived on this side.
#[allow(clippy::collapsible_match)] // same reason: a draw belongs in the arm, not in the guard
pub fn on_switch_in(state: &mut State, side: usize, log: &mut Log) {
    let other = 1 - side;
    let ability = state.sides[side].active_pokemon().ability.clone();
    match ability.as_str() {
        "INTIMIDATE" => {
            // A sub blocks Intimidate (gen 8+).
            let opponent = state.sides[other].active_pokemon();
            if !opponent.fainted() && !opponent.volatiles.contains_key("SUBSTITUTE") {
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
                let rock = crate::inline::rock_for_weather(weather);
                state.field.weather_turns_left =
                    if rock.is_some_and(|r| state.sides[side].active_pokemon().item == r) { 8 } else { 5 };
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
                state.field.terrain_turns_left =
                    if state.sides[side].active_pokemon().item == "TERRAIN_EXTENDER" { 8 } else { 5 };
                let nickname = state.sides[side].active_pokemon().nickname.clone();
                log.push(Event::TerrainSetByAbility {
                    side: side as i32,
                    pokemon: nickname,
                    ability: ability.clone(),
                });
                // `_set_terrain_from_ability` sweeps both sides' seeds itself, on the spot — not
                // through the bus at all. This is what makes a seed fire for a Pokemon that is not
                // switching in this turn: an already-standing seed holder, or one auto-replacing a
                // fainted ally elsewhere on the field the instant this ability's terrain lands.
                consume_terrain_seeds_on_terrain_change(state, log);
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
    // `_bind_paradox` registers its own `ON_SWITCH_IN` handler separately from the match above,
    // rather than as one more arm in it.
    evaluate_paradox(state, side, log);
    // `_bind_terrain_seed`'s own `ON_SWITCH_IN` binding, at `EventPriority.ITEM` (1000) — after
    // every ability above it including Paradox's own 2000 — so a Pokemon that both sets and eats
    // its own seed (Electric Surge holding Electric Seed) still sees the terrain it just set, on
    // the same switch-in. This is only one of two paths to the same seed: the terrain-setter's own
    // direct sweep (`consume_terrain_seeds_on_terrain_change`, called from the match above and from
    // `turn::resolve_move`'s `TerrainEffect` arm) is the other, and fires even when this Pokemon
    // itself never switches in at all.
    consume_terrain_seed_for_side(state, side, log);
}

fn consume_terrain_seed_for_side(state: &mut State, side: usize, log: &mut Log) {
    if state.sides[side].active_pokemon().fainted() {
        return;
    }
    let (seed, stat) = match state.field.terrain.as_str() {
        "GRASSY" => ("GRASSY_SEED", "DEFENCE"),
        "ELECTRIC" => ("ELECTRIC_SEED", "DEFENCE"),
        "PSYCHIC" => ("PSYCHIC_SEED", "SP_DEFENCE"),
        "MISTY" => ("MISTY_SEED", "SP_DEFENCE"),
        _ => ("", ""),
    };
    if !seed.is_empty() && state.sides[side].active_pokemon().item == seed {
        let pokemon = state.sides[side].active_mut();
        pokemon.last_consumed_item = pokemon.item.clone();
        pokemon.item = "NONE".to_string();
        pokemon.item_consumed = true;
        apply_stage_changes(state, side, &[(stat.to_string(), 1)], "seed", log);
    }
}

/// `_set_terrain_from_ability` / `_apply_field_effect`'s own direct sweep: the moment terrain
/// actually changes, both sides' seeds are checked on the spot, independent of `ON_SWITCH_IN`.
pub fn consume_terrain_seeds_on_terrain_change(state: &mut State, log: &mut Log) {
    consume_terrain_seed_for_side(state, 0, log);
    consume_terrain_seed_for_side(state, 1, log);
}

/// Abilities and items implemented at `ON_RESIDUAL`, on top of everything above.
pub const PORTED_RESIDUAL_ABILITIES: [&str; 9] = [
    "BAD_DREAMS",
    "HYDRATION",
    "ICE_BODY",
    "POISON_HEAL",
    "RAIN_DISH",
    "SHED_SKIN",
    "SOLAR_POWER",
    "SPEED_BOOST",
    "AIR_LOCK",
];
pub const PORTED_RESIDUAL_ITEMS: [&str; 4] = ["BLACK_SLUDGE", "FLAME_ORB", "LEFTOVERS", "TOXIC_ORB"];

/// `effective_weather`: Air Lock suppresses what the weather *does* while its holder is out,
/// without clearing the weather itself. Everything that asks about weather asks through this.
pub fn effective_weather(state: &State) -> String {
    for side in &state.sides {
        let active = side.active_pokemon();
        if !active.fainted() && active.ability == "AIR_LOCK" {
            return "NONE".to_string();
        }
    }
    state.field.weather.clone()
}

/// The end-of-turn handlers an ability or item registers, run in `ResidualOrder`.
///
/// The order is the whole content of this function. Poison Heal has to heal before the status chip
/// would have hurt (and the chip checks for it and stands down); Leftovers recovers before the
/// chip, so a Pokemon at one HP under Toxic still dies; Speed Boost is last of everything.
pub fn residual_before_status(state: &mut State, side: usize, tape: &mut Tape, log: &mut Log) -> Result<(), Refusal> {
    let weather = effective_weather(state);
    let ability = state.sides[side].active_pokemon().ability.clone();

    // WEATHER_ABILITY (8500).
    let weather_heal = match ability.as_str() {
        "ICE_BODY" if weather == "SNOW" => true,
        "RAIN_DISH" if matches!(weather.as_str(), "RAIN" | "HEAVY_RAIN") => true,
        _ => false,
    };
    if weather_heal {
        heal_by(state, side, 16, Healer::Ability(&ability), log);
    }

    // TERRAIN (8400), below the weather abilities above and above the cures below.
    crate::field::terrain_residual(state, side, log);

    // CURE (8200). Harvest is absent: it regrows a berry, which needs the consumed-item memory.
    if ability == "HYDRATION"
        && state.sides[side].active_pokemon().status != Status::None
        && matches!(weather.as_str(), "RAIN" | "HEAVY_RAIN")
    {
        clear_status(state, side, "hydration", log);
    }
    if ability == "SHED_SKIN" && state.sides[side].active_pokemon().status != Status::None {
        // The draw happens whenever there is a status to shed, landed or not.
        if tape.probability()? < 1.0 / 3.0 {
            clear_status(state, side, "shed_skin", log);
        }
    }

    // ITEM_RECOVERY (8000). Both guard on the holder still standing — an earlier chip in the same
    // pass can have knocked it out, and a corpse does not eat sludge.
    // The guard belongs to these two and nothing else. Returning here skipped Leech Seed further
    // down, which the Python saps regardless — a Pokemon the sandstorm just felled still reports a
    // sap of zero.
    let item = state.sides[side].active_pokemon().item.clone();
    let standing = !state.sides[side].active_pokemon().fainted();
    if !standing {
        // fall through to Leech Seed and Poison Heal below
    } else if item == "LEFTOVERS" {
        heal_by(state, side, 16, Healer::Item(&item), log);
    } else if item == "BLACK_SLUDGE" {
        let poison = state.sides[side].active_pokemon().types.iter().flatten().any(|t| t == "POISON");
        if poison {
            heal_by(state, side, 16, Healer::Item(&item), log);
        } else {
            let holder = state.sides[side].active_mut();
            let dealt = holder.take_damage(std::cmp::max(1, holder.totals.hp / 16));
            let nickname = holder.nickname.clone();
            log.push(Event::ItemChipDamage {
                side: side as i32,
                pokemon: nickname,
                item: item.clone(),
                amount: dealt,
            });
        }
    }

    // LEECH_SEED (7000): a share of the victim's health, straight into whoever is opposite.
    let guarded = crate::inline::ignores_indirect_damage(state.sides[side].active_pokemon());
    if state.sides[side].active_pokemon().volatiles.contains_key("LEECH_SEED") && !guarded {
        let active = state.sides[side].active_mut();
        let dealt = active.take_damage(std::cmp::max(1, active.totals.hp / 8));
        let nickname = active.nickname.clone();
        log.push(Event::LeechSeedSap { side: side as i32, pokemon: nickname, amount: dealt });
        let drainer = state.sides[1 - side].active_mut();
        if !drainer.fainted() {
            drainer.hp = std::cmp::min(drainer.totals.hp, drainer.hp + dealt);
        }
    }

    // POISON_HEAL (6500), which the status chip then declines to undo.
    let poisoned = matches!(state.sides[side].active_pokemon().status, Status::Poison | Status::Toxic);
    if ability == "POISON_HEAL" && poisoned {
        heal_by(state, side, 8, Healer::Ability(&ability), log);
    }
    Ok(())
}

/// Everything below `ResidualOrder.STATUS`, run after the chip.
pub fn residual_after_status(state: &mut State, side: usize, tape: &mut Tape, log: &mut Log) -> Result<(), Refusal> {
    let other = 1 - side;
    let ability = state.sides[side].active_pokemon().ability.clone();

    let guarded = crate::inline::ignores_indirect_damage(state.sides[side].active_pokemon());

    // NIGHTMARE (5000): only while its victim is still asleep, and it lifts the moment they wake.
    if state.sides[side].active_pokemon().volatiles.contains_key("NIGHTMARE") {
        if state.sides[side].active_pokemon().status != Status::Sleep {
            state.sides[side].active_mut().volatiles.remove("NIGHTMARE");
        } else if !guarded {
            chip_named(state, side, 4, "nightmare", log);
        }
    }

    // PARTIAL_TRAP (4900). The counter is read before the damage and written after it, so the turn
    // a grip expires is still a turn its victim is squeezed on.
    if let Some(remaining) = state.sides[side].active_pokemon().volatiles.get("PARTIALLY_TRAPPED").copied() {
        if !guarded {
            let active = state.sides[side].active_mut();
            let dealt = active.take_damage(std::cmp::max(1, active.totals.hp / 8));
            let nickname = active.nickname.clone();
            log.push(Event::TrapSqueezed { side: side as i32, pokemon: nickname, amount: dealt });
        }
        if remaining <= 1 {
            state.sides[side].active_mut().volatiles.remove("PARTIALLY_TRAPPED");
            let nickname = state.sides[side].active_pokemon().nickname.clone();
            log.push(Event::TrapReleased { side: side as i32, pokemon: nickname });
        } else {
            state.sides[side].active_mut().volatiles.insert("PARTIALLY_TRAPPED".to_string(), remaining - 1);
        }
    }

    // SALT_CURE (4800): twice as fast against Water and Steel, which is the whole of the move.
    if state.sides[side].active_pokemon().volatiles.contains_key("SALT_CURE") && !guarded {
        let active = state.sides[side].active_pokemon();
        let brittle = active.types.iter().flatten().any(|t| t == "WATER" || t == "STEEL");
        chip_named(state, side, if brittle { 4 } else { 8 }, "salt_cure", log);
    }

    // CURSE (4700): a permanent quarter-HP chip on the Ghost's target, with no countdown at all —
    // it only ever ends when its victim faints or switches out.
    if state.sides[side].active_pokemon().volatiles.contains_key("CURSE") && !guarded {
        chip_named(state, side, 4, "curse", log);
    }

    // BAD_DREAMS (4500): the foe's sleep, not this Pokemon's.
    if ability == "BAD_DREAMS" {
        let foe = state.sides[other].active_pokemon();
        if !foe.fainted() && foe.status == Status::Sleep && !crate::inline::ignores_indirect_damage(foe) {
            let foe = state.sides[other].active_mut();
            let dealt = foe.take_damage(std::cmp::max(1, foe.totals.hp / 8));
            let nickname = foe.nickname.clone();
            let fainted = foe.fainted();
            log.push(Event::AbilityChipDamage {
                side: other as i32,
                pokemon: nickname.clone(),
                ability: "BAD_DREAMS".into(),
                amount: dealt,
            });
            if fainted {
                log.push(Event::Fainted { side: other as i32, pokemon: nickname });
            }
        }
    }

    // YAWN (3000): drowsy through this turn, asleep at the end of the next.
    if let Some(left) = state.sides[side].active_pokemon().volatiles.get("YAWN").copied() {
        if left - 1 > 0 {
            state.sides[side].active_mut().volatiles.insert("YAWN".to_string(), left - 1);
        } else {
            state.sides[side].active_mut().volatiles.remove("YAWN");
            if state.sides[side].active_pokemon().status == Status::None {
                let turns = tape.integer(2, 5)?;
                let active = state.sides[side].active_mut();
                active.status = Status::Sleep;
                active.status_turns = turns;
                let nickname = active.nickname.clone();
                log.push(Event::StatusInflicted {
                    side: side as i32,
                    pokemon: nickname,
                    status: "SLEEP".into(),
                });
            }
        }
    }

    // LOCKED_MOVE (2000): the countdown started when the rampage or the roll began. On the turn it
    // reaches zero the lock lifts unconditionally — but only a rampage leaves its user confused on
    // the way out; Rollout and Ice Ball run their course and stop clean.
    if let Some(left) = state.sides[side].active_pokemon().volatiles.get("LOCKED_MOVE").copied() {
        if left - 1 > 0 {
            state.sides[side].active_mut().volatiles.insert("LOCKED_MOVE".to_string(), left - 1);
        } else {
            let active = state.sides[side].active_pokemon();
            let was_rolling = active
                .locked_slot
                .and_then(|slot| active.moves.get(slot))
                .is_some_and(|name| crate::power::ROLLING_MOVES.contains(&name.as_str()));
            let already_confused = active.volatiles.contains_key("CONFUSION");
            let active = state.sides[side].active_mut();
            active.volatiles.remove("LOCKED_MOVE");
            active.locked_slot = None;
            active.rolling_hits = 0;
            if !was_rolling && !already_confused {
                let turns = tape.integer(2, 6)?;
                let active = state.sides[side].active_mut();
                active.volatiles.insert("CONFUSION".to_string(), turns);
                let nickname = active.nickname.clone();
                log.push(Event::VolatileInflicted {
                    side: side as i32,
                    pokemon: nickname,
                    volatile: "CONFUSION".into(),
                });
            }
        }
    }

    // ORB (1500): the orb poisons or burns whoever is carrying it, once there is room to.
    let item = state.sides[side].active_pokemon().item.clone();
    let status = match item.as_str() {
        "TOXIC_ORB" => Some(Status::Toxic),
        "FLAME_ORB" => Some(Status::Burn),
        _ => None,
    };
    if let Some(status) = status {
        if !state.sides[side].active_pokemon().fainted()
            && state.sides[side].active_pokemon().status == Status::None
        {
            apply_main_status_from(state, side, status, None, None, tape, log)?;
        }
    }

    // SPEED_BOOST (1000): every turn-end except the one it arrived on, and logged only if the
    // stage actually moved.
    if ability == "SPEED_BOOST" {
        let pokemon = state.sides[side].active_pokemon();
        if !pokemon.fainted() && !pokemon.just_switched_in {
            let pokemon = state.sides[side].active_mut();
            let before = pokemon.stage("SPEED");
            let after = (before + 1).clamp(-6, 6);
            pokemon.stages.insert("SPEED".to_string(), after);
            if after > before {
                let nickname = pokemon.nickname.clone();
                log.push(Event::StatStageChanged {
                    side: side as i32,
                    pokemon: nickname,
                    stat: "SPEED".into(),
                    delta: after - before,
                    requested: 1,
                    source: "speed_boost".into(),
                });
            }
        }
    }
    Ok(())
}

/// Whether a heal is announced under an ability's name or an item's — the only difference between
/// Leftovers and Rain Dish once the arithmetic is done.
enum Healer<'a> {
    Ability(&'a str),
    Item(&'a str),
}

/// No fainted guard, deliberately. Leftovers and Black Sludge check for one and are gated by
/// their caller; Poison Heal, Ice Body and Rain Dish do not, so a Pokemon the sandstorm just
/// knocked to zero is healed straight back off the floor. That is the Python's behaviour and the
/// sweep found the disagreement the first time a sandstorm and a Poison Heal met.
fn heal_by(state: &mut State, side: usize, divisor: i32, by: Healer, log: &mut Log) {
    let pokemon = state.sides[side].active_mut();
    let amount = std::cmp::max(1, pokemon.totals.hp / divisor);
    let before = pokemon.hp;
    pokemon.hp = std::cmp::min(pokemon.totals.hp, pokemon.hp + amount);
    let healed = pokemon.hp - before;
    if healed == 0 {
        return;
    }
    let nickname = pokemon.nickname.clone();
    log.push(match by {
        // The item entry carries no amount; the ability's does. That is the Python's choice, not
        // an oversight here.
        Healer::Item(item) => Event::ItemHealed { side: side as i32, pokemon: nickname, item: item.to_string() },
        Healer::Ability(ability) => Event::AbilityHealed {
            side: side as i32,
            pokemon: nickname,
            ability: ability.to_string(),
            amount: healed,
        },
    });
}

/// A residual chip reported under a source name, as `ResidualDamage`.
fn chip_named(state: &mut State, side: usize, divisor: i32, source: &str, log: &mut Log) {
    let active = state.sides[side].active_mut();
    let dealt = active.take_damage(std::cmp::max(1, active.totals.hp / divisor));
    let nickname = active.nickname.clone();
    log.push(Event::ResidualDamage {
        side: side as i32,
        pokemon: nickname,
        source: source.to_string(),
        amount: dealt,
    });
}

fn clear_status(state: &mut State, side: usize, clearance: &str, log: &mut Log) {
    let pokemon = state.sides[side].active_mut();
    pokemon.status = Status::None;
    pokemon.status_turns = 0;
    let nickname = pokemon.nickname.clone();
    log.push(Event::StatusCleared { side: side as i32, pokemon: nickname, clearance: clearance.to_string() });
}
