//! Weather, terrain, screens, hazards — everything that belongs to the battlefield rather than to
//! a Pokemon.
//!
//! This is the highest-leverage part of the port and the reason it came before the long tail of
//! abilities: the damage formula already reads screens and weather, `effective_speed` already
//! wants a Tailwind, and a great many abilities and items are inert descriptions until there is a
//! sandstorm for Sand Force to be strong in.
//!
//! Durations are the part to be careful with. A weather set to five turns is ticked at the end of
//! the turn it was set on, so it covers this turn and four more; a screen set to five is ticked in
//! the same pass but on its own side's schedule. The Python does both in `_apply_residuals`, field
//! first and sides last, with every Pokemon's residual chip in between — and that order decides
//! whether a sandstorm's last chip lands.

use crate::battle::{Pokemon, State};
use crate::data::Database;
use crate::log::{Event, Log};

/// `_SPIKES_FRACTION`: one layer an eighth, two a sixth, three a quarter.
const SPIKES_FRACTION: [i32; 4] = [0, 8, 6, 4];
/// `_MAX_HAZARD_LAYERS`. Everything absent is a single layer.
pub fn max_layers(hazard: &str) -> i32 {
    match hazard {
        "SPIKES" => 3,
        "TOXIC_SPIKES" => 2,
        _ => 1,
    }
}

/// `_SCREEN_HAZARDS`: the three that live in `side.screens` rather than `side.hazards`.
pub const SCREENS: [&str; 3] = ["REFLECT", "LIGHT_SCREEN", "AURORA_VEIL"];
/// `_SANDSTORM_IMMUNE_TYPES`.
const SANDSTORM_IMMUNE: [&str; 3] = ["ROCK", "GROUND", "STEEL"];

/// `Pokemon.is_grounded` — intrinsic only, as the Python's is: type, item, ability, and nothing
/// from the field. Gravity would change the answer and is not ported.
///
/// Levitate and Air Balloon are read here even though neither is on the ported list: the clause is
/// right, it is the *rest* of what they do that is missing, and a Pokemon carrying either is
/// refused before it can reach this.
pub fn is_grounded(pokemon: &Pokemon) -> bool {
    !pokemon.battle_types().iter().flatten().any(|t| t == "FLYING")
        && pokemon.item != "AIR_BALLOON"
        && pokemon.ability != "LEVITATE"
}

/// `_tick_field_durations`, run once at the top of the residual pass.
pub fn tick_field(state: &mut State, log: &mut Log) {
    if state.field.weather_turns_left > 0 {
        state.field.weather_turns_left -= 1;
        if state.field.weather_turns_left == 0 {
            let prior = std::mem::replace(&mut state.field.weather, "NONE".to_string());
            if prior != "NONE" {
                log.push(Event::WeatherFaded { weather: prior });
            }
        }
    }
    if state.field.terrain_turns_left > 0 {
        state.field.terrain_turns_left -= 1;
        if state.field.terrain_turns_left == 0 {
            let prior = std::mem::replace(&mut state.field.terrain, "NONE".to_string());
            if prior != "NONE" {
                log.push(Event::TerrainFaded { terrain: prior });
            }
        }
    }
}

/// `_tick_side_durations`, run per side at the end of its own residual pass.
pub fn tick_side(state: &mut State, side: usize, log: &mut Log) {
    if state.sides[side].tailwind_turns > 0 {
        state.sides[side].tailwind_turns -= 1;
        if state.sides[side].tailwind_turns == 0 {
            log.push(Event::TailwindFaded { side: side as i32 });
        }
    }
    let standing: Vec<String> = state.sides[side].screens.keys().cloned().collect();
    for screen in standing {
        let left = state.sides[side].screens.get_mut(&screen).expect("just listed");
        *left -= 1;
        if *left <= 0 {
            state.sides[side].screens.remove(&screen);
            log.push(Event::ScreenFaded { side: side as i32, screen });
        }
    }
}

/// The sandstorm chip, at `ResidualOrder.WEATHER` (9000) — the first thing in the pass.
pub fn weather_residual(state: &mut State, side: usize, log: &mut Log) {
    // Through `effective_weather`, so an Air Lock on either side stops the sand biting without
    // stopping the sandstorm.
    if crate::hooks::effective_weather(state) == "SANDSTORM" {
        let active = state.sides[side].active_pokemon();
        let immune = active.types.iter().flatten().any(|t| SANDSTORM_IMMUNE.contains(&t.as_str()))
            || matches!(active.ability.as_str(), "SAND_VEIL" | "OVERCOAT")
            || crate::inline::ignores_indirect_damage(active);
        if !immune {
            let active = state.sides[side].active_mut();
            let dealt = active.take_damage(std::cmp::max(1, active.totals.hp / 16));
            let nickname = active.nickname.clone();
            log.push(Event::ResidualDamage {
                side: side as i32,
                pokemon: nickname,
                source: "sandstorm".into(),
                amount: dealt,
            });
        }
    }
}

/// `_apply_entry_hazards`, in the Python's order: rocks, then spikes, then toxic spikes, then the
/// web. Rocks and spikes each stop the rest if they knock the arrival out.
pub fn entry_hazards(state: &mut State, side: usize, db: &Database, log: &mut Log) {
    // Magic Guard cancels the whole ON_ENTRY_HAZARD emit, so not even Sticky Web's speed drop
    // lands — it is not "no damage from hazards", it is "no hazards".
    if crate::inline::ignores_indirect_damage(state.sides[side].active_pokemon()) {
        return;
    }
    if stealth_rock(state, side, db, log) {
        return;
    }
    let grounded = is_grounded(state.sides[side].active_pokemon());
    if grounded && spikes(state, side, log) {
        return;
    }
    if grounded && state.sides[side].hazards.contains_key("TOXIC_SPIKES") {
        toxic_spikes(state, side, log);
    }
    if grounded && state.sides[side].hazards.contains_key("STICKY_WEB") {
        sticky_web(state, side, log);
    }
}

/// True if the arrival fainted to it.
/// The grassy-terrain heal, at `TERRAIN` (8400) — *below* the weather abilities at 8500, which is
/// why it is a separate function rather than sharing one with the sandstorm above it.
pub fn terrain_residual(state: &mut State, side: usize, log: &mut Log) {
    if state.field.terrain != "GRASSY" || !is_grounded(state.sides[side].active_pokemon()) {
        return;
    }
    let active = state.sides[side].active_mut();
    let amount = std::cmp::max(1, active.totals.hp / 16);
    let before = active.hp;
    active.hp = std::cmp::min(active.totals.hp, active.hp + amount);
    let healed = active.hp - before;
    if healed > 0 {
        let nickname = active.nickname.clone();
        log.push(Event::Healed { side: side as i32, pokemon: nickname, amount: healed });
    }
}

fn stealth_rock(state: &mut State, side: usize, db: &Database, log: &mut Log) -> bool {
    if !state.sides[side].hazards.contains_key("STEALTH_ROCK") {
        return false;
    }
    let types = state.sides[side].active_pokemon().types.clone();
    let effectiveness = db.effectiveness("ROCK", &types);
    if effectiveness <= 0.0 {
        return false;
    }
    let incoming = state.sides[side].active_mut();
    // The Python multiplies into a float and floors on the way out of `int(... // 8)`, so a
    // quarter-weak Pokemon takes a full half: the division is by eight *after* the multiplier.
    let chip = std::cmp::max(1, (incoming.totals.hp as f64 * effectiveness / 8.0).floor() as i32);
    let dealt = incoming.take_damage(chip);
    let nickname = incoming.nickname.clone();
    let fainted = incoming.fainted();
    log.push(Event::HazardDamage {
        side: side as i32,
        pokemon: nickname.clone(),
        hazard: "STEALTH_ROCK".into(),
        amount: dealt,
    });
    if fainted {
        log.push(Event::Fainted { side: side as i32, pokemon: nickname });
    }
    fainted
}

fn spikes(state: &mut State, side: usize, log: &mut Log) -> bool {
    let layers = state.sides[side].hazards.get("SPIKES").copied().unwrap_or(0);
    if layers <= 0 {
        return false;
    }
    let incoming = state.sides[side].active_mut();
    let dealt = incoming.take_damage(std::cmp::max(1, incoming.totals.hp / SPIKES_FRACTION[layers as usize]));
    let nickname = incoming.nickname.clone();
    let fainted = incoming.fainted();
    log.push(Event::HazardDamage {
        side: side as i32,
        pokemon: nickname.clone(),
        hazard: "SPIKES".into(),
        amount: dealt,
    });
    if fainted {
        log.push(Event::Fainted { side: side as i32, pokemon: nickname });
    }
    fainted
}

/// The web goes straight to `change_stat_stage`, not through `apply_stage_changes`. So Simple does
/// not double it, Clear Body does not refuse it, and — the visible half — it is logged only if the
/// stage actually moved, where an ordinary drop reports a delta of zero and says so.
fn sticky_web(state: &mut State, side: usize, log: &mut Log) {
    let incoming = state.sides[side].active_mut();
    let before = incoming.stage("SPEED");
    let after = (before - 1).clamp(-6, 6);
    incoming.stages.insert("SPEED".to_string(), after);
    if after - before < 0 {
        let nickname = incoming.nickname.clone();
        log.push(Event::StatStageChanged {
            side: side as i32,
            pokemon: nickname,
            stat: "SPEED".into(),
            delta: after - before,
            requested: -1,
            source: "sticky_web".into(),
        });
    }
}

fn toxic_spikes(state: &mut State, side: usize, log: &mut Log) {
    let layers = state.sides[side].hazards.get("TOXIC_SPIKES").copied().unwrap_or(0);
    let incoming = state.sides[side].active_pokemon();
    let types: Vec<&str> = incoming.types.iter().flatten().map(|t| t.as_str()).collect();
    let nickname = incoming.nickname.clone();
    if types.contains(&"POISON") {
        // A grounded Poison type soaks the spikes up on the way in, clearing them for good.
        state.sides[side].hazards.remove("TOXIC_SPIKES");
        log.push(Event::HazardAbsorbed { side: side as i32, pokemon: nickname });
        return;
    }
    if types.contains(&"STEEL") || incoming.status != crate::battle::Status::None {
        return;
    }
    let status = if layers >= 2 { crate::battle::Status::Toxic } else { crate::battle::Status::Poison };
    let incoming = state.sides[side].active_mut();
    incoming.status = status;
    incoming.status_turns = 0;
    log.push(Event::HazardStatus {
        side: side as i32,
        pokemon: nickname,
        status: status.name().to_string(),
    });
}
