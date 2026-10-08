//! A JSON snapshot of a battle, for anything that needs to look at the state rather than just the
//! log — `bin/replay.rs`'s per-turn trace, and `python.rs`'s per-step return value alike. Shaped to
//! match what the Python differential harness already expects on its own side of the same trace.

use crate::battle::{Pokemon, State, STAGE_NAMES};
use serde_json::json;

pub fn pokemon(p: &Pokemon) -> serde_json::Value {
    json!({
        "nickname": p.nickname,
        "species": p.species_name,
        "hp": p.hp,
        "max_hp": p.totals.hp,
        "status": p.status.name(),
        "status_turns": p.status_turns,
        "fainted": p.fainted(),
        "item": p.item,
        "item_consumed": p.item_consumed,
        "ability": p.ability,
        // `battle_types`, as the Python's digest reads it — Roost's Flying-less typing included, which
        // a Pokemon knocked out mid-turn keeps, since only the standing have their volatiles ticked.
        "types": p.battle_types().into_iter().flatten().collect::<Vec<_>>(),
        "stages": STAGE_NAMES.iter().map(|s| (s.to_string(), json!(p.stage(s))))
            .collect::<serde_json::Map<String, serde_json::Value>>(),
        "volatiles": p.volatiles,
        "pp": p.pp,
        "lives_used": p.lives_used,
        "made_last_stand": p.made_last_stand,
    })
}

pub fn state(state: &State) -> serde_json::Value {
    json!({
        "turn": state.turn,
        "outcome": state.outcome.map(|o| o.name()),
        "weather": state.field.weather,
        "weather_turns_left": state.field.weather_turns_left,
        "terrain": state.field.terrain,
        "terrain_turns_left": state.field.terrain_turns_left,
        "sides": state.sides.iter().map(|side| json!({
            "active": serde_json::Value::Null,
            "team": side.team.iter().map(pokemon).collect::<Vec<_>>(),
            "hazards": side.hazards,
            "screens": side.screens,
        })).collect::<Vec<_>>(),
    })
}
