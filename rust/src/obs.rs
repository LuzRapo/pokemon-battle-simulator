//! What a policy network sees: one side's view of a battle, as fixed-shape arrays.
//!
//! Mirror battles are full-information — both sides field the same six, sets and all — so nothing
//! here is hidden: every Pokemon is encoded completely, the viewer's own team first. The layout is
//! mirrored exactly by `battle_sim/rl/encode.py`, which encodes a Python `BattleState` the same way
//! so a trained network can play inside the Python engine too; a cross-check test asserts the two
//! produce identical arrays at every decision of recorded battles.
//!
//! Integer ids index `rust/data/vocab.json` (0 = none) and become embeddings on the Python side.

use crate::battle::{Pokemon, State, Status, SLOT_NAMES, STAGE_NAMES};
use crate::data::Database;

pub const SLOTS: usize = 12;
pub const POKEMON_IDS: usize = 7; // species, ability, item, four moves
pub const TYPES: [&str; 18] = [
    "NORMAL", "FIRE", "WATER", "GRASS", "ELECTRIC", "ICE", "FIGHTING", "POISON", "GROUND", "FLYING", "PSYCHIC", "BUG",
    "ROCK", "GHOST", "DRAGON", "DARK", "STEEL", "FAIRY",
];
pub const STATUSES: [Status; 6] =
    [Status::Burn, Status::Poison, Status::Toxic, Status::Paralysis, Status::Sleep, Status::Freeze];
pub const VOLATILES: [&str; 25] = [
    "CONFUSION", "DISABLE", "ENCORE", "FLINCH", "FOCUS_ENERGY", "IDENTIFIED", "MIRACLE_EYE", "LEECH_SEED",
    "LOCKED_MOVE", "NIGHTMARE", "PROTECT", "YAWN", "SUBSTITUTE", "TAUNT", "SALT_CURE", "CURSE", "PERISH",
    "DESTINY_BOND", "ENDURE", "CHARGING", "MUST_RECHARGE", "SLOW_START", "LOAFING", "ROOSTED",
    "PARTIALLY_TRAPPED",
];
pub const WEATHERS: [&str; 7] = ["SUN", "RAIN", "HARSH_SUN", "HEAVY_RAIN", "SANDSTORM", "SNOW", "STRONG_WINDS"];
pub const TERRAINS: [&str; 4] = ["ELECTRIC", "GRASSY", "MISTY", "PSYCHIC"];
pub const PSEUDO_WEATHERS: [&str; 4] = ["TRICK_ROOM", "GRAVITY", "MAGIC_ROOM", "WONDER_ROOM"];
pub const HAZARDS: [(&str, f32); 4] = [("STEALTH_ROCK", 1.0), ("SPIKES", 3.0), ("TOXIC_SPIKES", 2.0), ("STICKY_WEB", 1.0)];
pub const SCREENS: [&str; 3] = ["REFLECT", "LIGHT_SCREEN", "AURORA_VEIL"];

/// present, hp fraction, max hp, five stats, seven stages, six statuses, status turns, fainted,
/// active, transformed, item consumed, eighteen types, the volatiles, four PP, and a
/// choice-locked / encored / disabled flag per slot.
pub const POKEMON_FLOATS: usize = 1 + 1 + 1 + 5 + 7 + 6 + 1 + 1 + 1 + 1 + 1 + 18 + VOLATILES.len() + 4 + 12;
/// Per side: hazards, screens, tailwind, the four once-per-battle flags, wish, future sight,
/// healing wish.
pub const SIDE_FLOATS: usize = HAZARDS.len() + SCREENS.len() + 1 + 4 + 3;
/// Weather and its turns, terrain and its turns, the pseudo-weathers, both sides, the turn number,
/// and which kind of decision this is.
pub const FIELD_FLOATS: usize = WEATHERS.len() + 1 + TERRAINS.len() + 1 + PSEUDO_WEATHERS.len() + 2 * SIDE_FLOATS + 1 + 3;

/// The decision being asked for: the lead before turn 0, a turn's action, or a replacement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    Lead,
    Turn,
    Switch,
}

/// One side's view: `ids` is `SLOTS * POKEMON_IDS`, `pokemon` is `SLOTS * POKEMON_FLOATS`, `field`
/// is `FIELD_FLOATS`, row-major, own team in slots 0..6 and the opponent's in 6..12.
pub struct Observation {
    pub ids: Vec<i64>,
    pub pokemon: Vec<f32>,
    pub field: Vec<f32>,
}

pub fn encode(state: &State, viewer: usize, decision: Decision, db: &Database) -> Observation {
    let mut observation = Observation {
        ids: vec![0; SLOTS * POKEMON_IDS],
        pokemon: vec![0.0; SLOTS * POKEMON_FLOATS],
        field: Vec::with_capacity(FIELD_FLOATS),
    };
    for (block, side) in [viewer, 1 - viewer].into_iter().enumerate() {
        let own = &state.sides[side];
        for (index, pokemon) in own.team.iter().enumerate() {
            let slot = block * 6 + index;
            let active = decision != Decision::Lead && index == own.active;
            let transformed = own.transforms.contains_key(&index);
            encode_pokemon(
                pokemon,
                active,
                transformed,
                db,
                &mut observation.ids[slot * POKEMON_IDS..(slot + 1) * POKEMON_IDS],
                &mut observation.pokemon[slot * POKEMON_FLOATS..(slot + 1) * POKEMON_FLOATS],
            );
        }
    }
    encode_field(state, viewer, decision, &mut observation.field);
    debug_assert_eq!(observation.field.len(), FIELD_FLOATS);
    observation
}

fn encode_pokemon(p: &Pokemon, active: bool, transformed: bool, db: &Database, ids: &mut [i64], out: &mut [f32]) {
    ids[0] = db.vocab.species(&p.species_name);
    ids[1] = db.vocab.ability(&p.ability);
    ids[2] = db.vocab.item(&p.item);
    for (slot, name) in p.moves.iter().take(4).enumerate() {
        ids[3 + slot] = db.vocab.move_id(name);
    }
    let mut at = 0;
    let mut push = |value: f32| {
        out[at] = value;
        at += 1;
    };
    push(1.0);
    push(if p.totals.hp > 0 { p.hp as f32 / p.totals.hp as f32 } else { 0.0 });
    push(p.totals.hp as f32 / 500.0);
    for stat in [p.totals.attack, p.totals.defence, p.totals.sp_attack, p.totals.sp_defence, p.totals.speed] {
        push(stat as f32 / 500.0);
    }
    for stage in STAGE_NAMES {
        push(p.stage(stage) as f32 / 6.0);
    }
    for status in STATUSES {
        push(if p.status == status { 1.0 } else { 0.0 });
    }
    push((p.status_turns as f32 / 10.0).min(1.0));
    push(if p.fainted() { 1.0 } else { 0.0 });
    push(if active { 1.0 } else { 0.0 });
    push(if transformed { 1.0 } else { 0.0 });
    push(if p.item_consumed { 1.0 } else { 0.0 });
    let types = p.battle_types();
    for kind in TYPES {
        push(if types.iter().flatten().any(|t| t == kind) { 1.0 } else { 0.0 });
    }
    for volatile in VOLATILES {
        push(if p.volatiles.get(volatile).is_some_and(|v| *v > 0) { 1.0 } else { 0.0 });
    }
    for slot in SLOT_NAMES {
        push((p.pp.get(slot).copied().unwrap_or(0) as f32 / 40.0).min(1.0));
    }
    for lock in [p.choice_locked_move, p.encored_slot, p.disabled_slot] {
        for slot in 0..4 {
            push(if lock == Some(slot) { 1.0 } else { 0.0 });
        }
    }
    debug_assert_eq!(at, POKEMON_FLOATS);
}

fn encode_field(state: &State, viewer: usize, decision: Decision, out: &mut Vec<f32>) {
    let field = &state.field;
    for weather in WEATHERS {
        out.push(if field.weather == weather { 1.0 } else { 0.0 });
    }
    out.push(field.weather_turns_left as f32 / 8.0);
    for terrain in TERRAINS {
        out.push(if field.terrain == terrain { 1.0 } else { 0.0 });
    }
    out.push(field.terrain_turns_left as f32 / 8.0);
    for pseudo in PSEUDO_WEATHERS {
        out.push(if field.pseudo_weather.contains_key(pseudo) { 1.0 } else { 0.0 });
    }
    for side in [viewer, 1 - viewer] {
        let own = &state.sides[side];
        for (hazard, most) in HAZARDS {
            out.push(own.hazards.get(hazard).unwrap_or(0) as f32 / most);
        }
        for screen in SCREENS {
            out.push(own.screens.get(screen).unwrap_or(0) as f32 / 8.0);
        }
        out.push(own.tailwind_turns as f32 / 4.0);
        for used in [own.has_mega_evolved, own.has_primal_reverted, own.has_ultra_bursted, own.has_used_z_move] {
            out.push(if used { 1.0 } else { 0.0 });
        }
        out.push(own.wish_turns as f32 / 2.0);
        out.push(own.future_sight_turns as f32 / 3.0);
        out.push(if own.healing_wish_pending { 1.0 } else { 0.0 });
    }
    out.push((state.turn as f32 / 100.0).min(1.0));
    for kind in [Decision::Lead, Decision::Turn, Decision::Switch] {
        out.push(if decision == kind { 1.0 } else { 0.0 });
    }
}
