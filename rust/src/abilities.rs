//! Abilities, as they affect the damage formula.
//!
//! The Python wires these to an event bus; this dispatches them directly. The difference is only
//! in the plumbing — what each one *does*, and crucially the order the modifiers are folded in,
//! has to be identical, because `chain` rounds and two folds in the other order differ by a point.
//!
//! Order is registration order at equal priority (see `EventBus.on`, whose sort is stable), and a
//! Pokemon registers when it is sent out. So the two actives are visited by when each last came in,
//! and within one Pokemon its ability is registered before its item. `registered_at` on `Pokemon`
//! carries that, and `switch_out` stamps a fresh one.
//!
//! Everything here is `ON_DAMAGE_CALC` and nothing else. An ability that also hooks another event,
//! or that the Python reads inline somewhere outside `mechanics/abilities.py`, is deliberately not
//! on this list: porting half of one is exactly the silent wrong answer this project is built to
//! avoid, so it stays refused until all of it is written.

use crate::battle::{Pokemon, State, Status};
use crate::damage::Payload;
use crate::data::{Database, Effect, Move};

/// Every ability implemented here. `unsupported_pokemon` refuses anything live and absent.
pub const PORTED: [&str; 48] = [
    "ADAPTABILITY",
    "ANALYTIC",
    "AURA_BREAK",
    "BEADS_OF_RUIN",
    "BLAZE",
    "DARK_AURA",
    "DEFEATIST",
    "DRAGONS_MAW",
    "FAIRY_AURA",
    "FILTER",
    "FLOWER_GIFT",
    "FLUFFY",
    "FUR_COAT",
    "GUTS",
    "HEATPROOF",
    "HUGE_POWER",
    "HUSTLE",
    "ICE_SCALES",
    "IRON_FIST",
    "MARVEL_SCALE",
    "MEGA_LAUNCHER",
    "MULTISCALE",
    "NEUROFORCE",
    "OVERGROW",
    "PARENTAL_BOND",
    "PRISM_ARMOR",
    "PUNK_ROCK",
    "PURE_POWER",
    "RECKLESS",
    "ROCKY_PAYLOAD",
    "SAND_FORCE",
    "SHADOW_SHIELD",
    "SHARPNESS",
    "STAKEOUT",
    "STEELWORKER",
    "STRONG_JAW",
    "SUPREME_OVERLORD",
    "SWARM",
    "SWORD_OF_RUIN",
    "TABLETS_OF_RUIN",
    "TECHNICIAN",
    "THICK_FAT",
    "TINTED_LENS",
    "TORRENT",
    "TOUGH_CLAWS",
    "TRANSISTOR",
    "UNAWARE",
    "VESSEL_OF_RUIN",
];

/// What the Python puts in `hit_payload_base` plus the few board facts these handlers read.
pub struct Calc<'a> {
    pub move_type: &'a str,
    pub category: &'a str,
    pub contact: bool,
    pub the_move: &'a Move,
    pub attacker: &'a Pokemon,
    pub defender: &'a Pokemon,
    /// Supreme Overlord counts its own side's graves.
    pub fallen_on_attacker_side: usize,
    /// Analytic: has the other side already moved this turn?
    pub defender_side_acted: bool,
    pub weather: &'a str,
    pub db: &'a Database,
    /// Read by the resist berries, which must not react to a hit a substitute is about to take —
    /// unlike everything else in this struct, which still runs the same either way (the Python's
    /// own `ON_DAMAGE_CALC` emit happens *before* the substitute check, so a berry reacting to
    /// projected low HP still triggers; only the resist berries ask this specifically).
    pub behind_substitute: bool,
}

/// Types Sand Force boosts, from `_SAND_FORCE_TYPES`.
const SAND_FORCE_TYPES: [&str; 3] = ["ROCK", "GROUND", "STEEL"];
/// `_OVERLORD_POWER_4096`: +10% per fallen teammate, five deep.
const OVERLORD_POWER_4096: [i64; 6] = [4096, 4506, 4915, 5325, 5734, 6144];

/// Apply what both actives' abilities and items contribute, in the order the Python's bus fires.
///
/// Every ability first, then every item — `EventPriority.ABILITY` is 2000 and `ITEM` is 1000, and
/// the bus sorts by priority descending before falling back on registration order. Running each
/// Pokemon's ability and item together instead, which reads more naturally, put one fold in the
/// wrong place and came back a point light on a 65-damage hit.
///
/// Returns whatever wants consuming: a resist berry is eaten by the same handler that halves the
/// hit, but this walk only holds the state by shared reference, so the eating is left to the
/// caller rather than smuggled in behind it.
pub fn apply_damage_calc(
    state: &State,
    attacker_side: usize,
    calc: &Calc,
    payload: &mut Payload,
) -> Option<crate::items::Consumed> {
    let defender_side = 1 - attacker_side;
    let mut order = [attacker_side, defender_side];
    order.sort_by_key(|side| state.sides[*side].active_pokemon().registered_at);
    // An aura is field-wide and Aura Break cancels it from either side, so this is decided once
    // before anybody's handler runs rather than inside one of them.
    let broken = state
        .sides
        .iter()
        .any(|side| side.active_pokemon().ability == "AURA_BREAK");
    for side in order {
        let pokemon = state.sides[side].active_pokemon();
        handle(&pokemon.ability, pokemon, side == attacker_side, broken, calc, payload);
    }
    let mut consumed = None;
    for side in order {
        let pokemon = state.sides[side].active_pokemon();
        if let Some(eaten) = crate::items::on_damage_calc(pokemon, side, side == attacker_side, calc, payload) {
            consumed = Some(eaten);
        }
    }
    consumed
}

fn first_damage_effect(the_move: &Move) -> Option<&Effect> {
    the_move.effects.iter().find(|e| matches!(e, Effect::DamageEffect { .. }))
}

/// One ability's contribution. `is_actor` is the Python's `context.actor is pokemon`; with two
/// active Pokemon the other one is always the defender, so no third case exists.
fn handle(ability: &str, pokemon: &Pokemon, is_actor: bool, aura_broken: bool, calc: &Calc, payload: &mut Payload) {
    let physical = calc.category == "PHYSICAL";
    let special = calc.category == "SPECIAL";
    match ability {
        // -- the attacker's own output ------------------------------------------------------
        "ADAPTABILITY" if is_actor => payload.stab_4096 = 8192,
        "HUGE_POWER" | "PURE_POWER" if is_actor && physical => payload.attack_mods_4096.push(8192),
        // Hustle trades accuracy for power; the other half of it is in `inline::accuracy_multiplier`.
        "HUSTLE" if is_actor && physical => payload.attack_mods_4096.push(6144),
        "GUTS" if is_actor => {
            if physical && pokemon.status != Status::None {
                payload.attack_mods_4096.push(6144);
            }
            // Returned unconditionally in the Python, boost or no boost: Guts means a burn never
            // halves your Attack, even on the turn the boost itself does not apply.
            payload.ignore_burn = true;
        }
        "DEFEATIST" if is_actor && pokemon.hp * 2 <= pokemon.totals.hp => payload.attack_mods_4096.push(2048),
        "STEELWORKER" if is_actor && calc.move_type == "STEEL" => payload.attack_mods_4096.push(6144),
        "DRAGONS_MAW" if is_actor && calc.move_type == "DRAGON" => payload.attack_mods_4096.push(6144),
        "ROCKY_PAYLOAD" if is_actor && calc.move_type == "ROCK" => payload.attack_mods_4096.push(6144),
        "TRANSISTOR" if is_actor && calc.move_type == "ELECTRIC" => payload.attack_mods_4096.push(6144),
        "STAKEOUT" if is_actor && calc.defender.just_switched_in => payload.attack_mods_4096.push(8192),
        "BLAZE" | "TORRENT" | "OVERGROW" | "SWARM" if is_actor => {
            let wanted = match ability {
                "BLAZE" => "FIRE",
                "TORRENT" => "WATER",
                "OVERGROW" => "GRASS",
                _ => "BUG",
            };
            if calc.move_type == wanted && 3 * pokemon.hp <= pokemon.totals.hp {
                payload.power_mods_4096.push(6144);
            }
        }
        "SHARPNESS" if is_actor && calc.the_move.slicing => payload.power_mods_4096.push(6144),
        "IRON_FIST" if is_actor && calc.the_move.punching => payload.power_mods_4096.push(4915),
        "STRONG_JAW" if is_actor && calc.the_move.biting => payload.power_mods_4096.push(6144),
        "MEGA_LAUNCHER" if is_actor && calc.the_move.pulse => payload.power_mods_4096.push(6144),
        "TOUGH_CLAWS" if is_actor && calc.contact => payload.power_mods_4096.push(5325),
        "TECHNICIAN" if is_actor => {
            if let Some(Effect::DamageEffect { power: Some(power), .. }) = first_damage_effect(calc.the_move) {
                if *power <= 60 {
                    payload.power_mods_4096.push(6144);
                }
            }
        }
        "RECKLESS" if is_actor => {
            if let Some(Effect::DamageEffect { recoil_percent: Some(_), .. }) = first_damage_effect(calc.the_move) {
                payload.power_mods_4096.push(4915);
            }
        }
        "PARENTAL_BOND" if is_actor => {
            // A flat 1.25x rather than a real second strike, matching the Python's approximation
            // exactly — including the fact that it does not apply to an already multi-hit move.
            let multi = matches!(
                first_damage_effect(calc.the_move),
                Some(Effect::DamageEffect { multi_hit: Some(_), .. })
            );
            if !multi {
                payload.power_mods_4096.push(5120);
            }
        }
        "SUPREME_OVERLORD" if is_actor => {
            let fallen = calc.fallen_on_attacker_side.min(5);
            if fallen > 0 {
                payload.power_mods_4096.push(OVERLORD_POWER_4096[fallen]);
            }
        }
        "SAND_FORCE" if is_actor && SAND_FORCE_TYPES.contains(&calc.move_type) && calc.weather == "SANDSTORM" => {
            payload.power_mods_4096.push(5325)
        }
        "ANALYTIC" if is_actor && calc.defender_side_acted => payload.power_mods_4096.push(5325),
        "TINTED_LENS" if is_actor => {
            // The plain type chart against the defender's own types, which is what the Python
            // reads here — not the fully-resolved effectiveness the formula uses.
            let effectiveness = calc.db.effectiveness(calc.move_type, &calc.defender.battle_types());
            if effectiveness > 0.0 && effectiveness < 1.0 {
                payload.final_mods_4096.push(8192);
            }
        }
        "NEUROFORCE" if is_actor => {
            if calc.db.effectiveness(calc.move_type, &calc.defender.battle_types()) > 1.0 {
                payload.final_mods_4096.push(5120);
            }
        }
        "UNAWARE" if is_actor => payload.ignore_defense_stages = true,

        // -- the defender's own mitigation --------------------------------------------------
        "MULTISCALE" | "SHADOW_SHIELD" if !is_actor && pokemon.hp == pokemon.totals.hp => {
            payload.final_mods_4096.push(2048)
        }
        "FUR_COAT" if !is_actor && physical => payload.defense_mods_4096.push(8192),
        "MARVEL_SCALE" if !is_actor && physical && pokemon.status != Status::None => {
            payload.defense_mods_4096.push(6144)
        }
        "ICE_SCALES" if !is_actor && special => payload.final_mods_4096.push(2048),
        "HEATPROOF" if !is_actor && calc.move_type == "FIRE" => payload.attack_mods_4096.push(2048),
        "THICK_FAT" if !is_actor && (calc.move_type == "FIRE" || calc.move_type == "ICE") => {
            payload.attack_mods_4096.push(2048)
        }
        "PRISM_ARMOR" | "FILTER" if !is_actor => {
            if calc.db.effectiveness(calc.move_type, &pokemon.battle_types()) >= 2.0 {
                payload.final_mods_4096.push(3072);
            }
        }
        "FLUFFY" if !is_actor => {
            // Both apply, in this order, so a contact Fire move lands for exactly its usual amount.
            if calc.contact {
                payload.final_mods_4096.push(2048);
            }
            if calc.move_type == "FIRE" {
                payload.final_mods_4096.push(8192);
            }
        }
        "UNAWARE" if !is_actor => payload.ignore_attack_stages = true,

        // -- effects that read which side you are on, not whether you are attacking ---------
        "TABLETS_OF_RUIN" if !is_actor && physical => payload.attack_mods_4096.push(3072),
        "VESSEL_OF_RUIN" if !is_actor && special => payload.attack_mods_4096.push(3072),
        "SWORD_OF_RUIN" if is_actor && physical => payload.defense_mods_4096.push(3072),
        "BEADS_OF_RUIN" if is_actor && special => payload.defense_mods_4096.push(3072),

        // -- field-wide, from whichever side holds it ---------------------------------------
        "FAIRY_AURA" | "DARK_AURA" => {
            let aura = if ability == "FAIRY_AURA" { "FAIRY" } else { "DARK" };
            if calc.move_type == aura {
                payload.power_mods_4096.push(if aura_broken { 3072 } else { 5461 });
            }
        }
        "PUNK_ROCK" if calc.the_move.sound => {
            if is_actor {
                payload.power_mods_4096.push(5325);
            } else {
                payload.final_mods_4096.push(2048);
            }
        }
        "FLOWER_GIFT" if calc.weather == "SUN" || calc.weather == "HARSH_SUN" => {
            if is_actor && physical {
                payload.attack_mods_4096.push(6144);
            }
            if !is_actor && special {
                payload.defense_mods_4096.push(6144);
            }
        }
        _ => {}
    }
}
