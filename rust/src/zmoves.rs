//! Z-moves: once per battle, a held Z-Crystal upgrades one of the holder's moves.
//!
//! Mirrors `battle_sim/zmoves.py` exactly, reading the same two facts Python's own export now
//! writes out (`Database::z_moves`, the same `Move` shape `moves.json` already uses): a generic
//! type crystal's template carries placeholder power (1) and a `category` that ignores whichever
//! base move triggers it, so both are taken from the base move instead — a signature crystal's
//! template already carries real data, and only its accuracy is forced to never miss.
//!
//! Two kinds exist, and only one is modelled here (matching Python): status moves get a bonus
//! effect instead of a Z-attack in the real games, and that half is not modelled on either engine.

use crate::data::{Database, Effect, Move};

const PLACEHOLDER_POWER: i32 = 1;

/// `_SIGNATURE_BASES`: which base move each signature crystal upgrades. Only the pairing is
/// hand-written — the Z-move itself, with its real power and type, comes from the vendored data
/// like everything else. Kept in exact sync with the Python's own dict.
const SIGNATURE_BASES: [(&str, &str); 15] = [
    ("aloraichiumz", "Thunderbolt"),
    ("decidiumz", "Spirit Shackle"),
    ("inciniumz", "Darkest Lariat"),
    ("kommoniumz", "Clanging Scales"),
    ("lunaliumz", "Moongeist Beam"),
    ("lycaniumz", "Stone Edge"),
    ("marshadiumz", "Spectral Thief"),
    ("mewniumz", "Psychic"),
    ("mimikiumz", "Play Rough"),
    ("pikaniumz", "Volt Tackle"),
    ("pikashuniumz", "Thunderbolt"),
    ("primariumz", "Sparkling Aria"),
    ("snorliumz", "Giga Impact"),
    ("solganiumz", "Sunsteel Strike"),
    ("ultranecroziumz", "Photon Geyser"),
];

/// `_Z_POWER_TABLE`: Gen 7's fixed conversion, the base move's power decides the Z-move's, in bands.
const Z_POWER_TABLE: [(i32, i32); 9] =
    [(55, 100), (65, 120), (75, 140), (85, 160), (95, 175), (100, 180), (110, 185), (125, 190), (130, 195)];
const MAX_Z_POWER: i32 = 200;

fn z_power(base_power: i32) -> i32 {
    for (threshold, power) in Z_POWER_TABLE {
        if base_power <= threshold {
            return power;
        }
    }
    MAX_Z_POWER
}

fn damage_effect(m: &Move) -> Option<&Effect> {
    m.effects.iter().find(|e| matches!(e, Effect::DamageEffect { .. }))
}

/// This crystal's Z-move template, but only if it is a *generic* type crystal. Generic-ness is
/// decided by the crystal's own entry carrying placeholder power, never by its type — signature
/// crystals share types with generic ones (Ghostium/Decidium/Mimikium/Marshadium Z are all Ghost),
/// so keying on type would hand a signature crystal the generic move instead of its own.
fn generic_template<'a>(item: &str, db: &'a Database) -> Option<&'a Move> {
    let z_move = db.z_moves.get(&Database::normalize_id(item))?;
    match damage_effect(z_move) {
        Some(Effect::DamageEffect { power: Some(power), .. }) if *power == PLACEHOLDER_POWER => Some(z_move),
        _ => None,
    }
}

/// The type a generic Z-Crystal upgrades, or `None` if it is not a generic crystal.
pub fn crystal_type(item: &str, db: &Database) -> Option<String> {
    generic_template(item, db).map(|m| m.move_type.clone())
}

/// This crystal's signature Z-move, if `base` is the one move it upgrades.
fn signature_move(item: &str, base_name: &str, db: &Database) -> Option<Move> {
    let key = Database::normalize_id(item);
    let wants = SIGNATURE_BASES.iter().find(|(crystal, _)| *crystal == key)?.1;
    if wants != base_name {
        return None;
    }
    let mut z_move = db.z_moves.get(&key)?.clone();
    z_move.accuracy_probability = None; // Z-moves never miss
    Some(z_move)
}

/// The Z-move this crystal makes of this move, or `None` if the pairing does nothing.
pub fn z_move_for(item: &str, base: &Move, db: &Database) -> Option<Move> {
    if let Some(signature) = signature_move(item, &base.name, db) {
        return Some(signature);
    }
    let template = generic_template(item, db)?;
    if base.move_type != template.move_type {
        return None;
    }
    let (base_power, base_category, base_contact) = match damage_effect(base) {
        // Status moves get a bonus effect instead of a Z-attack: not modelled.
        Some(Effect::DamageEffect { power: Some(power), category, contact, .. }) => {
            (*power, category.clone(), *contact)
        }
        _ => return None,
    };
    let template_effect =
        damage_effect(template).expect("generic_template only returns moves with a DamageEffect");
    let Effect::DamageEffect { crit_stage, drain_percent, recoil_percent, multi_hit, struggle_recoil, .. } =
        template_effect
    else {
        unreachable!("just matched DamageEffect")
    };
    let effect = Effect::DamageEffect {
        power: Some(z_power(base_power)),
        category: base_category.clone(),
        crit_stage: *crit_stage,
        contact: base_contact,
        drain_percent: *drain_percent,
        recoil_percent: *recoil_percent,
        multi_hit: multi_hit.clone(),
        struggle_recoil: *struggle_recoil,
    };
    let mut upgraded = template.clone();
    upgraded.category = base_category;
    upgraded.effects = vec![effect];
    upgraded.accuracy_probability = None; // Z-moves never miss
    Some(upgraded)
}
