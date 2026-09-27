//! Held items, as they affect the damage formula.
//!
//! Same contract as `abilities.rs`, and the same discipline: only items whose whole behaviour is
//! `ON_DAMAGE_CALC`, and which the Python does not also read inline somewhere outside
//! `mechanics/items.py`. Choice Band and Choice Specs are absent for that reason — they also lock
//! their holder into a move, which lives in the move flow — as are the plates, which double as
//! Judgment's type selector, and the Griseous Orb, which forces a forme.
//!
//! Items fire after *every* ability on the field, not just their holder's: the bus sorts by
//! priority first, and `EventPriority.ITEM` is half `ABILITY`. Within the items it is registration
//! order, which `apply_damage_calc` walks.

use crate::battle::Pokemon;
use crate::damage::Payload;
use crate::log::{Event, Log};

/// Every item implemented here.
pub const PORTED: [&str; 21] = [
    "ADAMANT_CRYSTAL",
    "ASSAULT_VEST",
    "BLACK_GLASSES",
    "CHOPLE_BERRY",
    "COLBUR_BERRY",
    "CORNERSTONE_MASK",
    "EVIOLITE",
    "EXPERT_BELT",
    "LUSTROUS_GLOBE",
    "MEOWFREDS_MONOCLE",
    "METAL_COAT",
    "MIRACLE_SEED",
    "MUSCLE_BAND",
    "MYSTIC_WATER",
    "NEVER_MELT_ICE",
    "SHUCA_BERRY",
    "SILK_SCARF",
    "SILVER_POWDER",
    "SOUL_DEW",
    "WELLSPRING_MASK",
    "WISE_GLASSES",
];

/// `MONOCLE_POWER_CAP` / `MONOCLE_MOD_4096`: Technician in an item, and not from the games.
const MONOCLE_POWER_CAP: i32 = 60;
const MONOCLE_MOD_4096: i64 = 6144;

/// The 1.2x type boosters: item, and the type it boosts.
const TYPE_BOOSTERS: [(&str, &str); 7] = [
    ("BLACK_GLASSES", "DARK"),
    ("METAL_COAT", "STEEL"),
    ("MIRACLE_SEED", "GRASS"),
    ("MYSTIC_WATER", "WATER"),
    ("NEVER_MELT_ICE", "ICE"),
    ("SILK_SCARF", "NORMAL"),
    ("SILVER_POWDER", "BUG"),
];

/// Soul Dew and the Origin orbs: item, the line that may hold it, and the two types it boosts.
const LEGEND_ORBS: [(&str, [&str; 2], [&str; 2]); 3] = [
    ("SOUL_DEW", ["Latios", "Latias"], ["PSYCHIC", "DRAGON"]),
    ("ADAMANT_CRYSTAL", ["Dialga", "Dialga-Origin"], ["STEEL", "DRAGON"]),
    ("LUSTROUS_GLOBE", ["Palkia", "Palkia-Origin"], ["WATER", "DRAGON"]),
];

/// The Ogerpon masks: item, and the forme it belongs to.
const OGERPON_MASKS: [(&str, &str); 2] =
    [("WELLSPRING_MASK", "Ogerpon-Wellspring"), ("CORNERSTONE_MASK", "Ogerpon-Cornerstone")];

/// The pinch berries that halve one super-effective hit: item, and the type it answers.
const RESIST_BERRIES: [(&str, &str); 3] =
    [("CHOPLE_BERRY", "FIGHTING"), ("SHUCA_BERRY", "GROUND"), ("COLBUR_BERRY", "DARK")];

/// One item's contribution. Unlike an ability, a resist berry *changes* its holder — it is eaten —
/// so this reports back what needs consuming rather than doing it behind the caller's back.
pub struct Consumed {
    pub side: usize,
    pub item: String,
}

pub fn on_damage_calc(
    pokemon: &Pokemon,
    side: usize,
    is_actor: bool,
    calc: &crate::abilities::Calc,
    payload: &mut Payload,
) -> Option<Consumed> {
    let item = pokemon.item.as_str();
    let physical = calc.category == "PHYSICAL";
    let special = calc.category == "SPECIAL";

    if is_actor {
        // Muscle Band and Wise Glasses share Choice Band's binder in the Python, and therefore its
        // 1.5x rather than the games' 1.1x. Ported as written: this engine's job is to agree with
        // that one, not to correct it.
        if (item == "MUSCLE_BAND" && physical) || (item == "WISE_GLASSES" && special) {
            payload.final_mods_4096.push(6144);
        }
        if let Some((_, boosted)) = TYPE_BOOSTERS.iter().find(|(name, _)| *name == item) {
            if calc.move_type == *boosted {
                payload.final_mods_4096.push(4915);
            }
        }
        if let Some((_, bearers, types)) = LEGEND_ORBS.iter().find(|(name, _, _)| *name == item) {
            if bearers.contains(&pokemon.species_name.as_str()) && types.contains(&calc.move_type) {
                payload.final_mods_4096.push(4915);
            }
        }
        if let Some((_, bearer)) = OGERPON_MASKS.iter().find(|(name, _)| *name == item) {
            if pokemon.species_name == *bearer {
                payload.power_mods_4096.push(4915);
            }
        }
        if item == "MEOWFREDS_MONOCLE" {
            if let Some(power) = base_power(calc) {
                if power <= MONOCLE_POWER_CAP {
                    payload.power_mods_4096.push(MONOCLE_MOD_4096);
                }
            }
        }
        if item == "EXPERT_BELT" && calc.db.effectiveness(calc.move_type, &calc.defender.types) >= 2.0 {
            payload.final_mods_4096.push(4915);
        }
        return None;
    }

    if item == "ASSAULT_VEST" && special {
        payload.defense_mods_4096.push(6144);
    }
    if item == "EVIOLITE" && !calc.db.species_named(&pokemon.species_name).is_some_and(|s| s.fully_evolved) {
        payload.defense_mods_4096.push(6144);
    }
    if let Some((_, weakened)) = RESIST_BERRIES.iter().find(|(name, _)| *name == item) {
        // `behind_substitute` gates this in the Python; substitutes are still refused, so there is
        // no state here that could be true. It belongs in this condition when they land.
        if calc.move_type == *weakened && calc.db.effectiveness(weakened, &pokemon.types) >= 2.0 {
            payload.final_mods_4096.push(2048);
            return Some(Consumed { side, item: item.to_string() });
        }
    }
    None
}

/// The listed power of the move's first damage effect, before any modifier.
fn base_power(calc: &crate::abilities::Calc) -> Option<i32> {
    calc.the_move.effects.iter().find_map(|e| match e {
        crate::data::Effect::DamageEffect { power, .. } => *power,
        _ => None,
    })
}

/// Eat the berry: `Pokemon.consume_item`, plus the entry the Python logs alongside it.
pub fn consume(pokemon: &mut Pokemon, side: usize, item: &str, log: &mut Log) {
    pokemon.last_consumed_item = pokemon.item.clone();
    pokemon.item = "NONE".to_string();
    pokemon.item_consumed = true;
    log.push(Event::BerryWeakened {
        side: side as i32,
        pokemon: pokemon.nickname.clone(),
        item: item.to_string(),
    });
}
