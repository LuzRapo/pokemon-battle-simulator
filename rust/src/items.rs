//! Held items, as they affect the damage formula.
//!
//! Same contract as `abilities.rs`, and the same discipline: only items whose whole behaviour is
//! `ON_DAMAGE_CALC`, and which the Python does not also read inline somewhere outside
//! `mechanics/items.py`. Choice Band and Choice Specs are absent for that reason — they also lock
//! their holder into a move, which lives in the move flow. The plates are a near miss — they also double as Judgment's type selector — but that half
//! already lives in `power::plate_type`, a query the Judgment case reads independently, so their
//! ordinary 1.2x boost (`_bind_type_boost`, the same binder Black Glasses/Miracle Seed/etc. use)
//! joins `TYPE_BOOSTERS` below like any other. Only six of the seventeen plates actually have that
//! binder registered in `mechanics/items.py` today (Iron/Earth/Spooky/Pixie/Splash/Stone) — the
//! other eleven are real, live items (Multitype/RKS reads them for the type-tracking half in
//! `hooks::sync_type_from_item`) that this reference engine simply has not wired up a damage boost
//! for yet. A generic `_bind_type_boost(item, type)` one-liner per plate would be trivial to add on
//! the Python side, but that is a Python change, not a Rust-port one — ported here exactly as
//! Python currently stands (a real divergence was caught by a sweep exactly this way: Rust boosted
//! all seventeen, Python only six, and `compare()` reported the extra 1.2x as a plain wrong number
//! with nothing in the trace to explain it).
//!
//! Items fire after *every* ability on the field, not just their holder's: the bus sorts by
//! priority first, and `EventPriority.ITEM` is half `ABILITY`. Within the items it is registration
//! order, which `apply_damage_calc` walks.

use crate::battle::Pokemon;
use crate::damage::Payload;
use crate::log::{Event, Log};

/// Every item implemented here.
pub const PORTED: [&str; 29] = [
    "ADAMANT_CRYSTAL",
    "ASSAULT_VEST",
    "BLACK_GLASSES",
    "CHOPLE_BERRY",
    "COLBUR_BERRY",
    "CORNERSTONE_MASK",
    "EARTH_PLATE",
    "EVIOLITE",
    "EXPERT_BELT",
    "GRISEOUS_CORE",
    "GRISEOUS_ORB",
    "IRON_PLATE",
    "LUSTROUS_GLOBE",
    "MEOWFREDS_MONOCLE",
    "METAL_COAT",
    "MIRACLE_SEED",
    "MUSCLE_BAND",
    "MYSTIC_WATER",
    "NEVER_MELT_ICE",
    "PIXIE_PLATE",
    "SHUCA_BERRY",
    "SILK_SCARF",
    "SILVER_POWDER",
    "SOUL_DEW",
    "SPLASH_PLATE",
    "SPOOKY_PLATE",
    "STONE_PLATE",
    "WELLSPRING_MASK",
    "WISE_GLASSES",
];

/// The eleven plates with no damage-boost binder in Python yet — Multitype/RKS's own type-tracking
/// in `hooks::sync_type_from_item` is their entire effect on this engine today, same shape as
/// `hooks::PORTED_RKS_MEMORIES`.
pub const PORTED_TYPE_ONLY_PLATES: [&str; 11] = [
    "DRACO_PLATE",
    "DREAD_PLATE",
    "FIST_PLATE",
    "FLAME_PLATE",
    "ICICLE_PLATE",
    "INSECT_PLATE",
    "MEADOW_PLATE",
    "MIND_PLATE",
    "SKY_PLATE",
    "TOXIC_PLATE",
    "ZAP_PLATE",
];

/// `MONOCLE_POWER_CAP` / `MONOCLE_MOD_4096`: Technician in an item, and not from the games.
const MONOCLE_POWER_CAP: i32 = 60;
const MONOCLE_MOD_4096: i64 = 6144;

/// The 1.2x type boosters: item, and the type it boosts. Of the seventeen plates
/// (`_MULTITYPE_PLATES` in the Python, which `hooks::sync_type_from_item` also reads for
/// Multitype's own type-tracking), only these six join the six standalone boosters here — see
/// `PORTED_TYPE_ONLY_PLATES` for why the other eleven do not.
const TYPE_BOOSTERS: [(&str, &str); 13] = [
    ("BLACK_GLASSES", "DARK"),
    ("EARTH_PLATE", "GROUND"),
    ("IRON_PLATE", "STEEL"),
    ("METAL_COAT", "STEEL"),
    ("MIRACLE_SEED", "GRASS"),
    ("MYSTIC_WATER", "WATER"),
    ("NEVER_MELT_ICE", "ICE"),
    ("PIXIE_PLATE", "FAIRY"),
    ("SILK_SCARF", "NORMAL"),
    ("SILVER_POWDER", "BUG"),
    ("SPLASH_PLATE", "WATER"),
    ("SPOOKY_PLATE", "GHOST"),
    ("STONE_PLATE", "ROCK"),
];

/// Soul Dew and the Origin orbs: item, the line that may hold it, and the two types it boosts. The
/// Griseous Orb and Core are one item under its Gen 7 and Gen 9 names, registered twice in the
/// Python for the same reason.
const LEGEND_ORBS: [(&str, [&str; 2], [&str; 2]); 5] = [
    ("SOUL_DEW", ["Latios", "Latias"], ["PSYCHIC", "DRAGON"]),
    ("ADAMANT_CRYSTAL", ["Dialga", "Dialga-Origin"], ["STEEL", "DRAGON"]),
    ("LUSTROUS_GLOBE", ["Palkia", "Palkia-Origin"], ["WATER", "DRAGON"]),
    ("GRISEOUS_CORE", ["Giratina", "Giratina-Origin"], ["GHOST", "DRAGON"]),
    ("GRISEOUS_ORB", ["Giratina", "Giratina-Origin"], ["GHOST", "DRAGON"]),
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
        if item == "EXPERT_BELT" && calc.db.effectiveness(calc.move_type, &calc.defender.battle_types()) >= 2.0 {
            payload.final_mods_4096.push(4915);
        }
        // Punching Glove's other half: its contact-negation already lives at `makes_contact`'s own
        // inline site. 1.1x on `power_mods_4096`, unconditional of category since every punching
        // move is physical anyway.
        if item == "PUNCHING_GLOVE" && calc.the_move.punching {
            payload.power_mods_4096.push(4506);
        }
        // Choice Band / Choice Specs: 1.5x, on `final_mods_4096` like every other flat item boost
        // here. Not in `PORTED` above, nor Choice Scarf beside them — all three also lock their
        // holder into a move, which lives in the move flow (`turn::resolve_move`), not here.
        if (item == "CHOICE_BAND" && physical) || (item == "CHOICE_SPECS" && special) {
            payload.final_mods_4096.push(6144);
        }
        // Life Orb: ~1.3x on every hit, unconditionally — its own `ON_ACTION_RESOLVE` recoil chip
        // is `hooks::life_orb_recoil`, called once per move from `turn::apply_damage`.
        if item == "LIFE_ORB" {
            payload.final_mods_4096.push(5324);
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
        // A hit a substitute is about to take is not one this berry reacts to — the Python's own
        // handler asks `payload["behind_substitute"]` before anything else.
        if !calc.behind_substitute
            && calc.move_type == *weakened
            && calc.db.effectiveness(weakened, &pokemon.battle_types()) >= 2.0
        {
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
