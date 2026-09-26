//! The abilities and items the Python reads inline, at a site, rather than binding to its bus.
//!
//! There are more of these than there are bound ones — 85 abilities and 59 items have live
//! behaviour without ever touching the event bus — and they are mostly one condition in a function
//! that already exists: Levitate is a clause in `is_grounded`, Sniper is a different constant in
//! the crit fold, Chlorophyll is a branch in `effective_speed`. So the code lives at those sites,
//! and this module holds only the shared predicates and the list of what has been done.
//!
//! Magic Guard is the exception worth naming. It is not one site: it suppresses the sandstorm chip,
//! the status chip, entry hazards, move recoil, crash damage and ability chip damage. Porting a
//! subset of those would be a Pokemon that takes a little indirect damage, which is worse than one
//! that takes all of it — so all six are here, and the differential is what says whether that is
//! the whole list.

use crate::battle::{Pokemon, Status};

/// Abilities implemented at their inline sites.
///
/// Levitate is deliberately absent, and it is the cautionary one. Its grounding clause is a single
/// line in `is_grounded`, but it is *also* bound to the bus, where it cancels an incoming Ground
/// move with its own log line — so porting the clause alone would be a Levitate that takes no
/// damage and never says why. Air Balloon, Punching Glove and Choice Scarf are out for the same
/// reason: each does more somewhere else.
pub const PORTED_ABILITIES: [&str; 51] = [
    "BATTLE_ARMOR",
    "BIG_PECKS",
    "CHLOROPHYLL",
    "CLEAR_BODY",
    "COMATOSE",
    "COMPETITIVE",
    "CONTRARY",
    "COMPOUND_EYES",
    "DEFIANT",
    "DRIZZLE",
    "DROUGHT",
    "ELECTRIC_SURGE",
    "FULL_METAL_BODY",
    "GRASSY_SURGE",
    "HUSTLE",
    "HYPER_CUTTER",
    "IMMUNITY",
    "INNER_FOCUS",
    "INSOMNIA",
    "KEEN_EYE",
    "LEAF_GUARD",
    "LIMBER",
    "LIQUID_OOZE",
    "LONG_REACH",
    "MAGIC_GUARD",
    "MAGMA_ARMOR",
    "MERCILESS",
    "MISTY_SURGE",
    "NO_GUARD",
    "OVERCOAT",
    "OWN_TEMPO",
    "PSYCHIC_SURGE",
    "PURIFYING_SALT",
    "QUICK_FEET",
    "ROCK_HEAD",
    "SAND_RUSH",
    "SAND_STREAM",
    "SAND_VEIL",
    "SHELL_ARMOR",
    "SIMPLE",
    "SLUSH_RUSH",
    "SNIPER",
    "SNOW_WARNING",
    "SNOW_CLOAK",
    "SUPER_LUCK",
    "SWIFT_SWIM",
    "TANGLED_FEET",
    "VICTORY_STAR",
    "VITAL_SPIRIT",
    "WATER_VEIL",
    "WHITE_SMOKE",
];

/// Items implemented at their inline sites.
pub const PORTED_ITEMS: [&str; 3] = ["PROTECTIVE_PADS", "SCOPE_LENS", "WIDE_LENS"];

/// `_WEATHER_SPEED_DOUBLERS`: the ability, and the weather it runs in.
pub fn doubles_speed_in(ability: &str, weather: &str) -> bool {
    matches!(
        (ability, weather),
        ("SWIFT_SWIM", "RAIN" | "HEAVY_RAIN")
            | ("CHLOROPHYLL", "SUN" | "HARSH_SUN")
            | ("SAND_RUSH", "SANDSTORM")
            | ("SLUSH_RUSH", "SNOW")
    )
}

/// Magic Guard: no indirect damage at all. Asked at every site that deals some.
pub fn ignores_indirect_damage(pokemon: &Pokemon) -> bool {
    pokemon.ability == "MAGIC_GUARD"
}

/// `_GUARANTEED`: large enough that any nonzero base accuracy clears the final `min(1.0, ..)`.
const GUARANTEED: f64 = 1e6;

/// `_weather_accuracy_multiplier`: what the weather alone does, by move name.
///
/// Folded into the ordinary chain rather than short-circuiting, so a move that both always-hits in
/// this weather and would otherwise have missed still reads as one accuracy roll. Three moves are
/// in these tables and all three are also named elsewhere — Thunder is a semi-invulnerability
/// reacher too, and freeing it on that basis without this cost an afternoon.
pub fn weather_accuracy_multiplier(move_name: &str, weather: &str) -> f64 {
    match (move_name, weather) {
        ("Thunder" | "Hurricane", "RAIN" | "HEAVY_RAIN") => GUARANTEED,
        ("Blizzard", "SNOW") => GUARANTEED,
        ("Thunder" | "Hurricane", "SUN" | "HARSH_SUN") => 0.5,
        _ => 1.0,
    }
}

/// The accuracy multiplier the two sides' abilities and items contribute, folded in the Python's
/// order — which starts with the weather's own contribution.
pub fn accuracy_multiplier(
    move_name: &str,
    attacker: &Pokemon,
    defender: &Pokemon,
    category: &str,
    weather: &str,
) -> f64 {
    let mut multiplier = weather_accuracy_multiplier(move_name, weather);
    if defender.ability == "SAND_VEIL" && weather == "SANDSTORM" {
        multiplier *= 0.8;
    }
    if defender.ability == "SNOW_CLOAK" && weather == "SNOW" {
        multiplier *= 0.8;
    }
    if defender.ability == "TANGLED_FEET" && defender.volatiles.contains_key("CONFUSION") {
        multiplier *= 0.8;
    }
    if attacker.ability == "COMPOUND_EYES" {
        multiplier *= 1.3;
    }
    if attacker.ability == "HUSTLE" && category == "PHYSICAL" {
        multiplier *= 0.8;
    }
    if attacker.ability == "VICTORY_STAR" {
        multiplier *= 1.1;
    }
    if attacker.item == "WIDE_LENS" {
        multiplier *= 1.1;
    }
    multiplier
}

/// `_ability_immune_to_status`: the half of status immunity that announces itself.
pub fn ability_blocks_status(target: &Pokemon, status: Status, weather: &str) -> bool {
    let blocked: &[Status] = match target.ability.as_str() {
        // Komala is permanently asleep and acts anyway, so nothing further can be inflicted. What
        // is not modelled — here or in the Python — is counting as asleep for Rest and Sleep Talk.
        "PURIFYING_SALT" | "COMATOSE" => {
            return status != Status::None;
        }
        "LEAF_GUARD" => {
            return matches!(weather, "SUN" | "HARSH_SUN") && status != Status::None;
        }
        "WATER_BUBBLE" | "THERMAL_EXCHANGE" | "WATER_VEIL" => &[Status::Burn],
        "LIMBER" => &[Status::Paralysis],
        "INSOMNIA" | "VITAL_SPIRIT" => &[Status::Sleep],
        "MAGMA_ARMOR" => &[Status::Freeze],
        "IMMUNITY" => &[Status::Poison, Status::Toxic],
        _ => &[],
    };
    blocked.contains(&status)
}

/// `_VOLATILE_ABILITY_IMMUNITY`: Inner Focus never flinches, Own Tempo never gets confused.
pub fn ability_blocks_volatile(target: &Pokemon, volatile: &str) -> bool {
    matches!((target.ability.as_str(), volatile), ("INNER_FOCUS", "FLINCH") | ("OWN_TEMPO", "CONFUSION"))
}

/// `ABILITY_WEATHER` and `ABILITY_TERRAIN`: what a Pokemon brings with it onto the field.
pub fn weather_from_ability(ability: &str) -> Option<&'static str> {
    Some(match ability {
        "DROUGHT" => "SUN",
        "DRIZZLE" => "RAIN",
        "SAND_STREAM" => "SANDSTORM",
        "SNOW_WARNING" => "SNOW",
        _ => return None,
    })
}

pub fn terrain_from_ability(ability: &str) -> Option<&'static str> {
    Some(match ability {
        "GRASSY_SURGE" => "GRASSY",
        "ELECTRIC_SURGE" => "ELECTRIC",
        "PSYCHIC_SURGE" => "PSYCHIC",
        "MISTY_SURGE" => "MISTY",
        _ => return None,
    })
}

/// Whether the accuracy roll is skipped entirely. No Guard on *either* side is enough.
pub fn never_misses(attacker: &Pokemon, defender: &Pokemon) -> bool {
    attacker.ability == "NO_GUARD" || defender.ability == "NO_GUARD"
}

/// Merciless crits outright against a poisoned target rather than raising the stage, and the two
/// armors refuse a crit however it was earned.
pub fn crit_overrides(attacker: &Pokemon, defender: &Pokemon, rolled: bool) -> bool {
    let merciless = attacker.ability == "MERCILESS" && matches!(defender.status, Status::Poison | Status::Toxic);
    (rolled || merciless) && !matches!(defender.ability.as_str(), "BATTLE_ARMOR" | "SHELL_ARMOR")
}

/// The extra crit stages an item or ability buys. Focus Energy's two are added at the call site,
/// because they come off the volatile rather than off either of these.
pub fn crit_stage_bonus(attacker: &Pokemon) -> i32 {
    i32::from(attacker.item == "SCOPE_LENS") + i32::from(attacker.ability == "SUPER_LUCK")
}
