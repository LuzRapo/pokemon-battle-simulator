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
pub const PORTED_ABILITIES: [&str; 22] = [
    "BATTLE_ARMOR",
    "CHLOROPHYLL",
    "COMPOUND_EYES",
    "HUSTLE",
    "LIQUID_OOZE",
    "LONG_REACH",
    "MAGIC_GUARD",
    "MERCILESS",
    "NO_GUARD",
    "OVERCOAT",
    "QUICK_FEET",
    "ROCK_HEAD",
    "SAND_RUSH",
    "SAND_VEIL",
    "SHELL_ARMOR",
    "SLUSH_RUSH",
    "SNIPER",
    "SNOW_CLOAK",
    "SUPER_LUCK",
    "SWIFT_SWIM",
    "TANGLED_FEET",
    "VICTORY_STAR",
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

/// The accuracy multiplier the two sides' abilities and items contribute, folded in the Python's
/// order. Weather's own contribution is absent: `_weather_accuracy_multiplier` is keyed by move
/// name — Thunder, Hurricane, Blizzard — and all three are refused as special-cased anyway.
pub fn accuracy_multiplier(attacker: &Pokemon, defender: &Pokemon, category: &str, weather: &str) -> f64 {
    let mut multiplier = 1.0;
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

/// The extra crit stages an item or ability buys. Focus Energy's two are a volatile, still unported.
pub fn crit_stage_bonus(attacker: &Pokemon) -> i32 {
    i32::from(attacker.item == "SCOPE_LENS") + i32::from(attacker.ability == "SUPER_LUCK")
}
