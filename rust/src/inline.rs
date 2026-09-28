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
pub const PORTED_ABILITIES: [&str; 74] = [
    "BATTLE_ARMOR",
    "BIG_PECKS",
    "CHLOROPHYLL",
    "CLEAR_BODY",
    "COMATOSE",
    "COMPETITIVE",
    "CONTRARY",
    "COMPOUND_EYES",
    "CORROSION",
    "DEFIANT",
    // Primal weather. Mega Rayquaza's own ability has no held item to key it to `formes.rs`'s
    // table (it is move-gated), but it sets weather the exact same way these two do, so it rides
    // in on this same array rather than needing one of its own.
    "DELTA_STREAM",
    "DESOLATE_LAND",
    "DRIZZLE",
    "DROUGHT",
    "ELECTRIC_SURGE",
    "FULL_METAL_BODY",
    "GALE_WINGS",
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
    "LIQUID_VOICE",
    "LONG_REACH",
    "MAGIC_GUARD",
    "MAGMA_ARMOR",
    "MERCILESS",
    "MINDS_EYE",
    "MISTY_SURGE",
    "MYCELIUM_MIGHT",
    "NO_GUARD",
    "OVERCOAT",
    "OWN_TEMPO",
    "POISON_PUPPETEER",
    "PRANKSTER",
    "PRESSURE",
    "PRIMORDIAL_SEA",
    "PSYCHIC_SURGE",
    "PURIFYING_SALT",
    "QUICK_DRAW",
    "QUICK_FEET",
    "ROCK_HEAD",
    "SAND_RUSH",
    "SAND_STREAM",
    "SAND_VEIL",
    "SCRAPPY",
    "SERENE_GRACE",
    "SHEER_FORCE",
    "SHELL_ARMOR",
    "SHIELD_DUST",
    "SIMPLE",
    "SKILL_LINK",
    "SLUSH_RUSH",
    "SNIPER",
    "SNOW_WARNING",
    "SNOW_CLOAK",
    "STEADFAST",
    "STURDY",
    "SUPER_LUCK",
    "SURGE_SURFER",
    "SWIFT_SWIM",
    "SYNCHRONIZE",
    "TANGLED_FEET",
    "TRIAGE",
    "UNBURDEN",
    "VICTORY_STAR",
    "VITAL_SPIRIT",
    "WATER_VEIL",
    "WHITE_SMOKE",
];

/// Items implemented at their inline sites.
pub const PORTED_ITEMS: [&str; 33] = [
    "ADRENALINE_ORB",
    "BURN_DRIVE",
    "CHESTO_BERRY",
    "CHILL_DRIVE",
    "CHOICE_BAND",
    "CHOICE_SCARF",
    "CHOICE_SPECS",
    "CLEAR_AMULET",
    "COVERT_CLOAK",
    "CUSTAP_BERRY",
    "DAMP_ROCK",
    "DOUSE_DRIVE",
    "EJECT_PACK",
    "FOCUS_SASH",
    "HEAT_ROCK",
    "HEAVY_DUTY_BOOTS",
    "ICY_ROCK",
    "LEPPA_BERRY",
    "LIGHT_CLAY",
    "LOADED_DICE",
    "LUM_BERRY",
    "MENTAL_HERB",
    "MIRROR_HERB",
    "POWER_HERB",
    "PROTECTIVE_PADS",
    "PUNCHING_GLOVE",
    "QUICK_CLAW",
    "SCOPE_LENS",
    "SHOCK_DRIVE",
    "SMOOTH_ROCK",
    "TERRAIN_EXTENDER",
    "WHITE_HERB",
    "WIDE_LENS",
];

/// `_priority_modifiers`: Prankster bumps a status move, Gale Wings a full-HP Flying move, Triage
/// a healing move — each by a fixed amount, folded together the way the Python adds them rather
/// than short-circuiting (a Pokemon can only hold one ability, so at most one of these ever fires,
/// but the addition is what the source does and costs nothing to keep).
pub fn priority_bonus(ability: &str, move_type: &str, category: &str, healing: bool, at_full_hp: bool) -> i32 {
    let mut bonus = 0;
    if ability == "PRANKSTER" && category == "STATUS" {
        bonus += 1;
    }
    if ability == "GALE_WINGS" && move_type == "FLYING" && at_full_hp {
        bonus += 1;
    }
    if ability == "TRIAGE" && healing {
        bonus += 3;
    }
    bonus
}

/// `_tuned_status_secondary`: a *secondary* status effect (one riding a damaging move, not a pure
/// status move) is blocked outright by Shield Dust or Covert Cloak on the defender — no probability
/// draw at all — unless it targets the move's own user, or doubled by Serene Grace on the attacker.
/// `None` means skip the draw entirely; `Some` carries the probability actually to be drawn. A
/// non-secondary effect is returned unchanged either way. Sheer Force's own nullification is checked
/// first, ahead of the block.
pub fn tune_status_secondary(
    is_secondary: bool,
    probability: f64,
    aimed_at_self: bool,
    attacker_ability: &str,
    defender_ability: &str,
    defender_item: &str,
) -> Option<f64> {
    if !is_secondary {
        return Some(probability);
    }
    // Sheer Force trades the whole secondary for its own power boost — checked before
    // `_blocks_secondaries`'s block, matching the Python's own order, though both paths land on
    // the same `None`.
    if attacker_ability == "SHEER_FORCE" {
        return None;
    }
    // `_blocks_secondaries`: Shield Dust or Covert Cloak, either one refusing the same way.
    if !aimed_at_self && (defender_ability == "SHIELD_DUST" || defender_item == "COVERT_CLOAK") {
        return None;
    }
    if attacker_ability == "SERENE_GRACE" {
        return Some((probability * 2.0).min(1.0));
    }
    Some(probability)
}

/// `_tuned_stage_secondary`: the same idea for a secondary stat-stage effect, which carries its
/// own `target` ("SELF" or "TARGET") rather than the status effect's `to_self` flag.
pub fn tune_stage_secondary(
    is_secondary: bool,
    probability: f64,
    target: &str,
    attacker_ability: &str,
    defender_ability: &str,
    defender_item: &str,
) -> Option<f64> {
    if !is_secondary {
        return Some(probability);
    }
    if attacker_ability == "SHEER_FORCE" {
        return None;
    }
    if target == "TARGET" && (defender_ability == "SHIELD_DUST" || defender_item == "COVERT_CLOAK") {
        return None;
    }
    if attacker_ability == "SERENE_GRACE" {
        return Some((probability * 2.0).min(1.0));
    }
    Some(probability)
}

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
///
/// Desolate Land / Primordial Sea / Delta Stream are read here exactly like the ordinary four —
/// this engine, like the Python it mirrors, does not make their weather immune to being overwritten
/// by an ordinary weather move afterward. That is a real cartridge behaviour missing from the
/// Python reference itself (its own comments say so), and porting it here would be a divergence
/// from the engine this is checked against, not a fix.
pub fn weather_from_ability(ability: &str) -> Option<&'static str> {
    Some(match ability {
        "DROUGHT" => "SUN",
        "DRIZZLE" => "RAIN",
        "SAND_STREAM" => "SANDSTORM",
        "SNOW_WARNING" => "SNOW",
        "DESOLATE_LAND" => "HARSH_SUN",
        "PRIMORDIAL_SEA" => "HEAVY_RAIN",
        "DELTA_STREAM" => "STRONG_WINDS",
        _ => return None,
    })
}

/// `WEATHER_ROCKS`: the item that stretches a weather's ordinary 5-turn duration to 8, whether the
/// weather was set by a move or by one of the four abilities above. Primal weather (`HARSH_SUN`,
/// `HEAVY_RAIN`) is not in this table in the Python either — a rock cannot touch it.
pub fn rock_for_weather(weather: &str) -> Option<&'static str> {
    Some(match weather {
        "RAIN" => "DAMP_ROCK",
        "SUN" => "HEAT_ROCK",
        "SANDSTORM" => "SMOOTH_ROCK",
        "SNOW" => "ICY_ROCK",
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
