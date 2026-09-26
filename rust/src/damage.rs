//! The Gen-9 damage chain, ported from `battle_sim/maths/damage.py`.
//!
//! The sequence *is* the contract. Every `_chain` in the Python is an integer fold with a `+2047`
//! rounding step, and folding two of them in the other order changes the answer by a point — which
//! is enough to turn a survival into a knockout. So this is written in the same order, with the
//! same intermediate truncations, and deliberately not tidied.
//!
//! The `payload` is the Python's: a bag of modifiers the abilities and items fill in before the
//! formula runs. Its fold *positions* are what matter — a modifier applied one step earlier or
//! later changes the answer by a point, which is enough to turn a survival into a knockout — so
//! each list is read exactly where `maths/damage.py` reads it, and in the order the handlers
//! pushed into it.

use crate::battle::{Field, Pokemon, Side};
use crate::data::{Database, Effect, Move};

/// Gen-9 crit odds by stage, from `_CRIT_PROBABILITIES`.
const CRIT_PROBABILITIES: [f64; 4] = [1.0 / 24.0, 1.0 / 8.0, 1.0 / 2.0, 1.0];

pub struct Hit {
    pub amount: i32,
    pub is_crit: bool,
}

impl Hit {
    pub fn nothing() -> Hit {
        Hit { amount: 0, is_crit: false }
    }
}

/// The 4096-denominator fold the whole chain is built from. `+2047` before the shift is the
/// round-half-up the cartridge does; a plain divide loses a point here and there.
fn chain(value: i32, modifier_4096: i64) -> i32 {
    (((value as i64 * modifier_4096) + 2047) / 4096) as i32
}

fn crit_chance(stage: i32) -> f64 {
    CRIT_PROBABILITIES[stage.clamp(0, CRIT_PROBABILITIES.len() as i32 - 1) as usize]
}

/// A stat for damage, with the crit rule: an attacker's negative boosts are ignored on a crit, and
/// a defender's positive ones are.
fn crit_aware(value: i32, stage: i32, is_crit: bool, attacking: bool, ignore_stages: bool) -> i32 {
    // Unaware's flag is read before the crit rule, as `_crit_aware_*_stat` does: the stage is
    // simply gone, so there is nothing left for the crit to clamp.
    let stage = if ignore_stages { 0 } else { stage };
    let stage = if is_crit {
        if attacking {
            stage.max(0)
        } else {
            stage.min(0)
        }
    } else {
        stage
    };
    crate::stats::with_stage(value, stage)
}

fn weather_modifier(move_type: &str, weather: &str) -> i64 {
    match (weather, move_type) {
        ("SUN" | "HARSH_SUN", "FIRE") => 6144,
        ("SUN" | "HARSH_SUN", "WATER") => 2048,
        ("RAIN" | "HEAVY_RAIN", "WATER") => 6144,
        ("RAIN" | "HEAVY_RAIN", "FIRE") => 2048,
        _ => 4096,
    }
}

/// `damage_effect` is the first `DamageEffect` on the move, as the Python takes it.
fn damage_effect(the_move: &Move) -> Option<(&Option<i32>, &str, i32)> {
    the_move.effects.iter().find_map(|e| match e {
        Effect::DamageEffect { power, category, crit_stage, .. } => Some((power, category.as_str(), *crit_stage)),
        _ => None,
    })
}

pub fn is_damaging(the_move: &Move) -> bool {
    damage_effect(the_move).is_some_and(|(power, _, _)| power.unwrap_or(0) > 0)
}

/// The two draws this formula consumes, taken by the caller so the order is visible where the
/// tape is read rather than buried in here.
///
/// Safe to take in advance *only* because `resolve_move` has already established everything the
/// Python checks before its own draws: the move has a fixed, non-zero power, and the matchup is
/// not an immunity. Both draws are therefore certain to be consumed. Push another early return
/// into this function and that stops being true, and the tape silently slips by one.
pub struct Rolls {
    pub crit: f64,
    pub damage: i32,
}

/// The Python's `Payload`, restricted to the keys the damage formula reads.
///
/// Defaulted to "nobody changed anything", which is the same thing as an empty dict on that side.
#[derive(Debug, Default)]
pub struct Payload {
    pub attack_mods_4096: Vec<i64>,
    pub defense_mods_4096: Vec<i64>,
    pub power_mods_4096: Vec<i64>,
    pub pre_screen_mods_4096: Vec<i64>,
    pub final_mods_4096: Vec<i64>,
    pub ignore_attack_stages: bool,
    pub ignore_defense_stages: bool,
    pub ignore_burn: bool,
    pub bypass_screens: bool,
    /// `stab_4096`, which Adaptability raises from the usual 6144.
    pub stab_4096: i64,
    /// Air Lock: the weather still blows, it just stops mattering.
    pub weather_suppressed: bool,
}

impl Payload {
    pub fn new() -> Payload {
        Payload { stab_4096: 6144, ..Default::default() }
    }
}

// Eight arguments, because the Python's `calculate_hit` takes the same eight things and this is
// kept readable against it rather than against a style rule. Bundling them into a context struct
// would make the two harder to diff, which is the only thing keeping them honest.
#[allow(clippy::too_many_arguments)]
pub fn calculate_hit(
    attacker: &Pokemon,
    defender: &Pokemon,
    the_move: &Move,
    field: &Field,
    defender_side: &Side,
    db: &Database,
    rolls: Rolls,
    payload: &Payload,
) -> Hit {
    let Some((power, category, crit_stage)) = damage_effect(the_move) else {
        return Hit::nothing();
    };
    let Some(mut power) = *power else {
        return Hit::nothing(); // variable power: not modelled until the coded moves are ported
    };
    if power == 0 {
        return Hit::nothing();
    }

    let type_multiplier = if the_move.typeless {
        1.0
    } else {
        db.effectiveness(&the_move.move_type, &defender.types)
    };
    if type_multiplier == 0.0 {
        return Hit::nothing();
    }

    // The crit is rolled here and only here, which is what keeps the tape aligned: every early
    // return above happens *before* a draw, exactly as in the Python.
    //
    // Merciless does not raise the stage, it overrules the roll — and the two armors refuse the
    // result however it was reached. So the draw happens either way; only its verdict changes.
    let stage = crit_stage
        + crate::inline::crit_stage_bonus(attacker)
        + if attacker.volatiles.contains_key("FOCUS_ENERGY") { 2 } else { 0 };
    let is_crit = crate::inline::crit_overrides(attacker, defender, rolls.crit < crit_chance(stage));

    let (attack_stat, defense_stat) = if category == "PHYSICAL" {
        ("ATTACK", "DEFENCE")
    } else {
        ("SP_ATTACK", "SP_DEFENCE")
    };
    let mut attack = crit_aware(
        attacker.stat(attack_stat),
        attacker.stage(attack_stat),
        is_crit,
        true,
        payload.ignore_attack_stages,
    );
    for modifier in &payload.attack_mods_4096 {
        attack = chain(attack, *modifier);
    }
    let mut defense = crit_aware(
        defender.stat(defense_stat),
        defender.stage(defense_stat),
        is_crit,
        false,
        payload.ignore_defense_stages,
    );
    for modifier in &payload.defense_mods_4096 {
        defense = chain(defense, *modifier);
    }
    for modifier in &payload.power_mods_4096 {
        power = chain(power, *modifier);
    }

    let mut damage = ((2 * attacker.level) / 5 + 2) * power * attack / defense / 50 + 2;

    let active_weather = if payload.weather_suppressed { "NONE" } else { field.weather.as_str() };
    damage = chain(damage, weather_modifier(&the_move.move_type, active_weather));
    if is_crit {
        // Sniper's crits are half again as hard as anybody else's.
        damage = chain(damage, if attacker.ability == "SNIPER" { 9216 } else { 6144 });
    }

    damage = damage * rolls.damage / 100;

    if attacker.types.iter().flatten().any(|t| t == &the_move.move_type) {
        damage = chain(damage, payload.stab_4096);
    }

    damage = (damage as f64 * type_multiplier) as i32;

    if attacker.status == crate::battle::Status::Burn && category == "PHYSICAL" && !payload.ignore_burn {
        damage = chain(damage, 2048);
    }

    for modifier in &payload.pre_screen_mods_4096 {
        damage = chain(damage, *modifier);
    }

    if !is_crit && !payload.bypass_screens {
        damage = chain(damage, screen_modifier(category, defender_side));
    }

    damage = chain(damage, terrain_modifier(&the_move.move_type, attacker, defender, &field.terrain));
    for modifier in &payload.final_mods_4096 {
        damage = chain(damage, *modifier);
    }

    Hit { amount: damage.max(1), is_crit }
}

/// `_terrain_modifier`. Three terrains strengthen their own type by a third for a grounded user;
/// Misty weakens Dragon against a grounded target instead — which is the point of it.
fn terrain_modifier(move_type: &str, attacker: &Pokemon, defender: &Pokemon, terrain: &str) -> i64 {
    let grounded = crate::field::is_grounded(attacker);
    match (terrain, move_type) {
        ("ELECTRIC", "ELECTRIC") | ("GRASSY", "GRASS") | ("PSYCHIC", "PSYCHIC") if grounded => 5324,
        ("MISTY", "DRAGON") if crate::field::is_grounded(defender) => 2048,
        _ => 4096,
    }
}

fn screen_modifier(category: &str, side: &Side) -> i64 {
    let wanted = if category == "PHYSICAL" { "REFLECT" } else { "LIGHT_SCREEN" };
    if side.screens.contains_key(wanted) || side.screens.contains_key("AURORA_VEIL") {
        2048
    } else {
        4096
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_fold_rounds_the_way_the_cartridge_does() {
        // 4096 is identity; 6144 is x1.5 with the half rounded up.
        // Values taken from the Python `_chain`, not from arithmetic done in my head — the first
        // version of this test asserted 2 for the third case and was simply wrong.
        assert_eq!(chain(100, 4096), 100);
        assert_eq!(chain(100, 6144), 150);
        assert_eq!(chain(1, 6144), 1);
        assert_eq!(chain(7, 2048), 3);
        assert_eq!(chain(333, 3072), 250);
    }

    #[test]
    fn crit_odds_match_the_table() {
        assert_eq!(crit_chance(0), 1.0 / 24.0);
        assert_eq!(crit_chance(3), 1.0);
        assert_eq!(crit_chance(99), 1.0); // clamped
    }

    #[test]
    fn a_crit_ignores_the_attackers_drops_and_the_defenders_boosts() {
        assert_eq!(crit_aware(100, -2, true, true, false), 100);
        assert_eq!(crit_aware(100, -2, false, true, false), 50);
        assert_eq!(crit_aware(100, 2, true, false, false), 100);
        assert_eq!(crit_aware(100, 2, false, false, false), 200);
        // Unaware: the stage is gone before the crit rule ever looks at it.
        assert_eq!(crit_aware(100, 2, false, false, true), 100);
        assert_eq!(crit_aware(100, -2, false, true, true), 100);
    }
}
