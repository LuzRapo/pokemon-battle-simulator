//! Stat calculation, ported line for line from `battle_sim/maths/stats.py`.
//!
//! Every `floor` here is deliberate and in the same place as the Python. The formula truncates
//! three separate times, and moving any one of them changes a handful of stats by a point —
//! which changes a damage roll, which changes whether something survives, which changes the
//! battle. This is the cheapest possible place for the two engines to part company, so it is
//! written to be boringly literal rather than tidy.

use crate::data::{BaseStats, NatureEffect};

/// Shedinja: 1 HP, whatever the level or the investment says.
const SHEDINJA_BASE_HP: i32 = 1;
/// The stage table's base, from `StageBases.STATS`.
const STAGE_BASE: i32 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StatTotals {
    pub hp: i32,
    pub attack: i32,
    pub defence: i32,
    pub sp_attack: i32,
    pub sp_defence: i32,
    pub speed: i32,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Spread {
    pub hp: i32,
    pub attack: i32,
    pub defence: i32,
    pub sp_attack: i32,
    pub sp_defence: i32,
    pub speed: i32,
}

pub fn total_hp(base: i32, iv: i32, ev: i32, level: i32) -> i32 {
    if base == SHEDINJA_BASE_HP {
        return 1;
    }
    ((2 * base + iv + ev / 4) * level) / 100 + level + 10
}

pub fn total_stat(base: i32, iv: i32, ev: i32, level: i32, multiplier: f64) -> i32 {
    let raw = ((2 * base + iv + ev / 4) * level) / 100 + 5;
    (raw as f64 * multiplier).floor() as i32
}

/// 1.1 on the raised stat, 0.9 on the lowered one, and 1.0 for a neutral nature — where "neutral"
/// means the two name the same stat, which is how the Python encodes Hardy and friends.
pub fn nature_multiplier(nature: &NatureEffect, stat: &str) -> f64 {
    if nature.up == nature.down {
        return 1.0;
    }
    if nature.up == stat {
        1.1
    } else if nature.down == stat {
        0.9
    } else {
        1.0
    }
}

pub fn totals(base: &BaseStats, ivs: &Spread, evs: &Spread, level: i32, nature: &NatureEffect) -> StatTotals {
    let of = |stat: &str, b: i32, iv: i32, ev: i32| total_stat(b, iv, ev, level, nature_multiplier(nature, stat));
    StatTotals {
        hp: total_hp(base.hp, ivs.hp, evs.hp, level),
        attack: of("ATTACK", base.attack, ivs.attack, evs.attack),
        defence: of("DEFENCE", base.defence, ivs.defence, evs.defence),
        sp_attack: of("SP_ATTACK", base.sp_attack, ivs.sp_attack, evs.sp_attack),
        sp_defence: of("SP_DEFENCE", base.sp_defence, ivs.sp_defence, evs.sp_defence),
        speed: of("SPEED", base.speed, ivs.speed, evs.speed),
    }
}

/// A stat after its stage multiplier. Integer division, floored, never below 1 — matching
/// `apply_stage_multiplier`, which uses `//` rather than a float divide for exactly this reason.
pub fn with_stage(value: i32, stage: i32) -> i32 {
    let (numerator, denominator) = if stage >= 0 {
        (STAGE_BASE + stage, STAGE_BASE)
    } else {
        (STAGE_BASE, STAGE_BASE - stage)
    };
    std::cmp::max(1, value * numerator / denominator)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn neutral() -> NatureEffect {
        NatureEffect { up: "ATTACK".into(), down: "ATTACK".into() }
    }

    #[test]
    fn shedinja_always_has_one_hit_point() {
        assert_eq!(total_hp(1, 31, 252, 100), 1);
        assert_eq!(total_hp(1, 0, 0, 50), 1);
    }

    #[test]
    fn a_neutral_nature_changes_nothing() {
        assert_eq!(nature_multiplier(&neutral(), "ATTACK"), 1.0);
    }

    #[test]
    fn a_nature_raises_one_stat_and_lowers_another() {
        let jolly = NatureEffect { up: "SPEED".into(), down: "SP_ATTACK".into() };
        assert_eq!(nature_multiplier(&jolly, "SPEED"), 1.1);
        assert_eq!(nature_multiplier(&jolly, "SP_ATTACK"), 0.9);
        assert_eq!(nature_multiplier(&jolly, "ATTACK"), 1.0);
    }

    #[test]
    fn stages_move_a_stat_the_way_the_table_says() {
        assert_eq!(with_stage(100, 0), 100);
        assert_eq!(with_stage(100, 1), 150);
        assert_eq!(with_stage(100, 2), 200);
        assert_eq!(with_stage(100, -1), 66); // floored, not rounded
        assert_eq!(with_stage(100, -6), 25);
        assert_eq!(with_stage(1, -6), 1); // never below one
    }
}
