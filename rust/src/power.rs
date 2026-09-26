//! The moves whose power is not the number in their data.
//!
//! Fifty of the hundred-and-something moves the Python special-cases by name are here, and they
//! are the mechanical half: a formula that reads the board (Gyro Ball off the speed difference,
//! Eruption off the user's health) or a condition that doubles it (Facade when statused, Brine
//! against a target under half).
//!
//! `effective_power` runs **once per move**, not once per hit, and after the hit count has been
//! rolled. That matters for exactly one move — Magnitude draws its own number — but it is the kind
//! of thing that is invisible until a multi-hit variant appears and then wrong forever.

use crate::battle::{Pokemon, State, Status};
use crate::data::{Database, Move};
use crate::tape::Tape;
use crate::turn::Refusal;

/// Coded moves whose power rule is implemented here. They come off the by-name refusal.
pub const PORTED: [&str; 46] = [
    // `_POWER_FORMULAS`
    "Low Kick",
    "Grass Knot",
    "Heavy Slam",
    "Heat Crash",
    "Electro Ball",
    "Gyro Ball",
    "Stored Power",
    "Power Trip",
    "Last Respects",
    "Rage Fist",
    "Beat Up",
    "Fury Cutter",
    "Water Spout",
    "Eruption",
    "Dragon Energy",
    "Return",
    "Frustration",
    "Flail",
    "Reversal",
    "Crush Grip",
    "Wring Out",
    "Punishment",
    "Magnitude",
    // `_POWER_CONDITIONS`
    "Facade",
    "Hex",
    "Infernal Parade",
    "Barb Barrage",
    "Acrobatics",
    "Avalanche",
    "Revenge",
    "Payback",
    "Assurance",
    "Brine",
    "Venoshock",
    "Wake-Up Slap",
    "Smelling Salts",
    "Rising Voltage",
    "Knock Off",
    "Expanding Force",
    "Psyblade",
    "Hydro Steam",
    // `_SE_BONUS_MOVES`, `_HITS_PHYSICAL_DEFENCE`, and two of `payload_overrides`
    "Electro Drift",
    "Collision Course",
    "Psyshock",
    "Psystrike",
    "Secret Sword",
];

/// `_STAGED_STATS`, in the Python's enum order.
const STAGED: [&str; 7] = ["ATTACK", "DEFENCE", "SP_ATTACK", "SP_DEFENCE", "SPEED", "ACCURACY", "EVASION"];

fn positive_stages(pokemon: &Pokemon) -> i32 {
    STAGED.iter().map(|stat| pokemon.stage(stat).max(0)).sum()
}

/// Rollout, Ice Ball and Pursuit are absent. The first two lock their user into a run, which needs
/// the locked-move volatile; Pursuit reads the *other side's chosen action*, which this engine does
/// not keep past the ordering step.
fn formula(
    name: &str,
    state: &State,
    side: usize,
    speed: impl Fn(usize) -> i32,
    tape: &mut Tape,
) -> Result<Option<i32>, Refusal> {
    let attacker = state.sides[side].active_pokemon();
    let defender = state.sides[1 - side].active_pokemon();
    Ok(Some(match name {
        "Low Kick" | "Grass Knot" => match defender.weight_kg {
            w if w >= 200.0 => 120,
            w if w >= 100.0 => 100,
            w if w >= 50.0 => 80,
            w if w >= 25.0 => 60,
            w if w >= 10.0 => 40,
            _ => 20,
        },
        "Heavy Slam" | "Heat Crash" => match attacker.weight_kg / defender.weight_kg {
            r if r >= 5.0 => 120,
            r if r >= 4.0 => 100,
            r if r >= 3.0 => 80,
            r if r >= 2.0 => 60,
            _ => 40,
        },
        "Electro Ball" => match speed(side) as f64 / std::cmp::max(1, speed(1 - side)) as f64 {
            r if r >= 4.0 => 150,
            r if r >= 3.0 => 120,
            r if r >= 2.0 => 80,
            r if r >= 1.0 => 60,
            _ => 40,
        },
        // The slower the user next to its target, the harder it hits.
        "Gyro Ball" => std::cmp::min(150, 25 * speed(1 - side) / std::cmp::max(1, speed(side)) + 1),
        "Stored Power" | "Power Trip" => 20 + 20 * positive_stages(attacker),
        "Punishment" => std::cmp::min(200, 60 + 20 * positive_stages(defender)),
        "Last Respects" => 50 * (1 + state.sides[side].team.iter().filter(|p| p.fainted()).count() as i32),
        "Rage Fist" => std::cmp::min(350, 50 * (1 + attacker.times_hit)),
        // PS hits once per healthy ally; approximated as one hit carrying the summed power.
        "Beat Up" => {
            let total: i32 = state.sides[side]
                .team
                .iter()
                .filter(|m| !m.fainted() && m.status == Status::None)
                .map(|m| 5 + m.base_attack / 10)
                .sum();
            if total == 0 {
                5
            } else {
                total
            }
        }
        "Fury Cutter" => [40, 80, 160][std::cmp::min(attacker.rolling_hits as usize, 2)],
        "Water Spout" | "Eruption" | "Dragon Energy" => {
            std::cmp::max(1, 150 * attacker.hp / attacker.totals.hp)
        }
        // Both come out at 102: their power is friendship-scaled, friendship is not modelled, and
        // a competitive set is always built at whichever end its move wants.
        "Return" | "Frustration" => 102,
        "Flail" | "Reversal" => match 48 * attacker.hp / std::cmp::max(1, attacker.totals.hp) {
            s if s < 1 => 200,
            s if s < 4 => 150,
            s if s < 9 => 100,
            s if s < 16 => 80,
            s if s < 32 => 40,
            _ => 20,
        },
        "Crush Grip" | "Wring Out" => {
            std::cmp::max(1, 120 * defender.hp / std::cmp::max(1, defender.totals.hp))
        }
        "Magnitude" => match tape.integer(1, 20)? {
            r if r <= 1 => 10,
            r if r <= 3 => 30,
            r if r <= 7 => 50,
            r if r <= 13 => 70,
            r if r <= 17 => 90,
            r if r <= 19 => 110,
            _ => 150,
        },
        _ => return Ok(None),
    }))
}

/// `_POWER_CONDITIONS`: the multiplier and the question that earns it.
fn condition(name: &str, state: &State, side: usize) -> Option<(bool, i32, i32)> {
    let attacker = state.sides[side].active_pokemon();
    let defender = state.sides[1 - side].active_pokemon();
    let grounded = |p: &Pokemon| crate::field::is_grounded(p);
    let met = match name {
        "Facade" => attacker.status != Status::None,
        "Hex" | "Infernal Parade" => defender.status != Status::None,
        "Barb Barrage" | "Venoshock" => matches!(defender.status, Status::Poison | Status::Toxic),
        "Acrobatics" => attacker.item == "NONE",
        // "Hit by the target this turn", which is what `last_hit_taken` records and why it is
        // cleared at the top of every turn.
        "Avalanche" | "Revenge" => attacker.last_hit_taken > 0,
        "Payback" => state.sides[1 - side].acted_this_turn,
        "Assurance" => defender.last_hit_taken > 0,
        "Brine" => defender.hp * 2 <= defender.totals.hp,
        "Wake-Up Slap" => defender.status == Status::Sleep,
        "Smelling Salts" => defender.status == Status::Paralysis,
        "Rising Voltage" => state.field.terrain == "ELECTRIC" && grounded(defender),
        "Knock Off" => return Some((defender.item != "NONE", 3, 2)),
        "Expanding Force" => return Some((state.field.terrain == "PSYCHIC" && grounded(attacker), 3, 2)),
        "Psyblade" => return Some((state.field.terrain == "ELECTRIC" && grounded(attacker), 3, 2)),
        "Hydro Steam" => {
            return Some((matches!(crate::hooks::effective_weather(state).as_str(), "SUN" | "HARSH_SUN"), 3, 2))
        }
        _ => return None,
    };
    Some((met, 2, 1))
}

/// `effective_power`, whole: the formula, then the condition, then the super-effective bonus.
pub fn effective_power(
    the_move: &Move,
    listed: Option<i32>,
    state: &State,
    side: usize,
    db: &Database,
    tape: &mut Tape,
) -> Result<Option<i32>, Refusal> {
    let speed = |which: usize| {
        crate::turn::effective_speed(state.sides[which].active_pokemon(), &state.sides[which], &state.field)
    };
    let mut power = match formula(&the_move.name, state, side, speed, tape)? {
        Some(computed) => computed,
        None => match listed {
            Some(listed) => listed,
            None => return Ok(None),
        },
    };
    if let Some((met, numerator, denominator)) = condition(&the_move.name, state, side) {
        if met {
            power = power * numerator / denominator;
        }
    }
    if matches!(the_move.name.as_str(), "Electro Drift" | "Collision Course") {
        let types = state.sides[1 - side].active_pokemon().types.clone();
        if db.effectiveness(&the_move.move_type, &types) >= 2.0 {
            power = power * 5461 / 4096;
        }
    }
    Ok(Some(power))
}

/// `payload_overrides`, for the handful whose stats are not the ones their category implies.
/// Body Press, Photon Geyser and Foul Play are absent: each needs a different stat *owner*, not
/// just a different stat, and that is a wider change to the formula's inputs.
pub fn defense_stat_override(the_move: &Move) -> Option<&'static str> {
    matches!(the_move.name.as_str(), "Psyshock" | "Psystrike" | "Secret Sword").then_some("DEFENCE")
}

/// The two flag overrides. Facade is the reason its own doubling is worth anything: a burned
/// Pokemon using it does not also take the burn's halving, so the move really is twice as hard
/// rather than exactly as hard as before.
pub fn ignores_burn(the_move: &Move) -> bool {
    the_move.name == "Facade"
}

/// Hydro Steam thrives in the sun instead of wilting in it.
pub fn ignores_weather_drop(the_move: &Move) -> bool {
    the_move.name == "Hydro Steam"
}
