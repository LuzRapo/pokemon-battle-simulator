//! Replay a recorded scenario through this engine and print the trace, for diffing against the
//! Python's.
//!
//!     replay <scenario.json> <data-dir>
//!
//! Exit 0 with a trace on stdout; exit 2 with a reason on stderr when the scenario needs something
//! this engine has not learned yet; exit 3 when the recorded randomness shows the two engines have
//! already taken different paths.
//!
//! 2 and 3 are different on purpose. A 2 is this engine declining to guess, which is the only
//! honest thing it can do while most of the game is unported, and the harness may skip it. A 3 is a
//! real disagreement and must fail the run.

use pokemon_engine::battle::{Pokemon, Side, Spec, State};
use pokemon_engine::data::Database;
use pokemon_engine::digest;
use pokemon_engine::tape::{Draw, Tape};
use pokemon_engine::turn::{parse_action, step, unsupported_pokemon};
use serde::Deserialize;
use serde_json::json;
use std::path::Path;

#[derive(Debug, Deserialize)]
struct Scenario {
    version: u32,
    teams: Vec<Vec<Spec>>,
    actions: Vec<Vec<String>>,
    tape: Vec<serde_json::Value>,
}

const FORMAT_VERSION: u32 = 1;

fn draws(raw: &[serde_json::Value]) -> Vec<Draw> {
    raw.iter()
        .map(|value| {
            // A recorded integer arrives as a JSON integer and a probability as a float. `is_i64`
            // is the only thing separating them, and the distinction is load-bearing: asking for
            // the wrong kind is how the harness notices the engines have parted.
            if value.is_i64() {
                Draw::Integer(value.as_i64().expect("checked"))
            } else {
                Draw::Probability(value.as_f64().expect("a draw is a number"))
            }
        })
        .collect()
}

/// A set as a sorted list, so `--ported` reads the same way twice running.
fn sorted(names: &std::collections::HashSet<&'static str>) -> Vec<&'static str> {
    let mut all: Vec<&str> = names.iter().copied().collect();
    all.sort_unstable();
    all
}

fn main() {
    // `replay --ported` prints what this engine has actually implemented, so the scenario
    // generator can restrict itself to that without a second copy of the list drifting out of
    // step with the first. A hand-kept duplicate is how the coded-move list fell behind.
    if std::env::args().nth(1).as_deref() == Some("--ported") {
        println!(
            "{}",
            json!({
                "abilities": sorted(pokemon_engine::turn::ported_abilities()),
                "items": sorted(pokemon_engine::turn::ported_items()),
                "coded_moves": sorted(pokemon_engine::turn::ported_coded_moves()),
                "coded_kinds": pokemon_engine::turn::PORTED_CODED_KINDS.as_slice(),
                "volatiles": pokemon_engine::turn::PORTED_VOLATILES
                    .iter()
                    .chain(pokemon_engine::turn::PORTED_BESPOKE_VOLATILES.iter())
                    .collect::<Vec<_>>(),
            })
        );
        return;
    }
    // `replay --coverage <data-dir>` asks the engine how much of the game it can currently play,
    // decided by the same `unsupported_reason` a real scenario is refused by. A progress number
    // counted from the rules themselves cannot flatter the port the way a hand-kept tally can.
    if std::env::args().nth(1).as_deref() == Some("--coverage") {
        let data = std::env::args().nth(2).unwrap_or_else(|| "rust/data".to_string());
        let db = Database::load(Path::new(&data)).unwrap_or_else(|e| fail(&e));
        let mut refused: std::collections::BTreeMap<&str, usize> = Default::default();
        let mut playable = 0usize;
        for the_move in db.moves.values() {
            match pokemon_engine::turn::unsupported(the_move, &db) {
                None => playable += 1,
                Some(gap) => *refused.entry(gap.label()).or_default() += 1,
            }
        }
        let ported_abilities = pokemon_engine::turn::ported_abilities().len();
        let ported_items = pokemon_engine::turn::ported_items().len();
        println!(
            "{}",
            json!({
                "moves": {"playable": playable, "total": db.moves.len(), "refused_by_cause": refused},
                "abilities": {"ported": ported_abilities, "live": db.live_abilities.len()},
                "items": {"ported": ported_items, "live": db.live_items.len()},
            })
        );
        return;
    }
    let mut args = std::env::args().skip(1);
    let (scenario_path, data_dir) = match (args.next(), args.next()) {
        (Some(s), Some(d)) => (s, d),
        _ => {
            eprintln!("usage: replay <scenario.json> <data-dir>");
            std::process::exit(64);
        }
    };
    let text = std::fs::read_to_string(&scenario_path).unwrap_or_else(|e| fail(&format!("{scenario_path}: {e}")));
    let scenario: Scenario = serde_json::from_str(&text).unwrap_or_else(|e| fail(&format!("scenario: {e}")));
    if scenario.version != FORMAT_VERSION {
        fail(&format!("scenario format v{}, this build reads v{FORMAT_VERSION}", scenario.version));
    }
    let db = Database::load(Path::new(&data_dir)).unwrap_or_else(|e| fail(&e));

    let build = |team: &Vec<Spec>| -> Side {
        Side::new(
            team.iter()
                .map(|spec| Pokemon::build(spec, &db).unwrap_or_else(|e| fail(&e)))
                .collect(),
        )
    };
    let mut state = State::new(build(&scenario.teams[0]), build(&scenario.teams[1]));
    // Checked over the whole roster, not just whoever leads: a Pokemon on the bench with an
    // unported ability will be sent out later, and finding that out mid-battle would mean half a
    // comparison had already been reported as agreement.
    for side in &state.sides {
        for pokemon in &side.team {
            if let Some(why) = unsupported_pokemon(pokemon, &db) {
                fail(&why);
            }
        }
    }
    let mut tape = Tape::new(draws(&scenario.tape));

    let mut turns = Vec::new();
    for pair in &scenario.actions {
        if state.outcome.is_some() {
            break;
        }
        let chosen = [
            parse_action(&pair[0], &state.sides[0]).unwrap_or_else(|e| fail(&e)),
            parse_action(&pair[1], &state.sides[1]).unwrap_or_else(|e| fail(&e)),
        ];
        let log = match step(&mut state, chosen, &db, &mut tape) {
            Ok(log) => log,
            Err(why) => stop(why.exit_code(), why.reason()),
        };
        turns.push(json!({
            "actions": pair,
            "events": log.entries,
            "state": digest::state(&state),
            // How far into the tape this engine has read. The Python writes the same number, so a
            // turn where the two took a different count of draws is caught at that turn rather
            // than whenever the mismatch happens to change the kind of number being asked for.
            "drawn": tape.position(),
        }));
    }
    println!("{}", serde_json::Value::Array(turns));
}

/// Exit 2 and say why: the scenario needs something this engine has not learned. Loading and
/// parsing problems come through here too — they are all "we cannot play this", which the harness
/// is allowed to skip.
fn fail(why: &str) -> ! {
    stop(2, why)
}

/// Exit with the code the refusal chose. 2 is unported, 3 is a divergence the harness must never
/// skip; keeping them apart in the exit code is what stops a real disagreement reading as a gap.
fn stop(code: i32, why: &str) -> ! {
    eprintln!("{why}");
    std::process::exit(code);
}
