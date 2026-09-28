//! Throughput, measured the way the search will actually pay for it.
//!
//!     bench <scenario-dir> <data-dir> [repeats]
//!
//! Loads the database once and replays every scenario in the directory, because that is the shape
//! of the work the tree search does: one process, one copy of the data, an enormous number of
//! battles. Timing `replay` in a loop instead measures process startup and JSON parsing, which is
//! most of that binary's runtime and none of the engine's.

use pokemon_engine::battle::{Pokemon, Side, Spec, State, SLOT_NAMES};
use pokemon_engine::data::Database;
use pokemon_engine::tape::{Draw, Tape};
use pokemon_engine::turn::{step, Action};
use serde::Deserialize;
use std::path::Path;
use std::time::Instant;

#[derive(Debug, Deserialize)]
struct Scenario {
    teams: Vec<Vec<Spec>>,
    actions: Vec<Vec<String>>,
    tape: Vec<serde_json::Value>,
}

fn draws(raw: &[serde_json::Value]) -> Vec<Draw> {
    raw.iter()
        .map(|value| {
            if value.is_i64() {
                Draw::Integer(value.as_i64().expect("checked"))
            } else {
                Draw::Probability(value.as_f64().expect("a draw is a number"))
            }
        })
        .collect()
}

fn parse_action(named: &str, side: &Side) -> Option<Action> {
    let (kind, rest) = named.split_once(':')?;
    match kind {
        "move" => {
            let (slot_name, _) = rest.split_once(':')?;
            SLOT_NAMES.iter().position(|s| *s == slot_name).map(|slot| Action::Move { slot, z_move: false })
        }
        "switch" => side.team.iter().position(|p| p.nickname == rest).map(|to| Action::Switch { to }),
        _ => None,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (dir, data) = match (args.first(), args.get(1)) {
        (Some(d), Some(x)) => (d.clone(), x.clone()),
        _ => {
            eprintln!("usage: bench <scenario-dir> <data-dir> [repeats]");
            std::process::exit(64);
        }
    };
    let repeats: usize = args.get(2).and_then(|r| r.parse().ok()).unwrap_or(1);

    let loading = Instant::now();
    let db = Database::load(Path::new(&data)).unwrap_or_else(|e| panic!("{e}"));
    let load_time = loading.elapsed();

    let mut scenarios = Vec::new();
    for entry in std::fs::read_dir(&dir).expect("scenario directory").flatten() {
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "json") {
            let text = std::fs::read_to_string(&path).expect("scenario");
            scenarios.push(serde_json::from_str::<Scenario>(&text).expect("scenario json"));
        }
    }

    let started = Instant::now();
    let (mut battles, mut turns) = (0usize, 0usize);
    for _ in 0..repeats {
        for scenario in &scenarios {
            let build = |team: &Vec<Spec>| -> Side {
                Side::new(team.iter().map(|spec| Pokemon::build(spec, &db).expect("build")).collect())
            };
            let mut state = State::new(build(&scenario.teams[0]), build(&scenario.teams[1]));
            let mut tape = Tape::new(draws(&scenario.tape));
            for pair in &scenario.actions {
                if state.outcome.is_some() {
                    break;
                }
                let chosen = match (parse_action(&pair[0], &state.sides[0]), parse_action(&pair[1], &state.sides[1])) {
                    (Some(a), Some(b)) => [a, b],
                    _ => break,
                };
                if step(&mut state, chosen, &db, &mut tape).is_err() {
                    break;
                }
                turns += 1;
            }
            battles += 1;
        }
    }
    let elapsed = started.elapsed();

    println!("database loaded in {:.3}s (once)", load_time.as_secs_f64());
    println!(
        "{battles} battles, {turns} turns in {:.3}s -- {:.0} turns/s, {:.0} battles/s",
        elapsed.as_secs_f64(),
        turns as f64 / elapsed.as_secs_f64(),
        battles as f64 / elapsed.as_secs_f64()
    );
}
