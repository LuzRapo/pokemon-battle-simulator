//! Every species' stats, computed by this engine, as JSON on stdout.
//!
//! The first thing worth checking between two engines: before anybody simulates a turn, do they
//! even agree what a Pokemon's numbers are? A single stat off by one changes a damage roll, which
//! changes whether something survives, which changes the battle — and it would be nearly
//! impossible to find from the far end.

use pokemon_engine::data::Database;
use pokemon_engine::stats::{totals, Spread};
use std::path::Path;

fn main() {
    let directory = std::env::args().nth(1).unwrap_or_else(|| "data".into());
    let db = match Database::load(Path::new(&directory)) {
        Ok(db) => db,
        Err(why) => {
            eprintln!("could not load {directory}: {why}");
            std::process::exit(1);
        }
    };
    // Three cases, chosen to exercise the three places the formula truncates: a neutral nature with
    // perfect IVs, a nature that raises and lowers, and an EV spread that is not a multiple of four.
    let cases: [(&str, i32, Spread, Spread); 3] = [
        ("HARDY", 50, Spread { hp: 31, attack: 31, defence: 31, sp_attack: 31, sp_defence: 31, speed: 31 }, Spread::default()),
        (
            "JOLLY",
            50,
            Spread { hp: 31, attack: 31, defence: 31, sp_attack: 31, sp_defence: 31, speed: 31 },
            Spread { hp: 4, attack: 252, defence: 0, sp_attack: 0, sp_defence: 0, speed: 252 },
        ),
        (
            "MODEST",
            100,
            Spread { hp: 13, attack: 7, defence: 29, sp_attack: 30, sp_defence: 3, speed: 19 },
            Spread { hp: 74, attack: 11, defence: 6, sp_attack: 251, sp_defence: 85, speed: 83 },
        ),
    ];
    let mut out = serde_json::Map::new();
    for (nature_name, level, ivs, evs) in cases {
        let nature = db.natures.get(nature_name).expect("exported natures include this one");
        let mut per_species = serde_json::Map::new();
        for (key, species) in &db.species {
            let t = totals(&species.base_stats, &ivs, &evs, level, nature);
            per_species.insert(
                key.clone(),
                serde_json::json!([t.hp, t.attack, t.defence, t.sp_attack, t.sp_defence, t.speed]),
            );
        }
        out.insert(format!("{nature_name}-{level}"), serde_json::Value::Object(per_species));
    }
    println!("{}", serde_json::Value::Object(out));
}
