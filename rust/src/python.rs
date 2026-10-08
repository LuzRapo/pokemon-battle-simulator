//! The minimal PyO3 bridge: Python can call into this engine's `step()` in-process, on the exact
//! same replay-shaped surface `bin/replay.rs` already drives (`Database::load`, `State`, `Action`,
//! `step`) — a scenario's teams, then one `step()` per turn against a pair of canonical action
//! strings ("move:SLOT:Name" / "zmove:SLOT:Name" / "switch:Nickname", the same strings
//! `differential.py::name_action` already produces and `turn::parse_action` already reads).
//!
//! What this deliberately does not do: decide which actions are legal. That stays Python's job,
//! exactly as it is for every mechanic in this project already (`battle_sim/engine/choices.py`) —
//! this module only executes whatever action it is handed, fast. A battle built here can run two
//! ways:
//!
//!   * **Replay mode** (`PyBattle.new` with a tape): the recorded randomness from a Python
//!     `differential.record()` run, replayed in-process instead of through the `replay` binary's
//!     stdin/stdout — for verifying this bridge agrees with the subprocess path and, through it,
//!     with the Python reference, using the exact same contract every other test in this project
//!     already trusts.
//!   * **Live mode** (`PyBattle.live` with a seed): this engine's own fresh, seeded randomness —
//!     for actually playing a battle nothing else is watching, which self-play needs and which a
//!     replay tape structurally cannot provide (there is no Python run alongside to have recorded
//!     one). See `tape::Tape::live`.
//!
//! Both modes return the same per-turn payload: the log as JSON (`log::Event` already derives
//! `Serialize`) and a state digest (`digest::state`) shaped like `bin/replay.rs`'s own trace, so a
//! caller on either side of this boundary can reuse the same JSON-shaped tooling either way.
//!
//! Still missing before a self-play loop can run unattended: something on the Python side to keep
//! a `BattleState` mirror in sync from this digest each turn, since `legal_actions` reads
//! `BattleState`/`SideState` fields directly and has no way to ask a Rust `State` the same
//! question. That is Python-side glue, not a Rust-port gap, and is flagged in
//! `docs/rust-port-plan.md` as the concrete next step rather than bundled into this bridge.

// The `#[pymethods]`/`#[pymodule]` macro expansion in this module trips `useless_conversion` on a
// `PyResult` bubbling into a `PyResult` wherever a constructor's body uses `?` — a false positive
// in generated code this module does not write, not a real double conversion anywhere here.
// `create_exception!` separately probes a `gil-refs` cfg pyo3 0.22 does not declare in this
// crate's `Cargo.toml` — also its own macro's business, not this module's.
#![allow(clippy::useless_conversion, unexpected_cfgs)]

use crate::battle::{Pokemon, Side, Spec, State};
use crate::data::Database;
use crate::digest;
use crate::tape::{Draw, Tape};
use crate::turn::{parse_action, step, unsupported_pokemon, Refusal};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::path::Path;
use std::sync::Arc;

// A scenario needs something this engine has not learned yet — the same "exit 2" the `replay`
// binary reports, raised here as a distinct Python exception so a caller can skip it rather than
// mistaking it for a real divergence.
pyo3::create_exception!(pokemon_engine_rs, Unported, PyRuntimeError);

// The recorded randomness shows the two engines have already taken different paths — the same
// "exit 3" the `replay` binary reports. Never skippable.
pyo3::create_exception!(pokemon_engine_rs, Diverged, PyRuntimeError);

fn raise(why: Refusal) -> PyErr {
    match why {
        Refusal::Unported(reason) => Unported::new_err(reason),
        Refusal::Diverged(reason) => Diverged::new_err(reason),
    }
}

#[pyclass(name = "Database")]
pub struct PyDatabase(pub(crate) Arc<Database>);

#[pymethods]
impl PyDatabase {
    /// Load once, from the same `rust/data` directory `replay`/`bench` read. Cheap to share: every
    /// `PyBattle` built from this holds an `Arc` clone rather than its own copy.
    #[new]
    fn new(data_dir: &str) -> PyResult<Self> {
        Database::load(Path::new(data_dir)).map(|db| PyDatabase(Arc::new(db))).map_err(PyValueError::new_err)
    }
}

fn draws_from_json(raw: &str) -> PyResult<Vec<Draw>> {
    let values: Vec<serde_json::Value> = serde_json::from_str(raw).map_err(|e| PyValueError::new_err(e.to_string()))?;
    values
        .iter()
        .map(|value| {
            if value.is_i64() {
                value.as_i64().map(Draw::Integer).ok_or_else(|| PyValueError::new_err("bad integer draw"))
            } else {
                value.as_f64().map(Draw::Probability).ok_or_else(|| PyValueError::new_err("bad probability draw"))
            }
        })
        .collect()
}

fn build_side(team_json: &str, db: &Database) -> PyResult<Side> {
    let specs: Vec<Spec> = serde_json::from_str(team_json).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let team = specs
        .iter()
        .map(|spec| Pokemon::build(spec, db))
        .collect::<Result<Vec<Pokemon>, String>>()
        .map_err(PyValueError::new_err)?;
    Ok(Side::new(team))
}

/// Every Pokemon on both rosters, checked up front exactly as `bin/replay.rs` checks them: a bench
/// Pokemon with an unported ability failing mid-battle would mean turns already reported as played
/// had quietly been wrong.
fn check_playable(state: &State, db: &Database) -> PyResult<()> {
    for side in &state.sides {
        for pokemon in &side.team {
            if let Some(why) = unsupported_pokemon(pokemon, db) {
                return Err(Unported::new_err(why));
            }
        }
    }
    Ok(())
}

#[pyclass(name = "Battle")]
pub struct PyBattle {
    state: State,
    db: Arc<Database>,
    tape: Tape,
}

#[pymethods]
impl PyBattle {
    /// A battle whose randomness is a pre-recorded tape — `teams_json` mirrors `Scenario.teams`
    /// (a JSON array of two `Spec` arrays), `tape_json` mirrors `Scenario.tape` (the flat array of
    /// recorded draws). Use this to check the bridge against a Python `differential.record()` run.
    #[staticmethod]
    fn new(db: &PyDatabase, teams_json: [String; 2], tape_json: &str) -> PyResult<Self> {
        let database = db.0.clone();
        let state = State::new(build_side(&teams_json[0], &database)?, build_side(&teams_json[1], &database)?);
        check_playable(&state, &database)?;
        let tape = Tape::new(draws_from_json(tape_json)?);
        Ok(PyBattle { state, db: database, tape })
    }

    /// A battle whose randomness is freshly drawn from a seeded RNG, with no recording behind it —
    /// what self-play actually runs on, since there is no Python run alongside to have recorded one.
    #[staticmethod]
    fn live(db: &PyDatabase, teams_json: [String; 2], seed: u64) -> PyResult<Self> {
        let database = db.0.clone();
        let state = State::new(build_side(&teams_json[0], &database)?, build_side(&teams_json[1], &database)?);
        check_playable(&state, &database)?;
        Ok(PyBattle { state, db: database, tape: Tape::live(seed) })
    }

    /// Resolve one turn. `actions` is `["move:FIRST:Tackle", "switch:Onix"]`-shaped, the same
    /// strings `differential.py::name_action` produces and `bin/replay.rs` parses. Returns the
    /// turn's log (a JSON array of `log::Event`) — raises `Unported` or `Diverged` rather than
    /// returning either, so a caller cannot forget to check which one happened.
    fn step(&mut self, actions: [String; 2]) -> PyResult<String> {
        let chosen = [
            parse_action(&actions[0], &self.state.sides[0]).map_err(PyValueError::new_err)?,
            parse_action(&actions[1], &self.state.sides[1]).map_err(PyValueError::new_err)?,
        ];
        let log = step(&mut self.state, chosen, &self.db, &mut self.tape).map_err(raise)?;
        serde_json::to_string(&log.entries).map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// A snapshot of both sides and the field, shaped like `bin/replay.rs`'s own per-turn trace —
    /// what a caller reads to decide the next pair of legal actions, and what self-play needs to
    /// keep a Python-side bookkeeping mirror in sync (see the module doc for what that still
    /// requires beyond this bridge).
    fn digest(&self) -> String {
        digest::state(&self.state).to_string()
    }

    /// `Some("WON")`/`Some("LOST")`/`Some("DRAW")` once the battle has an outcome, `None` while it
    /// is still live.
    fn outcome(&self) -> Option<&'static str> {
        self.state.outcome.map(|o| o.name())
    }

    /// How many draws this battle has taken so far — meaningful in replay mode (the Python writes
    /// the same number, so a mismatched count is a divergence caught the moment it happens) and
    /// just a counter in live mode.
    fn drawn(&self) -> usize {
        self.tape.position()
    }
}

#[pymodule]
fn pokemon_engine_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDatabase>()?;
    m.add_class::<PyBattle>()?;
    m.add("Unported", m.py().get_type_bound::<Unported>())?;
    m.add("Diverged", m.py().get_type_bound::<Diverged>())?;
    Ok(())
}
