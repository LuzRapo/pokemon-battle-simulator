//! The minimal PyO3 bridge: Python can call into this engine's `step()` in-process, on the exact
//! same replay-shaped surface `bin/replay.rs` already drives (`Database::load`, `State`, `Action`,
//! `step`) — a scenario's teams, then one `step()` per turn against a pair of canonical action
//! strings ("move:SLOT:Name" / "zmove:SLOT:Name" / "switch:Nickname", the same strings
//! `differential.py::name_action` already produces and `turn::parse_action` already reads).
//!
//! It also answers what each side may do (`legal_actions`, via `choices.rs`) and lets each side
//! make its own mid-turn and between-turn replacements (`begin_turn` / `resume` / `forced_switch`),
//! as `runner.run_battle` does. A battle built here can run two ways:
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
//! Self-play does not step `Battle`s one at a time: `VecEnv` (over `env.rs`) holds many, pauses each
//! at its next decision, and hands observations and masks over as numpy arrays.

// The `#[pymethods]`/`#[pymodule]` macro expansion in this module trips `useless_conversion` on a
// `PyResult` bubbling into a `PyResult` wherever a constructor's body uses `?` — a false positive
// in generated code this module does not write, not a real double conversion anywhere here.
// `create_exception!` separately probes a `gil-refs` cfg pyo3 0.22 does not declare in this
// crate's `Cargo.toml` — also its own macro's business, not this module's.
#![allow(clippy::useless_conversion, unexpected_cfgs)]

use crate::battle::{Pokemon, Side, Spec, State};
use crate::data::Database;
use crate::tape::{Draw, Tape};
use crate::turn::{
    apply_forced_switch, begin_turn, parse_action, resume_with_switch, step, unsupported_pokemon, Action, Refusal,
    Replacements, TurnStatus,
};
use crate::{choices, digest, env, matchup, obs, tournament};
use numpy::{PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray1};
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

    /// Whether this engine can play the move at all: known, and nothing in it left unported.
    fn move_playable(&self, name: &str) -> bool {
        self.0.move_named(name).is_some_and(|m| crate::turn::unsupported(m, &self.0).is_none())
    }

    /// Whether a Pokemon holding this ability can be built into a battle here.
    fn ability_playable(&self, name: &str) -> bool {
        !self.0.live_abilities.contains(name) || crate::turn::ported_abilities().contains(name)
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

/// `runner.build_side`: the team in `order`, whose first entry leads. The in-battle team indices
/// follow the permuted order, exactly as the Python's do.
fn build_side(team_json: &str, order: Option<&Vec<usize>>, db: &Database) -> PyResult<Side> {
    let specs: Vec<Spec> = serde_json::from_str(team_json).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let order: Vec<usize> = match order {
        Some(order) => {
            let mut sorted = order.clone();
            sorted.sort_unstable();
            if sorted != (0..specs.len()).collect::<Vec<_>>() {
                return Err(PyValueError::new_err(format!("order must be a permutation of 0..{}", specs.len())));
            }
            order.clone()
        }
        None => (0..specs.len()).collect(),
    };
    let team = order
        .iter()
        .map(|&index| Pokemon::build(&specs[index], db))
        .collect::<Result<Vec<Pokemon>, String>>()
        .map_err(PyValueError::new_err)?;
    Ok(Side::new(team))
}

fn build_state(teams_json: &[String; 2], orders: &Option<[Vec<usize>; 2]>, db: &Database) -> PyResult<State> {
    let order = |side: usize| orders.as_ref().map(|orders| &orders[side]);
    Ok(State::new(build_side(&teams_json[0], order(0), db)?, build_side(&teams_json[1], order(1), db)?))
}

/// What a turn call hands back: `("done", log_json)`, or `("switch", side)` while it waits on that
/// side to name a replacement.
fn turn_status(status: TurnStatus) -> PyResult<(String, PyObject)> {
    Python::with_gil(|py| match status {
        TurnStatus::Done(log) => {
            let json = serde_json::to_string(&log.entries).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(("done".to_string(), json.into_py(py)))
        }
        TurnStatus::NeedsSwitch { side } => Ok(("switch".to_string(), side.into_py(py))),
    })
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
    /// One `MatchupPlayer` per side, kept for the whole battle as the Python's players are, so
    /// their offense caches fill in the same order.
    scorers: [matchup::MatchupAi; 2],
}

#[pymethods]
impl PyBattle {
    /// A battle whose randomness is a pre-recorded tape — `teams_json` mirrors `Scenario.teams`
    /// (a JSON array of two `Spec` arrays), `tape_json` mirrors `Scenario.tape` (the flat array of
    /// recorded draws). Use this to check the bridge against a Python `differential.record()` run.
    #[staticmethod]
    #[pyo3(signature = (db, teams_json, tape_json, orders=None))]
    fn new(db: &PyDatabase, teams_json: [String; 2], tape_json: &str, orders: Option<[Vec<usize>; 2]>) -> PyResult<Self> {
        let database = db.0.clone();
        let state = build_state(&teams_json, &orders, &database)?;
        check_playable(&state, &database)?;
        let tape = Tape::new(draws_from_json(tape_json)?);
        Ok(PyBattle { state, db: database, tape, scorers: fresh_scorers() })
    }

    /// A battle whose randomness is freshly drawn from a seeded RNG, with no recording behind it —
    /// what self-play actually runs on, since there is no Python run alongside to have recorded one.
    #[staticmethod]
    #[pyo3(signature = (db, teams_json, seed, orders=None))]
    fn live(db: &PyDatabase, teams_json: [String; 2], seed: u64, orders: Option<[Vec<usize>; 2]>) -> PyResult<Self> {
        let database = db.0.clone();
        let state = build_state(&teams_json, &orders, &database)?;
        check_playable(&state, &database)?;
        Ok(PyBattle { state, db: database, tape: Tape::live(seed), scorers: fresh_scorers() })
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

    /// Start a turn in which the players answer their own mid-turn replacements (`runner.run_battle`'s
    /// `switch_chooser`): `("switch", side)` means that side has to name one, via `resume`.
    fn begin_turn(&mut self, actions: [String; 2]) -> PyResult<(String, PyObject)> {
        let chosen = [
            parse_action(&actions[0], &self.state.sides[0]).map_err(PyValueError::new_err)?,
            parse_action(&actions[1], &self.state.sides[1]).map_err(PyValueError::new_err)?,
        ];
        let status = begin_turn(&mut self.state, chosen, Replacements::Caller, &self.db, &mut self.tape).map_err(raise)?;
        turn_status(status)
    }

    /// Answer the replacement a paused turn is waiting on, with a `"switch:Nickname"` action.
    fn resume(&mut self, action: String) -> PyResult<(String, PyObject)> {
        let side = self.waiting_on().ok_or_else(|| PyValueError::new_err("no turn is waiting on a replacement"))?;
        let to = match parse_action(&action, &self.state.sides[side]).map_err(PyValueError::new_err)? {
            Action::Switch { to } => to,
            Action::Move { .. } => return Err(PyValueError::new_err("a replacement has to be a switch")),
        };
        let status = resume_with_switch(&mut self.state, to, &self.db, &mut self.tape).map_err(raise)?;
        turn_status(status)
    }

    /// `apply_forced_switch`, between turns: a faint replacement. Returns its log as JSON.
    fn forced_switch(&mut self, side: usize, action: String) -> PyResult<String> {
        let to = match parse_action(&action, &self.state.sides[side]).map_err(PyValueError::new_err)? {
            Action::Switch { to } => to,
            Action::Move { .. } => return Err(PyValueError::new_err("a replacement has to be a switch")),
        };
        let log = apply_forced_switch(&mut self.state, side, to, &self.db);
        serde_json::to_string(&log.entries).map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// The side a paused turn is waiting on, if any.
    fn waiting_on(&self) -> Option<usize> {
        self.state.turn_in_progress.as_ref().map(|turn| turn.waiting_on())
    }

    /// `runner.run_battle`'s replacement loop condition: this side must switch before anything else.
    fn needs_replacement(&self, side: usize) -> bool {
        let own = &self.state.sides[side];
        own.active_pokemon().fainted() || own.needs_switch
    }

    /// Every legal action for `side`, named as `differential.name_action` names them.
    fn legal_actions(&self, side: usize) -> Vec<String> {
        crate::choices::legal_actions(&self.state, side, &self.db)
            .into_iter()
            .map(|action| crate::choices::name_action(&self.state, side, action))
            .collect()
    }

    /// `MatchupPlayer.feature_actions` for `side`'s legal actions, in order: each action's name and
    /// its 28 features, or `None` for one scored as dead.
    fn matchup_features(&mut self, side: usize) -> Vec<(String, Option<Vec<f64>>)> {
        let legal = choices::legal_actions(&self.state, side, &self.db);
        let features = self.scorers[side].features(&self.state, side, &legal, &self.db);
        legal
            .iter()
            .zip(features)
            .map(|(&action, f)| (choices::name_action(&self.state, side, action), f.map(|f| f.to_vec())))
            .collect()
    }

    /// `obs::encode` from `viewer`'s side, as flat `(ids, pokemon, field)` lists — for checking the
    /// Python encoder against; training reads the same arrays from `VecEnv` without the copies.
    fn observe(&self, viewer: usize, decision: &str) -> PyResult<(Vec<i64>, Vec<f32>, Vec<f32>)> {
        let decision = match decision {
            "lead" => crate::obs::Decision::Lead,
            "turn" => crate::obs::Decision::Turn,
            "switch" => crate::obs::Decision::Switch,
            other => return Err(PyValueError::new_err(format!("no decision called {other:?}"))),
        };
        let observation = crate::obs::encode(&self.state, viewer, decision, &self.db);
        Ok((observation.ids, observation.pokemon, observation.field))
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

fn fresh_scorers() -> [matchup::MatchupAi; 2] {
    [matchup::MatchupAi::new(matchup::Weights::default()), matchup::MatchupAi::new(matchup::Weights::default())]
}

fn team_from_json(raw: &str) -> PyResult<Vec<Spec>> {
    serde_json::from_str(raw).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// `MatchupPlayer().choose_order(own, opponent)`: team indices, the lead first.
#[pyfunction]
fn matchup_order(db: &PyDatabase, own_json: &str, opponent_json: &str) -> PyResult<Vec<usize>> {
    let build = |raw: &str| -> PyResult<Vec<Pokemon>> {
        team_from_json(raw)?
            .iter()
            .map(|spec| Pokemon::build(spec, &db.0))
            .collect::<Result<Vec<_>, String>>()
            .map_err(PyValueError::new_err)
    };
    Ok(matchup::choose_order(build(own_json)?, build(opponent_json)?, &db.0))
}

/// `(side, index, move, foe index, low, high)`.
type Estimate = (usize, usize, String, usize, i32, i32);
/// `(outcome, turns, survivors_a, survivors_b, error)`.
type TournamentResult = (Option<&'static str>, i32, usize, usize, Option<String>);

/// `analysis.damage_range` for every move of every Pokemon on each team against every Pokemon on
/// the other, on the scratch board `choose_order` reads: `(side, index, move, foe index, low, high)`.
/// For finding which estimate the two engines disagree on.
#[pyfunction]
fn matchup_damage(db: &PyDatabase, own_json: &str, opponent_json: &str) -> PyResult<Vec<Estimate>> {
    let build = |raw: &str| -> PyResult<Vec<Pokemon>> {
        team_from_json(raw)?
            .iter()
            .map(|spec| Pokemon::build(spec, &db.0))
            .collect::<Result<Vec<_>, String>>()
            .map_err(PyValueError::new_err)
    };
    let state = State::new(Side::new(build(own_json)?), Side::new(build(opponent_json)?));
    let mut out = Vec::new();
    for side in 0..2 {
        for (index, pokemon) in state.sides[side].team.iter().enumerate() {
            for name in &pokemon.moves {
                let Some(the_move) = db.0.move_named(name) else { continue };
                for foe in 0..state.sides[1 - side].team.len() {
                    let (low, high) = matchup::damage_range(the_move, (side, index), (1 - side, foe), &state, &db.0);
                    out.push((side, index, name.clone(), foe, low, high));
                }
            }
        }
    }
    Ok(out)
}

/// Battles piloted by `MatchupPlayer` on both sides, played in parallel with the GIL released.
/// Each job is `(team_a_json, team_b_json, seed)`; each result is `(outcome, turns, survivors_a,
/// survivors_b, error)`, where `outcome` is "P1_WIN", "P2_WIN", "DRAW" or `None` at the turn cap,
/// and `error` names why a battle could not be played.
#[pyfunction]
#[pyo3(signature = (db, jobs, weights_json=None, max_turns=1000, threads=0))]
fn play_matchup_battles(
    py: Python<'_>,
    db: &PyDatabase,
    jobs: Vec<(String, String, u64)>,
    weights_json: Option<&str>,
    max_turns: i32,
    threads: usize,
) -> PyResult<Vec<TournamentResult>> {
    let weights = match weights_json {
        Some(raw) => matchup::Weights::from_json(raw).map_err(PyValueError::new_err)?,
        None => matchup::Weights::default(),
    };
    let battles = jobs
        .iter()
        .map(|(a, b, seed)| Ok(([team_from_json(a)?, team_from_json(b)?], *seed)))
        .collect::<PyResult<Vec<_>>>()?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let database = db.0.clone();
    let results = py.allow_threads(|| pool.install(|| tournament::play_all(&battles, weights, max_turns, &database)));
    Ok(results
        .into_iter()
        .map(|result| match result {
            Ok(played) => {
                (played.outcome.map(|o| o.name()), played.turns, played.survivors[0], played.survivors[1], None)
            }
            Err(why) => (None, 0, 0, 0, Some(why)),
        })
        .collect())
}

fn specs_from_json(teams_json: &[String; 2]) -> PyResult<[Vec<Spec>; 2]> {
    let parse = |raw: &str| serde_json::from_str::<Vec<Spec>>(raw).map_err(|e| PyValueError::new_err(e.to_string()));
    Ok([parse(&teams_json[0])?, parse(&teams_json[1])?])
}

type Observed<'py> = (
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray3<i64>>,
    Bound<'py, PyArray3<f32>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<bool>>,
);

/// Self-play's environment: `size` battles, each paused at its next real decision (`env::Battle`).
///
/// One round is `observe()` — every open request across every battle, as arrays — then `act()`
/// with an answer per row, which plays each battle on to its next decision, in parallel on a pool
/// of `threads` with the GIL released. Finished battles are handed back by `collect()` and their
/// slot sits empty until `reset` fills it.
#[pyclass(name = "VecEnv")]
pub struct PyVecEnv {
    db: Arc<Database>,
    battles: Vec<Option<env::Battle>>,
    max_turns: i32,
    pool: rayon::ThreadPool,
}

#[pymethods]
impl PyVecEnv {
    #[new]
    #[pyo3(signature = (db, size, max_turns=300, threads=0))]
    fn new(db: &PyDatabase, size: usize, max_turns: i32, threads: usize) -> PyResult<Self> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyVecEnv { db: db.0.clone(), battles: (0..size).map(|_| None).collect(), max_turns, pool })
    }

    fn __len__(&self) -> usize {
        self.battles.len()
    }

    /// Start a live battle in slot `index`; it waits on both leads (or on less, if forced).
    fn reset(&mut self, index: usize, teams_json: [String; 2], seed: u64) -> PyResult<()> {
        let battle = env::Battle::new(specs_from_json(&teams_json)?, Tape::live(seed), self.max_turns, &self.db)
            .map_err(Unported::new_err)?;
        *self.slot(index)? = Some(battle);
        Ok(())
    }

    /// Start a battle that replays a Python recording's draws — for checking this environment's
    /// sequencing against `differential.record_played`, never for training.
    fn replay(&mut self, index: usize, teams_json: [String; 2], tape_json: &str) -> PyResult<()> {
        let tape = Tape::new(draws_from_json(tape_json)?);
        let battle = env::Battle::new(specs_from_json(&teams_json)?, tape, self.max_turns, &self.db)
            .map_err(Unported::new_err)?;
        *self.slot(index)? = Some(battle);
        Ok(())
    }

    /// Every open request: `(env, side, decision, ids, pokemon, field, mask)`, one row each, with
    /// `decision` 0 lead / 1 turn / 2 switch and `mask` the legal actions of `choices`' 14.
    fn observe<'py>(&self, py: Python<'py>) -> PyResult<Observed<'py>> {
        let batch = py.allow_threads(|| self.pool.install(|| env::observe_all(&self.battles, &self.db)));
        let rows = batch.env.len();
        let shaped = |error: PyErr| PyRuntimeError::new_err(format!("observation shape: {error}"));
        Ok((
            PyArray1::from_vec_bound(py, batch.env),
            PyArray1::from_vec_bound(py, batch.side),
            PyArray1::from_vec_bound(py, batch.decision),
            PyArray1::from_vec_bound(py, batch.ids).reshape([rows, obs::SLOTS, obs::POKEMON_IDS]).map_err(shaped)?,
            PyArray1::from_vec_bound(py, batch.pokemon)
                .reshape([rows, obs::SLOTS, obs::POKEMON_FLOATS])
                .map_err(shaped)?,
            PyArray1::from_vec_bound(py, batch.field).reshape([rows, obs::FIELD_FLOATS]).map_err(shaped)?,
            PyArray1::from_vec_bound(py, batch.mask).reshape([rows, choices::ACTION_SPACE]).map_err(shaped)?,
        ))
    }

    /// Answer requests: row `r` gives battle `env[r]`'s `side[r]` the action `action[r]`. Every
    /// answer is applied; an illegal one is refused and reported after the rest have been.
    fn act(
        &mut self,
        py: Python<'_>,
        env: PyReadonlyArray1<'_, i64>,
        side: PyReadonlyArray1<'_, i64>,
        action: PyReadonlyArray1<'_, i64>,
    ) -> PyResult<()> {
        let (env, side, action) = (env.as_slice()?, side.as_slice()?, action.as_slice()?);
        if env.len() != side.len() || env.len() != action.len() {
            return Err(PyValueError::new_err("env, side and action must be the same length"));
        }
        let answers: Vec<(usize, usize, u8)> =
            (0..env.len()).map(|r| (env[r] as usize, side[r] as usize, action[r] as u8)).collect();
        let (battles, db, pool) = (&mut self.battles, &self.db, &self.pool);
        py.allow_threads(|| pool.install(|| env::act_all(battles, &answers, db))).map_err(PyValueError::new_err)
    }

    /// Finished battles, emptied from their slots: `(index, ending, winner, turns)`, where `ending`
    /// is "won", "draw", "timeout" or "failed: why", and `winner` is the winning side or `None`.
    fn collect(&mut self) -> Vec<(usize, String, Option<usize>, i32)> {
        let mut finished = Vec::new();
        for (index, slot) in self.battles.iter_mut().enumerate() {
            let Some(ending) = slot.as_ref().and_then(|battle| battle.ending().cloned()) else { continue };
            let turns = slot.as_ref().map_or(0, |battle| battle.state().turn);
            let (name, winner) = match ending {
                env::Ending::Won(side) => ("won".to_string(), Some(side)),
                env::Ending::Draw => ("draw".to_string(), None),
                env::Ending::TimedOut => ("timeout".to_string(), None),
                env::Ending::Failed(why) => (format!("failed: {why}"), None),
            };
            finished.push((index, name, winner, turns));
            *slot = None;
        }
        finished
    }

    /// Slots with no battle in them, to `reset`.
    fn empty(&self) -> Vec<usize> {
        (0..self.battles.len()).filter(|&i| self.battles[i].is_none()).collect()
    }

    /// Battle `index`'s legal actions for `side`, as numbers and as `differential.name_action`
    /// strings — for tests that drive this environment from a Python recording.
    fn legal(&self, index: usize, side: usize) -> PyResult<Vec<(u8, String)>> {
        let battle = self.battle(index)?;
        let names = |action: u8| match battle.phase_is_lead() {
            true => format!("lead:{}", action - choices::SWITCH),
            false => choices::name_action(battle.state(), side, action),
        };
        Ok(battle.legal(side, &self.db).into_iter().map(|action| (action, names(action))).collect())
    }

    fn digest(&self, index: usize) -> PyResult<String> {
        Ok(digest::state(self.battle(index)?.state()).to_string())
    }

    fn drawn(&self, index: usize) -> PyResult<usize> {
        Ok(self.battle(index)?.drawn())
    }
}

impl PyVecEnv {
    fn slot(&mut self, index: usize) -> PyResult<&mut Option<env::Battle>> {
        let size = self.battles.len();
        self.battles.get_mut(index).ok_or_else(|| PyValueError::new_err(format!("no slot {index} of {size}")))
    }

    fn battle(&self, index: usize) -> PyResult<&env::Battle> {
        self.battles
            .get(index)
            .and_then(Option::as_ref)
            .ok_or_else(|| PyValueError::new_err(format!("no battle in slot {index}")))
    }
}

#[pymodule]
fn pokemon_engine_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDatabase>()?;
    m.add_class::<PyBattle>()?;
    m.add_class::<PyVecEnv>()?;
    m.add_function(wrap_pyfunction!(matchup_order, m)?)?;
    m.add_function(wrap_pyfunction!(matchup_damage, m)?)?;
    m.add_function(wrap_pyfunction!(play_matchup_battles, m)?)?;
    m.add("Unported", m.py().get_type_bound::<Unported>())?;
    m.add("Diverged", m.py().get_type_bound::<Diverged>())?;
    Ok(())
}
