//! Many battles at once, stopped only where a player has something to decide.
//!
//! A `Battle` walks one battle through the decisions `runner.run_battle` asks a player for, in its
//! order: each side's lead before turn 0; both sides' actions for a turn; a pivot's replacement the
//! moment it is forced (`turn::begin_turn` with `Replacements::Caller`); and faint replacements
//! after the turn, side 0 then side 1, each side repeating while hazards knock out its arrival.
//! Every decision is one number in `choices`' fixed action space — for a lead, `SWITCH + i` sends
//! team member `i` out first and the rest keep their listed order, as the bot's `_lead_order` does.
//!
//! A decision with only one legal answer is answered here, not asked: a network learns nothing
//! from it, and in a long battle they are common (a lone survivor's replacement, a rampage, a
//! charge). `VecEnv` holds many battles and advances them in parallel between rounds of asking.

use crate::battle::{Outcome, Pokemon, Side, Spec, State};
use crate::choices::{legal_actions, ACTION_SPACE, SWITCH, Z_MOVE};
use crate::data::Database;
use crate::obs::{encode, Decision, Observation};
use crate::tape::Tape;
use crate::turn::{apply_forced_switch, begin_turn, resume_with_switch, Action, Refusal, Replacements, TurnStatus};
use rayon::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Lead,
    Turn,
    /// A turn paused on this side's mid-turn replacement.
    Pivot(usize),
    /// Between turns, this side must replace a fainted (or ejected) Pokemon.
    Replace(usize),
    Over,
}

/// How a battle ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Ending {
    Won(usize),
    Draw,
    /// The turn limit came first. Not a result: a trainer should bootstrap from it, not score it.
    TimedOut,
    Failed(String),
}

/// One decision being asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Request {
    pub side: usize,
    pub decision: Decision,
}

pub struct Battle {
    specs: [Vec<Spec>; 2],
    state: State,
    tape: Tape,
    phase: Phase,
    /// This round's choices, while the other side's is still outstanding (leads and turns).
    chosen: [Option<u8>; 2],
    max_turns: i32,
    ending: Option<Ending>,
}

fn build(specs: &[Spec], order: &[usize], db: &Database) -> Result<Side, String> {
    let team = order.iter().map(|&index| Pokemon::build(&specs[index], db)).collect::<Result<Vec<_>, _>>()?;
    Ok(Side::new(team))
}

fn lead_order(lead: usize, size: usize) -> Vec<usize> {
    std::iter::once(lead).chain((0..size).filter(|&i| i != lead)).collect()
}

pub(crate) fn to_action(action: u8) -> Action {
    if action >= SWITCH {
        Action::Switch { to: (action - SWITCH) as usize }
    } else if action >= Z_MOVE {
        Action::Move { slot: (action - Z_MOVE) as usize, z_move: true }
    } else {
        Action::Move { slot: action as usize, z_move: false }
    }
}

fn describe(why: Refusal) -> String {
    match why {
        Refusal::Unported(reason) => format!("unported: {reason}"),
        Refusal::Diverged(reason) => format!("diverged: {reason}"),
    }
}

impl Battle {
    /// A battle waiting on both leads, drawing from `tape` — `Tape::live(seed)` to play, or a
    /// Python recording to check this sequencing against `differential.record_played`. The teams
    /// are checked up front, as `bin/replay.rs` checks them: a bench Pokemon the engine cannot play
    /// must not surface halfway through.
    pub fn new(specs: [Vec<Spec>; 2], tape: Tape, max_turns: i32, db: &Database) -> Result<Self, String> {
        let listed = |side: usize| (0..specs[side].len()).collect::<Vec<_>>();
        let state = State::new(build(&specs[0], &listed(0), db)?, build(&specs[1], &listed(1), db)?);
        for side in &state.sides {
            for pokemon in &side.team {
                if let Some(why) = crate::turn::unsupported_pokemon(pokemon, db) {
                    return Err(why);
                }
            }
        }
        let mut battle = Battle {
            specs,
            state,
            tape,
            phase: Phase::Lead,
            chosen: [None, None],
            max_turns,
            ending: None,
        };
        battle.settle(db);
        Ok(battle)
    }

    pub fn state(&self) -> &State {
        &self.state
    }

    /// Still waiting on leads: its actions are `SWITCH + i` for team member `i`, and name nobody yet.
    pub fn phase_is_lead(&self) -> bool {
        self.phase == Phase::Lead
    }

    pub fn drawn(&self) -> usize {
        self.tape.position()
    }

    pub fn ending(&self) -> Option<&Ending> {
        self.ending.as_ref()
    }

    /// The decisions this battle is waiting on, side 0's first.
    pub fn requests(&self) -> Vec<Request> {
        let open = |decision| (0..2).filter(|&side| self.chosen[side].is_none()).map(move |side| Request { side, decision });
        match self.phase {
            Phase::Lead => open(Decision::Lead).collect(),
            Phase::Turn => open(Decision::Turn).collect(),
            Phase::Pivot(side) | Phase::Replace(side) => vec![Request { side, decision: Decision::Switch }],
            Phase::Over => Vec::new(),
        }
    }

    /// What `side` may answer the request it has open.
    pub fn legal(&self, side: usize, db: &Database) -> Vec<u8> {
        match self.phase {
            Phase::Lead => (0..self.specs[side].len() as u8).map(|i| SWITCH + i).collect(),
            Phase::Over => Vec::new(),
            _ => legal_actions(&self.state, side, db),
        }
    }

    /// Answer `side`'s open request, then play on until a real decision is needed or it ends. An
    /// illegal answer is refused rather than played: the caller's mask and this engine disagree.
    pub fn act(&mut self, side: usize, action: u8, db: &Database) -> Result<(), String> {
        if !self.requests().iter().any(|r| r.side == side) {
            return Err(format!("side {side} has nothing to decide"));
        }
        if !self.legal(side, db).contains(&action) {
            return Err(format!("action {action} is not legal for side {side}"));
        }
        self.answer(side, action, db);
        self.settle(db);
        Ok(())
    }

    fn answer(&mut self, side: usize, action: u8, db: &Database) {
        match self.phase {
            Phase::Lead | Phase::Turn => {
                self.chosen[side] = Some(action);
                if let [Some(first), Some(second)] = self.chosen {
                    self.chosen = [None, None];
                    if self.phase == Phase::Lead {
                        self.send_out_leads(first, second, db);
                    } else {
                        let status = begin_turn(&mut self.state, [to_action(first), to_action(second)], Replacements::Caller, db, &mut self.tape);
                        self.after_turn_step(status);
                    }
                }
            }
            Phase::Pivot(_) => {
                let status = resume_with_switch(&mut self.state, (action - SWITCH) as usize, db, &mut self.tape);
                self.after_turn_step(status);
            }
            Phase::Replace(side) => {
                apply_forced_switch(&mut self.state, side, (action - SWITCH) as usize, db);
                self.phase = self.next_replacement(side);
            }
            Phase::Over => {}
        }
    }

    fn send_out_leads(&mut self, first: u8, second: u8, db: &Database) {
        let order = |side: usize, lead: u8| lead_order((lead - SWITCH) as usize, self.specs[side].len());
        let sides = build(&self.specs[0], &order(0, first), db).and_then(|a| Ok((a, build(&self.specs[1], &order(1, second), db)?)));
        match sides {
            Ok((a, b)) => {
                self.state = State::new(a, b);
                self.phase = Phase::Turn;
            }
            Err(why) => self.finish(Ending::Failed(why)),
        }
    }

    fn after_turn_step(&mut self, status: Result<TurnStatus, Refusal>) {
        match status {
            Ok(TurnStatus::NeedsSwitch { side }) => self.phase = Phase::Pivot(side),
            Ok(TurnStatus::Done(_)) => self.phase = self.next_replacement(0),
            Err(why) => self.finish(Ending::Failed(describe(why))),
        }
    }

    /// `run_battle`'s loop after a turn: side 0 replaces until it stands, then side 1. `from` is
    /// the side being worked through, so side 1's arrival never sends the loop back to side 0.
    fn next_replacement(&self, from: usize) -> Phase {
        if self.state.outcome.is_none() {
            for side in from..2 {
                let own = &self.state.sides[side];
                if own.active_pokemon().fainted() || own.needs_switch {
                    return Phase::Replace(side);
                }
            }
        }
        Phase::Turn
    }

    fn finish(&mut self, ending: Ending) {
        self.phase = Phase::Over;
        self.chosen = [None, None];
        self.ending = Some(ending);
    }

    /// Answer every request with a single legal answer, until one has a real choice or the battle
    /// is over.
    fn settle(&mut self, db: &Database) {
        loop {
            if self.phase == Phase::Turn {
                match self.state.outcome {
                    Some(Outcome::P1Win) => return self.finish(Ending::Won(0)),
                    Some(Outcome::P2Win) => return self.finish(Ending::Won(1)),
                    Some(Outcome::Draw) => return self.finish(Ending::Draw),
                    None if self.state.turn >= self.max_turns && self.chosen == [None, None] => {
                        return self.finish(Ending::TimedOut)
                    }
                    None => {}
                }
            }
            let forced = self.requests().into_iter().find_map(|request| {
                let legal = self.legal(request.side, db);
                (legal.len() == 1).then(|| (request.side, legal[0]))
            });
            match forced {
                Some((side, action)) => self.answer(side, action, db),
                None => return,
            }
        }
    }
}

/// Every open request across a set of battles, encoded: row `r` is battle `env[r]`'s `side[r]`.
pub struct Batch {
    pub env: Vec<i64>,
    pub side: Vec<i64>,
    pub decision: Vec<i64>,
    pub ids: Vec<i64>,
    pub pokemon: Vec<f32>,
    pub field: Vec<f32>,
    pub mask: Vec<bool>,
}

fn decision_index(decision: Decision) -> i64 {
    match decision {
        Decision::Lead => 0,
        Decision::Turn => 1,
        Decision::Switch => 2,
    }
}

/// `requests` over every live battle, observed in parallel.
pub fn observe_all(battles: &[Option<Battle>], db: &Database) -> Batch {
    let rows: Vec<(usize, Request, Observation, Vec<u8>)> = battles
        .par_iter()
        .enumerate()
        .flat_map_iter(|(index, battle)| {
            let requests = battle.as_ref().map(Battle::requests).unwrap_or_default();
            requests.into_iter().map(move |request| {
                let battle = battle.as_ref().expect("only live battles have requests");
                let observation = encode(battle.state(), request.side, request.decision, db);
                (index, request, observation, battle.legal(request.side, db))
            })
        })
        .collect();
    let mut batch = Batch {
        env: Vec::with_capacity(rows.len()),
        side: Vec::with_capacity(rows.len()),
        decision: Vec::with_capacity(rows.len()),
        ids: Vec::new(),
        pokemon: Vec::new(),
        field: Vec::new(),
        mask: Vec::with_capacity(rows.len() * ACTION_SPACE),
    };
    for (index, request, observation, legal) in rows {
        batch.env.push(index as i64);
        batch.side.push(request.side as i64);
        batch.decision.push(decision_index(request.decision));
        batch.ids.extend(observation.ids);
        batch.pokemon.extend(observation.pokemon);
        batch.field.extend(observation.field);
        batch.mask.extend((0..ACTION_SPACE as u8).map(|action| legal.contains(&action)));
    }
    batch
}

/// Apply `(env, side, action)` answers, each battle's in the order given, battles in parallel.
/// Returns the first refusal, if any, after applying everything else.
pub fn act_all(battles: &mut [Option<Battle>], answers: &[(usize, usize, u8)], db: &Database) -> Result<(), String> {
    let mut per_battle: Vec<Vec<(usize, u8)>> = vec![Vec::new(); battles.len()];
    for &(env, side, action) in answers {
        per_battle.get_mut(env).ok_or_else(|| format!("no battle {env}"))?.push((side, action));
    }
    let refusals: Vec<String> = battles
        .par_iter_mut()
        .zip(per_battle.into_par_iter())
        .filter_map(|(battle, answers)| {
            if answers.is_empty() {
                return None;
            }
            let Some(battle) = battle.as_mut() else { return Some("that battle is not running".to_string()) };
            answers.into_iter().find_map(|(side, action)| battle.act(side, action, db).err())
        })
        .collect();
    refusals.into_iter().next().map_or(Ok(()), Err)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use std::path::Path;

    fn database() -> Database {
        Database::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("data")).expect("rust/data loads")
    }

    fn team() -> Vec<Spec> {
        let spec = |species: &str, nickname: &str, moves: &[&str]| {
            let spread = |value: i32| {
                serde_json::json!({"HP": value, "ATTACK": value, "DEFENCE": value, "SP_ATTACK": value,
                                   "SP_DEFENCE": value, "SPEED": value})
            };
            serde_json::from_value::<Spec>(serde_json::json!({
                "species": species, "nickname": nickname, "level": 50, "moves": moves,
                "nature": "HARDY", "ability": "NONE", "item": "NONE",
                "effort_values": spread(0), "individual_values": spread(31),
            }))
            .expect("a spec")
        };
        vec![
            spec("Tauros", "P0", &["Tackle", "U-turn"]),
            spec("Rhydon", "P1", &["Rock Slide", "Earthquake"]),
            spec("Machamp", "P2", &["Cross Chop", "Stealth Rock"]),
        ]
    }

    #[test]
    fn random_play_always_reaches_an_ending_through_masked_decisions() {
        let db = database();
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        for seed in 0..50 {
            let mut battle = Battle::new([team(), team()], Tape::live(seed), 300, &db).expect("playable");
            while battle.ending().is_none() {
                let requests = battle.requests();
                assert!(!requests.is_empty(), "a running battle is waiting on someone");
                for request in requests {
                    let legal = battle.legal(request.side, &db);
                    assert!(legal.len() > 1, "a single-answer decision was asked: {legal:?}");
                    battle.act(request.side, legal[rng.gen_range(0..legal.len())], &db).expect("legal");
                }
            }
            assert!(!matches!(battle.ending(), Some(Ending::Failed(_))), "{:?}", battle.ending());
        }
    }

    #[test]
    fn a_lead_choice_sends_that_pokemon_out_first() {
        let db = database();
        let mut battle = Battle::new([team(), team()], Tape::live(0), 300, &db).expect("playable");
        battle.act(0, SWITCH + 2, &db).expect("legal");
        battle.act(1, SWITCH + 1, &db).expect("legal");
        assert_eq!(battle.state().sides[0].active_pokemon().nickname, "P2");
        assert_eq!(battle.state().sides[1].active_pokemon().nickname, "P1");
    }

    #[test]
    fn an_illegal_answer_is_refused() {
        let db = database();
        let mut battle = Battle::new([team(), team()], Tape::live(0), 300, &db).expect("playable");
        assert!(battle.act(0, 0, &db).is_err(), "a move is no lead");
    }
}
