//! Whole battles piloted by `matchup::MatchupAi` on both sides, many at once.
//!
//! One battle runs as `runner.run_battle` runs one: each side orders its team with `choose_order`,
//! both pick an action each turn, a pivot's replacement is answered mid-turn by the side that has
//! to switch, and faint replacements follow the turn, side 0 then side 1, each repeating while
//! hazards knock out the arrival. What it returns is what `species_rating` writes down: the result,
//! the turn count and each side's survivors.

use crate::battle::{Outcome, Pokemon, Side, Spec, State};
use crate::choices::legal_actions;
use crate::data::Database;
use crate::env::to_action;
use crate::matchup::{choose_order, MatchupAi, Weights};
use crate::tape::Tape;
use crate::turn::{apply_forced_switch, begin_turn, resume_with_switch, unsupported_pokemon, Replacements, TurnStatus};
use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct Played {
    /// `None` when the turn cap came first.
    pub outcome: Option<Outcome>,
    pub turns: i32,
    pub survivors: [usize; 2],
}

fn build(specs: &[Spec], db: &Database) -> Result<Vec<Pokemon>, String> {
    specs.iter().map(|spec| Pokemon::build(spec, db)).collect()
}

fn ordered(specs: &[Spec], order: &[usize], db: &Database) -> Result<Side, String> {
    Ok(Side::new(order.iter().map(|&i| Pokemon::build(&specs[i], db)).collect::<Result<Vec<_>, _>>()?))
}

/// One battle. An `Err` is a battle the engine could not play: a Pokemon it refuses up front, or a
/// mechanic it meets partway through that it has not learned.
pub fn play(teams: &[Vec<Spec>; 2], seed: u64, weights: Weights, max_turns: i32, db: &Database) -> Result<Played, String> {
    let orders = [
        choose_order(build(&teams[0], db)?, build(&teams[1], db)?, db),
        choose_order(build(&teams[1], db)?, build(&teams[0], db)?, db),
    ];
    let mut state = State::new(ordered(&teams[0], &orders[0], db)?, ordered(&teams[1], &orders[1], db)?);
    for side in &state.sides {
        if let Some(why) = side.team.iter().find_map(|p| unsupported_pokemon(p, db)) {
            return Err(why);
        }
    }
    let mut tape = Tape::live(seed);
    let mut players = [MatchupAi::new(weights), MatchupAi::new(weights)];
    let pick = |state: &State, side: usize, players: &mut [MatchupAi; 2]| {
        let legal = legal_actions(state, side, db);
        players[side].choose(state, side, &legal, db)
    };
    while state.outcome.is_none() && state.turn < max_turns {
        let actions = [pick(&state, 0, &mut players), pick(&state, 1, &mut players)];
        let mut status = begin_turn(&mut state, [to_action(actions[0]), to_action(actions[1])], Replacements::Caller, db, &mut tape)
            .map_err(|why| why.reason().to_string())?;
        while let TurnStatus::NeedsSwitch { side } = status {
            let to = pick(&state, side, &mut players) - crate::choices::SWITCH;
            status = resume_with_switch(&mut state, to as usize, db, &mut tape).map_err(|why| why.reason().to_string())?;
        }
        for side in 0..2 {
            while state.outcome.is_none() && {
                let own = &state.sides[side];
                own.active_pokemon().fainted() || own.needs_switch
            } {
                let to = pick(&state, side, &mut players) - crate::choices::SWITCH;
                apply_forced_switch(&mut state, side, to as usize, db);
            }
        }
    }
    let survivors = [0, 1].map(|side| state.sides[side].team.iter().filter(|p| !p.fainted()).count());
    Ok(Played { outcome: state.outcome, turns: state.turn, survivors })
}

/// Many battles in parallel, each `(teams, seed)`, results in the same order.
pub fn play_all(
    battles: &[([Vec<Spec>; 2], u64)],
    weights: Weights,
    max_turns: i32,
    db: &Database,
) -> Vec<Result<Played, String>> {
    battles.par_iter().map(|(teams, seed)| play(teams, *seed, weights, max_turns, db)).collect()
}
