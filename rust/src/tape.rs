//! The randomness a battle consumes — recorded-and-replayed for differential testing, or freshly
//! drawn for a battle nothing else is watching.
//!
//! Differential testing records every draw the Python's RNG made and hands them back in the same
//! order: if this engine asks for a draw the recording does not have, the two have taken different
//! paths, which is a divergence to report, never a shortage to paper over with a fresh random
//! number. Self-play has no Python run alongside to have recorded one — there `Tape::live` draws
//! from this engine's own seeded RNG instead. Every call site in the engine goes through the same
//! `probability`/`integer` pair either way, so nothing downstream of this module needs to know or
//! care which mode it is running in.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Draw {
    Probability(f64),
    Integer(i64),
}

#[derive(Debug)]
pub struct Tape {
    draws: Vec<Draw>,
    position: usize,
    live: Option<StdRng>,
}

#[derive(Debug)]
pub struct Exhausted(pub String);

impl From<Exhausted> for crate::turn::Refusal {
    fn from(e: Exhausted) -> Self {
        // Always a divergence, never a missing feature: the tape only runs short or changes
        // kind when this engine has asked for randomness the Python did not take here.
        crate::turn::Refusal::Diverged(e.0)
    }
}

impl Tape {
    pub fn new(draws: Vec<Draw>) -> Self {
        Tape { draws, position: 0, live: None }
    }

    /// A tape with nothing recorded to replay: every draw this battle takes is freshly generated
    /// from a `StdRng` seeded here, instead of read back from a Python run. `low`/`high` on
    /// `integer` — dead weight in replay mode, since the recorded value is returned regardless of
    /// what a call site claims its own range is — become load-bearing under this mode, so a call
    /// site's bounds have to be right rather than merely documentary.
    pub fn live(seed: u64) -> Self {
        Tape { draws: Vec::new(), position: 0, live: Some(StdRng::seed_from_u64(seed)) }
    }

    pub fn position(&self) -> usize {
        self.position
    }

    pub fn len(&self) -> usize {
        self.draws.len()
    }

    pub fn is_empty(&self) -> bool {
        self.draws.is_empty()
    }

    fn next(&mut self) -> Result<Draw, Exhausted> {
        let draw = self.draws.get(self.position).copied().ok_or_else(|| {
            Exhausted(format!(
                "the recording holds {} draws and a {}th was asked for",
                self.draws.len(),
                self.position + 1
            ))
        })?;
        self.position += 1;
        Ok(draw)
    }

    /// A uniform draw in `[0, 1)`, matching Python's own `random.random()` — the same shape every
    /// `ON_DAMAGE_CALC`/status/miss roll in the engine already compares against a threshold with.
    pub fn probability(&mut self) -> Result<f64, Exhausted> {
        if let Some(rng) = &mut self.live {
            self.position += 1;
            return Ok(rng.gen::<f64>());
        }
        match self.next()? {
            Draw::Probability(value) => Ok(value),
            Draw::Integer(value) => Err(Exhausted(format!(
                "draw {} was recorded as the integer {value}, and a probability was asked for -- \
                 the engines are taking different paths",
                self.position - 1
            ))),
        }
    }

    /// A uniform draw in `[low, high)` — every call site already writes its bounds this way (see
    /// e.g. `turn.rs`'s `tape.integer(85, 101)` for the 85-100 damage roll), matching Python's own
    /// `random.randrange(low, high)`.
    pub fn integer(&mut self, low: i32, high: i32) -> Result<i32, Exhausted> {
        if let Some(rng) = &mut self.live {
            self.position += 1;
            return Ok(rng.gen_range(low..high));
        }
        match self.next()? {
            Draw::Integer(value) => Ok(value as i32),
            Draw::Probability(value) => Err(Exhausted(format!(
                "draw {} was recorded as the probability {value}, and an integer was asked for -- \
                 the engines are taking different paths",
                self.position - 1
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn live_probability_stays_in_unit_range() {
        let mut tape = Tape::live(42);
        for _ in 0..10_000 {
            let value = tape.probability().unwrap();
            assert!((0.0..1.0).contains(&value), "{value} outside [0, 1)");
        }
        assert_eq!(tape.position(), 10_000);
    }

    #[test]
    fn live_integer_respects_its_half_open_bounds() {
        let mut tape = Tape::live(7);
        for _ in 0..10_000 {
            let value = tape.integer(85, 101).unwrap();
            assert!((85..101).contains(&value), "{value} outside [85, 101)");
        }
    }

    #[test]
    fn live_tape_never_reads_a_recording_and_never_exhausts() {
        // A live tape has no recording at all -- confirms `next()` (the replay-only path) is never
        // reached from either accessor once `live` is set, however many draws are taken.
        let mut tape = Tape::live(1);
        assert!(tape.is_empty());
        for _ in 0..5 {
            tape.probability().unwrap();
            tape.integer(0, 6).unwrap();
        }
    }

    #[test]
    fn two_live_tapes_with_the_same_seed_agree() {
        let mut a = Tape::live(99);
        let mut b = Tape::live(99);
        for _ in 0..100 {
            assert_eq!(a.probability().unwrap(), b.probability().unwrap());
            assert_eq!(a.integer(0, 1000).unwrap(), b.integer(0, 1000).unwrap());
        }
    }
}
