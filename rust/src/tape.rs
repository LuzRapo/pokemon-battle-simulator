//! The recorded randomness, replayed.
//!
//! The Python side records every draw its RNG made; this hands them back in the same order. If
//! this engine asks for a draw the recording does not have, the two have taken different paths —
//! which is a divergence to report, never a shortage to paper over with a fresh random number.

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Draw {
    Probability(f64),
    Integer(i64),
}

#[derive(Debug)]
pub struct Tape {
    draws: Vec<Draw>,
    position: usize,
}

#[derive(Debug)]
pub struct Exhausted(pub String);

impl From<Exhausted> for crate::turn::Unsupported {
    fn from(e: Exhausted) -> Self {
        crate::turn::Unsupported(e.0)
    }
}

impl Tape {
    pub fn new(draws: Vec<Draw>) -> Self {
        Tape { draws, position: 0 }
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

    pub fn probability(&mut self) -> Result<f64, Exhausted> {
        match self.next()? {
            Draw::Probability(value) => Ok(value),
            Draw::Integer(value) => Err(Exhausted(format!(
                "draw {} was recorded as the integer {value}, and a probability was asked for -- \
                 the engines are taking different paths",
                self.position - 1
            ))),
        }
    }

    pub fn integer(&mut self, _low: i32, _high: i32) -> Result<i32, Exhausted> {
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
