//! The events a turn emits, in the shape `differential.entry_json` writes them.
//!
//! Field names and the `type` tag have to match the Python dataclasses exactly, because the
//! comparison is a plain equality check on parsed JSON. An event this engine spells differently is
//! indistinguishable from one it got wrong.

use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum Event {
    MoveUsed {
        side: i32,
        pokemon: String,
        #[serde(rename = "move")]
        the_move: String,
        // Always written, never skipped: the Python dataclass has the field and `entry_json`
        // emits it as null, so omitting it here is a difference the comparator rightly reports.
        unleashed_as: Option<String>,
    },
    MoveMissed,
    NoEffect {
        side: i32,
        pokemon: String,
    },
    Effectiveness {
        level: String,
    },
    CriticalHit,
    DamageDealt {
        side: i32,
        pokemon: String,
        amount: i32,
    },
    Fainted {
        side: i32,
        pokemon: String,
    },
    Switched {
        side: i32,
        withdrew: String,
        sent_out: String,
    },
    BattleEnded {
        outcome: String,
    },
}

#[derive(Debug, Default)]
pub struct Log {
    pub entries: Vec<Event>,
}

impl Log {
    pub fn new() -> Self {
        Log::default()
    }

    pub fn push(&mut self, event: Event) {
        self.entries.push(event);
    }

    /// Named and thresholded exactly as `_log_effectiveness` does it. The words are lowercase
    /// literals in the Python, not enum members, and the bands are `>= 2` and `<= 0.5` rather than
    /// "not 1" — which happens to agree for the powers of two a type chart produces, but would not
    /// if anything ever multiplied to 1.5.
    pub fn effectiveness(&mut self, multiplier: f64) {
        let level = if multiplier >= 2.0 {
            "super"
        } else if multiplier <= 0.5 {
            "resisted"
        } else {
            return;
        };
        self.push(Event::Effectiveness { level: level.into() });
    }
}
