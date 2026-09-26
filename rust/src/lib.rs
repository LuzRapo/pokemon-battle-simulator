//! A second implementation of the battle engine, fast enough to make experiments free.
//!
//! The Python engine in `battle_sim/` stays the reference: it is the one Discord talks to, and it
//! is the one this is checked against. The contract between them is
//! `battle_sim/differential.py` — a recorded battle, the tape of random draws it consumed, and the
//! trace it produced. This engine passes when it replays that tape into the same trace, turn for
//! turn, event for event.
//!
//! Nothing here re-interprets Showdown's data files; see `data.rs`.

pub mod abilities;
pub mod battle;
pub mod damage;
pub mod data;
pub mod log;
pub mod stats;
pub mod tape;
pub mod turn;
