//! The board: what a Pokémon is, what a side holds, and what a turn can change.
//!
//! Mirrors `battle_sim/models/pokemon.py` and `mechanics/battle.py` closely enough that the
//! differential harness can compare them field by field. Where a name differs from the Python it
//! is because Rust convention demands it (`hp` for `live_stats.HP`), never because the meaning
//! differs — the digest in `battle_sim/differential.py` is the list of things that must agree.

use crate::data::{Database, Species};
use crate::stats::{totals, with_stage, Spread, StatTotals};
use serde::Deserialize;
use std::collections::BTreeMap;

/// A non-volatile status. The names match the Python enum exactly because the trace compares them
/// as strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    None,
    Burn,
    Freeze,
    Paralysis,
    Poison,
    Toxic,
    Sleep,
}

impl Status {
    pub fn parse(name: &str) -> Option<Status> {
        Some(match name {
            "NONE" => Status::None,
            "BURN" => Status::Burn,
            "FREEZE" => Status::Freeze,
            "PARALYSIS" => Status::Paralysis,
            "POISON" => Status::Poison,
            "TOXIC" => Status::Toxic,
            "SLEEP" => Status::Sleep,
            _ => return None,
        })
    }

    /// The types that can never take this status, from `_STATUS_TYPE_IMMUNITY`. Corrosion is the
    /// one exception and belongs with the abilities, which are not ported yet.
    pub fn immune_types(&self) -> &'static [&'static str] {
        match self {
            Status::Burn => &["FIRE"],
            Status::Paralysis => &["ELECTRIC"],
            Status::Freeze => &["ICE"],
            Status::Poison | Status::Toxic => &["POISON", "STEEL"],
            _ => &[],
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Status::None => "NONE",
            Status::Burn => "BURN",
            Status::Freeze => "FREEZE",
            Status::Paralysis => "PARALYSIS",
            Status::Poison => "POISON",
            Status::Toxic => "TOXIC",
            Status::Sleep => "SLEEP",
        }
    }
}

/// One Pokémon on the field or the bench.
#[derive(Debug, Clone)]
pub struct Pokemon {
    pub nickname: String,
    pub species_name: String,
    pub level: i32,
    pub types: Vec<Option<String>>,
    pub totals: StatTotals,
    pub hp: i32,
    pub status: Status,
    pub status_turns: i32,
    pub item: String,
    pub item_consumed: bool,
    pub ability: String,
    pub stages: BTreeMap<String, i32>,
    pub volatiles: BTreeMap<String, i32>,
    pub moves: Vec<String>,
    pub pp: BTreeMap<String, i32>,
    pub lives_used: i32,
    pub made_last_stand: bool,
    /// True from when this Pokemon was sent out until the end of that turn. Stakeout reads it.
    pub just_switched_in: bool,
    /// When this Pokemon's effects were registered on the Python's bus. Handlers at equal priority
    /// fire in registration order, and a fold of `chain` modifiers is not commutative, so the
    /// order two abilities push into the same list is worth a point of damage.
    pub registered_at: u64,
    /// Dauntless Shield and Intrepid Sword fire once per battle, not once per switch-in.
    pub switch_in_boost_used: bool,
    /// What the last hit this turn took off, and whether it was physical or special. Counter and
    /// Mirror Coat are the readers; the Python clears both at the top of every turn, so they mean
    /// "this turn" rather than "ever".
    /// How many hits this Pokemon has taken all battle (Rage Fist) and how many times its
    /// current run of Fury Cutter has connected.
    pub times_hit: i32,
    pub rolling_hits: i32,
    /// The species' base Attack, which Beat Up sums over the whole team, and its weight, which
    /// Low Kick and Heavy Slam weigh against each other.
    pub base_attack: i32,
    pub weight_kg: f64,
    pub last_hit_taken: i32,
    pub last_hit_category: Option<String>,
    /// Consecutive stalling moves. Protect-likes fail with odds 1 - 1/3^n, and the counter is
    /// broken by anything else the Pokemon does — including a turn it could not act on.
    pub protect_streak: i32,
}

pub const STAGE_NAMES: [&str; 7] = [
    "ATTACK",
    "DEFENCE",
    "SP_ATTACK",
    "SP_DEFENCE",
    "SPEED",
    "ACCURACY",
    "EVASION",
];
pub const SLOT_NAMES: [&str; 4] = ["FIRST", "SECOND", "THIRD", "FOURTH"];

/// The spec a scenario carries, in the shape `differential.encode_spec` writes.
#[derive(Debug, Clone, Deserialize)]
pub struct Spec {
    pub species: String,
    pub nickname: Option<String>,
    pub level: i32,
    pub ability: String,
    pub item: String,
    pub nature: String,
    pub effort_values: SpreadJson,
    pub individual_values: SpreadJson,
    pub moves: Vec<String>,
    #[serde(default)]
    pub pp_ups: i32,
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub struct SpreadJson {
    #[serde(rename = "HP")]
    pub hp: i32,
    #[serde(rename = "ATTACK")]
    pub attack: i32,
    #[serde(rename = "DEFENCE")]
    pub defence: i32,
    #[serde(rename = "SP_ATTACK")]
    pub sp_attack: i32,
    #[serde(rename = "SP_DEFENCE")]
    pub sp_defence: i32,
    #[serde(rename = "SPEED")]
    pub speed: i32,
}

impl From<SpreadJson> for Spread {
    fn from(s: SpreadJson) -> Self {
        Spread {
            hp: s.hp,
            attack: s.attack,
            defence: s.defence,
            sp_attack: s.sp_attack,
            sp_defence: s.sp_defence,
            speed: s.speed,
        }
    }
}

impl Pokemon {
    pub fn build(spec: &Spec, db: &Database) -> Result<Pokemon, String> {
        let species: &Species = db
            .species_named(&spec.species)
            .ok_or_else(|| format!("unknown species {:?}", spec.species))?;
        let nature = db
            .natures
            .get(&spec.nature)
            .ok_or_else(|| format!("unknown nature {:?}", spec.nature))?;
        let stat_totals = totals(
            &species.base_stats,
            &spec.individual_values.into(),
            &spec.effort_values.into(),
            spec.level,
            nature,
        );
        // Four slots, always. `build_pokemon` pads a short set by repeating the *first* move
        // rather than leaving the slot empty, so a two-move Pokemon really does carry four moves
        // and four PP counters — which is what the digest compares, and how this was found.
        let mut filled: Vec<String> = spec.moves.clone();
        while filled.len() < 4 {
            filled.push(filled.first().cloned().unwrap_or_else(|| "Tackle".to_string()));
        }
        // PP Ups, as `Pokemon._init_pp` applies them: each is a fifth of the listed PP, floored
        // after the multiplication rather than per-up.
        let bonus = 1.0 + spec.pp_ups as f64 / 5.0;
        let mut pp = BTreeMap::new();
        for (index, name) in filled.iter().enumerate() {
            let listed = db
                .move_named(name)
                .ok_or_else(|| format!("unknown move {name:?}"))?
                .pp;
            pp.insert(SLOT_NAMES[index].to_string(), (listed as f64 * bonus) as i32);
        }
        Ok(Pokemon {
            nickname: spec.nickname.clone().unwrap_or_else(|| species.name.clone()),
            species_name: species.name.clone(),
            level: spec.level,
            types: species.types.clone(),
            totals: stat_totals,
            hp: stat_totals.hp,
            status: Status::None,
            status_turns: 0,
            item: spec.item.clone(),
            item_consumed: false,
            ability: spec.ability.clone(),
            stages: STAGE_NAMES.iter().map(|s| (s.to_string(), 0)).collect(),
            volatiles: BTreeMap::new(),
            moves: filled,
            pp,
            lives_used: 0,
            made_last_stand: false,
            // True from the moment it is built, as the Python's field default is: a lead has
            // just as much arrived as anything switched in later, and Stakeout punishes it on turn
            // zero exactly the same way.
            just_switched_in: true,
            registered_at: 0,
            switch_in_boost_used: false,
            times_hit: 0,
            rolling_hits: 0,
            base_attack: species.base_stats.attack,
            weight_kg: species.weight_kg,
            last_hit_taken: 0,
            last_hit_category: None,
            protect_streak: 0,
        })
    }

    pub fn fainted(&self) -> bool {
        self.hp <= 0
    }

    pub fn stage(&self, name: &str) -> i32 {
        self.stages.get(name).copied().unwrap_or(0)
    }

    pub fn stat(&self, name: &str) -> i32 {
        match name {
            "ATTACK" => self.totals.attack,
            "DEFENCE" => self.totals.defence,
            "SP_ATTACK" => self.totals.sp_attack,
            "SP_DEFENCE" => self.totals.sp_defence,
            "SPEED" => self.totals.speed,
            "HP" => self.totals.hp,
            other => panic!("no such stat {other}"),
        }
    }

    pub fn effective(&self, name: &str) -> i32 {
        with_stage(self.stat(name), self.stage(name))
    }

    /// Damage, clamped at zero, returning how much actually landed. Nine Lives is deliberately not
    /// here yet: it belongs with the ability work, and until then a butler in a Rust battle simply
    /// faints — which the differential harness will say, loudly, the moment one is in a scenario.
    pub fn take_damage(&mut self, amount: i32) -> i32 {
        let before = self.hp;
        self.hp = (self.hp - amount.abs()).max(0);
        before - self.hp
    }
}

#[derive(Debug, Clone)]
pub struct Side {
    pub team: Vec<Pokemon>,
    pub active: usize,
    pub hazards: BTreeMap<String, i32>,
    pub screens: BTreeMap<String, i32>,
    pub tailwind_turns: i32,
    /// Whether this side has already taken its action this turn. Analytic reads it.
    pub acted_this_turn: bool,
}

impl Side {
    pub fn new(team: Vec<Pokemon>) -> Self {
        Side {
            team,
            active: 0,
            hazards: BTreeMap::new(),
            screens: BTreeMap::new(),
            tailwind_turns: 0,
            acted_this_turn: false,
        }
    }

    pub fn active_pokemon(&self) -> &Pokemon {
        &self.team[self.active]
    }

    pub fn active_mut(&mut self) -> &mut Pokemon {
        &mut self.team[self.active]
    }

    pub fn all_fainted(&self) -> bool {
        self.team.iter().all(|p| p.fainted())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    P1Win,
    P2Win,
    Draw,
}

impl Outcome {
    pub fn name(&self) -> &'static str {
        match self {
            Outcome::P1Win => "P1_WIN",
            Outcome::P2Win => "P2_WIN",
            Outcome::Draw => "DRAW",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Field {
    pub weather: String,
    pub weather_turns_left: i32,
    pub terrain: String,
    pub terrain_turns_left: i32,
}

impl Default for Field {
    fn default() -> Self {
        Field {
            weather: "NONE".into(),
            weather_turns_left: 0,
            terrain: "NONE".into(),
            terrain_turns_left: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct State {
    pub sides: [Side; 2],
    pub field: Field,
    pub turn: i32,
    pub outcome: Option<Outcome>,
    /// Hands out `registered_at` stamps. Side 0's lead registers before side 1's, which is the
    /// order `BattleState` wires them up in.
    pub registrations: u64,
}

impl State {
    pub fn new(first: Side, second: Side) -> Self {
        let mut state =
            State { sides: [first, second], field: Field::default(), turn: 0, outcome: None, registrations: 0 };
        for side in 0..2 {
            state.register_active(side);
        }
        state
    }

    /// Stamp whoever is active on this side as freshly registered.
    pub fn register_active(&mut self, side: usize) {
        self.registrations += 1;
        let stamp = self.registrations;
        self.sides[side].active_mut().registered_at = stamp;
    }

    /// The Python's `_update_outcome`: a side with nothing left standing has lost, and both at once
    /// is a draw.
    pub fn update_outcome(&mut self) {
        if self.outcome.is_some() {
            return;
        }
        let (first, second) = (self.sides[0].all_fainted(), self.sides[1].all_fainted());
        self.outcome = match (first, second) {
            (true, true) => Some(Outcome::Draw),
            (true, false) => Some(Outcome::P2Win),
            (false, true) => Some(Outcome::P1Win),
            _ => None,
        };
    }
}
