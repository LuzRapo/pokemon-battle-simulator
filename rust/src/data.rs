//! The engine's data, as Python normalised it.
//!
//! Nothing here interprets Showdown's raw files. `battle_sim/export_data.py` already decided which
//! of Showdown's fields map to which effect, which volatiles are modelled and which are skipped,
//! and wrote the answer out. Reading that instead of re-deriving it is the difference between one
//! interpretation of the game and two that have to be kept in agreement by hand.

use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Exactly the shape `export_data.py` writes. A field it does not know about is ignored rather
/// than fatal, so adding one to the Python side never breaks the build here — but a field this
/// engine *needs* and cannot find is a hard error at load, not a silent default at turn 40.
#[derive(Debug, Clone, Deserialize)]
pub struct Species {
    pub name: String,
    pub base_stats: BaseStats,
    /// Two entries, the second `None` for a monotype. Kept as written rather than flattened on
    /// load: "has one type" and "has two, the second of which happens to be missing" are the same
    /// thing here, and the Python side writes the pair.
    pub types: Vec<Option<String>>,
    #[serde(default)]
    pub weight_kg: f64,
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub struct BaseStats {
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

// `Default` only under test: it lets a unit test name the two or three fields it cares about
// instead of every field on the struct, which is what let the priority test rot unnoticed. Outside
// tests a Move must come from the exported data, so the derive stays out of production builds.
#[derive(Debug, Clone, Deserialize)]
#[cfg_attr(test, derive(Default))]
pub struct Move {
    pub name: String,
    #[serde(default)]
    pub defrosts_user: bool,
    /// The short off-type list that thaws whoever it hits: Scald, Steam Eruption, Matcha Gotcha.
    /// Every damaging Fire move does it too, without being on the list.
    #[serde(default)]
    pub thaws_target: bool,
    #[serde(rename = "type")]
    pub move_type: String,
    pub category: String,
    pub accuracy_probability: Option<f64>,
    pub priority: String,
    pub pp: i32,
    pub target: String,
    #[serde(default)]
    pub effects: Vec<Effect>,
    #[serde(default)]
    pub protectable: bool,
    #[serde(default)]
    pub healing: bool,
    #[serde(default)]
    pub typeless: bool,
    #[serde(default)]
    pub self_switch: bool,
    #[serde(default)]
    pub force_switch: bool,
    #[serde(default)]
    pub recharges: bool,
    #[serde(default)]
    pub charge: bool,
    #[serde(default)]
    pub self_destructs: bool,
    /// PS `hasCrashDamage`: (High) Jump Kick loses half its user's max HP when it fails.
    pub has_crash_damage: bool,
}

/// Effects are a tagged union in the export (`kind` names the Python dataclass). Only the shapes
/// this engine has learned to resolve are modelled; anything else parses as `Unmodelled` and is
/// refused loudly when a battle tries to use it, rather than silently doing nothing.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind")]
pub enum Effect {
    DamageEffect {
        /// `None` for the variable-power moves — Gyro Ball, Low Kick, Return and the rest compute
        /// theirs from the board at the moment they land, so the data cannot hold a number.
        power: Option<i32>,
        category: String,
        #[serde(default)]
        crit_stage: i32,
        #[serde(default)]
        contact: bool,
        drain_percent: Option<f64>,
        recoil_percent: Option<f64>,
        multi_hit: Option<serde_json::Value>,
        #[serde(default)]
        struggle_recoil: bool,
    },
    InflictStatusEffect {
        status: String,
        probability: f64,
        #[serde(default)]
        to_self: bool,
        #[serde(default)]
        is_secondary: bool,
    },
    StatStageChangeEffect {
        /// Ordered pairs, not a map: one log entry is emitted per stat in this order, so the
        /// order is mechanics. See the note in `export_data.move_json`.
        stages: Vec<(String, i32)>,
        probability: f64,
        target: String,
        #[serde(default)]
        is_secondary: bool,
    },
    #[serde(other)]
    Unmodelled,
}

#[derive(Debug, Clone, Deserialize)]
pub struct NatureEffect {
    pub up: String,
    pub down: String,
}

#[derive(Debug, Deserialize)]
struct MovesFile {
    moves: HashMap<String, Move>,
}

#[derive(Debug, Deserialize)]
struct SpeciesFile {
    species: HashMap<String, Species>,
}

#[derive(Debug, Deserialize)]
struct RulesFile {
    coded_moves: Vec<String>,
    live_abilities: Vec<String>,
    live_items: Vec<String>,
    types: Vec<String>,
    type_chart: HashMap<String, HashMap<String, f64>>,
    natures: HashMap<String, NatureEffect>,
}

/// Every table the engine needs, loaded once.
#[derive(Debug)]
pub struct Database {
    pub moves: HashMap<String, Move>,
    pub species: HashMap<String, Species>,
    /// Moves whose power is computed from the board rather than read from the data — Revenge,
    /// Gyro Ball, Weather Ball and the rest. Exported by Python rather than listed here, because a
    /// second copy of this list is a second thing to keep in step.
    pub coded_moves: std::collections::HashSet<String>,
    /// What the Python wires to its event bus. Anything in here that this engine has not
    /// implemented makes a scenario unplayable rather than quietly inert.
    pub live_abilities: std::collections::HashSet<String>,
    pub live_items: std::collections::HashSet<String>,
    pub types: Vec<String>,
    pub type_chart: HashMap<String, HashMap<String, f64>>,
    pub natures: HashMap<String, NatureEffect>,
}

impl Database {
    pub fn load(directory: &Path) -> Result<Self, String> {
        let read = |name: &str| -> Result<String, String> {
            fs::read_to_string(directory.join(name)).map_err(|e| format!("{name}: {e}"))
        };
        let moves: MovesFile =
            serde_json::from_str(&read("moves.json")?).map_err(|e| format!("moves.json: {e}"))?;
        let species: SpeciesFile =
            serde_json::from_str(&read("species.json")?).map_err(|e| format!("species.json: {e}"))?;
        let rules: RulesFile =
            serde_json::from_str(&read("rules.json")?).map_err(|e| format!("rules.json: {e}"))?;
        Ok(Database {
            moves: moves.moves,
            species: species.species,
            coded_moves: rules.coded_moves.into_iter().collect(),
            live_abilities: rules.live_abilities.into_iter().collect(),
            live_items: rules.live_items.into_iter().collect(),
            types: rules.types,
            type_chart: rules.type_chart,
            natures: rules.natures,
        })
    }

    /// Normalised the way Python's `normalize_id` does it: lowercase, letters and digits only.
    /// Both engines have to agree on what "Tapu Fini" and "tapufini" mean or half the lookups miss.
    pub fn normalize_id(name: &str) -> String {
        name.chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .map(|c| c.to_ascii_lowercase())
            .collect()
    }

    pub fn species_named(&self, name: &str) -> Option<&Species> {
        self.species
            .get(name)
            .or_else(|| self.species.get(&Self::normalize_id(name)))
    }

    pub fn move_named(&self, name: &str) -> Option<&Move> {
        self.moves
            .get(name)
            .or_else(|| self.moves.get(&Self::normalize_id(name)))
    }

    /// One attacking type against one or two defending types, multiplied out.
    pub fn effectiveness(&self, attacking: &str, defending: &[Option<String>]) -> f64 {
        defending
            .iter()
            .flatten()
            .map(|d| {
                self.type_chart
                    .get(attacking)
                    .and_then(|row| row.get(d))
                    .copied()
                    .unwrap_or(1.0)
            })
            .product()
    }
}
