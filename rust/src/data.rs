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
    /// Eviolite asks this and nothing else does yet.
    #[serde(default)]
    pub fully_evolved: bool,
    /// The species a forme's mega stone (or Primal orb, or a Rusted Sword/Shield) reaches from —
    /// `None` for a species that is not a forme reached that way. Read by `is_fused_to`.
    #[serde(default)]
    pub base_species: Option<String>,
    /// The item that *is* this forme rather than something it merely holds — already resolved to
    /// this engine's own item name by the export, not Showdown's display name. Read by
    /// `is_fused_to`, which asks it of every forme sharing a base species, not just this one.
    #[serde(default)]
    pub fused_item: Option<String>,
    /// This forme's first regular ability, already in this engine's own name — `None` if it has
    /// none listed or Python does not model it. Read by a forme swap (Mega Evolution, Primal
    /// Reversion, Ultra Burst) to know what ability the new forme hands over, and by `formes`'s own
    /// playability check to refuse a swap into an ability this engine has not ported, rather than
    /// silently handing out one nothing will ever dispatch.
    #[serde(default)]
    pub regular_ability: Option<String>,
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
    /// Magic Bounce sends this move back at its user.
    #[serde(default)]
    pub reflectable: bool,
    #[serde(default)]
    pub healing: bool,
    #[serde(default)]
    pub typeless: bool,
    // The move flags the abilities read. Each names a class of move — Sharpness boosts every
    // `slicing` one, Iron Fist every `punching` one — so they are carried through rather than
    // re-derived from a list of names in a second place.
    /// Protect and its relatives, which fail when used twice running.
    #[serde(default)]
    pub stalling: bool,
    #[serde(default)]
    pub slicing: bool,
    #[serde(default)]
    pub punching: bool,
    #[serde(default)]
    pub biting: bool,
    #[serde(default)]
    pub pulse: bool,
    #[serde(default)]
    pub sound: bool,
    /// Bulletproof's class of move.
    #[serde(default)]
    pub bullet: bool,
    /// PS flag `powder`: no effect on a Grass type, or on Overcoat.
    #[serde(default)]
    pub powder: bool,
    /// PS flag `wind`: absorbed by Wind Rider.
    #[serde(default)]
    pub wind: bool,
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
    /// PS `bypasssub`: sound moves, Chatter and the rest reach straight through a Substitute.
    #[serde(default)]
    pub bypass_substitute: bool,
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
        /// `[low, high]` — an inclusive span, equal ends meaning a fixed count.
        multi_hit: Option<Vec<i32>>,
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
    /// Damage that is not the formula's: a set number, the user's level, a share of somebody's HP.
    FixedDamageEffect {
        amount_formula: String,
        set_amount: Option<i32>,
    },
    /// A move that heals its user by a fraction of its maximum.
    HealEffect {
        fraction: f64,
    },
    /// Weather, terrain and the rooms — `variant` says which, `kind` says which family.
    WeatherEffect {
        variant: String,
        duration_turns: Option<i32>,
    },
    TerrainEffect {
        variant: String,
        duration_turns: Option<i32>,
    },
    PseudoWeatherEffect {
        variant: String,
        duration_turns: Option<i32>,
    },
    /// Screens, hazards and Tailwind — everything that belongs to one side of the field.
    SideConditionEffect {
        variant: String,
        duration_turns: Option<i32>,
    },
    RemoveHazardsEffect {
        style: String,
    },
    /// The long tail the Python implements by hand: Haze, Rest, Trick, Pain Split and thirty more.
    /// Named rather than anonymous so a refusal says which one, and so the coverage report can
    /// count them.
    CodedEffect {
        variant: String,
    },
    /// Every effect kind this engine cannot read at all. The catch-all is what makes a new kind
    /// appearing in the export a refusal rather than a silent no-op.
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

/// A Z-Crystal's id (already `Database::normalize_id`'d, same as every other lookup key this file
/// hands out) -> the Z-move it unleashes. Deliberately separate from `moves.json` the same reason
/// Python's `get_all_z_moves()` is separate from `get_all_moves()` — a Z-move is never an ordinary
/// move slot. The 18 generic type Z-moves carry placeholder power (1) and a `category` that ignores
/// whichever base move triggers them; `zmoves::z_move_for` supplies both from the base move.
#[derive(Debug, Deserialize)]
struct ZMovesFile {
    zmoves: HashMap<String, Move>,
}

/// `battle_sim/rl/vocab.py`: the network's embedding tables, sorted, each id one past its index so
/// that 0 means none — or a name the table has never seen.
#[derive(Debug, Deserialize)]
struct VocabFile {
    species: Vec<String>,
    moves: Vec<String>,
    abilities: Vec<String>,
    items: Vec<String>,
}

#[derive(Debug)]
pub struct Vocab {
    species: HashMap<String, i64>,
    moves: HashMap<String, i64>,
    abilities: HashMap<String, i64>,
    items: HashMap<String, i64>,
}

impl Vocab {
    fn from_file(file: VocabFile) -> Self {
        let table = |names: Vec<String>| names.into_iter().zip(1..).collect::<HashMap<_, _>>();
        Vocab {
            species: table(file.species),
            moves: table(file.moves),
            abilities: table(file.abilities),
            items: table(file.items),
        }
    }

    pub fn species(&self, name: &str) -> i64 {
        self.species.get(&Database::normalize_id(name)).copied().unwrap_or(0)
    }

    pub fn move_id(&self, name: &str) -> i64 {
        self.moves.get(&Database::normalize_id(name)).copied().unwrap_or(0)
    }

    pub fn ability(&self, name: &str) -> i64 {
        self.abilities.get(name).copied().unwrap_or(0)
    }

    pub fn item(&self, name: &str) -> i64 {
        self.items.get(name).copied().unwrap_or(0)
    }
}

#[derive(Debug, Deserialize)]
struct RulesFile {
    coded_moves: Vec<String>,
    live_abilities: Vec<String>,
    live_items: Vec<String>,
    #[serde(default)]
    move_gated_formes: Vec<MoveGatedForme>,
    #[serde(default)]
    mega_formes: Vec<ItemGatedForme>,
    #[serde(default)]
    ultra_burst_formes: Vec<ItemGatedForme>,
    types: Vec<String>,
    type_chart: HashMap<String, HashMap<String, f64>>,
    natures: HashMap<String, NatureEffect>,
}

/// `_forme_by_base_and_move`: a species that Mega Evolves (or Primal Reverts) just by knowing a
/// particular move, no item at all. One entry in this dex — Mega Rayquaza, gated on Dragon Ascent
/// — but read from the export rather than hardcoded, since the whole reason it exists as a table
/// on the Python side is that a hand-kept "it's just Rayquaza" note is exactly the kind of thing
/// that falls behind the day a second entry is added.
#[derive(Debug, Deserialize)]
pub struct MoveGatedForme {
    pub base_species: String,
    #[serde(rename = "move")]
    pub move_name: String,
    /// The forme this pairing reaches — Python's own export used to throw this away (it iterated
    /// `_forme_by_base_and_move()`'s *keys* only), which was fine while nothing here read it and
    /// silently wrong the day something needed to.
    pub forme: String,
}

/// `_forme_by_base_and_item` (`mega_formes`) and `_ULTRA_BURST` (`ultra_burst_formes`): the same
/// shape, one row per (base species, held item) pairing and the forme it reaches. Both are already
/// filtered by Python's own `_is_transformed_forme`/`_is_playable`/`_source_forme` at export time —
/// `formes::mega_forme` does not re-derive any of that, only adds the one filter Python's own
/// playability check cannot know about: whether *this* engine has ported the forme's ability yet.
#[derive(Debug, Deserialize)]
pub struct ItemGatedForme {
    pub base_species: String,
    pub item: String,
    pub forme: String,
}

/// Every table the engine needs, loaded once.
#[derive(Debug)]
pub struct Database {
    pub moves: HashMap<String, Move>,
    pub species: HashMap<String, Species>,
    /// Crystal id (`Database::normalize_id`'d) -> the Z-move it unleashes. See `ZMovesFile`.
    pub z_moves: HashMap<String, Move>,
    /// Moves whose power is computed from the board rather than read from the data — Revenge,
    /// Gyro Ball, Weather Ball and the rest. Exported by Python rather than listed here, because a
    /// second copy of this list is a second thing to keep in step.
    pub coded_moves: std::collections::HashSet<String>,
    /// What the Python wires to its event bus. Anything in here that this engine has not
    /// implemented makes a scenario unplayable rather than quietly inert.
    pub live_abilities: std::collections::HashSet<String>,
    pub live_items: std::collections::HashSet<String>,
    pub move_gated_formes: Vec<MoveGatedForme>,
    pub mega_formes: Vec<ItemGatedForme>,
    pub ultra_burst_formes: Vec<ItemGatedForme>,
    pub types: Vec<String>,
    pub type_chart: HashMap<String, HashMap<String, f64>>,
    pub natures: HashMap<String, NatureEffect>,
    /// The observation encoder's embedding ids; nothing that simulates reads it.
    pub vocab: Vocab,
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
        let z_moves: ZMovesFile =
            serde_json::from_str(&read("zmoves.json")?).map_err(|e| format!("zmoves.json: {e}"))?;
        let rules: RulesFile =
            serde_json::from_str(&read("rules.json")?).map_err(|e| format!("rules.json: {e}"))?;
        let vocab: VocabFile =
            serde_json::from_str(&read("vocab.json")?).map_err(|e| format!("vocab.json: {e}"))?;
        Ok(Database {
            vocab: Vocab::from_file(vocab),
            moves: moves.moves,
            species: species.species,
            z_moves: z_moves.zmoves,
            coded_moves: rules.coded_moves.into_iter().collect(),
            live_abilities: rules.live_abilities.into_iter().collect(),
            live_items: rules.live_items.into_iter().collect(),
            move_gated_formes: rules.move_gated_formes,
            mega_formes: rules.mega_formes,
            ultra_burst_formes: rules.ultra_burst_formes,
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

    /// `formes.is_fused_to`. Z-Crystals are refused on any holder — checked by name, `_Z`, which
    /// covers this dex's real ones and its three deliberately invented ones (Absolite Z, Garchompite
    /// Z, Lucarionite Z) alike. Plus the Python's two special cases the species table cannot supply:
    /// Arceus's formes name no required item (Multitype supplies the plate), and Giratina-Origin's
    /// names the Gen 9 "Griseous Core" while a Gen 7 set holds the "Griseous Orb" — one item, two
    /// names. Both used to be unreachable, every item they name being refused outright; porting
    /// plates and the Orb made them live, and Knock Off took a plate off Arceus until they landed.
    pub fn is_fused_to(&self, species_name: &str, item: &str) -> bool {
        if item.ends_with("_Z") {
            return true;
        }
        let Some(entry) = self.species_named(species_name) else { return false };
        let base = entry.base_species.as_deref().unwrap_or(species_name);
        if base == "Arceus" && item.ends_with("_PLATE") {
            return true;
        }
        let fused = |wanted: &str| {
            self.species.values().any(|s| {
                s.base_species.as_deref().unwrap_or(s.name.as_str()) == base && s.fused_item.as_deref() == Some(wanted)
            })
        };
        const GRISEOUS: [&str; 2] = ["GRISEOUS_CORE", "GRISEOUS_ORB"];
        fused(item) || (GRISEOUS.contains(&item) && GRISEOUS.iter().any(|name| fused(name)))
    }

    pub fn move_named(&self, name: &str) -> Option<&Move> {
        self.moves
            .get(name)
            .or_else(|| self.moves.get(&Self::normalize_id(name)))
    }

    /// One attacking type against one or two defending types, multiplied out.
    pub fn effectiveness(&self, attacking: &str, defending: &[Option<String>]) -> f64 {
        self.effectiveness_bypassing(attacking, defending, &[])
    }

    /// `type_effectiveness`'s `immunity_bypass`: a defending type named here has its 0x immunity
    /// treated as 1x — Foresight/Odor Sleuth's Ghost, Miracle Eye's Dark — and nothing else about
    /// it changes, so a Ghost/Flying target bypassed only on Ghost still halves a Flying-weak move.
    pub fn effectiveness_bypassing(&self, attacking: &str, defending: &[Option<String>], bypass: &[&str]) -> f64 {
        defending
            .iter()
            .flatten()
            .map(|d| {
                let raw = self.type_chart.get(attacking).and_then(|row| row.get(d)).copied().unwrap_or(1.0);
                if raw == 0.0 && bypass.contains(&d.as_str()) {
                    1.0
                } else {
                    raw
                }
            })
            .product()
    }
}
