//! The board: what a Pokémon is, what a side holds, and what a turn can change.
//!
//! Mirrors `battle_sim/models/pokemon.py` and `mechanics/battle.py` closely enough that the
//! differential harness can compare them field by field. Where a name differs from the Python it
//! is because Rust convention demands it (`hp` for `live_stats.HP`), never because the meaning
//! differs — the digest in `battle_sim/differential.py` is the list of things that must agree.

use crate::data::{BaseStats, Database, NatureEffect, Species};
use crate::stats::{totals, with_stage, Spread, StatTotals};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// A small string-keyed counter that remembers *insertion* order rather than sorting by key —
/// hazards, screens and pseudo-weather all need this, because the Python's `dict` does the same
/// and more than one of them can end its turn on the same residual pass. A `BTreeMap` here logged
/// `TRICK_ROOM` before `WONDER_ROOM` fading together on the same turn regardless of which was cast
/// first, which the Python never does — it iterates in cast order, and this was found by two rooms
/// disagreeing about which one the log said ended first.
///
/// Updating an existing key does not move it, matching a Python `dict`'s own behaviour; only a new
/// key is appended. Every collection here stays under half a dozen entries for the life of a
/// battle, so the linear scans this does instead of a map's O(1) lookup cost nothing that matters.
#[derive(Debug, Clone, Default)]
pub struct OrderedCounts(Vec<(String, i32)>);

impl OrderedCounts {
    pub fn new() -> Self {
        OrderedCounts(Vec::new())
    }

    pub fn get(&self, key: &str) -> Option<i32> {
        self.0.iter().find(|(k, _)| k == key).map(|(_, v)| *v)
    }

    pub fn contains_key(&self, key: &str) -> bool {
        self.0.iter().any(|(k, _)| k == key)
    }

    /// Insert a new key at the end, or overwrite an existing one in place.
    pub fn insert(&mut self, key: String, value: i32) {
        match self.0.iter_mut().find(|(k, _)| *k == key) {
            Some(entry) => entry.1 = value,
            None => self.0.push((key, value)),
        }
    }

    pub fn get_mut(&mut self, key: &str) -> Option<&mut i32> {
        self.0.iter_mut().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    pub fn remove(&mut self, key: &str) -> Option<i32> {
        let index = self.0.iter().position(|(k, _)| k == key)?;
        Some(self.0.remove(index).1)
    }

    /// Keys in insertion order — the one thing a `BTreeMap` could not give this.
    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.0.iter().map(|(k, _)| k)
    }
}

impl Serialize for OrderedCounts {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut map = serializer.serialize_map(Some(self.0.len()))?;
        for (key, value) in &self.0 {
            map.serialize_entry(key, value)?;
        }
        map.end()
    }
}

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
    /// The inputs `totals` was folded from. Kept around, rather than discarded once `totals` is
    /// computed, only for Transform's sake: it borrows another Pokemon's base stats, nature, EVs
    /// and IVs wholesale (HP's base stat excepted) and has to be able to give its own back on
    /// switch-out.
    pub base_stats: BaseStats,
    pub nature: NatureEffect,
    pub ivs: Spread,
    pub evs: Spread,
    pub totals: StatTotals,
    pub hp: i32,
    pub status: Status,
    pub status_turns: i32,
    pub item: String,
    pub item_consumed: bool,
    /// What was eaten, which Belch will not fire without.
    pub last_consumed_item: String,
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
    /// Whole turns since this Pokemon's stint began, which Fake Out reads and a switch resets.
    pub turns_active: i32,
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
    /// The slot a charge move committed to. `CHARGING` in `volatiles` says a charge is in
    /// progress; this says which move it is, so the same name repeated across several slots (a
    /// short moveset padded by repetition) still resolves the one that was actually announced.
    pub charging_slot: Option<usize>,
    /// The slot a rampage (Outrage and the rest) or a Rollout/Ice Ball run has committed to, the
    /// same shape as `charging_slot` and for the same reason. `LOCKED_MOVE` in `volatiles` says
    /// which state it is; this says which move.
    pub locked_slot: Option<usize>,
    /// The slot most recently used, whatever it resolved to (Struggle included, since the Python
    /// records it after the substitution). Encore and Disable both read this to pick their target;
    /// a rampage sets `locked_slot` from it directly rather than reading it again later.
    pub last_move_slot: Option<usize>,
    /// The slot Encore is forcing, while `ENCORE` is in `volatiles`.
    pub encored_slot: Option<usize>,
    /// The slot Disable has silenced, while `DISABLE` is in `volatiles`.
    pub disabled_slot: Option<usize>,
    /// Flash Fire: set the first time a Fire move is absorbed, cleared on switch-out. Boosts every
    /// Fire move this Pokemon uses afterward, until then.
    pub flash_fire_active: bool,
    /// Protosynthesis / Quark Drive: which stat is currently boosted, if any. `None` until the
    /// field condition (or a consumed Booster Energy) activates it; cleared on switch-out.
    pub paradox_boost: Option<String>,
    /// Whether the current `paradox_boost` came from Booster Energy rather than the field — a
    /// booster-sourced boost outlives the field condition ending, a field-sourced one does not.
    pub paradox_from_booster: bool,
    /// The slot a Choice item has locked this Pokemon into, once it has used a move while holding
    /// one. `None` until the first such move, cleared on switch-out.
    pub choice_locked_move: Option<usize>,
    /// Eject Pack: armed by `apply_stage_changes_from` the moment an opponent's move actually drops
    /// one of this Pokemon's stats, and drained (whether or not the switch actually happens) by
    /// `resolve_eject_pack` right after the action that armed it finishes resolving.
    pub eject_pending: bool,
}

/// A pre-Transform form, restored on switch-out. Mirrors `models.pokemon.FormSnapshot` exactly —
/// everything Transform overwrites, and nothing it doesn't (the Pokemon's own HP base stat is
/// deliberately excluded there and here, since Transform never touches it).
#[derive(Debug, Clone)]
pub struct FormSnapshot {
    pub base_stats: BaseStats,
    pub nature: NatureEffect,
    pub ivs: Spread,
    pub evs: Spread,
    pub types: Vec<Option<String>>,
    pub ability: String,
    pub moves: Vec<String>,
    pub pp: BTreeMap<String, i32>,
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
        let ivs: Spread = spec.individual_values.into();
        let evs: Spread = spec.effort_values.into();
        let stat_totals = totals(&species.base_stats, &ivs, &evs, spec.level, nature);
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
            base_stats: species.base_stats,
            nature: nature.clone(),
            ivs,
            evs,
            totals: stat_totals,
            hp: stat_totals.hp,
            status: Status::None,
            status_turns: 0,
            item: spec.item.clone(),
            item_consumed: false,
            last_consumed_item: "NONE".to_string(),
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
            turns_active: 0,
            times_hit: 0,
            rolling_hits: 0,
            base_attack: species.base_stats.attack,
            weight_kg: species.weight_kg,
            last_hit_taken: 0,
            last_hit_category: None,
            protect_streak: 0,
            charging_slot: None,
            locked_slot: None,
            last_move_slot: None,
            encored_slot: None,
            disabled_slot: None,
            flash_fire_active: false,
            paradox_boost: None,
            paradox_from_booster: false,
            choice_locked_move: None,
            eject_pending: false,
        })
    }

    pub fn fainted(&self) -> bool {
        self.hp <= 0
    }

    /// `refresh_stats`, folded straight through rather than just invalidating a cache: Transform
    /// and its restore are the only two callers, both of which just replaced `base_stats`/`nature`/
    /// `ivs`/`evs` and need `totals` to reflect it immediately, not lazily on next read. `hp` (the
    /// current wound, not the cap) is deliberately left alone — Transform never heals or clamps it,
    /// same as the Python.
    pub fn recompute_totals(&mut self) {
        self.totals = totals(&self.base_stats, &self.ivs, &self.evs, self.level, &self.nature);
    }

    /// `Pokemon.battle_types`: what this Pokemon counts as right now, which is not always what it
    /// is. Roost puts a bird on the ground for the rest of the turn — its Flying type is ignored,
    /// so Earthquake hits it and Ice Beam stops being doubly effective. Computed fresh rather than
    /// stored, so a Pokemon that switches out, faints or changes forme mid-turn never carries the
    /// edit with it. A pure Flying-type keeps its typing regardless — Tornadus alone, and the
    /// alternative is a Pokemon with no type at all, which nothing here can represent.
    pub fn battle_types(&self) -> Vec<Option<String>> {
        if !self.volatiles.contains_key("ROOSTED") || !self.types.iter().flatten().any(|t| t == "FLYING") {
            return self.types.clone();
        }
        let remaining: Vec<Option<String>> =
            self.types.iter().flatten().filter(|t| *t != "FLYING").map(|t| Some(t.clone())).collect();
        if remaining.is_empty() {
            self.types.clone()
        } else {
            remaining
        }
    }

    /// `immunity_bypass`: Foresight/Odor Sleuth's `IDENTIFIED` lets a Normal or Fighting move
    /// through a Ghost type; Miracle Eye's `MIRACLE_EYE` does the same for a Psychic move against
    /// Dark. Only the named type's own 0x is affected — a Ghost/Flying target identified on Ghost
    /// alone still resists a Flying-weak move exactly as much as it did before. Shared between the
    /// immunity gate in `turn::resolve_move` and `damage::calculate_hit`'s own effectiveness check,
    /// which independently recomputes the same multiplier for the damage formula — missing either
    /// site left an identified Ghost immune to the hit that should have landed on it.
    pub fn identify_bypass(&self) -> Vec<&'static str> {
        let mut bypass = Vec::new();
        let has_type = |t: &str| self.types.iter().flatten().any(|d| d == t);
        if has_type("GHOST") && self.volatiles.contains_key("IDENTIFIED") {
            bypass.push("GHOST");
        }
        if has_type("DARK") && self.volatiles.contains_key("MIRACLE_EYE") {
            bypass.push("DARK");
        }
        bypass
    }

    /// `move_effectiveness`'s Scrappy/Mind's Eye clause, unioned onto `identify_bypass` the same
    /// way the Python ORs a Ghost bypass into `immunity_bypass(defender)` — attacker-side rather
    /// than defender-side, so it cannot live in `identify_bypass` itself, but it lands in the same
    /// bypass set at all three sites that read one: the immunity gate in `resolve_move`, the
    /// residual landing site for Future Sight/Doom Desire, and `calculate_hit`'s own independent
    /// recomputation. `_apply_fixed_damage`'s own gate is deliberately not one of the three: the
    /// Python's fixed-damage path calls the bare `immunity_bypass(defender)`, attacker never
    /// passed, so a Ground/Ghost hit by Seismic Toss from a Scrappy Pokemon is immune regardless.
    pub fn effective_bypass(&self, attacker_ability: &str, move_type: &str) -> Vec<&'static str> {
        let mut bypass = self.identify_bypass();
        let scrappy = matches!(attacker_ability, "SCRAPPY" | "MINDS_EYE") && matches!(move_type, "NORMAL" | "FIGHTING");
        if scrappy && self.types.iter().flatten().any(|t| t == "GHOST") && !bypass.contains(&"GHOST") {
            bypass.push("GHOST");
        }
        bypass
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
    pub hazards: OrderedCounts,
    pub screens: OrderedCounts,
    pub tailwind_turns: i32,
    /// Whether this side has already taken its action this turn. Analytic reads it.
    pub acted_this_turn: bool,
    /// The move this side picked for the turn, by name — what Sucker Punch is trying to read.
    pub chosen_move: Option<String>,
    /// Wish: turns left, and how much it will heal when it lands (fixed at cast time, off the
    /// caster's own max HP — not whoever is standing there when it lands).
    pub wish_turns: i32,
    pub wish_pending: i32,
    /// Healing Wish / Lunar Dance: granted to whichever Pokemon next switches in on this side.
    pub healing_wish_pending: bool,
    /// Shed Tail's parting gift: the substitute HP the next switch-in arrives with already up.
    pub pending_substitute: i32,
    /// Future Sight / Doom Desire: turns left, which (side, team index) queued it — a stable
    /// identity across switches, since Python holds the live Pokemon object itself and reads its
    /// *current* stats at landing time, not a snapshot from when it was queued — and the move
    /// name, re-looked-up in the database at landing rather than carried as a whole `Move`.
    pub future_sight_turns: i32,
    pub future_sight_attacker: Option<(usize, usize)>,
    pub future_sight_move: Option<String>,
    /// Transform: team index -> the form it had before. Keyed by index rather than by the Python's
    /// `id(pokemon)` because index is this engine's own stable identity for a team slot — the same
    /// scheme `future_sight_attacker` already uses. Popped and restored on that Pokemon's next
    /// switch-out; a no-op for every Pokemon that never transformed, which is nearly all of them.
    pub transforms: BTreeMap<usize, FormSnapshot>,
    /// Eject Button: armed (with `SelfSwitchPending` already logged and the item already consumed)
    /// the instant a hit lands, but the actual switch waits for `resolve_pending_switch` — called
    /// once per completed action, same as Eject Pack's own deferred switch — so a move's own
    /// `DamageDealt` summary and any of its other completion events still log under the Pokemon
    /// that is, for the moment, still standing.
    pub needs_switch: bool,
    /// Mega Evolution / Primal Reversion, once per battle per side. Never reset — a fresh `State`
    /// is a fresh battle, exactly as the Python's own flag is only ever reset by starting one.
    pub has_mega_evolved: bool,
    /// Ultra Burst, Necrozma's own once-per-battle. Kept apart from `has_mega_evolved` because a
    /// side may do both in the same battle — they are different Pokemon, or different turns.
    pub has_ultra_bursted: bool,
    /// Z-move, once per battle per side — the crystal is spent after one use, but never removed
    /// from the holder (Z-Crystals are permanently fused, same as a Mega Stone).
    pub has_used_z_move: bool,
}

impl Side {
    pub fn new(team: Vec<Pokemon>) -> Self {
        Side {
            team,
            active: 0,
            hazards: OrderedCounts::new(),
            screens: OrderedCounts::new(),
            tailwind_turns: 0,
            acted_this_turn: false,
            chosen_move: None,
            wish_turns: 0,
            wish_pending: 0,
            healing_wish_pending: false,
            pending_substitute: 0,
            future_sight_turns: 0,
            future_sight_attacker: None,
            future_sight_move: None,
            transforms: BTreeMap::new(),
            needs_switch: false,
            has_mega_evolved: false,
            has_ultra_bursted: false,
            has_used_z_move: false,
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
    /// Trick Room, Gravity, Magic Room, Wonder Room: kind -> turns left. Several can stand at
    /// once, unlike weather or terrain, so this is a map rather than a single slot.
    pub pseudo_weather: OrderedCounts,
}

impl Default for Field {
    fn default() -> Self {
        Field {
            weather: "NONE".into(),
            weather_turns_left: 0,
            terrain: "NONE".into(),
            terrain_turns_left: 0,
            pseudo_weather: OrderedCounts::new(),
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
            State {
                sides: [first, second],
                field: Field::default(),
                turn: 0,
                outcome: None,
                registrations: 0,
            };
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
