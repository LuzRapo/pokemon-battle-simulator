//! `battle_sim/matchup.py`'s `MatchupPlayer`, and the parts of `analysis.py` it stands on.
//!
//! The Python AI scores every legal action as a weighted sum of 28 matchup features (the genome) and
//! plays the best. This is that scorer, feature for feature, so battles can be piloted without Python
//! in the loop. `tests/test_rust_matchup.py` checks every feature of every action against the
//! Python's at each decision of played battles.
//!
//! It is the scorer as it reads the *true* board — both sides' sets known. That is how
//! `MatchupPlayer` scores when handed the real `BattleState`, and what a full-information battle (a
//! Mirror, or a tournament whose sets are dealt openly) is; the Python bot's belief machinery, which
//! stands in guessed sets for the opponent's, is not ported.
//!
//! Pokemon are named by `(side, team index)`, which is stable for a whole battle in both engines
//! (Mega Evolution and Transform change a Pokemon in place), so the per-battle offense cache below
//! is keyed exactly as the Python's `id()`-keyed one is.
//!
//! One deliberate difference: the Python estimator rolls Magnitude off the battle's own RNG, which a
//! scorer here must not touch. Magnitude is estimated at its most likely strength, 70.

use crate::battle::{Pokemon, Side, State, Status, SLOT_NAMES};
use crate::choices::{SWITCH, Z_MOVE};
use crate::damage::{estimate_hit, Payload};
use crate::data::{Database, Effect, Move};
use crate::power::{power_between, Duel};
use crate::turn::effective_speed;
use std::collections::HashMap;

pub const GENE_NAMES: [&str; 28] = [
    "ko_now",
    "hko_progress",
    "exchange_edge",
    "timer_value",
    "para_speed_control",
    "sleep_tempo",
    "burn_disable",
    "knock_progress",
    "hazard_value",
    "hazard_removal",
    "heal_turns",
    "entry_cost",
    "setup_value",
    "fodder_exploit",
    "wincon_preservation",
    "wincon_support",
    "matchup_gain",
    "tempo_cost",
    "phaze_value",
    "lock_risk",
    "incoming_damage_taken",
    "incoming_damage_dealt",
    "incoming_outspeeds",
    "lock_relief",
    "volatile_relief",
    "debuff_relief",
    "setup_concession",
    "setup_denial",
];
pub const GENES: usize = GENE_NAMES.len();

/// `MatchupWeights()`'s defaults, in `GENE_NAMES` order.
const DEFAULT_WEIGHTS: [f64; GENES] = [
    6.0, 1.0, 0.6, 0.5, 0.5, 1.2, 0.5, 0.4, 0.6, 0.4, 0.7, 0.8, 0.6, 0.4, 0.7, 0.7, 0.8, 0.5, 0.4, 0.8, 0.8, 0.8,
    0.3, 0.4, 0.4, 0.4, 0.6, 0.6,
];

/// A genome. `evolution.load_weights`: a gene missing from the file keeps its default; an unknown
/// one is an error, because then the file and the code disagree.
#[derive(Debug, Clone, Copy)]
pub struct Weights(pub [f64; GENES]);

impl Default for Weights {
    fn default() -> Self {
        Weights(DEFAULT_WEIGHTS)
    }
}

impl Weights {
    pub fn from_json(raw: &str) -> Result<Self, String> {
        let genes: HashMap<String, f64> = serde_json::from_str(raw).map_err(|e| e.to_string())?;
        let mut weights = Weights::default();
        for (name, value) in genes {
            let index = GENE_NAMES.iter().position(|g| *g == name).ok_or_else(|| format!("unknown gene {name:?}"))?;
            weights.0[index] = value;
        }
        Ok(weights)
    }
}

// Feature indices, named as the genes are.
const KO_NOW: usize = 0;
const HKO_PROGRESS: usize = 1;
const EXCHANGE_EDGE: usize = 2;
const TIMER_VALUE: usize = 3;
const PARA_SPEED_CONTROL: usize = 4;
const SLEEP_TEMPO: usize = 5;
const BURN_DISABLE: usize = 6;
const KNOCK_PROGRESS: usize = 7;
const HAZARD_VALUE: usize = 8;
const HAZARD_REMOVAL: usize = 9;
const HEAL_TURNS: usize = 10;
const ENTRY_COST: usize = 11;
const SETUP_VALUE: usize = 12;
const FODDER_EXPLOIT: usize = 13;
const WINCON_PRESERVATION: usize = 14;
const WINCON_SUPPORT: usize = 15;
const MATCHUP_GAIN: usize = 16;
const TEMPO_COST: usize = 17;
const PHAZE_VALUE: usize = 18;
const LOCK_RISK: usize = 19;
const INCOMING_DAMAGE_TAKEN: usize = 20;
const INCOMING_DAMAGE_DEALT: usize = 21;
const INCOMING_OUTSPEEDS: usize = 22;
const LOCK_RELIEF: usize = 23;
const VOLATILE_RELIEF: usize = 24;
const DEBUFF_RELIEF: usize = 25;
const SETUP_CONCESSION: usize = 26;
const SETUP_DENIAL: usize = 27;

pub type Features = [f64; GENES];

const DEAD_MOVE_MARGIN: f64 = 1e-6;
const FODDER_THREAT_FRACTION: f64 = 0.2;
const BOOST_STATS: [&str; 3] = ["ATTACK", "SP_ATTACK", "SPEED"];
const RELIEF_STATS: [&str; 5] = ["ATTACK", "DEFENCE", "SP_ATTACK", "SP_DEFENCE", "SPEED"];
const BOOST_CAP: f64 = 6.0;
const SLEEP_TURNS: f64 = 2.0;
const FREE_TURN_SETUP: f64 = 0.85;
const DENIAL_BASE: f64 = 0.4;
const DENIAL_VOLATILES: [&str; 3] = ["TAUNT", "ENCORE", "DISABLE"];
const CHOICE_ITEMS: [&str; 3] = ["CHOICE_BAND", "CHOICE_SPECS", "CHOICE_SCARF"];

pub const EDGE_CAP: f64 = 4.0;
const MIN_ROLL: i32 = 85;
const MAX_ROLL: i32 = 100;
const NEUTRAL_ROCK_DIVISOR: f64 = 8.0;
const FREE_SURVIVAL_DENIAL: f64 = 1.0;
const BREAKPOINT_CONVERSION: f64 = 1.0;
/// Magnitude's roll for an estimate: 11 is Magnitude 7, power 70, the likeliest outcome.
const MAGNITUDE_ESTIMATE_ROLL: i32 = 11;

/// `(side, team index)`.
pub type PokeRef = (usize, usize);

fn poke(state: &State, at: PokeRef) -> &Pokemon {
    &state.sides[at.0].team[at.1]
}

fn active(state: &State, side: usize) -> PokeRef {
    (side, state.sides[side].active)
}

fn slot_pp(pokemon: &Pokemon, slot: usize) -> i32 {
    pokemon.pp.get(SLOT_NAMES[slot]).copied().unwrap_or(0)
}

fn has_type(pokemon: &Pokemon, kind: &str) -> bool {
    pokemon.types.iter().flatten().any(|t| t == kind)
}

fn is_main_status(name: &str) -> bool {
    matches!(name, "BURN" | "POISON" | "TOXIC" | "PARALYSIS" | "SLEEP" | "FREEZE")
}

fn ceil_div(hp: i32, per_turn: f64) -> f64 {
    (hp as f64 / per_turn).ceil()
}

// --- analysis.py ------------------------------------------------------------------------------

/// `_sides_of`: the attacker's side is 0 only if it *is* side 0's active; anything else — including
/// a benched side-0 Pokemon — reads as side 1. Reproduced, because the power formulas read speeds
/// and fallen teammates off whichever side this names.
fn sides_of(state: &State, attacker: PokeRef) -> (usize, usize) {
    if attacker == active(state, 0) {
        (0, 1)
    } else {
        (1, 0)
    }
}

/// `_fixed_amount`, for any two Pokemon.
fn fixed_amount(formula: &str, set_amount: Option<i32>, attacker: &Pokemon, defender: &Pokemon) -> Option<i32> {
    match formula {
        "LEVEL" | "PSYWAVE" => Some(attacker.level),
        "SET" => set_amount,
        "HALF_TARGET_HP" => Some(std::cmp::max(1, defender.hp / 2)),
        "ENDEAVOR" => Some(defender.hp - attacker.hp).filter(|difference| *difference > 0),
        "COUNTER" | "MIRROR_COAT" => {
            let wanted = if formula == "COUNTER" { "PHYSICAL" } else { "SPECIAL" };
            let valid = attacker.last_hit_category.as_deref() == Some(wanted) && attacker.last_hit_taken > 0;
            valid.then(|| 2 * attacker.last_hit_taken)
        }
        "USER_HP" => Some(attacker.hp),
        "TARGET_HP" => Some(defender.hp),
        _ => None,
    }
}

const MOLD_BREAKERS: [&str; 3] = ["MOLD_BREAKER", "TERAVOLT", "TURBOBLAZE"];

fn absorbed_type(ability: &str) -> Option<&'static str> {
    Some(match ability {
        "VOLT_ABSORB" | "LIGHTNING_ROD" | "MOTOR_DRIVE" => "ELECTRIC",
        "WATER_ABSORB" | "STORM_DRAIN" | "DRY_SKIN" => "WATER",
        "SAP_SIPPER" => "GRASS",
        "FLASH_FIRE" | "WELL_BAKED_BODY" => "FIRE",
        "EARTH_EATER" => "GROUND",
        _ => return None,
    })
}

/// `ability_absorbs`: the defender simply cannot be hurt by this move.
fn ability_absorbs(move_type: &str, the_move: &Move, attacker: &Pokemon, defender: &Pokemon, db: &Database) -> bool {
    if MOLD_BREAKERS.contains(&attacker.ability.as_str()) {
        return false;
    }
    if move_type == "GROUND" && !crate::field::is_grounded(defender) {
        return true;
    }
    if absorbed_type(&defender.ability) == Some(move_type) {
        return true;
    }
    if defender.ability == "WONDER_GUARD" && db.effectiveness(move_type, &defender.types) < 2.0 {
        return true;
    }
    if defender.ability == "SOUNDPROOF" && the_move.sound {
        return true;
    }
    defender.ability == "BULLETPROOF" && the_move.bullet
}

/// A channel of `payload`'s modifiers, by the Python's key.
type Channels = Vec<(&'static str, Vec<i64>)>;

fn push(channels: &mut Channels, key: &'static str, value: i64) {
    match channels.iter_mut().find(|(k, _)| *k == key) {
        Some((_, values)) => values.push(value),
        None => channels.push((key, vec![value])),
    }
}

/// `static_damage_modifiers`: the items an estimate can price on its own.
fn item_modifiers(move_type: &str, category: &str, attacker: &Pokemon, defender: &Pokemon, base_power: Option<i32>, db: &Database) -> Channels {
    let mut final_mods = Vec::new();
    let mut defense = Vec::new();
    let mut power = Vec::new();
    if attacker.item == "MEOWFREDS_MONOCLE" && base_power.is_some_and(|p| p <= 60) {
        power.push(6144);
    }
    let choice = match attacker.item.as_str() {
        "CHOICE_BAND" => Some("PHYSICAL"),
        "CHOICE_SPECS" => Some("SPECIAL"),
        _ => None,
    };
    if choice == Some(category) {
        final_mods.push(6144);
    }
    if attacker.item == "LIFE_ORB" {
        final_mods.push(5324);
    }
    if attacker.item == "EXPERT_BELT" && db.effectiveness(move_type, &defender.types) >= 2.0 {
        final_mods.push(4915);
    }
    let orb: Option<(&[&str], &[&str])> = match attacker.item.as_str() {
        "SOUL_DEW" => Some((&["Latios", "Latias"], &["PSYCHIC", "DRAGON"])),
        "GRISEOUS_CORE" | "GRISEOUS_ORB" => Some((&["Giratina", "Giratina-Origin"], &["GHOST", "DRAGON"])),
        _ => None,
    };
    if let Some((owners, types)) = orb {
        if owners.contains(&attacker.species_name.as_str()) && types.contains(&move_type) {
            final_mods.push(4915);
        }
    }
    if defender.item == "EVIOLITE" && !defender.fully_evolved {
        defense.push(6144);
    }
    if defender.item == "ASSAULT_VEST" && category == "SPECIAL" {
        defense.push(6144);
    }
    let mut channels = Channels::new();
    for (key, values) in [("final_mods_4096", final_mods), ("defense_mods_4096", defense), ("power_mods_4096", power)] {
        if !values.is_empty() {
            channels.push((key, values));
        }
    }
    channels
}

/// `static_ability_modifiers`: the abilities an estimate can price on its own. Returns the channels
/// and the two scalars (`ignore_burn`, `stab_4096`), which replace rather than append.
fn ability_modifiers(
    move_type: &str,
    the_move: &Move,
    base_power: Option<i32>,
    attacker: &Pokemon,
    defender: &Pokemon,
    db: &Database,
) -> (Channels, bool, Option<i64>) {
    let effectiveness = db.effectiveness(move_type, &defender.types);
    let mut channels = Channels::new();
    let (mut ignore_burn, mut stab) = (false, None);
    let physical = the_move.category == "PHYSICAL";
    let ability = attacker.ability.as_str();
    if physical && matches!(ability, "HUGE_POWER" | "PURE_POWER") {
        push(&mut channels, "attack_mods_4096", 8192);
    }
    if ability == "GUTS" {
        if physical && attacker.status != Status::None {
            push(&mut channels, "attack_mods_4096", 6144);
        }
        ignore_burn = true;
    }
    if ability == "DEFEATIST" && attacker.hp * 2 <= attacker.totals.hp {
        push(&mut channels, "attack_mods_4096", 2048);
    }
    if ability == "ADAPTABILITY" && has_type(attacker, move_type) {
        stab = Some(8192);
    }
    if ability == "TECHNICIAN" && base_power.is_some_and(|p| p <= 60) {
        push(&mut channels, "power_mods_4096", 6144);
    }
    if ability == "SHEER_FORCE" && the_move.effects.iter().any(is_secondary) {
        push(&mut channels, "power_mods_4096", 5325);
    }
    if ability == "TINTED_LENS" && 0.0 < effectiveness && effectiveness < 1.0 {
        push(&mut channels, "final_mods_4096", 8192);
    }
    if !MOLD_BREAKERS.contains(&ability) {
        let defending = defender.ability.as_str();
        if defending == "THICK_FAT" && matches!(move_type, "FIRE" | "ICE") {
            push(&mut channels, "attack_mods_4096", 2048);
        }
        if defending == "MULTISCALE" && defender.hp == defender.totals.hp {
            push(&mut channels, "final_mods_4096", 2048);
        }
        if matches!(defending, "FILTER" | "PRISM_ARMOR") && effectiveness >= 2.0 {
            push(&mut channels, "final_mods_4096", 3072);
        }
        if defending == "FUR_COAT" && physical {
            push(&mut channels, "defense_mods_4096", 8192);
        }
    }
    (channels, ignore_burn, stab)
}

fn is_secondary(effect: &Effect) -> bool {
    matches!(
        effect,
        Effect::InflictStatusEffect { is_secondary: true, .. } | Effect::StatStageChangeEffect { is_secondary: true, .. }
    )
}

/// `static_field_modifiers`: an aura belongs to whoever stands on the field.
fn field_modifiers(move_type: &str, state: &State) -> Channels {
    let actives = [state.sides[0].active_pokemon(), state.sides[1].active_pokemon()];
    let aura = |p: &Pokemon| match p.ability.as_str() {
        "DARK_AURA" => Some("DARK"),
        "FAIRY_AURA" => Some("FAIRY"),
        _ => None,
    };
    if !actives.iter().any(|p| aura(p) == Some(move_type)) {
        return Channels::new();
    }
    let broken = actives.iter().any(|p| p.ability == "AURA_BREAK");
    vec![("power_mods_4096", vec![if broken { 3072 } else { 5461 }])]
}

fn channel<'p>(payload: &'p mut Payload, key: &str) -> &'p mut Vec<i64> {
    match key {
        "attack_mods_4096" => &mut payload.attack_mods_4096,
        "defense_mods_4096" => &mut payload.defense_mods_4096,
        "power_mods_4096" => &mut payload.power_mods_4096,
        "pre_screen_mods_4096" => &mut payload.pre_screen_mods_4096,
        _ => &mut payload.final_mods_4096,
    }
}

/// `damage_range`: (min, max) damage of one use against the defender, crits excluded; (0, 0) if it
/// cannot hurt.
pub fn damage_range(the_move: &Move, from: PokeRef, to: PokeRef, state: &State, db: &Database) -> (i32, i32) {
    let attacker = poke(state, from);
    let defender = poke(state, to);
    let damage_effect = the_move.effects.iter().find_map(|e| match e {
        Effect::DamageEffect { power, multi_hit, .. } => Some((*power, multi_hit.clone())),
        _ => None,
    });
    let Some((listed_power, multi_hit)) = damage_effect else {
        return the_move
            .effects
            .iter()
            .find_map(|e| match e {
                Effect::FixedDamageEffect { amount_formula, set_amount } => Some(
                    fixed_amount(amount_formula, *set_amount, attacker, defender).map_or((0, 0), |amount| (amount, amount)),
                ),
                _ => None,
            })
            .unwrap_or((0, 0));
    };

    let listed_type = the_move.move_type.clone();
    let mut resolved = the_move.clone();
    if let Some(became) = crate::power::type_override(the_move, attacker, state) {
        resolved.move_type = became;
    }
    let move_type = resolved.move_type.clone();
    if ability_absorbs(&move_type, &resolved, attacker, defender, db) {
        return (0, 0);
    }

    let mut payload = Payload::new();
    let overrides = crate::power::payload_overrides(&resolved, &listed_type, attacker);
    payload.power_mods_4096.extend(overrides.ate_power_mod);
    payload.attack_stat_override = overrides.attack_stat;
    payload.use_target_attack = overrides.use_target_attack;
    payload.defense_stat_override = overrides.defense_stat;
    payload.ignore_burn = overrides.ignore_burn;
    payload.ignore_weather_drop = overrides.ignore_weather_drop;
    let (attacker_side, defender_side) = sides_of(state, from);
    let duel = Duel { attacker, defender, attacker_side, defender_side };
    payload.power_override =
        power_between(&resolved, listed_power, duel, state, db, &mut || Ok(MAGNITUDE_ESTIMATE_ROLL)).ok().flatten();

    // `{**items, **abilities, **field}`: a channel two of them both fill keeps only the *later*
    // one's list — an item's Life Orb is dropped when a Tinted Lens also writes `final_mods_4096`.
    // The Python's dict merge does that; the estimate has to agree with it.
    let (ability_channels, ignore_burn, stab) =
        ability_modifiers(&move_type, &resolved, listed_power, attacker, defender, db);
    let mut merged = Channels::new();
    for (key, values) in item_modifiers(&move_type, &resolved.category, attacker, defender, listed_power, db)
        .into_iter()
        .chain(ability_channels)
        .chain(field_modifiers(&move_type, state))
    {
        match merged.iter_mut().find(|(k, _)| *k == key) {
            Some((_, existing)) => *existing = values,
            None => merged.push((key, values)),
        }
    }
    for (key, values) in merged {
        channel(&mut payload, key).extend(values);
    }
    if ignore_burn {
        payload.ignore_burn = true;
    }
    if let Some(stab) = stab {
        payload.stab_4096 = stab;
    }
    if defender.ability == "UNAWARE" && attacker.ability != "MOLD_BREAKER" {
        payload.ignore_attack_stages = true;
    }
    if attacker.ability == "UNAWARE" {
        payload.ignore_defense_stages = true;
    }
    if defender.item == "AIR_BALLOON" && move_type == "GROUND" {
        return (0, 0);
    }
    let side = &state.sides[to.0];
    let mut low = estimate_hit(attacker, defender, &resolved, &state.field, side, db, MIN_ROLL, &payload);
    let mut high = estimate_hit(attacker, defender, &resolved, &state.field, side, db, MAX_ROLL, &payload);
    if let Some(hits) = multi_hit {
        low *= hits[0];
        high *= hits[1];
    }
    (low, high)
}

fn accuracy(the_move: &Move) -> f64 {
    the_move.accuracy_probability.unwrap_or(1.0)
}

/// `best_expected_damage`: the best accuracy-weighted midpoint over moves with PP left.
fn best_expected_damage(from: PokeRef, to: PokeRef, state: &State, db: &Database) -> f64 {
    let attacker = poke(state, from);
    let mut best: Option<f64> = None;
    for (slot, name) in attacker.moves.iter().enumerate().take(4) {
        if slot_pp(attacker, slot) <= 0 {
            continue;
        }
        let Some(the_move) = db.move_named(name) else { continue };
        let (low, high) = damage_range(the_move, from, to, state, db);
        let expected = (low + high) as f64 / 2.0 * accuracy(the_move);
        best = Some(best.map_or(expected, |b: f64| b.max(expected)));
    }
    best.unwrap_or(0.0)
}

/// `exchange_edge_from`.
pub fn exchange_edge_from(my_hp: i32, their_hp: i32, my_best: f64, their_best: f64, faster: bool) -> f64 {
    let my_ttk = if my_best > 0.0 { ceil_div(their_hp, my_best) } else { f64::INFINITY };
    let their_ttk = if their_best > 0.0 { ceil_div(my_hp, their_best) } else { f64::INFINITY };
    if my_ttk.is_infinite() && their_ttk.is_infinite() {
        return 0.0;
    }
    (their_ttk - my_ttk + if faster { 0.5 } else { -0.5 }).clamp(-EDGE_CAP, EDGE_CAP)
}

/// `exchange_edge`, sets known.
fn exchange_edge(mine: PokeRef, theirs: PokeRef, state: &State, db: &Database) -> f64 {
    let (me, them) = (poke(state, mine), poke(state, theirs));
    let my_speed = effective_speed(me, &state.sides[mine.0], &state.field);
    let their_speed = effective_speed(them, &state.sides[theirs.0], &state.field);
    exchange_edge_from(
        me.hp,
        them.hp,
        best_expected_damage(mine, theirs, state, db),
        best_expected_damage(theirs, mine, state, db),
        my_speed >= their_speed,
    )
}

fn hazard_cap(kind: &str) -> Option<i32> {
    Some(match kind {
        "STEALTH_ROCK" | "STICKY_WEB" => 1,
        "SPIKES" => 3,
        "TOXIC_SPIKES" => 2,
        _ => return None,
    })
}

/// `entry_hazard_share`, against an arbitrary layout.
fn entry_hazard_share(incoming: &Pokemon, hazards: &[(String, i32)], db: &Database) -> f64 {
    if incoming.item == "HEAVY_DUTY_BOOTS" || incoming.ability == "MAGIC_GUARD" {
        return 0.0;
    }
    let count = |kind: &str| hazards.iter().find(|(k, _)| k == kind).map_or(0, |(_, n)| *n);
    let mut share = 0.0;
    if count("STEALTH_ROCK") > 0 {
        share += db.effectiveness("ROCK", &incoming.types) / 8.0;
    }
    let layers = count("SPIKES");
    if layers > 0 && crate::field::is_grounded(incoming) {
        share += match layers.min(3) {
            1 => 1.0 / 8.0,
            2 => 1.0 / 6.0,
            _ => 1.0 / 4.0,
        };
    }
    share
}

fn layout(side: &Side) -> Vec<(String, i32)> {
    side.hazards.keys().map(|k| (k.clone(), side.hazards.get(k).unwrap_or(0))).collect()
}

/// `entry_hazard_chip`.
fn entry_hazard_chip(incoming: &Pokemon, side: &Side, db: &Database) -> i32 {
    (incoming.totals.hp as f64 * entry_hazard_share(incoming, &layout(side), db)) as i32
}

fn has_free_survival(pokemon: &Pokemon) -> bool {
    pokemon.hp >= pokemon.totals.hp && (pokemon.item == "FOCUS_SASH" || pokemon.ability == "STURDY")
}

/// `hazard_toll`, with the attacker always supplied (the only way the scorer asks).
fn hazard_toll(side_index: usize, extra: Option<&str>, attacker: PokeRef, state: &State, db: &Database) -> f64 {
    let side = &state.sides[side_index];
    let mut hazards = layout(side);
    if let Some(extra) = extra {
        match hazards.iter_mut().find(|(k, _)| k == extra) {
            Some((_, n)) => *n += 1,
            None => hazards.push((extra.to_string(), 1)),
        }
    }
    let (mut health, mut bodies) = (0.0, 0.0);
    for (index, member) in side.team.iter().enumerate() {
        if index == side.active || member.fainted() || member.item == "HEAVY_DUTY_BOOTS" {
            continue;
        }
        let share = entry_hazard_share(member, &hazards, db);
        if share <= 0.0 {
            continue;
        }
        health += share;
        if has_free_survival(member) {
            bodies += FREE_SURVIVAL_DENIAL;
        } else {
            let best = best_expected_damage(attacker, (side_index, index), state, db);
            let hp = member.hp as f64;
            if best < hp && hp <= best + share * member.totals.hp as f64 {
                bodies += BREAKPOINT_CONVERSION;
            }
        }
    }
    (health * NEUTRAL_ROCK_DIVISOR + bodies) / 6.0
}

/// `residual_drain`: net HP the residuals take next turn.
fn residual_drain(pokemon: &Pokemon, state: &State) -> i32 {
    if pokemon.ability == "MAGIC_GUARD" {
        return 0;
    }
    let max_hp = pokemon.totals.hp;
    let mut drain = 0;
    let poisoned = matches!(pokemon.status, Status::Poison | Status::Toxic);
    if !(poisoned && pokemon.ability == "POISON_HEAL") {
        match pokemon.status {
            Status::Burn => drain += std::cmp::max(1, max_hp / 16),
            Status::Poison => drain += std::cmp::max(1, max_hp / 8),
            Status::Toxic => drain += std::cmp::max(1, max_hp * (pokemon.status_turns + 1) / 16),
            _ => {}
        }
    }
    if pokemon.volatiles.contains_key("LEECH_SEED") {
        drain += std::cmp::max(1, max_hp / 8);
    }
    if crate::hooks::effective_weather(state) == "SANDSTORM" {
        let immune = ["ROCK", "GROUND", "STEEL"].iter().any(|t| has_type(pokemon, t));
        if !immune && !matches!(pokemon.ability.as_str(), "SAND_VEIL" | "OVERCOAT") {
            drain += std::cmp::max(1, max_hp / 16);
        }
    }
    if matches!(pokemon.item.as_str(), "LEFTOVERS" | "BLACK_SLUDGE") {
        drain -= std::cmp::max(1, max_hp / 16);
    }
    drain
}

/// `bootless_bench`.
fn bootless_bench(side: &Side) -> usize {
    side.team
        .iter()
        .enumerate()
        .filter(|(i, p)| *i != side.active && !p.fainted() && p.item != "HEAVY_DUTY_BOOTS")
        .count()
}

/// `hazard_pressure`.
fn hazard_pressure(side: &Side) -> f64 {
    let layers: i32 = layout(side).iter().map(|(kind, count)| (*count).min(hazard_cap(kind).unwrap_or(1))).sum();
    if layers == 0 {
        return 0.0;
    }
    layers.min(4) as f64 / 4.0 * bootless_bench(side) as f64 / 6.0
}

// --- the move's own effects, read the way matchup.py reads them -------------------------------

fn inflicted(the_move: &Move, main: bool) -> Option<&str> {
    the_move.effects.iter().find_map(|e| match e {
        Effect::InflictStatusEffect { status, to_self: false, is_secondary: false, .. } if is_main_status(status) == main => {
            Some(status.as_str())
        }
        _ => None,
    })
}

fn self_boost(the_move: &Move, stat: &str) -> i32 {
    the_move
        .effects
        .iter()
        .map(|e| match e {
            Effect::StatStageChangeEffect { stages, target, is_secondary: false, .. } if target == "SELF" => {
                stages.iter().filter(|(s, _)| s == stat).map(|(_, n)| *n).sum()
            }
            _ => 0,
        })
        .sum()
}

fn heals(the_move: &Move) -> bool {
    the_move.healing || the_move.effects.iter().any(|e| matches!(e, Effect::HealEffect { .. }))
}

fn coded(the_move: &Move, variants: &[&str]) -> bool {
    the_move.effects.iter().any(|e| matches!(e, Effect::CodedEffect { variant } if variants.contains(&variant.as_str())))
}

fn set_hazard(the_move: &Move) -> Option<&str> {
    the_move.effects.iter().find_map(|e| match e {
        Effect::SideConditionEffect { variant, .. } if hazard_cap(variant).is_some() => Some(variant.as_str()),
        _ => None,
    })
}

fn stage_multiplier(stage: i32) -> f64 {
    if stage >= 0 {
        (2 + stage) as f64 / 2.0
    } else {
        2.0 / (2 - stage) as f64
    }
}

fn positive_boosts(pokemon: &Pokemon) -> f64 {
    BOOST_STATS.iter().map(|s| pokemon.stage(s).max(0)).sum::<i32>() as f64
}

fn cannot_act(pokemon: &Pokemon) -> bool {
    match pokemon.status {
        Status::Sleep => pokemon.status_turns > 1,
        Status::Freeze => true,
        _ => false,
    }
}

fn damage_share(damage: f64, target: &Pokemon, cap: f64) -> f64 {
    if target.hp <= 0 {
        return 0.0;
    }
    (damage / target.hp as f64).min(cap)
}

/// `status_cannot_land`.
fn status_cannot_land(status: Status, target: &Pokemon, inflictor: &Pokemon, state: &State) -> bool {
    let corrosive = inflictor.ability == "CORROSION" && matches!(status, Status::Poison | Status::Toxic);
    let type_immune = !corrosive && status.immune_types().iter().any(|t| has_type(target, t));
    type_immune || crate::inline::ability_blocks_status(target, status, &state.field.weather)
}

// --- the scorer -------------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Offense {
    best_expected: f64,
    /// `posterior_threat`, which with the set known is `best_expected_damage` — the same number.
    threat_expected: f64,
    strongest: i32,
    best_category: Option<String>,
}

fn offense(from: PokeRef, to: PokeRef, state: &State, db: &Database) -> Offense {
    let attacker = poke(state, from);
    let (mut best_expected, mut strongest, mut best_category) = (0.0, 0, None);
    for (slot, name) in attacker.moves.iter().enumerate().take(4) {
        if slot_pp(attacker, slot) == 0 {
            continue;
        }
        let Some(the_move) = db.move_named(name) else { continue };
        let (low, high) = damage_range(the_move, from, to, state, db);
        let expected = (low + high) as f64 / 2.0 * accuracy(the_move);
        if expected > best_expected {
            best_expected = expected;
            best_category = Some(the_move.category.clone());
        }
        strongest = strongest.max(high);
    }
    Offense { threat_expected: best_expected, best_expected, strongest, best_category }
}

struct Turn {
    side: usize,
    me: PokeRef,
    opponent: PokeRef,
    wincon: PokeRef,
    mine: Offense,
    theirs: Offense,
    edge: f64,
    opponent_faster: bool,
}

/// One side's `MatchupPlayer` for one battle: the genome and the full-strength offense cache.
pub struct MatchupAi {
    pub weights: Weights,
    cache: HashMap<(PokeRef, PokeRef), Offense>,
}

impl MatchupAi {
    pub fn new(weights: Weights) -> Self {
        MatchupAi { weights, cache: HashMap::new() }
    }

    /// `choose_action`: the first of the best-scoring actions, in the order offered.
    pub fn choose(&mut self, state: &State, side: usize, actions: &[u8], db: &Database) -> u8 {
        let scores = self.scores(state, side, actions, db);
        let mut best = 0;
        for (index, score) in scores.iter().enumerate() {
            if *score > scores[best] {
                best = index;
            }
        }
        actions[best]
    }

    /// `score_actions`: a dead action scores just below the worst real one.
    pub fn scores(&mut self, state: &State, side: usize, actions: &[u8], db: &Database) -> Vec<f64> {
        let featured = self.features(state, side, actions, db);
        let scored: Vec<Option<f64>> = featured
            .iter()
            .map(|f| f.map(|f| f.iter().zip(self.weights.0.iter()).fold(0.0, |sum, (x, w)| sum + x * w)))
            .collect();
        let real = scored.iter().flatten().copied().fold(None, |low: Option<f64>, s| Some(low.map_or(s, |l| l.min(s))));
        let dead = real.map_or(0.0, |low| low - DEAD_MOVE_MARGIN);
        scored.into_iter().map(|s| s.unwrap_or(dead)).collect()
    }

    /// `feature_actions`: each action's features, or `None` for one the engine will not play as
    /// scored (no PP left, or a coded failure).
    pub fn features(&mut self, state: &State, side: usize, actions: &[u8], db: &Database) -> Vec<Option<Features>> {
        let turn = self.turn(state, side, db);
        let own = &state.sides[side];
        let voluntary = !poke(state, turn.me).fainted() && !own.needs_switch;
        actions
            .iter()
            .map(|&action| {
                if action >= SWITCH {
                    Some(self.switch_features((side, (action - SWITCH) as usize), &turn, voluntary, state, db))
                } else {
                    let slot = (action % Z_MOVE) as usize;
                    if slot_pp(poke(state, turn.me), slot) == 0 {
                        None
                    } else {
                        self.move_features(slot, action >= Z_MOVE, &turn, state, db)
                    }
                }
            })
            .collect()
    }

    fn turn(&mut self, state: &State, side: usize, db: &Database) -> Turn {
        let (me, opponent) = (active(state, side), active(state, 1 - side));
        let mine = offense(me, opponent, state, db);
        let theirs = offense(opponent, me, state, db);
        let faster = effective_speed(poke(state, me), &state.sides[side], &state.field)
            >= effective_speed(poke(state, opponent), &state.sides[1 - side], &state.field);
        let edge =
            exchange_edge_from(poke(state, me).hp, poke(state, opponent).hp, mine.best_expected, theirs.threat_expected, faster);
        let wincon = self.wincon(state, side, db);
        Turn { side, me, opponent, wincon, mine, theirs, edge, opponent_faster: !faster }
    }

    // -- moves --

    fn move_features(&mut self, slot: usize, z: bool, turn: &Turn, state: &State, db: &Database) -> Option<Features> {
        let me = poke(state, turn.me);
        let opponent = poke(state, turn.opponent);
        let base = db.move_named(&me.moves[slot])?;
        let upgraded;
        let the_move = if z {
            upgraded = crate::zmoves::z_move_for(&me.item, base, db)?;
            &upgraded
        } else {
            base
        };
        if the_move.reflectable && opponent.ability == "MAGIC_BOUNCE" {
            return None;
        }
        if crate::power::coded_move_fails(the_move, state, turn.side, db) {
            return None;
        }
        let (low, high) = damage_range(the_move, turn.me, turn.opponent, state, db);
        let mut f = self.effect_features(the_move, turn, low, high, state, db);
        let any_effect = f.iter().any(|v| *v != 0.0);
        let acc = accuracy(the_move);
        let expected = (low + high) as f64 / 2.0 * acc;
        let (mut ko_now, mut hko, mut lock) = (0.0, 0.0, 0.0);
        if expected > 0.0 {
            let turns = ceil_div(opponent.hp, expected);
            ko_now = if low >= opponent.hp && low > 0 { acc } else { 0.0 };
            hko = 1.0 / turns;
            if CHOICE_ITEMS.contains(&me.item.as_str()) {
                lock = self.lock_risk(the_move, turn, state, db);
            }
        }
        let threatened = turn.theirs.strongest as f64 / me.totals.hp as f64;
        f[KO_NOW] = ko_now;
        f[HKO_PROGRESS] = hko;
        f[LOCK_RISK] = lock;
        f[EXCHANGE_EDGE] = turn.edge / EDGE_CAP;
        f[FODDER_EXPLOIT] = if any_effect && high == 0 && threatened < FODDER_THREAT_FRACTION { 1.0 } else { 0.0 };
        f[SETUP_CONCESSION] = -self.setup_concession(turn, state, db);
        f[WINCON_PRESERVATION] = if turn.me == turn.wincon && turn.edge < 0.0 { -1.0 } else { 0.0 };
        f[WINCON_SUPPORT] = self.wincon_support(low, high, turn, state, db);
        Some(f)
    }

    fn setup_concession(&self, turn: &Turn, state: &State, db: &Database) -> f64 {
        if !opponent_gains_from_a_free_turn(turn, state, db) {
            return 0.0;
        }
        if turn.mine.best_expected <= 0.0 {
            return 1.0;
        }
        let turns_to_ko = ceil_div(poke(state, turn.opponent).hp, turn.mine.best_expected);
        if turns_to_ko <= 2.0 {
            0.0
        } else {
            (turns_to_ko - 2.0).min(4.0) / 4.0
        }
    }

    fn lock_risk(&self, the_move: &Move, turn: &Turn, state: &State, db: &Database) -> f64 {
        let their = 1 - turn.side;
        let standing: Vec<usize> =
            (0..state.sides[their].team.len()).filter(|&i| !state.sides[their].team[i].fainted()).collect();
        let immune = standing.iter().filter(|&&i| damage_range(the_move, turn.me, (their, i), state, db).1 == 0).count();
        -(immune as f64) / standing.len() as f64
    }

    fn effect_features(&mut self, the_move: &Move, turn: &Turn, low: i32, high: i32, state: &State, db: &Database) -> Features {
        let mut f = status_features(the_move, turn, state);
        f[HAZARD_VALUE] = hazard_feature(the_move, turn, state, db);
        f[HAZARD_REMOVAL] = removal_feature(the_move, turn, state);
        f[KNOCK_PROGRESS] = knock_feature(the_move, turn, state);
        f[HEAL_TURNS] = heal_feature(the_move, turn, low, high, state);
        f[SETUP_VALUE] = setup_feature(the_move, turn, state);
        f[PHAZE_VALUE] = phaze_feature(the_move, turn, state);
        f[SETUP_DENIAL] = denial_feature(the_move, turn, state, db);
        f[TIMER_VALUE] = f[TIMER_VALUE].max(seed_feature(the_move, turn, state));
        f
    }

    fn wincon_support(&mut self, low: i32, high: i32, turn: &Turn, state: &State, db: &Database) -> f64 {
        if turn.me == turn.wincon {
            return 0.0;
        }
        match self.wincon_checker(turn, state, db) {
            Some(checker) if checker == turn.opponent => {
                damage_share((low + high) as f64 / 2.0, poke(state, checker), 1.0)
            }
            _ => 0.0,
        }
    }

    fn wincon_checker(&mut self, turn: &Turn, state: &State, db: &Database) -> Option<PokeRef> {
        let their = 1 - turn.side;
        let mut best: Option<(PokeRef, f64)> = None;
        for (index, foe) in state.sides[their].team.iter().enumerate() {
            if foe.fainted() {
                continue;
            }
            let edge = self.cached_edge(turn.wincon, (their, index), state, db);
            if best.is_none_or(|(_, low)| edge < low) {
                best = Some(((their, index), edge));
            }
        }
        best.map(|(at, _)| at)
    }

    // -- switches --

    fn switch_features(&mut self, incoming: PokeRef, turn: &Turn, voluntary: bool, state: &State, db: &Database) -> Features {
        let (my_side, their_side) = (&state.sides[turn.side], &state.sides[1 - turn.side]);
        let arriving = poke(state, incoming);
        let opponent = poke(state, turn.opponent);
        let incoming_edge = self.cached_edge(incoming, turn.opponent, state, db);
        let mut matchup_gain = incoming_edge / EDGE_CAP;
        let mut wincon = 0.0;
        if voluntary {
            matchup_gain -= turn.edge / EDGE_CAP;
            if turn.me == turn.wincon && turn.edge < 0.0 {
                wincon += 1.0;
            }
        }
        if incoming == turn.wincon && incoming_edge < 0.0 {
            wincon -= 1.0;
        }
        let their_hit = self.pair_offense(turn.opponent, incoming, state, db).threat_expected;
        let my_hit = self.pair_offense(incoming, turn.opponent, state, db).best_expected;
        let outspeeds =
            effective_speed(arriving, my_side, &state.field) > effective_speed(opponent, their_side, &state.field);
        let mut f = [0.0; GENES];
        f[MATCHUP_GAIN] = matchup_gain;
        f[TEMPO_COST] = if voluntary { -1.0 } else { 0.0 };
        f[ENTRY_COST] = -(entry_hazard_chip(arriving, my_side, db) as f64) / arriving.totals.hp as f64;
        f[WINCON_PRESERVATION] = wincon;
        f[INCOMING_DAMAGE_TAKEN] = -(their_hit / arriving.totals.hp as f64).min(2.0);
        f[INCOMING_DAMAGE_DEALT] = damage_share(my_hit, opponent, 2.0);
        f[INCOMING_OUTSPEEDS] = if outspeeds { 1.0 } else { 0.0 };
        if voluntary {
            f[LOCK_RELIEF] = lock_relief(turn, state, db);
            f[VOLATILE_RELIEF] = volatile_relief(poke(state, turn.me));
            f[DEBUFF_RELIEF] = debuff_relief(poke(state, turn.me));
        }
        f[SETUP_DENIAL] = if opponent_gains_from_a_free_turn(turn, state, db) {
            let my_hit = self.pair_offense(incoming, turn.opponent, state, db).best_expected;
            damage_share(my_hit, opponent, 1.0)
        } else {
            0.0
        };
        f
    }

    // -- matchup matrix and win condition --

    fn pair_offense(&mut self, from: PokeRef, to: PokeRef, state: &State, db: &Database) -> Offense {
        self.cache.entry((from, to)).or_insert_with(|| offense(from, to, state, db)).clone()
    }

    fn cached_edge(&mut self, mine: PokeRef, theirs: PokeRef, state: &State, db: &Database) -> f64 {
        let my_best = self.pair_offense(mine, theirs, state, db).best_expected;
        let their_threat = self.pair_offense(theirs, mine, state, db).threat_expected;
        let (me, them) = (poke(state, mine), poke(state, theirs));
        let faster = effective_speed(me, &state.sides[mine.0], &state.field)
            >= effective_speed(them, &state.sides[theirs.0], &state.field);
        exchange_edge_from(me.hp, them.hp, my_best, their_threat, faster)
    }

    /// The teammate with the best full-strength spread against their survivors; the first of equals.
    fn wincon(&mut self, state: &State, side: usize, db: &Database) -> PokeRef {
        let their = 1 - side;
        let survivors: Vec<usize> =
            (0..state.sides[their].team.len()).filter(|&i| !state.sides[their].team[i].fainted()).collect();
        let mut best: Option<(PokeRef, f64)> = None;
        for index in 0..state.sides[side].team.len() {
            if state.sides[side].team[index].fainted() {
                continue;
            }
            let total = survivors.iter().fold(0.0, |sum, &foe| sum + self.cached_edge((side, index), (their, foe), state, db));
            if best.is_none_or(|(_, top)| total > top) {
                best = Some(((side, index), total));
            }
        }
        // Nobody standing is only asked about in a finished battle; the active stands in.
        best.map_or(active(state, side), |(at, _)| at)
    }
}

fn opponent_gains_from_a_free_turn(turn: &Turn, state: &State, db: &Database) -> bool {
    let opponent = poke(state, turn.opponent);
    let moves: Vec<&Move> = opponent.moves.iter().take(4).filter_map(|name| db.move_named(name)).collect();
    let boosts = moves.iter().any(|m| ["ATTACK", "SP_ATTACK"].iter().any(|s| self_boost(m, s) > 0));
    boosts || moves.iter().any(|m| heals(m))
}

fn survival(opponent: &Pokemon, turn: &Turn) -> f64 {
    if turn.mine.best_expected > 0.0 {
        ceil_div(opponent.hp, turn.mine.best_expected)
    } else {
        f64::INFINITY
    }
}

fn status_features(the_move: &Move, turn: &Turn, state: &State) -> Features {
    let mut f = [0.0; GENES];
    let me = poke(state, turn.me);
    let opponent = poke(state, turn.opponent);
    let Some(status) = inflicted(the_move, true).and_then(Status::parse) else { return f };
    if opponent.status != Status::None || status_cannot_land(status, opponent, me, state) {
        return f;
    }
    f[TIMER_VALUE] = survival(opponent, turn).min(6.0) / 6.0;
    f[PARA_SPEED_CONTROL] = if status == Status::Paralysis && turn.opponent_faster { 1.0 } else { 0.0 };
    let physical = turn.theirs.best_category.as_deref() == Some("PHYSICAL");
    f[BURN_DISABLE] = if status == Status::Burn && physical { 1.0 } else { 0.0 };
    if status == Status::Sleep {
        f[SLEEP_TEMPO] = (turn.theirs.best_expected * SLEEP_TURNS / std::cmp::max(1, me.hp) as f64).min(1.0);
    }
    f
}

fn seed_feature(the_move: &Move, turn: &Turn, state: &State) -> f64 {
    if inflicted(the_move, false) != Some("LEECH_SEED") {
        return 0.0;
    }
    let opponent = poke(state, turn.opponent);
    if opponent.volatiles.contains_key("LEECH_SEED") || has_type(opponent, "GRASS") {
        return 0.0;
    }
    survival(opponent, turn).min(6.0) / 6.0
}

fn denial_feature(the_move: &Move, turn: &Turn, state: &State, db: &Database) -> f64 {
    let opponent = poke(state, turn.opponent);
    let theirs = positive_boosts(opponent);
    if coded(the_move, &["HAZE"]) {
        return ((theirs - positive_boosts(poke(state, turn.me))) / BOOST_CAP).max(0.0);
    }
    let Some(blocked) = inflicted(the_move, false) else { return 0.0 };
    if !DENIAL_VOLATILES.contains(&blocked) || opponent.volatiles.contains_key(blocked) {
        return 0.0;
    }
    if !opponent_gains_from_a_free_turn(turn, state, db) && theirs == 0.0 {
        return 0.0;
    }
    (DENIAL_BASE + theirs / BOOST_CAP).min(1.0)
}

fn hazard_feature(the_move: &Move, turn: &Turn, state: &State, db: &Database) -> f64 {
    let Some(hazard) = set_hazard(the_move) else { return 0.0 };
    let their = 1 - turn.side;
    if state.sides[their].hazards.get(hazard).unwrap_or(0) >= hazard_cap(hazard).unwrap_or(1) {
        return 0.0;
    }
    let with_it = hazard_toll(their, Some(hazard), turn.me, state, db);
    with_it - hazard_toll(their, None, turn.me, state, db)
}

fn removal_feature(the_move: &Move, turn: &Turn, state: &State) -> f64 {
    let Some(style) = the_move.effects.iter().find_map(|e| match e {
        Effect::RemoveHazardsEffect { style } => Some(style.as_str()),
        _ => None,
    }) else {
        return 0.0;
    };
    let mut relief = hazard_pressure(&state.sides[turn.side]);
    if style == "DEFOG" {
        relief -= hazard_pressure(&state.sides[1 - turn.side]);
    }
    relief
}

fn knock_feature(the_move: &Move, turn: &Turn, state: &State) -> f64 {
    let opponent = poke(state, turn.opponent);
    if !coded(the_move, &["KNOCK_OFF_ITEM", "TRICK"]) || opponent.item == "NONE" {
        return 0.0;
    }
    if opponent.item == "HEAVY_DUTY_BOOTS" && state.sides[1 - turn.side].hazards.keys().next().is_some() {
        return 1.0;
    }
    0.6
}

fn heal_feature(the_move: &Move, turn: &Turn, low: i32, high: i32, state: &State) -> f64 {
    if !heals(the_move) {
        return 0.0;
    }
    let me = poke(state, turn.me);
    let opponent = poke(state, turn.opponent);
    let drained = the_move.effects.iter().find_map(|e| match e {
        Effect::DamageEffect { drain_percent: Some(p), .. } if *p != 0.0 => Some(*p),
        _ => None,
    });
    let mut healed = match drained {
        Some(percent) => {
            let dealt = ((low + high) as f64 / 2.0).min(opponent.hp as f64);
            let healed = percent * dealt;
            let healed = if opponent.ability == "LIQUID_OOZE" { -healed } else { healed };
            if healed < 0.0 {
                return (healed / std::cmp::max(1, me.totals.hp) as f64).max(-1.0);
            }
            healed
        }
        None => {
            let fraction = the_move
                .effects
                .iter()
                .find_map(|e| match e {
                    Effect::HealEffect { fraction } => Some(*fraction),
                    _ => None,
                })
                .unwrap_or(0.5);
            fraction * me.totals.hp as f64
        }
    };
    healed = healed.min((me.totals.hp - me.hp) as f64);
    let kept = healed - residual_drain(me, state) as f64;
    if kept <= 0.0 {
        return 0.0;
    }
    (kept / std::cmp::max(1, turn.theirs.strongest) as f64).min(2.0) / 2.0
}

fn with_free_turn(value: f64, free: bool, the_move: &Move) -> f64 {
    if !free || !BOOST_STATS.iter().any(|s| self_boost(the_move, s) > 0) {
        return value;
    }
    value.clamp(FREE_TURN_SETUP, 1.0)
}

fn setup_feature(the_move: &Move, turn: &Turn, state: &State) -> f64 {
    let me = poke(state, turn.me);
    let opponent = poke(state, turn.opponent);
    let free = cannot_act(opponent);
    if !free && turn.theirs.threat_expected >= me.hp as f64 {
        return 0.0;
    }
    let speed_bonus = speed_flip_bonus(the_move, turn, state);
    let stat = match turn.mine.best_category.as_deref() {
        Some("PHYSICAL") if turn.mine.best_expected > 0.0 => "ATTACK",
        Some("SPECIAL") if turn.mine.best_expected > 0.0 => "SP_ATTACK",
        _ => return with_free_turn(speed_bonus, free, the_move),
    };
    let gain = self_boost(the_move, stat);
    if gain == 0 {
        return with_free_turn(speed_bonus, free, the_move);
    }
    let current = me.stage(stat);
    let boosted = turn.mine.best_expected * stage_multiplier((current + gain).min(6)) / stage_multiplier(current);
    let turns_now = ceil_div(opponent.hp, turn.mine.best_expected);
    let turns_after = ceil_div(opponent.hp, boosted);
    with_free_turn((turns_now - turns_after).min(3.0) / 3.0 + speed_bonus, free, the_move)
}

fn speed_flip_bonus(the_move: &Move, turn: &Turn, state: &State) -> f64 {
    let gain = self_boost(the_move, "SPEED");
    if gain <= 0 {
        return 0.0;
    }
    let me = poke(state, turn.me);
    let current = effective_speed(me, &state.sides[turn.side], &state.field);
    let theirs = effective_speed(poke(state, turn.opponent), &state.sides[1 - turn.side], &state.field);
    if current > theirs {
        return 0.0;
    }
    let stage = me.stage("SPEED");
    let boosted = current as f64 * stage_multiplier((stage + gain).min(6)) / stage_multiplier(stage);
    if boosted > theirs as f64 {
        1.0
    } else {
        0.0
    }
}

fn phaze_feature(the_move: &Move, turn: &Turn, state: &State) -> f64 {
    if !the_move.force_switch {
        return 0.0;
    }
    let their = &state.sides[1 - turn.side];
    let others = their.team.iter().enumerate().any(|(i, p)| !p.fainted() && (1 - turn.side, i) != turn.opponent);
    if !others {
        return 0.0;
    }
    let opponent = poke(state, turn.opponent);
    let boosts = ["ATTACK", "SP_ATTACK"].iter().map(|s| opponent.stage(s).max(0)).sum::<i32>() as f64 / 4.0;
    let chip = if their.hazards.keys().next().is_some() && bootless_bench(their) > 0 { 0.5 } else { 0.0 };
    (boosts + chip).min(1.5)
}

fn lock_relief(turn: &Turn, state: &State, db: &Database) -> f64 {
    let me = poke(state, turn.me);
    let Some(locked) = me.choice_locked_move else { return 0.0 };
    let Some(the_move) = me.moves.get(locked).and_then(|name| db.move_named(name)) else { return 1.0 };
    let stuck = damage_range(the_move, turn.me, turn.opponent, state, db).1;
    let best = turn.mine.strongest;
    if best <= 0 {
        1.0
    } else {
        1.0 - (stuck as f64 / best as f64).min(1.0)
    }
}

fn volatile_relief(me: &Pokemon) -> f64 {
    me.volatiles
        .keys()
        .map(|v| match v.as_str() {
            "CONFUSION" => 1.0,
            "LEECH_SEED" => 0.8,
            "TAUNT" | "ENCORE" => 0.5,
            _ => 0.0,
        })
        .sum()
}

fn debuff_relief(me: &Pokemon) -> f64 {
    let dropped = -RELIEF_STATS.iter().map(|s| me.stage(s).min(0)).sum::<i32>();
    dropped.min(6) as f64 / 6.0
}

/// `choose_order`: the team sorted by its summed exchange edge against theirs, best first, ties
/// kept in listed order. Asked of a scratch board with nobody's switch-in having run, as the
/// Python asks it.
pub fn choose_order(ours: Vec<Pokemon>, theirs: Vec<Pokemon>, db: &Database) -> Vec<usize> {
    let (size, foes) = (ours.len(), theirs.len());
    let scratch = State::new(Side::new(ours), Side::new(theirs));
    let totals: Vec<f64> = (0..size)
        .map(|mine| (0..foes).fold(0.0, |sum, foe| sum + exchange_edge((0, mine), (1, foe), &scratch, db)))
        .collect();
    let mut order: Vec<usize> = (0..size).collect();
    order.sort_by(|a, b| totals[*b].partial_cmp(&totals[*a]).unwrap_or(std::cmp::Ordering::Equal));
    order
}
