//! One turn, in the order the Python resolves it.
//!
//! The order matters twice over. Once because it decides the battle, and once because **it decides
//! where the tape is**: every draw either engine makes has to be the same draw in the same place,
//! or the two immediately start reading each other's randomness. The sequence within a turn is
//!
//!   1. one tie-break probability per side, in `order_actions`
//!   2. then, per move that resolves: the accuracy roll (only if the move has an accuracy), the
//!      crit roll, and the damage roll — in that order, and only on the paths that reach them
//!
//! Every early return in the Python happens *before* its draw would have been taken, which is why
//! a miss costs one draw and a no-effect costs none. Porting that faithfully is most of the work.
//!
//! Restricted on purpose, for now: damaging moves and switches, no abilities, no items, no
//! volatiles, no residuals. Anything outside that raises `Unsupported` rather than guessing, so a
//! scenario that wanders out of the ported subset fails loudly instead of diverging quietly.

use crate::battle::{FormSnapshot, Pokemon, Side, State, Status, SLOT_NAMES};
use crate::abilities::{apply_damage_calc, Calc};
use crate::damage::{calculate_hit, Payload, Rolls};
use crate::hooks::{on_after_hit, on_switch_in, Hit};
use crate::data::{BaseStats, Database, Effect, Move};
use crate::log::{Event, Log};
use crate::tape::Tape;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    Move {
        slot: usize,
        /// Unleash this slot's move through the held Z-Crystal, once per battle — `action.z_move`
        /// in Python's own `Action`. Same slot, same PP, resolved through the crystal's upgrade
        /// rather than as a separate action type.
        z_move: bool,
    },
    Switch { to: usize },
}

/// Why this engine stopped, and the distinction the whole project rests on.
///
/// `Unported` means the scenario asked for something not written yet — a fair, expected answer
/// while most of the game is still missing, and one the harness may skip over. `Diverged` means
/// the two engines have already parted: the tape it is reading was recorded by a Python that made
/// different draws in a different order. That is never skippable. Collapsing the two was how five
/// real disagreements sat in a green test run labelled "outside the ported slice".
#[derive(Debug)]
pub enum Refusal {
    Unported(String),
    Diverged(String),
}

impl Refusal {
    pub fn reason(&self) -> &str {
        match self {
            Refusal::Unported(why) | Refusal::Diverged(why) => why,
        }
    }

    /// The process exit code, which is the only channel the Python harness reads this on.
    pub fn exit_code(&self) -> i32 {
        match self {
            Refusal::Unported(_) => 2,
            Refusal::Diverged(_) => 3,
        }
    }
}

/// What a side's action counts as for ordering: switches resolve before moves, as in
/// `_CATEGORY_ORDER`.
fn category_of(action: &Action) -> i32 {
    match action {
        Action::Switch { .. } => 0,
        Action::Move { .. } => 1,
    }
}

/// The export writes the bracket by name; these are the values of the Python's `PriorityLevel`,
/// taken from the enum rather than guessed — the first version of this invented plausible names
/// like "QUICK" and panicked on contact with the real data.
fn priority_of(the_move: &Move) -> Result<i32, Refusal> {
    Ok(match the_move.priority.as_str() {
        "HELPING_HAND" => 5,
        "PROTECT" => 4,
        "FAKE_OUT" => 3,
        "E_SPEED" => 2,
        "QUICK_ATTACK" => 1,
        "NORMAL" => 0,
        "VITAL_THROW" => -1,
        "FOCUS_PUNCH" => -3,
        "AVALANCHE" => -4,
        "COUNTER" => -5,
        "ROAR" => -6,
        "TRICK_ROOM" => -7,
        other => return Err(Refusal::Unported(format!("unmapped priority bracket {other:?}"))),
    })
}

/// Sorts exactly as `_sort_key` does: category, then priority (descending), then speed
/// (descending), then the tie-break draw. Rust sorts ascending, so speed and priority are negated
/// the same way the Python negates speed.
fn order_actions(
    state: &mut State,
    actions: &[Action; 2],
    db: &Database,
    tape: &mut Tape,
) -> Result<Vec<usize>, Refusal> {
    // Drawn for both sides before anything resolves, in side order — the Python builds this dict
    // by comprehension over `actions`, which is insertion-ordered 0 then 1.
    let tie_breakers = [tape.probability()?, tape.probability()?];
    let mut keys: Vec<(i32, i32, i32, i32, f64, usize)> = Vec::new();
    for side in 0..2 {
        let speed = effective_speed(state.sides[side].active_pokemon(), &state.sides[side], &state.field);
        let mut category = category_of(&actions[side]);
        let mut in_bracket_jump = 0;
        let priority = match &actions[side] {
            Action::Switch { .. } => 0,
            Action::Move { slot, .. } => {
                // `the_move` borrows `db`, not the active Pokemon, so it stays valid across the
                // several fresh `active_pokemon()` re-borrows below — `bracket_jump` needs `state`
                // by mutable reference, for Custap Berry's own consumption, which an `actor`
                // reference held across the call would not allow.
                let the_move = move_in_slot(state.sides[side].active_pokemon(), *slot, db)?;
                // Pursuit's whole point: it catches its target on the way out, so it resolves
                // ahead of the switch that would otherwise take the target off the field first.
                // Switches already sort before every move, so this is the one thing that sorts
                // before them.
                if the_move.name == "Pursuit" && matches!(actions[1 - side], Action::Switch { .. }) {
                    category = category_of(&Action::Switch { to: 0 }) - 1;
                }
                let actor = state.sides[side].active_pokemon();
                let at_full_hp = actor.hp == actor.totals.hp;
                let bonus = crate::inline::priority_bonus(
                    &actor.ability,
                    &the_move.move_type,
                    &the_move.category,
                    the_move.healing,
                    at_full_hp,
                );
                // `_bracket_jump`: Quick Claw, then Quick Draw, then Custap Berry — each a chance
                // (or, for Custap, a guaranteed but HP-gated) jump to the front of the priority
                // bracket. Drawn unconditionally in that order, same as the Python — Mycelium
                // Might overrides the result below rather than skipping the draw, so the tape
                // still owes this a probability (Quick Claw's, then Quick Draw's) on every move a
                // Mycelium Might Pokemon makes, status or not.
                in_bracket_jump = -bracket_jump(state, side, tape)?;
                let actor = state.sides[side].active_pokemon();
                if actor.ability == "MYCELIUM_MIGHT" && the_move.category == "STATUS" {
                    in_bracket_jump = 1; // status moves go last within their bracket
                }
                priority_of(the_move)? + bonus
            }
        };
        // Trick Room inverts the speed sort itself, rather than the speed stat, so a paralysed
        // Pokemon halved by the status is still slower under Trick Room than one that is not.
        let speed_key = if state.field.pseudo_weather.contains_key("TRICK_ROOM") { speed } else { -speed };
        keys.push((category, -priority, in_bracket_jump, speed_key, tie_breakers[side], side));
    }
    keys.sort_by(|a, b| a.partial_cmp(b).expect("no NaNs in a sort key"));
    Ok(keys.into_iter().map(|k| k.5).collect())
}

/// `_bracket_jump`: Quick Claw's 20% roll, then Quick Draw's 30% roll, then Custap Berry's
/// guaranteed jump under a quarter HP — each an early return, so a Pokemon holding Quick Claw
/// *and* somehow carrying Quick Draw (impossible in practice, but the Python does not special-case
/// it either) draws for both before either wins. Takes `state` rather than a plain `&Pokemon`
/// because Custap Berry consumes itself.
fn bracket_jump(state: &mut State, side: usize, tape: &mut Tape) -> Result<i32, Refusal> {
    let actor = state.sides[side].active_pokemon();
    if actor.item == "QUICK_CLAW" && tape.probability()? < 0.2 {
        return Ok(1);
    }
    if actor.ability == "QUICK_DRAW" && tape.probability()? < 0.3 {
        return Ok(1);
    }
    if actor.item == "CUSTAP_BERRY" && 4 * actor.hp <= actor.totals.hp {
        let actor = state.sides[side].active_mut();
        actor.last_consumed_item = actor.item.clone();
        actor.item = "NONE".to_string();
        actor.item_consumed = true;
        return Ok(1);
    }
    Ok(0)
}

/// `priority.effective_speed`, as much of it as is ported.
///
/// Paralysis halves it, and that halving decides who moves first — which is not a detail. A
/// paralysed Kangaskhan that this engine let move first took a paralysis check off the tape at the
/// moment the Python was rolling accuracy for somebody else, and the two engines never recovered.
/// Everything else in the Python's version is an ability, an item or a field effect, none of them
/// ported, and each will have to be added here as it lands.
pub fn effective_speed(pokemon: &Pokemon, side: &Side, field: &crate::battle::Field) -> i32 {
    let mut speed = pokemon.effective("SPEED");
    if pokemon.status == Status::Paralysis && pokemon.ability != "QUICK_FEET" {
        speed /= 2;
    }
    if pokemon.ability == "QUICK_FEET" && pokemon.status != Status::None {
        speed = speed * 3 / 2;
    }
    if crate::inline::doubles_speed_in(&pokemon.ability, &field.weather) {
        speed *= 2;
    }
    if pokemon.ability == "SURGE_SURFER" && field.terrain == "ELECTRIC" {
        speed *= 2; // the weather doublers' terrain cousin, and the whole of Alolan Raichu's identity
    }
    if pokemon.ability == "UNBURDEN" && pokemon.item == "NONE" && pokemon.item_consumed {
        speed *= 2;
    }
    if pokemon.paradox_boost.as_deref() == Some("SPEED") {
        speed = speed * 3 / 2;
    }
    if side.tailwind_turns > 0 {
        speed *= 2;
    }
    if pokemon.item == "CHOICE_SCARF" {
        speed = speed * 3 / 2;
    }
    std::cmp::max(1, speed)
}

fn move_in_slot<'a>(actor: &Pokemon, slot: usize, db: &'a Database) -> Result<&'a Move, Refusal> {
    let name = actor
        .moves
        .get(slot)
        .ok_or_else(|| Refusal::Unported(format!("{} has no move in slot {slot}", actor.nickname)))?;
    db.move_named(name)
        .ok_or_else(|| Refusal::Unported(format!("unknown move {name:?}")))
}

/// Everything this engine has not learned yet. A scenario containing one of these is refused up
/// front rather than played wrongly — the whole point of the differential work is that silence is
/// the one unacceptable failure mode.
/// Moves the Python special-cases by name that this engine has nonetheless implemented, and which
/// the `coded_moves` net must therefore stop refusing.
///
/// Deliberately short and explicit. That net is why silent wrong answers have been rare; a move
/// only comes off it once its Python behaviour has been read, ported, and agreed about across a
/// sweep.
pub const PORTED_CODED_MOVES: [&str; 7] = [
    "Struggle",
    // Sleep Talk (the redirect at the top of `resolve_move` plus `power::sleep_talk_choice`) and
    // Roost (the `ROOSTED` volatile set in `apply_heal`) both have real, ported behaviour behind
    // the name. King's Shield does not: its only mention anywhere in the Python is Aegislash's
    // forme swap in `formes.stance_forme`, gated on Stance Change — an ability still refused, so
    // this name is "ordinary despite being named" the same way the Z-move bases and the reachers
    // are in `power::PORTED_ORDINARY_DESPITE_BEING_NAMED`. It lives here instead because that list
    // is about `power.rs`'s tables specifically, and this one isn't in any of them.
    "Sleep Talk",
    "Roost",
    "King's Shield",
    // Trick's only other mention anywhere is a turn-order special case for the butler's own
    // ability (eating whatever is about to be Tricked onto him *before* it lands) — Nine Lives is
    // still refused, so nobody can ever actually take that branch. Ordinary, like King's Shield.
    "Trick",
    // `_DELAYED_DAMAGE_MOVES = {"Future Sight", "Doom Desire"}` names both by literal string —
    // the check that keeps their own `DamageEffect` from landing immediately, which
    // `resolve_move`'s own `_DELAYED_DAMAGE_MOVES` check mirrors. Real ported behaviour, not a
    // false positive like the two above; grouped here anyway since this is where a move's name
    // gets freed from the by-name gate.
    "Future Sight",
    "Doom Desire",
];

/// Every coded move this engine has learned: Struggle, plus the rules in `power.rs`.
///
/// Built once. These are asked on every move resolution, and rebuilding a sorted Vec each time
/// cost half the engine's throughput — 284k turns a second down to 143k — for a list that cannot
/// change while the process is running.
pub fn ported_coded_moves() -> &'static std::collections::HashSet<&'static str> {
    static ONCE: std::sync::OnceLock<std::collections::HashSet<&'static str>> = std::sync::OnceLock::new();
    ONCE.get_or_init(|| {
        let mut all: std::collections::HashSet<&str> = PORTED_CODED_MOVES.into_iter().collect();
        all.extend(crate::power::PORTED);
        all.extend(crate::power::PORTED_TYPE_OVERRIDES);
        all.extend(crate::power::PORTED_FAILURES);
        all.extend(crate::power::SCREEN_BREAKERS);
        all.extend(crate::power::PORTED_ORDINARY_DESPITE_BEING_NAMED);
        // The AST sweep flags most charge moves too — their names sit in `_REACHES_THROUGH`,
        // `_CHARGE_TURN_BOOSTS` or `_SUN_SKIP_CHARGE` as string literals. Without this they would
        // be refused by the coded-move gate before the `the_move.charge` check below ever ran.
        all.extend(crate::power::PORTED_CHARGES);
        all
    })
}

/// `_DEFENDER_FACING_TARGETS`: the targets a Protect can stand in the way of.
pub const DEFENDER_FACING: [&str; 3] = ["SINGLE_OPPONENT", "ALL_ADJACENT_ENEMIES", "ALL_ADJACENT"];

/// The volatiles this engine knows. Everything else in `ExtraStatus` still makes a move unplayable.
///
/// These two are most of what volatiles actually are in practice: 32 moves can flinch and 18 can
/// confuse, against one apiece for Leech Seed, Taunt, Encore and the rest.
/// Volatiles carried by `InflictStatusEffect` but *not* dispatched through `apply_volatile` —
/// each has its own bespoke landing rule the Python gives it in `_apply_status` rather than the
/// generic one, so routing it through the generic path would silently play it wrong.
pub const PORTED_BESPOKE_VOLATILES: [&str; 4] = ["LOCKED_MOVE", "ENCORE", "DISABLE", "SUBSTITUTE"];

pub const PORTED_VOLATILES: [&str; 14] = [
    "FLINCH",
    "CONFUSION",
    "PROTECT",
    "ENDURE",
    "FOCUS_ENERGY",
    "LEECH_SEED",
    "NIGHTMARE",
    "SALT_CURE",
    "PARTIALLY_TRAPPED",
    "YAWN",
    "TAUNT",
    "DESTINY_BOND",
    "IDENTIFIED",
    "MIRACLE_EYE",
];

/// `CodedMoveKind` variants `apply_coded` has learned. Named by the exported enum member, which is
/// what `variant` carries — not by move name, since several moves share a kind (the four
/// `WEATHER_HEAL` moves, the two `CURE_PARTY` ones).
pub const PORTED_CODED_KINDS: [&str; 25] = [
    "REST",
    "WEATHER_HEAL",
    "PAIN_SPLIT",
    "STRENGTH_SAP",
    "BELLY_DRUM",
    "HAZE",
    "COURT_CHANGE",
    "CURSE",
    "TIDY_UP",
    "PERISH_SONG",
    "CURE_SELF",
    "CURE_PARTY",
    "KNOCK_OFF_ITEM",
    "TRICK",
    "SKILL_SWAP",
    "ROLE_PLAY",
    "ENTRAINMENT",
    "WORRY_SEED",
    "SIMPLE_BEAM",
    "WISH",
    "HEALING_WISH",
    "REVIVAL_BLESSING",
    "SHED_TAIL",
    "FUTURE_SIGHT",
    "TRANSFORM",
];

/// Shadow Tag / Arena Trap / Magnet Pull. Confirmed by reading `battle_sim/` end to end: these
/// three are registered nowhere except `battle_sim/engine/choices.py`'s `_trapped()` — no
/// damage-calc hook, no switch hook, nothing. Their entire effect is restricting which switches
/// `legal_actions` offers, which `step()` never consults for any action (Move and Switch are not
/// re-validated for legality either) — so there is genuinely no dispatch code for these to join
/// anywhere else. Ported here, with nothing behind them, rather than left refusing every Dugtrio/
/// Gothitelle/Magnezone team that fields one and never switches into a position where it matters.
const PORTED_NO_STEP_EFFECT_ABILITIES: [&str; 3] = ["SHADOW_TAG", "ARENA_TRAP", "MAGNET_PULL"];

/// Everything this engine has implemented, gathered from the modules that implement it so a name
/// cannot be claimed in one place and missing from the other.
///
/// A Pokemon carrying live behaviour absent from these is refused, not played with part of its
/// rules missing. 220 abilities and 193 items are live in the Python; these say how far along the
/// port is, and a name joins one only once it has been agreed across a sweep.
pub fn ported_abilities() -> &'static std::collections::HashSet<&'static str> {
    static ONCE: std::sync::OnceLock<std::collections::HashSet<&'static str>> = std::sync::OnceLock::new();
    ONCE.get_or_init(|| {
        let mut all: std::collections::HashSet<&str> = crate::abilities::PORTED.into_iter().collect();
        all.extend(crate::hooks::PORTED_ABILITIES);
        all.extend(crate::hooks::PORTED_BEFORE_MOVE_ABILITIES);
        all.extend(crate::hooks::PORTED_ON_FAINT_ABILITIES);
        all.extend(crate::hooks::PORTED_ON_SWITCH_OUT_ABILITIES);
        all.extend(crate::hooks::PORTED_PARADOX_ABILITIES);
        all.extend(crate::inline::PORTED_ABILITIES);
        all.extend(crate::hooks::PORTED_RESIDUAL_ABILITIES);
        all.extend(crate::power::PORTED_ATE_ABILITIES);
        all.extend(crate::hooks::PORTED_TYPE_SHIFTER_ABILITIES);
        all.extend(PORTED_NO_STEP_EFFECT_ABILITIES);
        all.extend(crate::hooks::MOLD_BREAKERS);
        all.extend(crate::formes::PORTED_FORME_ABILITIES);
        all
    })
}

pub fn ported_items() -> &'static std::collections::HashSet<&'static str> {
    static ONCE: std::sync::OnceLock<std::collections::HashSet<&'static str>> = std::sync::OnceLock::new();
    ONCE.get_or_init(|| {
        let mut all: std::collections::HashSet<&str> = crate::items::PORTED.into_iter().collect();
        all.extend(crate::hooks::PORTED_ITEMS);
        all.extend(crate::inline::PORTED_ITEMS);
        all.extend(crate::hooks::PORTED_RESIDUAL_ITEMS);
        all.extend(crate::hooks::PORTED_RKS_MEMORIES);
        all.extend(crate::items::PORTED_TYPE_ONLY_PLATES);
        all
    })
}

/// "move:SLOT:Name" / "zmove:SLOT:Name" / "switch:Nickname", resolved against who is actually out —
/// the one parser `bin/replay.rs` and `python.rs` both drive a battle from, so a scenario or a
/// self-play action string means exactly one thing everywhere it is read.
pub fn parse_action(named: &str, side: &Side) -> Result<Action, String> {
    let (kind, rest) = named.split_once(':').ok_or_else(|| format!("malformed action {named:?}"))?;
    match kind {
        // "move:THIRD:Power Whip" / "zmove:THIRD:Power Whip" — the slot decides, and the name is
        // carried alongside so a scenario stays readable and so a mismatch between the two is
        // caught rather than trusted. Two slots can hold the same move once a short set has been
        // padded. A Z-move is a flag on this same action in Python (`action.z_move`), not a
        // different action type — the base moves in `zmoves::SIGNATURE_BASES` are ordinary moves,
        // playable on that basis, that also happen to be unleashable through a held crystal.
        "move" | "zmove" => {
            let (slot_name, move_name) = rest
                .split_once(':')
                .ok_or_else(|| format!("action {named:?} does not name a slot"))?;
            let slot = SLOT_NAMES
                .iter()
                .position(|s| *s == slot_name)
                .ok_or_else(|| format!("no slot called {slot_name:?}"))?;
            let actor = side.active_pokemon();
            match actor.moves.get(slot) {
                Some(found) if found == move_name => Ok(Action::Move { slot, z_move: kind == "zmove" }),
                Some(found) => Err(format!("slot {slot_name} holds {found:?}, not {move_name:?}")),
                None => Err(format!("{} has no slot {slot_name}", actor.nickname)),
            }
        }
        "switch" => side
            .team
            .iter()
            .position(|p| p.nickname == rest)
            .map(|to| Action::Switch { to })
            .ok_or_else(|| format!("nobody on this side is called {rest:?}")),
        other => Err(format!("unsupported action kind {other:?}")),
    }
}

/// Why this Pokemon cannot be played, if it cannot.
///
/// Reading an ability off a Pokemon and doing nothing with it is the exact failure this project is
/// built to catch — a wrong answer delivered in silence. Until Intimidate is written here, a
/// scenario containing one stops the run.
pub fn unsupported_pokemon(pokemon: &Pokemon, db: &Database) -> Option<String> {
    if db.live_abilities.contains(&pokemon.ability) && !ported_abilities().contains(pokemon.ability.as_str()) {
        return Some(format!("{} has {}, which is not ported", pokemon.nickname, pokemon.ability));
    }
    // A Mega Stone, Primal orb or Ultranecrozium Z is `live` (Python's own `mega_stones()` union)
    // but will never appear in `ported_items()` — Mega Evolution is a whole feature, not an item
    // effect, so it is recognised here by table membership instead. Held for no reason at all
    // (a Mega Stone on a Pikachu) it is simply inert, exactly as the Python plays it.
    let holds_forme_item = db.mega_formes.iter().any(|f| f.item == pokemon.item)
        || db.ultra_burst_formes.iter().any(|f| f.item == pokemon.item);
    if db.live_items.contains(&pokemon.item) && !ported_items().contains(pokemon.item.as_str()) && !holds_forme_item {
        return Some(format!("{} is holding {}, which is not ported", pokemon.nickname, pokemon.item));
    }
    // This Pokemon's item or moveset would transform it per the game's own rules (a Mega Stone, a
    // Primal orb, Ultranecrozium Z on the right Necrozma, or Rayquaza knowing Dragon Ascent), but
    // `formes::mega_forme` refuses to actually do it — the target forme's own ability is not
    // ported yet. Refused here rather than left quietly sitting in its base forme all battle,
    // which would be a different Pokemon from the one Python is playing turn for turn.
    if crate::formes::wants_to_transform(&pokemon.species_name, &pokemon.item, &pokemon.moves, db)
        && crate::formes::mega_forme(&pokemon.species_name, &pokemon.item, &pokemon.moves, db).is_none()
    {
        return Some(format!("{} would transform into a forme this engine has not ported", pokemon.nickname));
    }
    None
}

/// A class of thing this engine has not learned. Named separately from the message so progress can
/// be counted by cause — a message has the move's name in it, and grouping on that counts moves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Gap {
    CodedByName,
    NoModelledEffect,
    UserOrFieldEffect,
    VariablePower,
    Volatile,
}

impl Gap {
    pub fn label(&self) -> &'static str {
        match self {
            Gap::CodedByName => "special-cased by name in the Python engine",
            Gap::NoModelledEffect => "carries an effect kind this engine cannot read",
            Gap::UserOrFieldEffect => "does something to its user or the field",
            Gap::VariablePower => "power is computed at use time",
            Gap::Volatile => "inflicts an unported volatile",
        }
    }
}

/// Everything this engine has not learned yet. A scenario containing one of these is refused up
/// front rather than played wrongly — the whole point of the differential work is that silence is
/// the one unacceptable failure mode.
pub fn unsupported(the_move: &Move, db: &Database) -> Option<Gap> {
    if db.coded_moves.contains(&the_move.name) && !ported_coded_moves().contains(the_move.name.as_str()) {
        return Some(Gap::CodedByName);
    }
    // `healing` is deliberately absent. It is Showdown's `heal` flag, and the Python branches on it
    // in exactly one rule — Triage's +3 priority — which is an ability this engine still refuses.
    // Everything else reading it is AI policy, not mechanics. A healing move that actually heals
    // carries a `HealEffect`, which arrives here as `Unmodelled` and is refused on its own merits;
    // refusing the flag as well cost the engine all twelve draining moves for nothing.
    // `force_switch`, `recharges` and `self_destructs` are ported; a pivot is not — it needs the
    // turn loop to send somebody in mid-turn. All 17 charge moves are ported; an unrecognised one
    // would mean the database grew an eighteenth without `power::PORTED_CHARGES` learning about it.
    if the_move.charge && !crate::power::PORTED_CHARGES.contains(&the_move.name.as_str()) {
        return Some(Gap::UserOrFieldEffect);
    }
    // An effect list that is empty is not a gap: the Python has nothing to apply either, so the
    // move announces itself, does nothing, and reports MoveFailed — which this engine already does.
    // 102 moves are in that state, and the differential is what decides whether that reading is
    // right, not this comment.
    for effect in &the_move.effects {
        match effect {
            Effect::DamageEffect { power: None, .. } if !crate::power::PORTED.contains(&the_move.name.as_str()) => {
                return Some(Gap::VariablePower)
            }
            Effect::InflictStatusEffect { status, .. }
                if Status::parse(status).is_none()
                    && !PORTED_VOLATILES.contains(&status.as_str())
                    && !PORTED_BESPOKE_VOLATILES.contains(&status.as_str()) =>
            {
                return Some(Gap::Volatile)
            }
            // Caught here as well as at use time. A refusal that only happens when the move is
            // actually reached is still a refusal — but it makes the coverage number a promise
            // rather than a measurement, and this report is supposed to be the honest one.
            Effect::CodedEffect { variant } if !PORTED_CODED_KINDS.contains(&variant.as_str()) => {
                return Some(Gap::CodedByName)
            }
            Effect::Unmodelled => return Some(Gap::NoModelledEffect),
            _ => {}
        }
    }
    None
}

/// The same question, answered with a sentence naming the move — what a refusal actually prints.
pub fn unsupported_reason(the_move: &Move, db: &Database) -> Option<String> {
    unsupported(the_move, db).map(|gap| format!("{}: {}", the_move.name, gap.label()))
}

pub fn step(
    state: &mut State,
    actions: [Action; 2],
    db: &Database,
    tape: &mut Tape,
) -> Result<Log, Refusal> {
    let mut log = Log::new();
    // Indexed rather than zipped: the loop touches `state.sides`, `actions` and `choosers` by the
    // same index, and a zip over one of them would only hide that.
    #[allow(clippy::needless_range_loop)]
    for side in 0..2 {
        state.sides[side].acted_this_turn = false;
        // What this side picked, so Sucker Punch can ask whether an attack is still coming.
        state.sides[side].chosen_slot = match &actions[side] {
            Action::Move { slot, .. } => Some(*slot),
            Action::Switch { .. } => None,
        };
        // "This turn" is what Counter and Mirror Coat mean, so the record starts empty.
        let active = state.sides[side].active_mut();
        active.last_hit_taken = 0;
        active.last_hit_category = None;
    }
    // `_send_out_leads`: the leads' switch-in abilities fire before the turn is ordered, in
    // descending speed, so the slower weather-setter's weather is the one that stands. It takes no
    // draws, but it does log, and the entries belong at the very top of turn zero.
    if state.turn == 0 {
        let mut leads = [0usize, 1];
        leads.sort_by_key(|side| {
            std::cmp::Reverse(effective_speed(
                state.sides[*side].active_pokemon(),
                &state.sides[*side],
                &state.field,
            ))
        });
        for side in leads {
            on_switch_in(state, side, &mut log);
        }
    }
    // `ON_TURN_START`: emitted once here, before `order_actions` reads `effective_speed` below —
    // which is the entire reason a Paradox ability needs this hook and not just `ON_SWITCH_IN` and
    // `ON_RESIDUAL`.
    crate::hooks::ability_on_turn_start(state, &mut log);
    // `_resolve_mega_evolution`: same reasoning, same placement — a Mega Evolution's new Speed has
    // to be visible before `order_actions` reads it below.
    crate::formes::resolve_forme_changes(state, &actions, db, &mut log);
    // `_resolve_hp_formes`, before the turn is ordered: a Pokemon that crossed its line last turn
    // (or arrived below it) sorts on the forme it is actually in.
    crate::formes::resolve_hp_formes(state, db, &mut log);
    // Who chose each action, by team slot. A Pokemon dragged out by Roar before it acted takes its
    // queued move with it — resolving the slot anyway means the replacement uses whatever happens
    // to be in that slot, which is a different move belonging to a different Pokemon.
    let choosers = [state.sides[0].active, state.sides[1].active];
    let order = order_actions(state, &actions, db, tape)?;
    for side in order {
        if state.outcome.is_some() {
            break;
        }
        if state.sides[side].active != choosers[side] {
            continue; // phazed out before acting
        }
        // A fainted Pokemon cannot move, but its side must still send out a replacement — and
        // that switch is the first thing to resolve. Skipping the side outright left the Rust
        // engine a Pokemon behind for the rest of the battle.
        if state.sides[side].active_pokemon().fainted() && matches!(actions[side], Action::Move { .. }) {
            continue;
        }
        match &actions[side] {
            Action::Switch { to } => {
                let sent_out = state.sides[side].team[*to].nickname.clone();
                let withdrew = switch_out(state, side, *to, &mut log);
                state.register_active(side);
                let arriving = state.sides[side].active_mut();
                arriving.just_switched_in = true;
                arriving.turns_active = 0; // a fresh stint, so Fake Out is live again
                log.push(Event::Switched { side: side as i32, withdrew, sent_out });
                // `_execute_switch` lays the hazards on before it emits ON_SWITCH_IN, so Stealth
                // Rock bites before Intimidate looks across the field.
                crate::field::entry_hazards(state, side, db, &mut log);
                // An arrival the hazards knock out is unregistered and returns: its switch-in
                // ability never fires. So a Pokemon that dies to Stealth Rock on the way in does
                // not get to Intimidate on the way past.
                if !state.sides[side].active_pokemon().fainted() {
                    grant_switch_in_bonuses(state, side, &mut log);
                    // `_execute_switch` logs the swap and then emits, so an Intimidate lands after
                    // the line announcing who arrived.
                    on_switch_in(state, side, &mut log);
                }
            }
            Action::Move { slot, z_move } => {
                // `_mold_breaker_window`: wraps the whole of `_execute_move`, can-act checks and all.
                let window = crate::hooks::open_mold_breaker_window(state, side);
                // Whether the *chosen* move melts its own user free, decided before anything is
                // rolled: Flame Wheel, Sacred Fire and Scald thaw and go off anyway, with no 20%
                // check taken. The Python reads the chosen move here too, so a Struggle
                // substitution later does not change it.
                let defrosting = {
                    let actor = state.sides[side].active_pokemon();
                    actor.status == Status::Freeze && move_in_slot(actor, *slot, db)?.defrosts_user
                };
                // Same idea, same reason: the *chosen* slot, read before any redirect could
                // substitute a different move in.
                let sleep_talking = {
                    let actor = state.sides[side].active_pokemon();
                    actor.status == Status::Sleep && move_in_slot(actor, *slot, db)?.name == "Sleep Talk"
                };
                if can_act(state, side, defrosting, sleep_talking, tape, &mut log)? {
                    resolve_move(state, side, *slot, *z_move, sleep_talking, db, tape, &mut log)?
                } else {
                    // A skipped turn breaks the consecutive-Protect chain.
                    state.sides[side].active_mut().protect_streak = 0;
                }
                crate::hooks::close_mold_breaker_window(state, side, window);
            }
        }
        // Set after the action, not before: Analytic asks whether the *other* side has already
        // moved, and a side that has just finished moving is exactly what that means.
        state.sides[side].acted_this_turn = true;
        // The hit that just landed may have crossed an HP forme's line.
        crate::formes::resolve_hp_formes(state, db, &mut log);
        // Both sides, after *every* completed action, not just the one that just acted — a drop or
        // an Eject Button this action triggered on the other side pulls that side's Pokemon before
        // it ever gets its own turn, if it hasn't acted yet this turn. Two full passes, in the
        // Python's own order: every pending Eject Pack is armed (and logged) before any switch at
        // all executes, so a side-0 pivot already holding `needs_switch` doesn't jump the queue
        // ahead of a side-1 Eject Pack this same action just triggered.
        for eject_side in 0..2 {
            resolve_eject_packs(state, eject_side, &mut log);
        }
        for switch_side in 0..2 {
            resolve_pending_switches(state, switch_side, db, &mut log);
        }
        let was_decided = state.outcome.is_some();
        state.update_outcome();
        // `_update_outcome` announces the result the moment it is decided, once.
        if !was_decided {
            if let Some(outcome) = state.outcome {
                log.push(Event::BattleEnded { outcome: outcome.name().to_string() });
            }
        }
    }
    if state.outcome.is_none() {
        residuals(state, db, tape, &mut log)?;
        // Residual chip crosses HP-forme lines too.
        crate::formes::resolve_hp_formes(state, db, &mut log);
        state.update_outcome();
        if let Some(outcome) = state.outcome {
            log.push(Event::BattleEnded { outcome: outcome.name().to_string() });
        }
    }
    // `_tick_turns_active`, which runs at turn end whether or not the battle is over: the flag's
    // rule is "every turn-end except the one I entered on", so it clears here rather than when its
    // owner acted. Clearing it after the action instead cost Stakeout its whole effect, since the
    // Pokemon it punishes is the one that came in *this* turn and has not moved yet.
    // Indexed rather than zipped: the body reaches into `state.sides` and `choosers` by the same
    // index, and zipping one of them would only disguise that.
    #[allow(clippy::needless_range_loop)]
    for side in 0..2 {
        if !state.sides[side].active_pokemon().fainted() {
            state.sides[side].active_mut().just_switched_in = false;
            // `turns_active` counts only for whoever actually chose this turn's action: a Pokemon
            // that arrived mid-turn has not had a turn of its own yet, and Fake Out is still live
            // for it next turn.
            if state.sides[side].active == choosers[side] {
                state.sides[side].active_mut().turns_active += 1;
            }
        }
    }
    // `ON_TURN_END`, emitted unconditionally after the turn-active tick — its only subscriber is
    // the balloon's pop, with an empty payload, so side 0 again.
    crate::hooks::check_air_balloons(state, 0, &mut log);
    state.turn += 1;
    Ok(log)
}

/// `_stall_check`: consecutive Protect-likes fail with odds 1 - 1/3^n, capped at n = 6.
///
/// Any non-stalling move clears the streak, so the counter really does mean "in a row". A failed
/// check clears it too — the chain is broken by the failure itself, not only by doing something
/// else.
/// `_break_rolling`: a miss or a Protect ends the run, power and commitment together — but the
/// commitment only if it was Rollout or Ice Ball's. A rampage's lock survives a miss; that is the
/// entire reason it needs its own `_is_rolling_slot` gate rather than clearing unconditionally.
fn break_rolling(state: &mut State, side: usize) {
    let active = state.sides[side].active_mut();
    active.rolling_hits = 0;
    let is_rolling_slot = active.locked_slot.is_some_and(|slot| {
        active.volatiles.contains_key("LOCKED_MOVE")
            && active.moves.get(slot).is_some_and(|name| crate::power::ROLLING_MOVES.contains(&name.as_str()))
    });
    if is_rolling_slot {
        active.volatiles.remove("LOCKED_MOVE");
        active.locked_slot = None;
    }
}

fn stall_check(state: &mut State, side: usize, the_move: &Move, tape: &mut Tape) -> Result<bool, Refusal> {
    let actor = state.sides[side].active_mut();
    if !the_move.stalling {
        actor.protect_streak = 0;
        return Ok(true);
    }
    if actor.protect_streak > 0 {
        let odds = 3.0_f64.powi(-actor.protect_streak.min(6));
        if tape.probability()? >= odds {
            state.sides[side].active_mut().protect_streak = 0;
            return Ok(false);
        }
    }
    state.sides[side].active_mut().protect_streak += 1;
    Ok(true)
}

/// `_confusion_allows_acting`: tick the counter, then a third of the time hurt yourself instead.
///
/// The odd branch is the last one, and it is the Python's: on the two-thirds where the Pokemon does
/// *not* hit itself, a `CantAct` entry is logged with reason "confused" and then the move goes off
/// anyway. Reproduced as written — this engine's job is to agree with that one.
fn confusion_allows_acting(state: &mut State, side: usize, tape: &mut Tape, log: &mut Log) -> Result<bool, Refusal> {
    let pokemon = state.sides[side].active_mut();
    let left = pokemon.volatiles.get("CONFUSION").copied().unwrap_or(0) - 1;
    pokemon.volatiles.insert("CONFUSION".to_string(), left);
    let nickname = pokemon.nickname.clone();
    if left <= 0 {
        state.sides[side].active_mut().volatiles.remove("CONFUSION");
        log.push(Event::StatusCleared {
            side: side as i32,
            pokemon: nickname,
            clearance: "confusion_ended".into(),
        });
        return Ok(true);
    }
    if tape.probability()? < 1.0 / 3.0 {
        let pokemon = state.sides[side].active_mut();
        // Its own Attack against its own Defence, 40 base power, typeless and never a crit.
        let attack = pokemon.effective("ATTACK");
        let defence = pokemon.effective("DEFENCE");
        let amount = std::cmp::max(1, ((2 * pokemon.level) / 5 + 2) * 40 * attack / defence / 50 + 2);
        pokemon.take_damage(amount);
        log.push(Event::ConfusionSelfHit { side: side as i32, pokemon: nickname, amount });
        return Ok(false);
    }
    log.push(Event::CantAct { side: side as i32, pokemon: nickname, reason: "confused".into() });
    Ok(true)
}

/// Withdraw whatever is active and send out `to`, returning the outgoing nickname.
///
/// The resets are `switching._execute_switch`'s, and they are the point of the function: a Pokemon
/// that leaves the field drops its stat stages, its volatiles and — the easy one to miss — its
/// toxic counter, so it comes back poisoned but counting from zero again. Leaving the stages on was
/// how a Rust battle reached -6 Special Attack against a Python that had long since reset to 0.
///
/// The Protect streak goes with them. It is invisible in the log — a Protect that fails to the
/// streak and one that fails for having no effect both print `MoveFailed` — so the only sign of
/// getting it wrong is the draw count, which is what caught it.
///
/// The rest of `_execute_switch` clears fields this engine does not have yet (choice lock, encore,
/// charging slot). They arrive with the volatiles milestone; until then there is nothing to clear.
fn switch_out(state: &mut State, side: usize, to: usize, log: &mut Log) -> String {
    // `ON_SWITCH_OUT`: the very first thing `_execute_switch` does, and only when the outgoing
    // Pokemon did not faint — a fainted switch is a replacement, not a choice, and the Python's own
    // comment on the event says as much ("not emitted for fainted switches").
    if !state.sides[side].active_pokemon().fainted() {
        crate::hooks::ability_on_switch_out(state, side, log);
    }
    // `_release_anyone_it_was_holding`: a wrap ends when whoever was doing the wrapping leaves. In
    // singles the side whose opponent is wrapped is the side holding them, because a wrapped
    // Pokemon is exactly the one that cannot switch — so the one walking out is always the one
    // letting go, whether it chose to or fainted.
    state.sides[1 - side].active_mut().volatiles.remove("PARTIALLY_TRAPPED");
    withdraw(&mut state.sides[side], to)
}

fn withdraw(side: &mut Side, to: usize) -> String {
    let departing_index = side.active;
    let outgoing = side.active_mut();
    for value in outgoing.stages.values_mut() {
        *value = 0;
    }
    outgoing.volatiles.clear();
    outgoing.protect_streak = 0;
    // A Fury Cutter run does not survive its owner leaving the field.
    outgoing.rolling_hits = 0;
    outgoing.charging_slot = None;
    outgoing.locked_slot = None;
    outgoing.last_move_slot = None;
    outgoing.encored_slot = None;
    outgoing.disabled_slot = None;
    outgoing.choice_locked_move = None;
    outgoing.flash_fire_active = false;
    outgoing.paradox_boost = None;
    outgoing.paradox_from_booster = false;
    if outgoing.status == Status::Toxic {
        outgoing.status_turns = 0;
    }
    let withdrew = outgoing.nickname.clone();
    side.active = to;
    // `restore_form`: a no-op for the near-totality of Pokemon that never transformed. Stat
    // stages are deliberately not part of the snapshot — the loop just above already zeroes them
    // on every switch-out, transformed or not, matching the Python's own separate reset.
    if let Some(snapshot) = side.transforms.remove(&departing_index) {
        let outgoing = &mut side.team[departing_index];
        outgoing.base_stats = snapshot.base_stats;
        outgoing.nature = snapshot.nature;
        outgoing.ivs = snapshot.ivs;
        outgoing.evs = snapshot.evs;
        outgoing.types = snapshot.types;
        outgoing.ability = snapshot.ability;
        outgoing.moves = snapshot.moves;
        outgoing.pp = snapshot.pp;
        outgoing.recompute_totals();
    }
    withdrew
}

/// End-of-turn status chip, from `residuals._status_chip`. Burn is a sixteenth, poison an eighth,
/// and toxic climbs by a sixteenth a turn — counting up *before* it bites, which is why a fresh
/// toxic takes a sixteenth rather than nothing.
///
/// Sides are ticked in order, which is the order the Python emits `ON_RESIDUAL` for them.
fn residuals(state: &mut State, db: &Database, tape: &mut Tape, log: &mut Log) -> Result<(), Refusal> {
    // `_apply_residuals` in order: the field's own clocks first, then each side — its chips, then
    // its durations. A sandstorm that expires this turn still chips on the way out only if the
    // tick and the chip are in this order, which is why the field goes first.
    crate::field::tick_field(state, log);
    // `_apply_residuals` calls `state.bus.emit(ON_RESIDUAL, ...)` once per standing side, and each
    // emit reaches *every* handler on the bus. Paradox's own handler never checks `context.actor is
    // pokemon`, so *both* actives re-evaluate at 9500 in *each* emit — a Quark Drive on side 1
    // activates during side 0's own emit, before side 0 has taken so much as a weather tick. The
    // item type shifter at the same priority does check its actor, so Multitype/RKS System re-sync
    // only in their own side's emit: after the whole of side 0's chips, for side 1. Interleaved by
    // registration order, as the bus sorts equal priorities.
    for side in 0..2 {
        if !state.sides[side].active_pokemon().fainted() {
            for holder in crate::hooks::by_ability_registration(state) {
                crate::hooks::evaluate_paradox(state, holder, log);
                if holder == side {
                    crate::hooks::sync_type_from_item(state, side, log);
                }
            }
            // ResidualOrder, top to bottom (Paradox and the type shifter just above): WEATHER
            // (9000), then the weather abilities (8500), the terrain (8400), the cures and the
            // recovery items, then the status chip at 6000, then everything below it.
            crate::field::weather_residual(state, side, log);
            crate::hooks::solar_power_chip(state, side, log);
            crate::hooks::residual_before_status(state, side, tape, log)?;
            status_chip(state, side, log);
            crate::hooks::residual_after_status(state, side, tape, log)?;
            // SPEED_BOOST (1000) and the balloon's pop (`EventPriority.ITEM`, also 1000), by
            // registration — the pop for *both* actives, since this emit reaches every handler,
            // and logged against side 0 whichever side holds the balloon (see `check_air_balloon`).
            // Three handlers at most, each at its own stamp — the actor's ability, and both
            // actives' items — and one Pokemon's ability ahead of its own item at a tie.
            let mut handlers = [(0u64, 0u8, side, true), (0, 1, 0, false), (0, 1, 1, false)];
            handlers[0].0 = state.sides[side].active_pokemon().ability_registered_at;
            for entry in handlers.iter_mut().skip(1) {
                entry.0 = state.sides[entry.2].active_pokemon().registered_at;
            }
            handlers.sort();
            for (_, _, holder, is_ability) in handlers {
                if is_ability {
                    crate::hooks::speed_boost_residual(state, holder, log);
                } else {
                    crate::hooks::check_air_balloon(state, holder, 0, log);
                }
            }
            // One faint line for the whole residual pass, whichever chip did it — the Python logs
            // it in `_apply_residuals` after the emit, not inside any handler. Announcing it from
            // the status chip alone was right until a sandstorm got a kill of its own.
            if state.sides[side].active_pokemon().fainted() {
                let nickname = state.sides[side].active_pokemon().nickname.clone();
                log.push(Event::Fainted { side: side as i32, pokemon: nickname });
            }
            // `_tick_volatiles`, which runs for the active whatever its status was — including one
            // that just fainted to a chip above. Each of these lasts exactly the turn it started.
            for gone in ["FLINCH", "PROTECT", "ENDURE", "ROOSTED"] {
                state.sides[side].active_mut().volatiles.remove(gone);
            }
            // Perish Song's own countdown, separate from `_tick_countdown` because it faints its
            // victim rather than merely clearing a status — three more turns of shared silence,
            // then the fourth turn's tick is the one that takes it down.
            if let Some(left) = state.sides[side].active_pokemon().volatiles.get("PERISH").copied() {
                if left - 1 > 0 {
                    state.sides[side].active_mut().volatiles.insert("PERISH".to_string(), left - 1);
                } else {
                    let active = state.sides[side].active_mut();
                    active.volatiles.remove("PERISH");
                    let all_of_it = active.hp;
                    active.take_damage(all_of_it);
                    let nickname = active.nickname.clone();
                    log.push(Event::Fainted { side: side as i32, pokemon: nickname });
                }
            }
            // `_tick_countdown`: Taunt, Encore and Disable each count down here, and Encore/Disable
            // also release the slot they were pinning the moment the count reaches zero.
            tick_countdown(state, side, "TAUNT", "taunt_ended", log);
            if tick_countdown(state, side, "ENCORE", "encore_ended", log) {
                state.sides[side].active_mut().encored_slot = None;
            }
            if tick_countdown(state, side, "DISABLE", "disable_ended", log) {
                state.sides[side].active_mut().disabled_slot = None;
            }
        }
        // `_tick_side_durations`, run per side whether or not its active fainted this same pass —
        // Wish, Future Sight and Tailwind are side-level clocks, not Pokemon-level ones.
        if state.sides[side].wish_turns > 0 {
            state.sides[side].wish_turns -= 1;
            if state.sides[side].wish_turns == 0 {
                let recipient = state.sides[side].active_pokemon();
                if !recipient.fainted() {
                    let pending = state.sides[side].wish_pending;
                    let recipient = state.sides[side].active_mut();
                    let before = recipient.hp;
                    recipient.hp = std::cmp::min(recipient.totals.hp, recipient.hp + pending);
                    let healed = recipient.hp - before;
                    if healed > 0 {
                        let nickname = recipient.nickname.clone();
                        log.push(Event::Healed { side: side as i32, pokemon: nickname, amount: healed });
                    }
                }
                state.sides[side].wish_pending = 0;
            }
        }
        if state.sides[side].future_sight_turns > 0 {
            state.sides[side].future_sight_turns -= 1;
            if state.sides[side].future_sight_turns == 0 {
                resolve_future_sight(state, side, db, tape, log)?;
            }
        }
        crate::field::tick_side(state, side, log);
    }
    Ok(())
}

/// `_tick_countdown`: decrement a plain volatile counter, clear it and log `StatusCleared` the
/// turn it reaches zero. Returns whether it just expired, so a caller with a slot to release along
/// with it (Encore, Disable) knows to do that too — mirroring the Python's `on_expire` callback.
fn tick_countdown(state: &mut State, side: usize, volatile: &str, clearance: &str, log: &mut Log) -> bool {
    let Some(left) = state.sides[side].active_pokemon().volatiles.get(volatile).copied() else {
        return false;
    };
    let left = left - 1;
    if left > 0 {
        state.sides[side].active_mut().volatiles.insert(volatile.to_string(), left);
        return false;
    }
    state.sides[side].active_mut().volatiles.remove(volatile);
    let nickname = state.sides[side].active_pokemon().nickname.clone();
    log.push(Event::StatusCleared { side: side as i32, pokemon: nickname, clearance: clearance.to_string() });
    true
}


fn status_chip(state: &mut State, side: usize, log: &mut Log) {
    let active = state.sides[side].active_pokemon();
    if crate::inline::ignores_indirect_damage(active) {
        return;
    }
    // Poison Heal already turned this into a heal, so there is nothing left to take.
    if active.ability == "POISON_HEAL" && matches!(active.status, Status::Poison | Status::Toxic) {
        return;
    }
    let active = state.sides[side].active_mut();
    let (source, amount) = match active.status {
        Status::Burn => ("burn", (active.totals.hp / 16).max(1)),
        Status::Poison => ("poison", (active.totals.hp / 8).max(1)),
        Status::Toxic => {
            active.status_turns += 1;
            ("toxic", (active.totals.hp * active.status_turns / 16).max(1))
        }
        _ => return,
    };
    let dealt = active.take_damage(amount);
    let nickname = active.nickname.clone();
    // No faint line here: `residuals` announces it once for the whole pass.
    log.push(Event::ResidualDamage {
        side: side as i32,
        pokemon: nickname,
        source: source.into(),
        amount: dealt,
    });
}

/// `_can_act`, minus the volatiles and abilities that are not ported. The order is the Python's
/// and so is where each draw falls: freeze rolls to thaw, sleep counts down without drawing, and
/// paralysis rolls last.
fn can_act(
    state: &mut State,
    side: usize,
    defrosting: bool,
    sleep_talking: bool,
    tape: &mut Tape,
    log: &mut Log,
) -> Result<bool, Refusal> {
    let (status, nickname) = {
        let active = state.sides[side].active_pokemon();
        (active.status, active.nickname.clone())
    };
    // First of every gate, ahead of the freeze roll — so a frozen Pokemon spending its recharge
    // turn does not take a thaw draw it was never entitled to.
    if state.sides[side].active_pokemon().volatiles.contains_key("MUST_RECHARGE") {
        state.sides[side].active_mut().volatiles.remove("MUST_RECHARGE");
        log.push(Event::CantAct { side: side as i32, pokemon: nickname, reason: "recharge".into() });
        return Ok(false);
    }
    if status == Status::Freeze {
        // No draw at all when the move defrosts its user — which is the point. Rolling the check
        // anyway thawed the right Pokemon for the wrong reason and left the tape one draw short
        // for the rest of the battle.
        if !defrosting && tape.probability()? >= 0.2 {
            log.push(Event::CantAct { side: side as i32, pokemon: nickname, reason: "frozen".into() });
            return Ok(false);
        }
        state.sides[side].active_mut().status = Status::None;
        log.push(Event::StatusCleared {
            side: side as i32,
            pokemon: nickname.clone(),
            clearance: "thawed".into(),
        });
    }
    if state.sides[side].active_pokemon().status == Status::Sleep {
        let active = state.sides[side].active_mut();
        active.status_turns -= 1;
        if active.status_turns <= 0 {
            active.status = Status::None;
            active.status_turns = 0;
            log.push(Event::StatusCleared {
                side: side as i32,
                pokemon: nickname.clone(),
                clearance: "woke".into(),
            });
        } else if !sleep_talking {
            log.push(Event::CantAct { side: side as i32, pokemon: nickname, reason: "asleep".into() });
            return Ok(false);
        }
    }
    // Flinch, then confusion, then paralysis — the Python's order, and therefore the order the
    // draws come off the tape. Confusion rolls before paralysis does, which matters on any turn
    // where both could fire.
    if state.sides[side].active_pokemon().volatiles.contains_key("FLINCH") {
        log.push(Event::CantAct { side: side as i32, pokemon: nickname, reason: "flinch".into() });
        return Ok(false);
    }
    if state.sides[side].active_pokemon().volatiles.contains_key("CONFUSION")
        && !confusion_allows_acting(state, side, tape, log)?
    {
        return Ok(false);
    }
    if state.sides[side].active_pokemon().status == Status::Paralysis && tape.probability()? < 0.25 {
        let nickname = state.sides[side].active_pokemon().nickname.clone();
        log.push(Event::CantAct { side: side as i32, pokemon: nickname, reason: "paralysis".into() });
        return Ok(false);
    }
    Ok(true)
}

/// `_targets_defender`: which effects a knocked-out target stops taking.
///
/// A status effect is aimed at the defender unless the *move* targets its user — the effect's own
/// `to_self` flag is not what decides this in the Python, and using it here would let Hypnosis-like
/// self-targeting cases drift. A stat change is aimed at whoever the effect names.
fn targets_defender(effect: &Effect, the_move: &Move) -> bool {
    match effect {
        Effect::DamageEffect { .. } => true,
        Effect::InflictStatusEffect { .. } => the_move.target != "SELF",
        Effect::StatStageChangeEffect { target, .. } => target == "TARGET",
        Effect::FixedDamageEffect { .. } => true,
        // Everything else lands on the user, on a side, or on the field, so a knocked-out target
        // does not stop it. That is the Python's `_targets_defender` returning False by default.
        _ => false,
    }
}

#[allow(clippy::too_many_arguments)] // reads against the Python's own `_execute_move`, one argument per thing it reads
fn resolve_move(
    state: &mut State,
    side: usize,
    slot: usize,
    z_move: bool,
    sleep_talking: bool,
    db: &Database,
    tape: &mut Tape,
    log: &mut Log,
) -> Result<(), Refusal> {
    let other = 1 - side;
    // The bond lasts until its user's next action, so it is cleared right here — before that action
    // is even decided — rather than by a countdown. A Destiny Bond that killed something last turn
    // does not still threaten to on this one.
    state.sides[side].active_mut().volatiles.remove("DESTINY_BOND");
    // A Choice item redirects to whatever it already locked, checked before Encore/rampage/charge —
    // the Python's own order. `None` both before the first use and for anyone not holding one.
    let choice_locked_slot = {
        let actor = state.sides[side].active_pokemon();
        if matches!(actor.item.as_str(), "CHOICE_BAND" | "CHOICE_SCARF" | "CHOICE_SPECS") {
            actor.choice_locked_move
        } else {
            None
        }
    };
    // Encore, a rampage or a charge each continue a use the Pokemon already committed to, whatever
    // slot the recorded action names — a short moveset pads itself by repeating its first move, so
    // the same move can sit in several slots and the recorded action can name a different one of
    // them. Applied in the Python's own order: Encore first, then a rampage, then a charge.
    let encored = {
        let actor = state.sides[side].active_pokemon();
        actor.volatiles.contains_key("ENCORE") && actor.encored_slot.is_some()
    };
    let rampaging = {
        let actor = state.sides[side].active_pokemon();
        actor.volatiles.contains_key("LOCKED_MOVE") && actor.locked_slot.is_some()
    };
    let releasing_charge = {
        let actor = state.sides[side].active_pokemon();
        actor.volatiles.contains_key("CHARGING") && actor.charging_slot.is_some()
    };
    let mut slot = slot;
    if let Some(locked) = choice_locked_slot {
        slot = locked;
    }
    if encored {
        slot = state.sides[side].active_pokemon().encored_slot.expect("checked above");
    }
    if rampaging {
        slot = state.sides[side].active_pokemon().locked_slot.expect("checked above");
    }
    if releasing_charge {
        slot = state.sides[side].active_pokemon().charging_slot.expect("checked above");
    }
    let slot = slot;
    let chosen = {
        let actor = state.sides[side].active_pokemon();
        move_in_slot(actor, slot, db)?.clone()
    };
    // `if slot is attacker.disabled_slot: DisabledBlocked; return` — *before* PP is spent or
    // Struggle substituted, and using the move that was actually chosen (Struggle never can be:
    // an empty slot is never the one Disable silenced, since Disable needs a move to have been
    // used first).
    if state.sides[side].active_pokemon().disabled_slot == Some(slot) {
        let nickname = state.sides[side].active_pokemon().nickname.clone();
        log.push(Event::DisabledBlocked { side: side as i32, pokemon: nickname, the_move: chosen.name });
        return Ok(());
    }
    // An empty slot is Struggle, and Struggle costs nothing — there is nothing left to spend. The
    // Python substitutes here rather than at choice time, so the recorded action still names the
    // move that was picked. Missing this only showed up past turn 60, once the PP had run out.
    // Never true mid rampage or while releasing a charge: PP for the whole run was spent up front.
    let empty = !rampaging
        && !releasing_charge
        && state.sides[side].active_pokemon().pp.get(crate::battle::SLOT_NAMES[slot]) == Some(&0);
    let the_move = if empty {
        db.move_named("Struggle")
            .ok_or_else(|| Refusal::Unported("the database has no Struggle".into()))?
            .clone()
    } else {
        chosen
    };
    if let Some(why) = unsupported_reason(&the_move, db) {
        return Err(Refusal::Unported(why));
    }

    // Spent before anything resolves, as `_spend_pp` does it: a move that misses still costs its
    // point, which is why this is here rather than after the hit lands. Rampaging or releasing a
    // charge spends nothing — the whole run was paid for on its first turn.
    if !empty && !rampaging && !releasing_charge {
        // Pressure doubles the cost of a move that faces its holder — not a move it merely
        // stands beside, which is what `DEFENDER_FACING` already tells apart everywhere else.
        let cost = if state.sides[other].active_pokemon().ability == "PRESSURE"
            && DEFENDER_FACING.contains(&the_move.target.as_str())
        {
            2
        } else {
            1
        };
        if let Some(left) = state.sides[side].active_mut().pp.get_mut(crate::battle::SLOT_NAMES[slot]) {
            *left = (*left - cost).max(0);
        }
        // Leppa Berry: only when *this* spend is what brought the slot to zero — a slot already
        // empty before this turn took the Struggle branch above instead, never reaching here.
        if state.sides[side].active_pokemon().pp[crate::battle::SLOT_NAMES[slot]] == 0
            && state.sides[side].active_pokemon().item == "LEPPA_BERRY"
        {
            let restored = std::cmp::min(10, the_move.pp);
            let attacker = state.sides[side].active_mut();
            attacker.last_consumed_item = attacker.item.clone();
            attacker.item = "NONE".to_string();
            attacker.item_consumed = true;
            attacker.pp.insert(crate::battle::SLOT_NAMES[slot].to_string(), restored);
            let nickname = attacker.nickname.clone();
            log.push(Event::PpRestored { side: side as i32, pokemon: nickname, the_move: the_move.name.clone() });
        }
    }
    // `action.z_move`: spent after the base move's own PP is paid — a Z-move costs the slot it
    // upgrades, not extra. The slot keeps its own move for logging: every `MoveUsed` name is read
    // as a move that fills a slot, and a Z-move name fills none. `unleashed_as` carries the Z-move's
    // own name through to whichever `MoveUsed` below actually fires, so it reads as one move
    // becoming its Z-move, not two.
    let mut unleashed_as: Option<String> = None;
    let the_move = if z_move && !state.sides[side].has_used_z_move {
        let attacker_item = state.sides[side].active_pokemon().item.clone();
        match crate::zmoves::z_move_for(&attacker_item, &the_move, db) {
            Some(upgraded) => {
                unleashed_as = Some(upgraded.name.clone());
                state.sides[side].has_used_z_move = true;
                Move { name: the_move.name.clone(), ..upgraded }
            }
            None => the_move,
        }
    } else {
        the_move
    };
    // Taunt: after PP is spent, before the move is announced. A status move turned aside here still
    // cost its user the point, which is the whole reason this sits after the PP-spend block above
    // rather than before it, unlike Disable's check.
    if state.sides[side].active_pokemon().volatiles.contains_key("TAUNT") && the_move.category == "STATUS" {
        let nickname = state.sides[side].active_pokemon().nickname.clone();
        log.push(Event::TauntBlocked { side: side as i32, pokemon: nickname, the_move: the_move.name.clone() });
        return Ok(());
    }
    // Prankster's Gen 7 drawback: a status move from a Prankster user, aimed at a Dark type, does
    // nothing. Here — after Taunt, before the choice lock and last-move slot are touched, and before
    // Magic Bounce is even asked — because that is where the Python returns. It reads the defender's
    // plain `types`, not Roost-adjusted ones, and applies whether or not Prankster's priority was what
    // sorted the move where it was.
    if state.sides[side].active_pokemon().ability == "PRANKSTER"
        && the_move.category == "STATUS"
        && DEFENDER_FACING.contains(&the_move.target.as_str())
        && state.sides[other].active_pokemon().types.iter().flatten().any(|t| t == "DARK")
    {
        log.push(Event::MoveUsed {
            side: side as i32,
            pokemon: state.sides[side].active_pokemon().nickname.clone(),
            the_move: the_move.name.clone(),
            unleashed_as,
        });
        let nickname = state.sides[other].active_pokemon().nickname.clone();
        log.push(Event::DoesNotAffect { side: other as i32, pokemon: nickname });
        return Ok(());
    }
    // `TRACE_DRAWS=1` prints where on the tape each move started. When the two engines disagree
    // about *how many* draws a turn took, the divergence message names the position but not the
    // move that got there — this closes that gap in one run.
    if std::env::var_os("TRACE_DRAWS").is_some() {
        eprintln!(
            "[draw {}] side {side} uses {} (slot {slot}, accuracy {:?})",
            tape.position(),
            the_move.name,
            the_move.accuracy_probability
        );
    }
    log.push(Event::MoveUsed {
        side: side as i32,
        pokemon: state.sides[side].active_pokemon().nickname.clone(),
        the_move: the_move.name.clone(),
        unleashed_as,
    });
    // Locked in on the first move used while holding one — never re-set once `Some`, and using
    // whatever slot was actually used (Struggle included), same as the Python's own `action.move`
    // read at this exact point, after its own choice/encore/rampage/charge redirects.
    if matches!(state.sides[side].active_pokemon().item.as_str(), "CHOICE_BAND" | "CHOICE_SCARF" | "CHOICE_SPECS")
        && state.sides[side].active_pokemon().choice_locked_move.is_none()
    {
        state.sides[side].active_mut().choice_locked_move = Some(slot);
    }
    // Read by Encore, Disable, and a rampage's own `InflictStatusEffect` — always the slot just
    // used, Struggle included, since the Python assigns this before the Struggle substitution
    // changes what `move` points at without changing `slot` itself.
    state.sides[side].active_mut().last_move_slot = Some(slot);

    // `ESCALATING_MOVES`: any move outside {Fury Cutter, Rollout, Ice Ball} ends the run, so the
    // next one of these starts from base. Set here rather than where the counter is incremented,
    // because the run ends whether or not *this* move lands — and an escalating move must not zero
    // its own count before its own power reads it a few lines below.
    let escalating = the_move.name == "Fury Cutter" || crate::power::ROLLING_MOVES.contains(&the_move.name.as_str());
    if !escalating {
        state.sides[side].active_mut().rolling_hits = 0;
    }

    // Sleep Talk: after it is announced (and after Taunt's check, which the *unsubstituted* move —
    // still named "Sleep Talk", a status move — has already had to clear), the real move is picked
    // and takes over for everything from here on, logged with its own second `MoveUsed`.
    let the_move = if sleep_talking {
        match crate::power::sleep_talk_choice(state.sides[side].active_pokemon(), db, tape)? {
            None => {
                log.push(Event::MoveFailed);
                return Ok(());
            }
            Some(picked) => {
                if let Some(why) = unsupported_reason(&picked, db) {
                    return Err(Refusal::Unported(why));
                }
                log.push(Event::MoveUsed {
                    side: side as i32,
                    pokemon: state.sides[side].active_pokemon().nickname.clone(),
                    the_move: picked.name.clone(),
                    unleashed_as: None,
                });
                picked
            }
        }
    } else {
        the_move
    };
    // The type it actually resolves as, decided before anything else looks at it — the Python
    // rebuilds the move with the new type, so every later reader sees only the new one. The listed
    // type is kept because the `-ate` boost is the one question that still needs it. Placed after
    // the Sleep Talk substitution rather than where the Struggle substitution is decided, because
    // that is where the Python's own `override_type` sits — Sleep Talk changes what move this even
    // is, and the type override has to ask about the move that is actually about to resolve.
    let listed_type = the_move.move_type.clone();
    let mut the_move = the_move;
    if let Some(became) = crate::power::type_override(&the_move, state.sides[side].active_pokemon(), state) {
        the_move.move_type = became;
    }
    let the_move = the_move;

    // A charge's first turn: the turn-boost (if it has one) applies whether or not the charge is
    // skipped, then — unless it is skipped — the user commits and the turn ends here. Nothing
    // after this point in the function is reached: no stall check, no accuracy roll, no draw.
    if the_move.charge && !releasing_charge {
        let skip = crate::power::skips_charge_turn(state, side, &the_move, log);
        if !skip {
            let attacker = state.sides[side].active_mut();
            attacker.volatiles.insert("CHARGING".to_string(), 1);
            attacker.charging_slot = Some(slot);
            let nickname = attacker.nickname.clone();
            log.push(Event::ChargingUp { side: side as i32, pokemon: nickname, the_move: the_move.name.clone() });
            return Ok(());
        }
    }
    if releasing_charge {
        state.sides[side].active_mut().volatiles.remove("CHARGING");
        state.sides[side].active_mut().charging_slot = None;
    }

    // `_stall_check`, which sits after the move is announced and before anything is rolled for it.
    // A second Protect in a row usually fails, and the roll it fails on is a real draw even though
    // nothing in the log says so — which is exactly how this was found: identical events, and the
    // two engines one draw apart.
    if !stall_check(state, side, &the_move, tape)? {
        log.push(Event::MoveFailed);
        return Ok(());
    }

    // `coded_move_fails`: the conditions a hand-written move checks before it will go off. After
    // the move is announced and after the stall check, which is where the Python asks.
    if crate::power::coded_move_fails(&the_move, state, side, db) {
        log.push(Event::MoveFailed);
        return Ok(());
    }

    // Magic Bounce: an inline read in `_execute_move`, not a bus handler — so a Mold Breaker does
    // not stop it — and placed before Protect. The Python swaps attacker and defender for the rest
    // of the move, so the bouncer rolls the accuracy, sets the hazards on the user's side, and so
    // on; rebinding `side`/`other` here does the same for everything below. There is no second
    // bounce: a user that also has Magic Bounce takes its own move.
    let (side, other) = if the_move.reflectable && state.sides[other].active_pokemon().ability == "MAGIC_BOUNCE" {
        let nickname = state.sides[other].active_pokemon().nickname.clone();
        log.push(Event::MoveBounced { side: other as i32, pokemon: nickname });
        (other, side)
    } else {
        (side, other)
    };

    // Protect and its relatives. *After* the stall check, which is where the Python puts it — a
    // move turned aside by a Protect has still spent its own stalling roll if it had one.
    if the_move.protectable
        && DEFENDER_FACING.contains(&the_move.target.as_str())
        && state.sides[other].active_pokemon().volatiles.contains_key("PROTECT")
    {
        let nickname = state.sides[other].active_pokemon().nickname.clone();
        log.push(Event::Protected { side: other as i32, pokemon: nickname });
        break_rolling(state, side); // `_break_rolling`
        crash_damage(state, side, &the_move, log);
        return Ok(());
    }

    // `_out_of_reach`: a defender mid-Fly/Dig/Phantom-Force is untouchable except by the handful of
    // moves that reach through. *After* Protect and *before* accuracy — same as the Python — and it
    // takes no draw either way, which is how an unlisted charge (Solar Beam) leaves its user
    // visible without costing anything on the tape.
    if DEFENDER_FACING.contains(&the_move.target.as_str())
        && crate::power::out_of_reach(state.sides[other].active_pokemon(), &the_move, db)
    {
        log.push(Event::MoveMissed);
        break_rolling(state, side); // `_break_rolling`
        crash_damage(state, side, &the_move, log);
        return Ok(());
    }

    // Accuracy first, and only when the move has one — `_accuracy_check` returns True without
    // drawing when `accuracy_probability` is None, which is how a never-missing move leaves the
    // tape untouched.
    let attacker = state.sides[side].active_pokemon();
    let defender = state.sides[other].active_pokemon();
    let never_misses = crate::inline::never_misses(attacker, defender);
    if let Some(accuracy) = the_move.accuracy_probability.filter(|_| !never_misses) {
        let attacker_stage = attacker.stage("ACCURACY");
        let defender_stage = defender.stage("EVASION");
        let net = (attacker_stage - defender_stage).clamp(-6, 6);
        let mut multiplier = if net >= 0 { (3 + net) as f64 / 3.0 } else { 3.0 / (3 - net) as f64 };
        // The *move's* category, not its damage effect's — Hustle reads `move.category`, and the
        // two are not always the same. Natural Gift's effect says one thing and the move says
        // another, which was enough to turn a miss into a hit.
        multiplier *= crate::inline::accuracy_multiplier(
            &the_move.name,
            attacker,
            defender,
            &the_move.category,
            &state.field.weather,
        );
        if tape.probability()? >= (accuracy * multiplier).min(1.0) {
            log.push(Event::MoveMissed);
            break_rolling(state, side); // `_break_rolling`
            crash_damage(state, side, &the_move, log);
            return Ok(());
        }
    }

    // `ON_BEFORE_MOVE`: absorption/immunity that cancels the move outright, for a damaging move or
    // one that targets the opponent directly. *After* accuracy, *before* the effectiveness/
    // immunity gate below — exactly where the Python emits it, so Volt Absorb still takes Thunder
    // Wave's own miss chance before it gets a say, and a cancelled move still pays its own crash
    // damage (Jump Kick into a Volt Absorb) the same way a Protect or an out-of-reach miss does.
    if crate::hooks::ability_before_move(state, side, &the_move, db, log) {
        crash_damage(state, side, &the_move, log);
        return Ok(());
    }

    let effectiveness = if the_move.typeless {
        1.0
    } else {
        let attacker_ability = state.sides[side].active_pokemon().ability.clone();
        let defender = state.sides[other].active_pokemon();
        let bypass = defender.effective_bypass(&attacker_ability, &the_move.move_type);
        let natural = db.effectiveness_bypassing(&the_move.move_type, &defender.battle_types(), &bypass);
        let natural =
            crate::power::strong_winds_negation(natural, &the_move.move_type, defender, &state.field.weather, db);
        crate::power::effectiveness_override(&the_move.name, defender, natural, db)
    };
    // The immunity gate is for *damaging* moves only, exactly as the Python writes it. Charge is
    // Electric and targets its user, so a Ground-type across the field does not stop it boosting —
    // and gating it here anyway skipped the boost's probability draw, which put every later draw
    // in the battle one place out.
    // Fixed damage counts as damaging, exactly as the Python's `any(isinstance(e, (DamageEffect,
    // FixedDamageEffect)))` does — so Night Shade into a Normal type announces NoEffect here rather
    // than falling through to the "nothing happened" MoveFailed at the end.
    let damaging = the_move
        .effects
        .iter()
        .any(|e| matches!(e, Effect::DamageEffect { .. } | Effect::FixedDamageEffect { .. }));
    if damaging && effectiveness == 0.0 {
        log.push(Event::NoEffect {
            side: other as i32,
            pokemon: state.sides[other].active_pokemon().nickname.clone(),
        });
        crash_damage(state, side, &the_move, log);
        return Ok(());
    }

    // The screen breakers, which take the wall down whether or not the hit that follows lands —
    // but only once accuracy and immunity have both let the move through. *Here*, not earlier: a
    // screen breaker that misses or is shrugged off as `NoEffect` must not break the screen either.
    if crate::power::SCREEN_BREAKERS.contains(&the_move.name.as_str()) {
        let standing: Vec<String> = state.sides[other].screens.keys().cloned().collect();
        for screen in standing {
            state.sides[other].screens.remove(&screen);
            log.push(Event::ScreenFaded { side: other as i32, screen });
        }
    }

    let log_before = log.entries.len();

    // `behind_substitute`: computed once, here, before any effect runs — not rechecked per effect.
    // Infiltrator is still refused, so it can never be the attacker's ability; the clause is kept
    // anyway so this reads the same as the Python and needs no revisiting once the ability lands.
    let behind_substitute = state.sides[other].active_pokemon().volatiles.contains_key("SUBSTITUTE")
        && !the_move.bypass_substitute
        && state.sides[side].active_pokemon().ability != "INFILTRATOR";

    // Effects resolve in the order the move lists them, which is the order `_apply_effect` is
    // called in and therefore the order their draws come off the tape.
    //
    // Red Card (inside the `DamageEffect` arm's own `apply_damage`) can switch `side`'s own active
    // mid-move. The Python reads everything from here on — every later effect, a rampage's own
    // `LOCKED_MOVE`, a recharge's own flag, this same move's recoil — against the one `attacker`
    // object reference it captured before any of this started, so all of it still lands on whoever
    // actually swung, not on whoever that side benched in behind them. `apply_damage` itself already
    // reproduces this for its own internal tail and for a second hit of the same multi-hit move;
    // `restore_active_to`, set the instant a switch is first seen below, pins the rest of this
    // function to match and is put back only once, right before `resolve_move` itself returns.
    // A disguise takes the whole move, not merely its damage — decided here, before any effect runs,
    // for the same reason the substitute is: by the time a flinch is reached the disguise has
    // already busted, and a Mimikyu that was never actually hit would flinch anyway. Only the damage
    // effect still runs, because absorbing it is what busts the disguise; a fixed-damage effect is
    // not a `DamageEffect`, so a Seismic Toss is simply blocked and leaves the disguise intact.
    let disguised = crate::formes::disguise_intercepts(state.sides[other].active_pokemon(), &the_move);
    let attacker_active_at_move_start = state.sides[side].active;
    let mut restore_active_to: Option<usize> = None;
    for effect in &the_move.effects {
        // `_DELAYED_DAMAGE_MOVES`: Future Sight and Doom Desire carry a real `DamageEffect` for the
        // residual to apply two turns on, and it must not also land immediately on the turn the
        // move is used — the `CodedEffect(FUTURE_SIGHT)` alongside it is what actually queues it.
        if matches!(effect, Effect::DamageEffect { .. })
            && matches!(the_move.name.as_str(), "Future Sight" | "Doom Desire")
        {
            continue;
        }
        // A knocked-out target takes no more of the move — not the burn from Steam Eruption, not
        // the speed drop from Icy Wind. What still lands is anything aimed elsewhere: the user's
        // own boost, a hazard, a side effect. Skipping the effect has to skip its probability draw
        // too, which is how this was found: the Python stopped after the faint and this engine
        // rolled on, so every draw from there wasread out of another turn.
        if state.sides[other].active_pokemon().fainted() && targets_defender(effect, &the_move) {
            continue;
        }
        if disguised && targets_defender(effect, &the_move) && !matches!(effect, Effect::DamageEffect { .. }) {
            continue;
        }
        match effect {
            Effect::DamageEffect { .. } => {
                // Effectiveness is logged inside, because the hit-count roll comes before it.
                apply_damage(state, side, &the_move, &listed_type, effectiveness, behind_substitute, db, tape, log)?;
                // `apply_damage` settles back to its own real, post-hit active index before it
                // returns (it needs the live board for its own multi-hit loop); catching the change
                // here, once, is what lets everything after this arm — for the rest of this whole
                // function — read the original attacker instead.
                if restore_active_to.is_none() && state.sides[side].active != attacker_active_at_move_start {
                    restore_active_to = Some(state.sides[side].active);
                    state.sides[side].active = attacker_active_at_move_start;
                }
            }
            Effect::InflictStatusEffect { status, probability, to_self, is_secondary } => {
                // `_apply_status_routed`: the *move* targeting its user is enough, whatever the
                // effect says. Endure's effect is not marked `to_self` and it is plainly not
                // something you do to somebody else. A substitute blocks a status effect aimed at
                // its owner completely — not even the probability draw happens, exactly as the
                // Python's `elif not behind_substitute` skips the whole call.
                let at_self = *to_self || the_move.target == "SELF";
                // `_tuned_status_secondary` runs first, unconditionally, ahead of the substitute
                // check — a Shield Dust block skips the draw even behind a substitute, since the
                // Python nullifies the effect object itself before `_apply_status_routed` (which
                // is where the substitute check lives) is ever called.
                let attacker_ability = state.sides[side].active_pokemon().ability.clone();
                let defender_ability = state.sides[other].active_pokemon().ability.clone();
                let defender_item = state.sides[other].active_pokemon().item.clone();
                let tuned = crate::inline::tune_status_secondary(
                    *is_secondary,
                    *probability,
                    at_self,
                    &attacker_ability,
                    &defender_ability,
                    &defender_item,
                );
                if let Some(tuned) = tuned {
                    if !at_self && behind_substitute {
                        // no draw
                    } else {
                        apply_status(state, side, status, tuned, at_self, tape, log)?;
                    }
                }
            }
            Effect::StatStageChangeEffect { stages, probability, target, is_secondary } => {
                // Same rule as the status case: a substitute blocks a stage drop aimed at its
                // owner before the probability draw, not after — and `_tuned_stage_secondary`
                // still runs ahead of both, for the same reason.
                let attacker_ability = state.sides[side].active_pokemon().ability.clone();
                let defender_ability = state.sides[other].active_pokemon().ability.clone();
                let defender_item = state.sides[other].active_pokemon().item.clone();
                let tuned = crate::inline::tune_stage_secondary(
                    *is_secondary,
                    *probability,
                    target,
                    &attacker_ability,
                    &defender_ability,
                    &defender_item,
                );
                if let Some(tuned) = tuned {
                    if target != "SELF" && behind_substitute {
                        // no draw
                    } else {
                        let fired = apply_stages(state, side, stages, tuned, target, tape, log)?;
                        // Mirror Herb: the *opponent's* own self-raise, copied onto its holder the
                        // instant it lands — checked against the request's own stages, unconditional
                        // of whether any of them actually moved a clamped stat.
                        if fired && target == "SELF" && !state.sides[other].active_pokemon().fainted() {
                            let raises: Vec<(String, i32)> =
                                stages.iter().filter(|(_, change)| *change > 0).cloned().collect();
                            if !raises.is_empty() && state.sides[other].active_pokemon().item == "MIRROR_HERB" {
                                let holder = state.sides[other].active_mut();
                                holder.last_consumed_item = holder.item.clone();
                                holder.item = "NONE".to_string();
                                holder.item_consumed = true;
                                apply_stage_changes(state, other, &raises, "seed", log);
                            }
                        }
                    }
                }
            }
            Effect::FixedDamageEffect { amount_formula, set_amount } => {
                apply_fixed_damage(state, side, &the_move, amount_formula, *set_amount, behind_substitute, db, log);
            }
            Effect::HealEffect { fraction } => apply_heal(state, side, *fraction, &the_move.name, log),
            Effect::WeatherEffect { variant, duration_turns } => {
                state.field.weather = variant.clone();
                let rock = crate::inline::rock_for_weather(variant);
                state.field.weather_turns_left =
                    if rock.is_some_and(|r| state.sides[side].active_pokemon().item == r) {
                        8
                    } else {
                        duration_turns.unwrap_or(5)
                    };
                log.push(Event::WeatherChanged { weather: variant.clone() });
            }
            Effect::TerrainEffect { variant, duration_turns } => {
                state.field.terrain = variant.clone();
                state.field.terrain_turns_left = if state.sides[side].active_pokemon().item == "TERRAIN_EXTENDER" {
                    8
                } else {
                    duration_turns.unwrap_or(5)
                };
                log.push(Event::TerrainChanged { terrain: variant.clone() });
                // `_apply_field_effect` sweeps both sides' seeds itself, unconditionally, every
                // time a terrain move resolves -- even a terrain re-set to what it already was.
                crate::hooks::consume_terrain_seeds_on_terrain_change(state, log);
            }
            Effect::SideConditionEffect { variant, duration_turns } => {
                apply_side_condition(state, side, &the_move, variant, *duration_turns, log);
            }
            Effect::RemoveHazardsEffect { style } => remove_hazards(state, side, style, log),
            Effect::PseudoWeatherEffect { variant, duration_turns } => {
                // `_apply_field_effect`: no item extends one of these (unlike weather's rocks or
                // terrain's Terrain Extender), and re-applying one already up just resets its
                // duration rather than refusing — Trick Room used again simply restarts the clock.
                state.field.pseudo_weather.insert(variant.clone(), duration_turns.unwrap_or(5));
                log.push(Event::PseudoWeatherStarted { kind: variant.clone() });
            }
            Effect::CodedEffect { variant } => {
                if !PORTED_CODED_KINDS.contains(&variant.as_str()) {
                    return Err(Refusal::Unported(format!("{variant} is hand-written in the Python engine")));
                }
                apply_coded(state, side, &the_move, variant, behind_substitute, db, log)?;
            }
            Effect::Unmodelled => return Err(Refusal::Unported(format!("{} has an unmodelled effect", the_move.name))),
        }
    }

    // Phazing drags a random healthy teammate in, and it counts as something happening — so it is
    // before the "nothing happened" check, exactly where the Python puts it.
    //
    // Unlike everything else past a mid-move switch, this one needs the *real* board, not the
    // pinned stale attacker: the arriving Pokemon's own `ON_SWITCH_IN` (Intimidate, entry hazards,
    // ...) is a fresh bus read in the Python, not a direct reference through `_execute_move`'s own
    // local variables, so it already sees whichever Pokemon Red Card actually left standing on
    // `side`. Un-pin for the call, then re-pin — the attacker is still stale for whatever of this
    // move's own resolution remains after it.
    if the_move.force_switch && !state.sides[other].active_pokemon().fainted() {
        let pinned = restore_active_to.is_some();
        if let Some(real_active) = restore_active_to {
            state.sides[side].active = real_active;
        }
        force_random_switch(state, other, db, tape, log)?;
        if pinned {
            state.sides[side].active = attacker_active_at_move_start;
        }
    }

    // Nothing at all happened: every effect was skipped, most often because the target had already
    // been knocked out by the other side this turn. The Python decides this by whether the log grew
    // rather than by inspecting the move, so this does too.
    //
    // Unreachable with `restore_active_to` set in practice — Red Card only ever fires once real
    // damage has already landed and been logged — but restoring before every return out of this
    // function, rather than trusting that, is the cheaper thing to get right.
    if log.entries.len() == log_before {
        log.push(Event::MoveFailed);
        crash_damage(state, side, &the_move, log);
        if let Some(real_active) = restore_active_to {
            state.sides[side].active = real_active;
        }
        return Ok(());
    }

    // `ESCALATING_MOVES`: Fury Cutter counts its run without being locked into it. Rollout and Ice
    // Ball pay for their doubling with a lock, taken here on the hit that starts the run.
    if the_move.name == "Fury Cutter" {
        state.sides[side].active_mut().rolling_hits += 1;
    } else if crate::power::ROLLING_MOVES.contains(&the_move.name.as_str()) {
        let attacker = state.sides[side].active_mut();
        attacker.rolling_hits += 1;
        if !attacker.volatiles.contains_key("LOCKED_MOVE") {
            attacker.volatiles.insert("LOCKED_MOVE".to_string(), crate::power::ROLLING_LOCK_TURNS);
            attacker.locked_slot = Some(slot);
        }
    }
    // A pivot's own switch is deferred exactly like Eject Button's: `SelfSwitchPending` logs and
    // `needs_switch` arms here, in `_execute_move` itself, but the actual switch waits for
    // `resolve_pending_switches` — called once per completed action, for both sides in side order.
    // That ordering is what a target still waiting to act this turn faces whoever arrived rather
    // than the pivot's user (nothing else happens between this action finishing and the next one
    // starting) — and it is also why a defender's own Eject Button, armed first (from `on_after_hit`,
    // strictly before this check), still resolves *after* this side's pivot if the pivot's own side
    // index is lower: `resolve_pending_switches` walks side 0 then side 1, not arming order.
    let bench = state.sides[side]
        .team
        .iter()
        .enumerate()
        .any(|(index, member)| index != state.sides[side].active && !member.fainted());
    if the_move.self_switch && !state.sides[side].active_pokemon().fainted() && bench {
        let nickname = state.sides[side].active_pokemon().nickname.clone();
        log.push(Event::SelfSwitchPending { side: side as i32, pokemon: nickname });
        state.sides[side].needs_switch = true;
    }
    if the_move.recharges {
        state.sides[side].active_mut().volatiles.insert("MUST_RECHARGE".to_string(), 1);
    }
    if the_move.self_destructs && !state.sides[side].active_pokemon().fainted() {
        let user = state.sides[side].active_mut();
        let all_of_it = user.hp;
        user.take_damage(all_of_it);
        let nickname = user.nickname.clone();
        log.push(Event::Fainted { side: side as i32, pokemon: nickname });
    }
    // Let go of the pin, if one was ever needed: whoever Red Card actually sent out is who the rest
    // of the turn — and the game — sees from here on.
    if let Some(real_active) = restore_active_to {
        state.sides[side].active = real_active;
    }
    Ok(())
}

/// `_grant_healing_wish` and Shed Tail's substitute handoff — the two side-level gifts a Pokemon
/// can arrive to. `_execute_switch` is the one place Python does either, whichever of the three
/// ways a switch happens (a chosen action, a pivot's replacement, a phazing drag), so this is
/// called from all three Rust equivalents rather than only the first one found to need it.
fn grant_switch_in_bonuses(state: &mut State, side: usize, log: &mut Log) {
    // `_grant_healing_wish`: only if there is actually something to fix — a healthy, unstatused
    // arrival leaves the wish pending for a later switch-in instead of spending it on nothing.
    if state.sides[side].healing_wish_pending {
        let incoming = state.sides[side].active_pokemon();
        let hurt = incoming.hp < incoming.totals.hp || incoming.status != Status::None;
        if hurt {
            state.sides[side].healing_wish_pending = false;
            let incoming = state.sides[side].active_mut();
            let before = incoming.hp;
            incoming.hp = incoming.totals.hp;
            incoming.status = Status::None;
            incoming.status_turns = 0;
            let healed = incoming.hp - before;
            let nickname = incoming.nickname.clone();
            log.push(Event::Healed { side: side as i32, pokemon: nickname, amount: healed });
        }
    }
    // Shed Tail's parting gift: the passed substitute has ordinary substitute HP, already computed
    // when the move that queued it was used.
    if state.sides[side].pending_substitute > 0
        && !state.sides[side].active_pokemon().volatiles.contains_key("SUBSTITUTE")
    {
        let given = state.sides[side].pending_substitute;
        state.sides[side].active_mut().volatiles.insert("SUBSTITUTE".to_string(), given);
        state.sides[side].pending_substitute = 0;
    }
}

/// The harness's `replacement_chooser`: the lowest-index healthy benched Pokemon, always.
///
/// Deterministic on purpose. It consumes no randomness, so it cannot shift the tape, and both
/// engines make the identical choice from the identical rule without the scenario recording it.
pub(crate) fn send_out_replacement(state: &mut State, side: usize, db: &Database, log: &mut Log) {
    let active = state.sides[side].active;
    let Some(to) = (0..state.sides[side].team.len())
        .find(|index| *index != active && !state.sides[side].team[*index].fainted())
    else {
        return;
    };
    let sent_out = state.sides[side].team[to].nickname.clone();
    let withdrew = switch_out(state, side, to, log);
    state.register_active(side);
    let arriving = state.sides[side].active_mut();
    arriving.just_switched_in = true;
    arriving.turns_active = 0;
    log.push(Event::Switched { side: side as i32, withdrew, sent_out });
    crate::field::entry_hazards(state, side, db, log);
    if !state.sides[side].active_pokemon().fainted() {
        grant_switch_in_bonuses(state, side, log);
        on_switch_in(state, side, log);
    }
}

/// `_resolve_eject_packs` followed by `_resolve_pending_switches`, both called once per completed
/// action in the Python — back to back, with nothing observable in between, so one function does
/// both here. Eject Pack's own flag is armed silently by a stat drop and is logged/consumed only
/// now, on resolution; `needs_switch` is the more general flag (Eject Button today) that arms
/// itself already logged and consumed, and just waits here for its actual switch — the Python's own
/// split between "announce it happened" and "who actually walks in", which for this harness's fixed
/// `replacement_chooser` collapse into the same deterministic pick either way.
/// `_resolve_eject_packs`: only the arming half — logging `SelfSwitchPending` and setting
/// `needs_switch`, never executing. Kept as its own full pass over both sides, called before any
/// switch actually executes, because the Python's own two functions are two separate passes:
/// `_resolve_eject_packs` (both sides) runs to completion, then `_resolve_pending_switches` (both
/// sides) runs to completion. Folding "arm this side" and "execute this side" into one call per
/// side, the way an earlier version of this function did, executed side 0's own already-armed pivot
/// switch *before* an Eject Pack a side-1 stat drop had just armed even got its own `SelfSwitchPending`
/// logged — the Python logs both pendings first, in side order, before either switch happens.
fn resolve_eject_packs(state: &mut State, side: usize, log: &mut Log) {
    if !state.sides[side].active_pokemon().eject_pending {
        return;
    }
    state.sides[side].active_mut().eject_pending = false;
    let active = state.sides[side].active;
    let has_bench =
        (0..state.sides[side].team.len()).any(|index| index != active && !state.sides[side].team[index].fainted());
    if !state.sides[side].active_pokemon().fainted() && has_bench {
        let pokemon = state.sides[side].active_mut();
        pokemon.last_consumed_item = pokemon.item.clone();
        pokemon.item = "NONE".to_string();
        pokemon.item_consumed = true;
        let nickname = pokemon.nickname.clone();
        log.push(Event::SelfSwitchPending { side: side as i32, pokemon: nickname });
        state.sides[side].needs_switch = true;
    }
}

/// `_resolve_pending_switches`: the execution half, for whichever side(s) `needs_switch` — armed
/// by a pivot, Eject Button, `resolve_eject_packs` above, or Shed Tail — by the time this runs.
fn resolve_pending_switches(state: &mut State, side: usize, db: &Database, log: &mut Log) {
    if state.sides[side].needs_switch {
        state.sides[side].needs_switch = false;
        send_out_replacement(state, side, db, log);
    }
}

/// `_force_random_switch`: Whirlwind, Roar and Dragon Tail drag somebody in at random.
///
/// The draw happens only when there *is* a bench to drag from, which is the Python's order — an
/// empty bench costs the tape nothing.
pub(crate) fn force_random_switch(
    state: &mut State,
    side: usize,
    db: &Database,
    tape: &mut Tape,
    log: &mut Log,
) -> Result<(), Refusal> {
    let active = state.sides[side].active;
    let bench: Vec<usize> = (0..state.sides[side].team.len())
        .filter(|index| *index != active && !state.sides[side].team[*index].fainted())
        .collect();
    if bench.is_empty() {
        return Ok(());
    }
    let chosen = bench[tape.integer(0, bench.len() as i32)? as usize];
    let sent_out = state.sides[side].team[chosen].nickname.clone();
    let withdrew = switch_out(state, side, chosen, log);
    state.register_active(side);
    let arriving = state.sides[side].active_mut();
    arriving.just_switched_in = true;
    arriving.turns_active = 0;
    log.push(Event::Switched { side: side as i32, withdrew, sent_out });
    crate::field::entry_hazards(state, side, db, log);
    if !state.sides[side].active_pokemon().fainted() {
        grant_switch_in_bonuses(state, side, log);
        on_switch_in(state, side, log);
    }
    // The drag *is* the replacement: an Eject Button/Pack armed on the Pokemon this just dragged
    // out does not also get to send in a second one behind it.
    state.sides[side].needs_switch = false;
    Ok(())
}

/// (High) Jump Kick and friends: half the user's own max HP whenever the attack does not land.
///
/// It applies to a miss, to an immunity, and to a move that simply did nothing — every way of
/// failing, which is why the Python calls it from six places and why it is a separate function
/// here rather than inlined into the miss path.
fn crash_damage(state: &mut State, side: usize, the_move: &Move, log: &mut Log) {
    if !the_move.has_crash_damage || crate::inline::ignores_indirect_damage(state.sides[side].active_pokemon()) {
        return;
    }
    let attacker = state.sides[side].active_mut();
    let amount = std::cmp::max(1, attacker.totals.hp / 2);
    let dealt = attacker.take_damage(amount);
    let nickname = attacker.nickname.clone();
    let fainted = attacker.fainted();
    log.push(Event::RecoilDamage { side: side as i32, pokemon: nickname.clone(), amount: dealt });
    if fainted {
        log.push(Event::Fainted { side: side as i32, pokemon: nickname });
    }
}

/// `_thaw_on_hit`: the hit itself melting a frozen defender.
/// `_damage_substitute`: the sub takes the hit, and its own HP is the only clamp — no Endure, no
/// survival at 1. Returns what was absorbed, which recoil and drain still count even though the
/// real Pokemon never felt it (PS gen 5+).
fn damage_substitute(state: &mut State, defender_side: usize, damage: i32, log: &mut Log) -> i32 {
    let defender = state.sides[defender_side].active_mut();
    let sub_hp = *defender.volatiles.get("SUBSTITUTE").expect("checked by the caller");
    let nickname = defender.nickname.clone();
    if damage >= sub_hp {
        defender.volatiles.remove("SUBSTITUTE");
        log.push(Event::SubstituteBroke { side: defender_side as i32, pokemon: nickname });
        sub_hp
    } else {
        defender.volatiles.insert("SUBSTITUTE".to_string(), sub_hp - damage);
        log.push(Event::SubstituteTookHit { side: defender_side as i32, pokemon: nickname });
        damage
    }
}

fn thaw_on_hit(state: &mut State, defender_side: usize, the_move: &Move, behind_substitute: bool, log: &mut Log) {
    let defender = state.sides[defender_side].active_pokemon();
    // Rechecked fresh, not the value carried in from the top of the move: a substitute that broke
    // on this very hit no longer stands between the move and the thaw.
    let soaked = behind_substitute && defender.volatiles.contains_key("SUBSTITUTE");
    if defender.status != Status::Freeze || defender.fainted() || soaked {
        return;
    }
    if !(the_move.thaws_target || (the_move.move_type == "FIRE" && the_move.category != "STATUS")) {
        return;
    }
    let defender = state.sides[defender_side].active_mut();
    defender.status = Status::None;
    defender.status_turns = 0;
    let nickname = defender.nickname.clone();
    log.push(Event::StatusCleared {
        side: defender_side as i32,
        pokemon: nickname,
        clearance: "thawed".into(),
    });
}

/// Everything the abilities and items want to say about this hit, gathered before the formula runs.
///
/// The Python emits `ON_DAMAGE_CALC` with a base payload and lets handlers fill it in; this builds
/// the same base and walks the same handlers in the same order. `contact` is a property of the hit
/// rather than of the move, which is why it is computed here and passed along.
/// The category and contact flag of the move's first damage effect — the two facts both the
/// damage payload and the after-hit handlers ask about.
fn hit_shape(the_move: &Move) -> (&str, bool) {
    the_move
        .effects
        .iter()
        .find_map(|e| match e {
            Effect::DamageEffect { category, contact, .. } => Some((category.as_str(), *contact)),
            _ => None,
        })
        .unwrap_or(("STATUS", false))
}

/// Whether this hit actually touches, which is not the same as whether the move is a contact move:
/// Protective Pads and Long Reach clear it outright, and a Punching Glove clears it for punches.
fn makes_contact(the_move: &Move, attacker: &Pokemon) -> bool {
    hit_shape(the_move).1
        && attacker.item != "PROTECTIVE_PADS"
        && attacker.ability != "LONG_REACH"
        && !(attacker.item == "PUNCHING_GLOVE" && the_move.punching)
}

fn collect_damage_payload(
    state: &State,
    side: usize,
    attacker_registered: bool,
    the_move: &Move,
    db: &Database,
    seed_power_mods: &[i64],
    behind_substitute: bool,
) -> (Payload, Option<crate::items::Consumed>) {
    let other = 1 - side;
    let weather = crate::hooks::effective_weather(state);
    let attacker = state.sides[side].active_pokemon();
    let defender = state.sides[other].active_pokemon();
    let (category, _) = hit_shape(the_move);
    let calc = Calc {
        move_type: &the_move.move_type,
        category,
        contact: makes_contact(the_move, attacker),
        the_move,
        attacker,
        defender,
        fallen_on_attacker_side: state.sides[side].team.iter().filter(|p| p.fainted()).count(),
        defender_side_acted: state.sides[other].acted_this_turn,
        weather: &weather,
        db,
        behind_substitute,
    };
    let mut payload = Payload::new();
    payload.power_mods_4096 = seed_power_mods.to_vec();
    payload.weather_suppressed = weather != state.field.weather;
    let consumed = apply_damage_calc(state, side, attacker_registered, &calc, &mut payload);
    (payload, consumed)
}

/// `_planned_hits`: how many times this move strikes, and the draw that decides it.
///
/// Taken *before* the effectiveness line is logged and before any crit or damage roll, which is
/// where the Python takes it. A fixed count — Double Kick's two, Triple Axel's three — costs no
/// draw at all. Skill Link always takes the top of the range, drawn from the tape not at all.
fn planned_hits(the_move: &Move, attacker_ability: &str, attacker_item: &str, tape: &mut Tape) -> Result<(bool, i32), Refusal> {
    let span = the_move.effects.iter().find_map(|e| match e {
        Effect::DamageEffect { multi_hit, .. } => multi_hit.as_ref(),
        _ => None,
    });
    let Some(span) = span else { return Ok((false, 1)) };
    let (low, high) = (span[0], span[1]);
    if low == high {
        return Ok((true, low));
    }
    if attacker_ability == "SKILL_LINK" {
        return Ok((true, high));
    }
    // Loaded Dice: only on the wider 2-5 spread, a 2-4-6 style roll wouldn't even qualify — folds
    // the distribution up to 4-or-5 rather than the plain 2-through-5 everybody else draws.
    if attacker_item == "LOADED_DICE" && high - low >= 2 {
        return Ok((true, tape.integer(high - 1, high + 1)?));
    }
    Ok((true, tape.integer(low, high + 1)?))
}

/// `_fixed_amount`: damage that is not the formula's. `None` means the condition was not met, and
/// the move then falls through to the empty-log "But it failed!".
fn fixed_amount(state: &State, side: usize, formula: &str, set_amount: Option<i32>) -> Option<i32> {
    let attacker = state.sides[side].active_pokemon();
    let defender = state.sides[1 - side].active_pokemon();
    match formula {
        "LEVEL" => Some(attacker.level),
        "SET" => set_amount,
        "HALF_TARGET_HP" => Some(std::cmp::max(1, defender.hp / 2)),
        "ENDEAVOR" => Some(defender.hp - attacker.hp).filter(|difference| *difference > 0),
        "COUNTER" | "MIRROR_COAT" => {
            let wanted = if formula == "COUNTER" { "PHYSICAL" } else { "SPECIAL" };
            let valid = attacker.last_hit_category.as_deref() == Some(wanted) && attacker.last_hit_taken > 0;
            valid.then(|| 2 * attacker.last_hit_taken)
        }
        "USER_HP" => Some(attacker.hp),
        // The OHKO moves, which their own 30% accuracy already gates.
        "TARGET_HP" => Some(defender.hp),
        // Really the level times a roll between 0.5 and 1.5, taken at its mean: `_fixed_amount` is
        // handed no RNG on the Python side either, so this takes no draw and must not.
        "PSYWAVE" => Some(attacker.level),
        _ => None,
    }
}

/// `_apply_fixed_damage`. No effectiveness line, no crit, no damage roll — it consumes no
/// randomness at all, which is the whole reason it is a separate path from the formula.
/// `_apply_side_condition`: screens, hazards and Tailwind.
///
/// Every one of them fails rather than refreshing. That is not a detail — without it a search
/// correctly sees a small gain in topping a screen up and will spend dying turns doing it.
fn apply_side_condition(
    state: &mut State,
    side: usize,
    the_move: &Move,
    variant: &str,
    duration_turns: Option<i32>,
    log: &mut Log,
) {
    let target = if the_move.target == "OPPONENT_SIDE" { 1 - side } else { side };
    if variant == "TAILWIND" {
        if state.sides[target].tailwind_turns > 0 {
            log.push(Event::MoveFailed);
            return;
        }
        state.sides[target].tailwind_turns = duration_turns.unwrap_or(4);
        log.push(Event::TailwindSet { side: target as i32 });
        return;
    }
    if crate::field::SCREENS.contains(&variant) {
        if state.sides[target].screens.contains_key(variant) {
            log.push(Event::MoveFailed);
            return;
        }
        let default_duration = if state.sides[side].active_pokemon().item == "LIGHT_CLAY" {
            8
        } else {
            duration_turns.unwrap_or(5)
        };
        state.sides[target].screens.insert(variant.to_string(), default_duration);
        log.push(Event::ScreenSet { side: target as i32, screen: variant.to_string() });
        return;
    }
    let standing = state.sides[target].hazards.get(variant).unwrap_or(0);
    if standing >= crate::field::max_layers(variant) {
        log.push(Event::MoveFailed);
        return;
    }
    state.sides[target].hazards.insert(variant.to_string(), standing + 1);
    log.push(Event::HazardSet { side: target as i32, hazard: variant.to_string() });
}

/// `_apply_remove_hazards`. Rapid Spin clears its own side and frees its user from Leech Seed;
/// Defog clears both sides, takes down the opponent's screens, and wipes the terrain.
fn remove_hazards(state: &mut State, side: usize, style: &str, log: &mut Log) {
    clear_hazards(state, side, log);
    if style == "RAPID_SPIN" {
        // The spin also tears off a Leech Seed, which is half of why the move is worth carrying.
        if state.sides[side].active_mut().volatiles.remove("LEECH_SEED").is_some() {
            let nickname = state.sides[side].active_pokemon().nickname.clone();
            log.push(Event::StatusCleared {
                side: side as i32,
                pokemon: nickname,
                clearance: "freed_from_leech_seed".into(),
            });
        }
        return;
    }
    let other = 1 - side;
    clear_hazards(state, other, log);
    let screens: Vec<String> = state.sides[other].screens.keys().cloned().collect();
    for screen in screens {
        state.sides[other].screens.remove(&screen);
        log.push(Event::ScreenFaded { side: other as i32, screen });
    }
    if state.field.terrain != "NONE" {
        let prior = std::mem::replace(&mut state.field.terrain, "NONE".to_string());
        state.field.terrain_turns_left = 0;
        log.push(Event::TerrainFaded { terrain: prior });
    }
}

fn clear_hazards(state: &mut State, side: usize, log: &mut Log) {
    let standing: Vec<String> = state.sides[side].hazards.keys().cloned().collect();
    for hazard in standing {
        state.sides[side].hazards.remove(&hazard);
        log.push(Event::HazardsCleared { side: side as i32, hazard });
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_fixed_damage(
    state: &mut State,
    side: usize,
    the_move: &Move,
    formula: &str,
    set_amount: Option<i32>,
    behind_substitute: bool,
    db: &Database,
    log: &mut Log,
) {
    let other = 1 - side;
    let defender_types = state.sides[other].active_pokemon().battle_types();
    let bypass = state.sides[other].active_pokemon().identify_bypass();
    if db.effectiveness_bypassing(&the_move.move_type, &defender_types, &bypass) == 0.0 {
        log.push(Event::NoEffect {
            side: other as i32,
            pokemon: state.sides[other].active_pokemon().nickname.clone(),
        });
        return;
    }
    let Some(amount) = fixed_amount(state, side, formula, set_amount) else {
        return;
    };
    // Unlike a DamageEffect hit, a soaked fixed-damage hit does not count toward Rage Fist — the
    // Python returns right after `_damage_substitute` with no `times_hit` increment at all here.
    if behind_substitute && state.sides[other].active_pokemon().volatiles.contains_key("SUBSTITUTE") {
        damage_substitute(state, other, amount, log);
        return;
    }
    let category = if formula == "MIRROR_COAT" { "SPECIAL" } else { "PHYSICAL" };
    let defender = state.sides[other].active_mut();
    let dealt = defender.take_damage(amount);
    if dealt > 0 {
        // Same bookkeeping as an ordinary hit: a Seismic Toss still counts toward Rage Fist.
        defender.times_hit += 1;
        defender.last_hit_taken = dealt;
        defender.last_hit_category = Some(category.to_string());
        let nickname = defender.nickname.clone();
        log.push(Event::DamageDealt { side: other as i32, pokemon: nickname, amount: dealt });
    }
    if state.sides[other].active_pokemon().fainted() {
        log.push(Event::Fainted {
            side: other as i32,
            pokemon: state.sides[other].active_pokemon().nickname.clone(),
        });
    }
}

/// `_apply_heal`. Roost's half of this — dropping the bird's Flying type for the turn — is folded
/// in here directly rather than given its own dispatch, since a `HealEffect` is all Roost's data
/// carries; nothing marks it as special ahead of time.
fn apply_heal(state: &mut State, side: usize, fraction: f64, move_name: &str, log: &mut Log) {
    let pokemon = state.sides[side].active_mut();
    let amount = std::cmp::max(1, (pokemon.totals.hp as f64 * fraction) as i32);
    let before = pokemon.hp;
    pokemon.hp = std::cmp::min(pokemon.totals.hp, pokemon.hp + amount);
    let healed = pokemon.hp - before;
    if healed > 0 {
        let nickname = pokemon.nickname.clone();
        log.push(Event::Healed { side: side as i32, pokemon: nickname, amount: healed });
    }
    // Roost: the bird comes down whether or not there was anything left to heal, and stays down
    // (Flying ignored by `Pokemon::battle_types`) for the rest of this turn regardless.
    if move_name == "Roost" {
        state.sides[side].active_mut().volatiles.insert("ROOSTED".to_string(), 1);
    }
}

/// `_apply_coded`, for the `CodedMoveKind`s this engine has learned. Dispatched by `variant`, the
/// exported enum member name, since several moves share a kind (the four `WEATHER_HEAL` moves).
fn apply_coded(
    state: &mut State,
    side: usize,
    the_move: &Move,
    variant: &str,
    behind_substitute: bool,
    db: &Database,
    log: &mut Log,
) -> Result<(), Refusal> {
    // `_SUB_BLOCKED_KINDS`: a substitute blocks these outright, and silently — no log line, which
    // is how this contributes to the empty-log `MoveFailed` exactly like a blocked status effect
    // does.
    const SUB_BLOCKED_KINDS: [&str; 4] = ["PAIN_SPLIT", "STRENGTH_SAP", "KNOCK_OFF_ITEM", "TRICK"];
    if behind_substitute && SUB_BLOCKED_KINDS.contains(&variant) {
        return Ok(());
    }
    match variant {
        "REST" => rest(state, side, log),
        "WEATHER_HEAL" => weather_heal(state, side, log),
        "PAIN_SPLIT" => pain_split(state, side, log),
        "STRENGTH_SAP" => strength_sap(state, side, log),
        "BELLY_DRUM" => belly_drum(state, side, log),
        "HAZE" => {
            for s in 0..2 {
                for value in state.sides[s].active_mut().stages.values_mut() {
                    *value = 0;
                }
            }
            log.push(Event::AllStatsReset);
        }
        "COURT_CHANGE" => {
            let (a, b) = state.sides.split_at_mut(1);
            std::mem::swap(&mut a[0].hazards, &mut b[0].hazards);
            std::mem::swap(&mut a[0].screens, &mut b[0].screens);
            std::mem::swap(&mut a[0].tailwind_turns, &mut b[0].tailwind_turns);
            log.push(Event::CourtChanged);
        }
        "CURSE" => curse(state, side, log),
        "TIDY_UP" => {
            for s in 0..2 {
                clear_hazards(state, s, log);
                state.sides[s].active_mut().volatiles.remove("SUBSTITUTE");
            }
        }
        "PERISH_SONG" => {
            for s in 0..2 {
                let active = state.sides[s].active_pokemon();
                if !active.fainted() && !active.volatiles.contains_key("PERISH") {
                    state.sides[s].active_mut().volatiles.insert("PERISH".to_string(), 4);
                    let nickname = state.sides[s].active_pokemon().nickname.clone();
                    log.push(Event::VolatileInflicted { side: s as i32, pokemon: nickname, volatile: "PERISH".into() });
                }
            }
        }
        "CURE_SELF" => {
            let attacker = state.sides[side].active_pokemon();
            if attacker.status != Status::None {
                let attacker = state.sides[side].active_mut();
                attacker.status = Status::None;
                attacker.status_turns = 0;
                let nickname = attacker.nickname.clone();
                log.push(Event::StatusCleared { side: side as i32, pokemon: nickname, clearance: "refreshed".into() });
            }
        }
        "CURE_PARTY" => {
            for i in 0..state.sides[side].team.len() {
                let member = &mut state.sides[side].team[i];
                if member.status != Status::None {
                    member.status = Status::None;
                    member.status_turns = 0;
                    let nickname = member.nickname.clone();
                    log.push(Event::StatusCleared { side: side as i32, pokemon: nickname, clearance: "refreshed".into() });
                }
            }
        }
        "KNOCK_OFF_ITEM" => knock_off_item(state, side, db, log),
        "TRICK" => trick(state, side, db, log),
        "SKILL_SWAP" => skill_swap(state, side, log),
        "ROLE_PLAY" => {
            let ability = state.sides[1 - side].active_pokemon().ability.clone();
            take_ability(state, side, &ability, log);
        }
        "ENTRAINMENT" => {
            let ability = state.sides[side].active_pokemon().ability.clone();
            take_ability(state, 1 - side, &ability, log);
        }
        "WORRY_SEED" => take_ability(state, 1 - side, "INSOMNIA", log),
        "SIMPLE_BEAM" => take_ability(state, 1 - side, "SIMPLE", log),
        "WISH" => {
            let attacker = state.sides[side].active_pokemon();
            if state.sides[side].wish_turns == 0 {
                let pending = std::cmp::max(1, attacker.totals.hp / 2);
                let nickname = attacker.nickname.clone();
                state.sides[side].wish_pending = pending;
                state.sides[side].wish_turns = 2;
                log.push(Event::WishMade { side: side as i32, pokemon: nickname });
            }
        }
        "HEALING_WISH" => {
            state.sides[side].healing_wish_pending = true;
            let nickname = state.sides[side].active_pokemon().nickname.clone();
            log.push(Event::WishMade { side: side as i32, pokemon: nickname });
        }
        "REVIVAL_BLESSING" => {
            let Some(fallen) = state.sides[side].team.iter_mut().find(|p| p.fainted()) else {
                log.push(Event::MoveFailed);
                return Ok(());
            };
            let amount = std::cmp::max(1, fallen.totals.hp / 2);
            fallen.hp = std::cmp::min(fallen.totals.hp, fallen.hp + amount);
            let nickname = fallen.nickname.clone();
            log.push(Event::Revived { side: side as i32, pokemon: nickname });
        }
        "SHED_TAIL" => {
            let attacker = state.sides[side].active_pokemon();
            let cost = std::cmp::max(1, attacker.totals.hp / 2);
            let active = state.sides[side].active;
            let has_healthy_bench =
                state.sides[side].team.iter().enumerate().any(|(i, p)| i != active && !p.fainted());
            if cost >= attacker.hp || !has_healthy_bench {
                log.push(Event::MoveFailed);
                return Ok(());
            }
            let attacker = state.sides[side].active_mut();
            attacker.take_damage(cost);
            let sub_hp = attacker.totals.hp / 4;
            let nickname = attacker.nickname.clone();
            state.sides[side].pending_substitute = sub_hp;
            log.push(Event::VolatileInflicted { side: side as i32, pokemon: nickname, volatile: "SUBSTITUTE".into() });
            // `side.needs_switch = True`, exactly like an ordinary pivot — deferred to
            // `resolve_pending_switches`, called once per completed action, not switched here
            // inline. Matters for the same reason the pivot fix did: a defender's own Eject Button,
            // armed earlier this same action, still resolves in side order, not arming order.
            state.sides[side].needs_switch = true;
        }
        "FUTURE_SIGHT" => {
            let other = 1 - side;
            if state.sides[other].future_sight_turns > 0 {
                return Ok(());
            }
            let attacker_index = state.sides[side].active;
            let nickname = state.sides[side].active_pokemon().nickname.clone();
            state.sides[other].future_sight_attacker = Some((side, attacker_index));
            state.sides[other].future_sight_move = Some(the_move.name.clone());
            state.sides[other].future_sight_turns = 3;
            log.push(Event::FutureAttackQueued { side: side as i32, pokemon: nickname, the_move: the_move.name.clone() });
        }
        "TRANSFORM" => transform(state, side, log),
        _ => unreachable!("gated by PORTED_CODED_KINDS"),
    }
    Ok(())
}

const TRANSFORM_PP: i32 = 5;

/// `transform_into`: copies base stats (HP's excepted), nature, EVs, IVs, types, ability, moves
/// (5 PP each) and stat stages off the target, snapshotting the attacker's own form first so a
/// later switch-out can give it back. Refuses — `MoveFailed`, not a refusal in the differential
/// sense — against a fainted target or a Pokemon already on either side of a transformation,
/// exactly as `id(attacker) in state.transforms or id(defender) in state.transforms` does; `side`/
/// `team_index` is this engine's stable identity in place of the Python's object identity, the
/// same substitution `future_sight_attacker` already makes.
pub(crate) fn transform(state: &mut State, side: usize, log: &mut Log) {
    let other = 1 - side;
    let attacker_index = state.sides[side].active;
    let defender_index = state.sides[other].active;
    let defender = state.sides[other].active_pokemon();
    if defender.fainted()
        || state.sides[side].transforms.contains_key(&attacker_index)
        || state.sides[other].transforms.contains_key(&defender_index)
    {
        log.push(Event::MoveFailed);
        return;
    }
    let defender_base_stats = defender.base_stats;
    let defender_nature = defender.nature.clone();
    let defender_ivs = defender.ivs;
    let defender_evs = defender.evs;
    let defender_types = defender.types.clone();
    let defender_ability = defender.ability.clone();
    let defender_moves = defender.moves.clone();
    let defender_stages = defender.stages.clone();
    let into = defender.nickname.clone();

    let attacker = state.sides[side].active_pokemon();
    let snapshot = FormSnapshot {
        base_stats: attacker.base_stats,
        nature: attacker.nature.clone(),
        ivs: attacker.ivs,
        evs: attacker.evs,
        types: attacker.types.clone(),
        ability: attacker.ability.clone(),
        moves: attacker.moves.clone(),
        pp: attacker.pp.clone(),
    };
    let own_hp_base = attacker.base_stats.hp;
    let nickname = attacker.nickname.clone();
    state.sides[side].transforms.insert(attacker_index, snapshot);

    let attacker = state.sides[side].active_mut();
    attacker.base_stats = BaseStats { hp: own_hp_base, ..defender_base_stats };
    attacker.nature = defender_nature;
    attacker.ivs = defender_ivs;
    attacker.evs = defender_evs;
    attacker.types = defender_types;
    attacker.ability = defender_ability;
    attacker.moves = defender_moves;
    attacker.pp = attacker
        .moves
        .iter()
        .enumerate()
        .map(|(i, _)| (crate::battle::SLOT_NAMES[i].to_string(), TRANSFORM_PP))
        .collect();
    attacker.stages = defender_stages;
    attacker.recompute_totals();
    // `unregister_active` ... `register_active` around the copy: the copied ability's handlers (and
    // the Pokemon's item's, afresh) join the bus behind everyone else's.
    state.register_active(side);
    log.push(Event::Transformed { side: side as i32, pokemon: nickname, into });
}

/// `_knock_off_item`: refuses silently (no log — contributes to the empty-log `MoveFailed`) with
/// nothing to take, a fainted target, or an item welded to what the target *is*.
fn knock_off_item(state: &mut State, side: usize, db: &Database, log: &mut Log) {
    let other = 1 - side;
    let defender = state.sides[other].active_pokemon();
    if defender.fainted() || defender.item == "NONE" || db.is_fused_to(&defender.species_name, &defender.item) {
        return;
    }
    let removed = defender.item.clone();
    let defender = state.sides[other].active_mut();
    defender.last_consumed_item = removed.clone();
    defender.item = "NONE".to_string();
    defender.item_consumed = true;
    let nickname = defender.nickname.clone();
    // `rewire_active`: the item's handlers go, and whatever is left re-registers behind the rest.
    state.register_active(other);
    log.push(Event::ItemRemoved { side: other as i32, pokemon: nickname, item: removed });
}

/// `_trick`: a trade needs both halves tradeable, so it refuses outright rather than taking the
/// one side it can — which is what stops it laundering a fused item off a Pokemon Knock Off
/// cannot touch either.
fn trick(state: &mut State, side: usize, db: &Database, log: &mut Log) {
    let other = 1 - side;
    let attacker = state.sides[side].active_pokemon();
    let defender = state.sides[other].active_pokemon();
    if attacker.item == "NONE" && defender.item == "NONE" {
        log.push(Event::MoveFailed);
        return;
    }
    if db.is_fused_to(&attacker.species_name, &attacker.item) || db.is_fused_to(&defender.species_name, &defender.item)
    {
        log.push(Event::MoveFailed);
        return;
    }
    let (a, b) = state.sides.split_at_mut(1);
    let (mine, theirs) = if side == 0 { (&mut a[0], &mut b[0]) } else { (&mut b[0], &mut a[0]) };
    std::mem::swap(&mut mine.active_mut().item, &mut theirs.active_mut().item);
    // `rewire_active` on both, attacker first: each binds the item it now holds.
    state.register_active(side);
    state.register_active(other);
    let nickname = state.sides[side].active_pokemon().nickname.clone();
    log.push(Event::ItemsSwapped { side: side as i32, pokemon: nickname });
}

/// `_skill_swap`. Both sides holding an untouchable ability refuse the whole swap rather than
/// half of it, same reasoning as Trick.
fn skill_swap(state: &mut State, side: usize, log: &mut Log) {
    let other = 1 - side;
    let attacker_ability = state.sides[side].active_pokemon().ability.clone();
    let defender_ability = state.sides[other].active_pokemon().ability.clone();
    if attacker_ability == "NONE" && defender_ability == "NONE" {
        log.push(Event::MoveFailed);
        return;
    }
    if is_untouchable(&attacker_ability) || is_untouchable(&defender_ability) {
        log.push(Event::MoveFailed);
        return;
    }
    let (a, b) = state.sides.split_at_mut(1);
    let (mine, theirs) = if side == 0 { (&mut a[0], &mut b[0]) } else { (&mut b[0], &mut a[0]) };
    std::mem::swap(&mut mine.active_mut().ability, &mut theirs.active_mut().ability);
    state.register_active(side);
    state.register_active(other);
    let nickname = state.sides[side].active_pokemon().nickname.clone();
    log.push(Event::AbilitiesSwapped { side: side as i32, pokemon: nickname });
}

/// `UNTOUCHABLE_ABILITIES`: refuses to be moved, copied, replaced or taken away. All thirteen are
/// still refused outright as abilities in their own right, so this can never actually fire today —
/// kept anyway so Role Play and friends read the same as the Python and need no revisiting once
/// one of the thirteen lands.
fn is_untouchable(ability: &str) -> bool {
    const UNTOUCHABLE_ABILITIES: [&str; 13] = [
        "NINE_LIVES",
        "MULTITYPE",
        "RKS_SYSTEM",
        "STANCE_CHANGE",
        "SCHOOLING",
        "SHIELDS_DOWN",
        "DISGUISE",
        "COMATOSE",
        "BATTLE_BOND",
        "POWER_CONSTRUCT",
        "ZEN_MODE",
        "ILLUSION",
        "IMPOSTER",
    ];
    UNTOUCHABLE_ABILITIES.contains(&ability)
}

/// `set_ability`: the one place an ability is rewritten, so the one place that checks whether it
/// may be. Returns whether it actually changed.
fn set_ability(state: &mut State, target_side: usize, ability: &str, log: &mut Log) -> bool {
    let target = state.sides[target_side].active_pokemon();
    if is_untouchable(&target.ability) {
        let nickname = target.nickname.clone();
        let stuck = target.ability.clone();
        log.push(Event::AbilityUnchanged { side: target_side as i32, pokemon: nickname, ability: stuck });
        return false;
    }
    if is_untouchable(ability) || target.ability == ability {
        return false;
    }
    state.sides[target_side].active_mut().ability = ability.to_string();
    state.register_active(target_side);
    let nickname = state.sides[target_side].active_pokemon().nickname.clone();
    log.push(Event::AbilityChanged { side: target_side as i32, pokemon: nickname, ability: ability.to_string() });
    true
}

/// `_take_ability`: Role Play, Entrainment, Worry Seed and Simple Beam, all the same operation
/// pointed in different directions. `set_ability` already announces an untouchable refusal; this
/// only adds `MoveFailed` on top when neither of the two things it could have said got said.
fn take_ability(state: &mut State, target_side: usize, ability: &str, log: &mut Log) {
    if ability != "NONE" && set_ability(state, target_side, ability, log) {
        return;
    }
    if !is_untouchable(&state.sides[target_side].active_pokemon().ability) {
        log.push(Event::MoveFailed);
    }
}

fn rest(state: &mut State, side: usize, log: &mut Log) {
    let attacker = state.sides[side].active_pokemon();
    if attacker.hp == attacker.totals.hp {
        log.push(Event::MoveFailed);
        return;
    }
    let attacker = state.sides[side].active_mut();
    attacker.status = Status::Sleep;
    attacker.status_turns = 3; // two full turns asleep
    let before = attacker.hp;
    attacker.hp = attacker.totals.hp;
    let healed = attacker.hp - before;
    let nickname = attacker.nickname.clone();
    log.push(Event::StatusInflicted { side: side as i32, pokemon: nickname.clone(), status: "SLEEP".into() });
    log.push(Event::Healed { side: side as i32, pokemon: nickname, amount: healed });
}

fn weather_heal(state: &mut State, side: usize, log: &mut Log) {
    let weather = crate::hooks::effective_weather(state);
    let (numerator, denominator) = match weather.as_str() {
        "SUN" | "HARSH_SUN" => (2, 3),
        "NONE" => (1, 2),
        _ => (1, 4),
    };
    let attacker = state.sides[side].active_mut();
    let amount = std::cmp::max(1, attacker.totals.hp * numerator / denominator);
    let before = attacker.hp;
    attacker.hp = std::cmp::min(attacker.totals.hp, attacker.hp + amount);
    let healed = attacker.hp - before;
    if healed > 0 {
        let nickname = attacker.nickname.clone();
        log.push(Event::Healed { side: side as i32, pokemon: nickname, amount: healed });
    }
}

/// `_pain_split`. Logs the amount it *asked* healing for, not what `apply_healing` actually
/// returned, exactly the same inconsistency as recoil's — see docs/python-oddities.md.
fn pain_split(state: &mut State, side: usize, log: &mut Log) {
    let other = 1 - side;
    let average = (state.sides[side].active_pokemon().hp + state.sides[other].active_pokemon().hp) / 2;
    for s in [side, other] {
        let pokemon = state.sides[s].active_mut();
        let delta = average - pokemon.hp;
        if delta > 0 {
            pokemon.hp = std::cmp::min(pokemon.totals.hp, pokemon.hp + delta);
            let nickname = pokemon.nickname.clone();
            log.push(Event::Healed { side: s as i32, pokemon: nickname, amount: delta });
        } else {
            pokemon.take_damage(-delta);
        }
    }
}

fn strength_sap(state: &mut State, side: usize, log: &mut Log) {
    let other = 1 - side;
    if state.sides[other].active_pokemon().stage("ATTACK") <= -6 {
        log.push(Event::MoveFailed);
        return;
    }
    let sapped = std::cmp::max(1, state.sides[other].active_pokemon().effective("ATTACK"));
    let attacker = state.sides[side].active_mut();
    let before = attacker.hp;
    attacker.hp = std::cmp::min(attacker.totals.hp, attacker.hp + sapped);
    let healed = attacker.hp - before;
    if healed > 0 {
        let nickname = attacker.nickname.clone();
        log.push(Event::Healed { side: side as i32, pokemon: nickname, amount: healed });
    }
    apply_stage_changes_from(state, other, &[("ATTACK".to_string(), -1)], "move", true, log);
}

fn belly_drum(state: &mut State, side: usize, log: &mut Log) {
    let attacker = state.sides[side].active_pokemon();
    let cost = attacker.totals.hp / 2;
    if cost >= attacker.hp || attacker.stage("ATTACK") >= 6 {
        log.push(Event::MoveFailed);
        return;
    }
    state.sides[side].active_mut().take_damage(cost);
    apply_stage_changes(state, side, &[("ATTACK".to_string(), 12)], "move", log);
}

/// `_curse`: a Ghost curses its target at half its own health; anything else boosts itself.
/// `attacker.types` on purpose, not `battle_types` — Curse cannot be used the same turn its own
/// user roosted, so the two can never actually disagree here, and the Python asks the plain field.
fn curse(state: &mut State, side: usize, log: &mut Log) {
    let other = 1 - side;
    let attacker_is_ghost = state.sides[side].active_pokemon().types.iter().flatten().any(|t| t == "GHOST");
    if attacker_is_ghost {
        let defender = state.sides[other].active_pokemon();
        if defender.volatiles.contains_key("CURSE") || defender.fainted() {
            log.push(Event::MoveFailed);
            return;
        }
        let cost = state.sides[side].active_pokemon().totals.hp / 2;
        state.sides[side].active_mut().take_damage(cost);
        state.sides[other].active_mut().volatiles.insert("CURSE".to_string(), 1);
        let nickname = state.sides[other].active_pokemon().nickname.clone();
        log.push(Event::VolatileInflicted { side: other as i32, pokemon: nickname, volatile: "CURSE".into() });
        return;
    }
    apply_stage_changes(
        state,
        side,
        &[("ATTACK".to_string(), 1), ("DEFENCE".to_string(), 1), ("SPEED".to_string(), -1)],
        "move",
        log,
    );
}

/// `_resolve_future_sight`: lands the queued hit on whoever is at this position now, using the
/// attacker's *current* stats — it may not be the same Pokemon that was here when the move was
/// used, if the position's owner switched in the meantime, and Python reads the same live object
/// either way. `side` is the *defending* side, whose countdown just reached zero.
///
/// The attacker not still being the one active on its own side is refused rather than guessed:
/// every damage-calc site downstream of `apply_damage` — ability, item, `calculate_hit` itself —
/// reads "the attacker" as "whoever is active on the attacking side", and there is no attacker
/// index threaded through any of them to say otherwise. Reproducing a switch in between correctly
/// would mean plumbing one through the whole pipeline for a single move's rarest case; refusing it
/// keeps that case honest instead of silently charging the damage to the wrong Pokemon's ability.
fn resolve_future_sight(state: &mut State, side: usize, db: &Database, tape: &mut Tape, log: &mut Log) -> Result<(), Refusal> {
    let (attacker_side, attacker_index) =
        state.sides[side].future_sight_attacker.take().expect("only called when the countdown hit zero");
    let move_name = state.sides[side].future_sight_move.take().expect("set alongside the attacker");
    if state.sides[attacker_side].active != attacker_index {
        return Err(Refusal::Unported(
            "Future Sight/Doom Desire landing after its attacker switched out is not ported".into(),
        ));
    }
    if state.sides[side].active_pokemon().fainted() {
        return Ok(());
    }
    let the_move = db
        .move_named(&move_name)
        .ok_or_else(|| Refusal::Unported(format!("unknown move {move_name:?}")))?
        .clone();
    let listed_type = the_move.move_type.clone();
    let behind_substitute = state.sides[side].active_pokemon().volatiles.contains_key("SUBSTITUTE")
        && !the_move.bypass_substitute
        && state.sides[attacker_side].active_pokemon().ability != "INFILTRATOR";
    let attacker_ability = state.sides[attacker_side].active_pokemon().ability.clone();
    let defender = state.sides[side].active_pokemon();
    let bypass = defender.effective_bypass(&attacker_ability, &the_move.move_type);
    let effectiveness = db.effectiveness_bypassing(&the_move.move_type, &defender.battle_types(), &bypass);
    let effectiveness =
        crate::power::strong_winds_negation(effectiveness, &the_move.move_type, defender, &state.field.weather, db);
    let nickname = defender.nickname.clone();
    log.push(Event::FutureAttackLands { side: side as i32, pokemon: nickname, the_move: the_move.name.clone() });
    // `_stopped_before_any_hit`: normally asked by the caller before `_apply_damage` is even
    // reached, but this path calls it directly, so the same immunity gate belongs here instead.
    if effectiveness == 0.0 {
        let nickname = state.sides[side].active_pokemon().nickname.clone();
        log.push(Event::NoEffect { side: side as i32, pokemon: nickname });
        return Ok(());
    }
    apply_damage(state, attacker_side, &the_move, &listed_type, effectiveness, behind_substitute, db, tape, log)
}

// Eight arguments, for the same reason `calculate_hit` takes eight: this reads against the
// Python's `_apply_damage`, and bundling them would make the two harder to diff.
#[allow(clippy::too_many_arguments)]
fn apply_damage(
    state: &mut State,
    side: usize,
    the_move: &Move,
    listed_type: &str,
    effectiveness: f64,
    behind_substitute: bool,
    db: &Database,
    tape: &mut Tape,
    log: &mut Log,
) -> Result<(), Refusal> {
    let other = 1 - side;
    // Red Card can switch the attacker's own side mid-resolution (`ON_AFTER_HIT`, inside the hit
    // loop below). The Python holds `attacker` as a plain object reference from the top of
    // `_apply_damage`, so every later hit of the same multi-hit move, and everything "once for the
    // whole move" after the loop (recoil, drain, Life Orb's chip) still lands on whoever actually
    // swung, not on whoever that side benched in to replace them. Re-pinned back to this index the
    // instant it changes, and only let go of for real once this function is done with it — the same
    // trick as the `Refusal`-free early exits elsewhere, minus needing a second notion of "the
    // attacker" in every stage-change/heal site downstream.
    // `_stopped_before_any_hit`'s second half (the first, the immunity gate, runs before this is
    // called): an intact disguise takes the whole move before a single hit is rolled.
    if crate::formes::disguise_absorbs(state, other, db, log) {
        return Ok(());
    }
    let attacker_active_before = state.sides[side].active;
    let mut attacker_active_after_hits = attacker_active_before;
    let attacker_ability = state.sides[side].active_pokemon().ability.clone();
    let attacker_item = state.sides[side].active_pokemon().item.clone();
    let (is_multi_hit, planned) = planned_hits(the_move, &attacker_ability, &attacker_item, tape)?;
    log.effectiveness(effectiveness);
    // Once for the whole move, after the hit count and the effectiveness line — which is where
    // the Python builds `hit_payload_base`. Only Magnitude notices, because only Magnitude draws.
    let listed = the_move.effects.iter().find_map(|e| match e {
        Effect::DamageEffect { power, .. } => Some(*power),
        _ => None,
    });
    let power_override = crate::power::effective_power(the_move, listed.flatten(), state, side, db, tape)?;
    let overrides = crate::power::payload_overrides(the_move, listed_type, state.sides[side].active_pokemon());
    // The Python copies `hit_payload_base` per hit with `dict()`, which is shallow — so when
    // `payload_overrides` returned a *list* (only the `-ate` branch does), every hit shares that
    // one list and each hit's handlers append to it permanently. A five-hit Fury Attack off a
    // Pixilate holder therefore escalates 6, 9, 13, 17, 28 rather than staying flat.
    //
    // Reproduced rather than fixed: see docs/python-oddities.md. Nothing else aliases, because
    // `setdefault` on a key absent from the copy makes a fresh list in that copy alone.
    let mut shared_power_mods: Vec<i64> = overrides.ate_power_mod.into_iter().collect();
    let aliased = overrides.ate_power_mod.is_some();

    let (mut total_dealt, mut hits_landed, mut critical) = (0, 0, false);
    // Flips the instant Red Card switches `side`'s own active (below) — from the *next* hit of the
    // same multi-hit move on, the attacker's own ability and item stop contributing to the damage
    // calc entirely, matching the bus unregistering the departed Pokemon's handlers. The hit that
    // triggers the switch still gets its own contribution: the switch happens at `on_after_hit`,
    // strictly after this same hit's own `ON_DAMAGE_CALC` already ran.
    let mut attacker_registered = true;
    for _ in 0..planned {
        // Re-collected every hit, as the Python re-emits ON_DAMAGE_CALC every hit: a berry eaten
        // on the first blow has to be gone by the second.
        let (mut payload, eaten) =
            collect_damage_payload(state, side, attacker_registered, the_move, db, &shared_power_mods, behind_substitute);
        payload.power_override = power_override;
        payload.attack_stat_override = overrides.attack_stat;
        payload.use_target_attack = overrides.use_target_attack;
        payload.defense_stat_override = overrides.defense_stat;
        payload.ignore_burn |= overrides.ignore_burn;
        payload.ignore_weather_drop = overrides.ignore_weather_drop;
        if aliased {
            shared_power_mods = payload.power_mods_4096.clone();
        }
        // An Air Balloon eats a Ground move whole, and the Python returns before rolling anything
        // — so the draws are skipped too, which is why this is here and not inside the formula.
        let absorbed = the_move.move_type == "GROUND"
            && state.sides[other].active_pokemon().item == "AIR_BALLOON";
        // Drawn here, in the Python's order: the crit first, then the damage roll. Both are certain
        // to be consumed by the time the formula is entered — see `Rolls`.
        let rolls = if absorbed {
            None
        } else {
            Some(Rolls { crit: tape.probability()?, damage: tape.integer(85, 101)? })
        };
        // The berry is spent whether or not the hit goes on to kill, exactly where the Python's
        // handler spends it: during the calculation, before the damage lands.
        if let Some(consumed) = eaten {
            let holder = state.sides[consumed.side].active_mut();
            crate::items::consume(holder, consumed.side, &consumed.item, log);
        }
        let hit = match rolls {
            None => crate::damage::Hit::nothing(),
            Some(rolls) => {
                let (mine, theirs) = state.sides.split_at(1);
                let (attacker, defender_side) = if side == 0 {
                    (mine[0].active_pokemon(), &theirs[0])
                } else {
                    (theirs[0].active_pokemon(), &mine[0])
                };
                calculate_hit(
                    attacker,
                    defender_side.active_pokemon(),
                    the_move,
                    &state.field,
                    defender_side,
                    db,
                    rolls,
                    &payload,
                )
            }
        };
        critical = hit.is_crit;
        // The substitute soaks the hit before anything else looks at it: no survival clamp
        // (Endure, an item cousin), no `on_after_hit` — no contact reaches the real Pokemon, so no
        // contact ability fires — and the loop moves straight to the next planned hit without ever
        // checking for a faint that could not have happened. Checked fresh every iteration rather
        // than cached, because a hit that breaks the sub partway through a multi-hit move leaves
        // the *remaining* hits landing on the real Pokemon instead — `damage_substitute` deletes
        // the volatile the instant it breaks, and nothing here remembers that it ever stood.
        // Reproduced rather than "fixed"; see docs/python-oddities.md.
        if behind_substitute && state.sides[other].active_pokemon().volatiles.contains_key("SUBSTITUTE") {
            let absorbed = damage_substitute(state, other, hit.amount, log);
            total_dealt += absorbed;
            hits_landed += 1;
            state.sides[other].active_mut().times_hit += 1;
            continue;
        }
        // `_land_hit`: Endure clamps the blow to leave exactly one hit point. Sturdy is the same
        // clamp, keyed off full HP rather than a volatile, and (per `ON_BEFORE_HIT` firing ahead
        // of `_land_hit`) checked first — moot in practice, since a hit Sturdy has already reduced
        // below the defender's current HP can never also satisfy Endure's own `>= defender.hp`.
        let mut incoming = hit.amount;
        // Focus Sash: the same full-HP clamp as Sturdy, checked first — historically the sash is
        // consumed in preference to the ability, which is why the Python binds it above the ABILITY
        // band rather than beside it.
        {
            let survives = {
                let defender = state.sides[other].active_pokemon();
                defender.item == "FOCUS_SASH" && defender.hp == defender.totals.hp && incoming >= defender.hp
            };
            if survives {
                let defender = state.sides[other].active_mut();
                incoming = defender.hp - 1;
                defender.last_consumed_item = defender.item.clone();
                defender.item = "NONE".to_string();
                defender.item_consumed = true;
                let nickname = defender.nickname.clone();
                log.push(Event::SurvivedAtOneHp { side: other as i32, pokemon: nickname, cause: "focus_sash".into() });
            }
        }
        {
            let defender = state.sides[other].active_pokemon();
            if defender.ability == "STURDY"
                && !defender.ability_suppressed
                && defender.hp == defender.totals.hp
                && incoming >= defender.hp
            {
                incoming = defender.hp - 1;
                let nickname = defender.nickname.clone();
                log.push(Event::SurvivedAtOneHp { side: other as i32, pokemon: nickname, cause: "sturdy".into() });
            }
        }
        {
            let defender = state.sides[other].active_pokemon();
            if defender.volatiles.contains_key("ENDURE") && incoming >= defender.hp {
                incoming = defender.hp - 1;
                let nickname = defender.nickname.clone();
                log.push(Event::SurvivedAtOneHp {
                    side: other as i32,
                    pokemon: nickname,
                    cause: "endure".into(),
                });
            }
        }
        let dealt = state.sides[other].active_mut().take_damage(incoming);
        if dealt > 0 {
            // `_land_hit`: what Counter and Mirror Coat read back on their own turn, and the
            // running tally Rage Fist charges itself from.
            let defender = state.sides[other].active_mut();
            defender.times_hit += 1;
            defender.last_hit_taken = dealt;
            defender.last_hit_category = Some(hit_shape(the_move).0.to_string());
        }
        total_dealt += dealt;
        hits_landed += 1;
        // A multi-hit move reports each blow where it happened; everything else reports one total
        // after the loop. Either way the crit is announced immediately before the damage it
        // explains, and nowhere else.
        if is_multi_hit {
            if hit.is_crit {
                log.push(Event::CriticalHit);
            }
            log.push(Event::DamageDealt {
                side: other as i32,
                pokemon: state.sides[other].active_pokemon().nickname.clone(),
                amount: dealt,
            });
        }
        // Inside the per-hit loop in the Python, which for a single hit means *before* the crit and
        // damage entries logged after it. So Rough Skin's chip is announced before the damage that
        // caused it, and a berry is eaten before the number that made it ripen is printed.
        let (category, _) = hit_shape(the_move);
        let contact = makes_contact(the_move, state.sides[side].active_pokemon());
        let shape = Hit { attacker_side: side, move_type: &the_move.move_type, category, contact, dealt };
        on_after_hit(state, &shape, attacker_registered, db, tape, log)?;
        // Red Card can switch `side`'s own active right here. Recorded once, then pinned straight
        // back for the rest of this loop — a second hit of the same multi-hit move still has to be
        // calculated off the Pokemon that is actually still swinging, the Python's own stale
        // reference, not whoever just got benched in behind it — everywhere except its own
        // ability/item damage-calc contribution, which `attacker_registered` turns off starting
        // next hit.
        if state.sides[side].active != attacker_active_before {
            attacker_active_after_hits = state.sides[side].active;
            state.sides[side].active = attacker_active_before;
            attacker_registered = false;
        }
        if state.sides[other].active_pokemon().fainted() {
            break;
        }
    }

    // `state.sides[side].active` is already pinned to `attacker_active_before` here — the loop
    // above re-pins it the instant Red Card moves it. `attacker_active_after_hits` remembers the
    // real destination so it can be restored below, once this function is done needing the
    // original — but it is *also* the tell for whether a switch happened at all, which matters
    // for what comes next: Destiny Bond and recoil/drain are plain function calls in the Python,
    // reading the same stale `attacker` object directly, so the pin alone reproduces them
    // correctly. Moxie's own `ON_FAINT` boost and Life Orb's own `ON_ACTION_RESOLVE` chip are not —
    // they are bus subscriptions, and `_execute_switch` unregisters the departing Pokemon's the
    // instant it leaves. A stale reference does not save a subscription that has already been torn
    // down, so both of those have to be skipped outright once a switch has happened, not pinned.
    let attacker_switched_mid_move = attacker_active_after_hits != attacker_active_before;
    if is_multi_hit {
        log.push(Event::MultiHitSummary { hits: hits_landed });
    } else if total_dealt > 0 {
        if critical {
            log.push(Event::CriticalHit);
        }
        log.push(Event::DamageDealt {
            side: other as i32,
            pokemon: state.sides[other].active_pokemon().nickname.clone(),
            amount: total_dealt,
        });
    }
    // Between the damage and the faint, in that order, as `_thaw_on_hit` sits between them: a
    // frozen defender that takes a Fire move — or one of the three off-type thawers — is free
    // again, and then takes its turn normally instead of rolling the 20% check.
    thaw_on_hit(state, other, the_move, behind_substitute, log);
    if state.sides[other].active_pokemon().fainted() {
        log.push(Event::Fainted {
            side: other as i32,
            pokemon: state.sides[other].active_pokemon().nickname.clone(),
        });
        // `ON_FAINT`: emitted right here and nowhere else, matching the one site the Python emits
        // it from — before Destiny Bond's retaliation, before recoil/drain. Skipped if the attacker
        // already left mid-move: Moxie's own handler went with it.
        if !attacker_switched_mid_move {
            crate::hooks::ability_on_faint(state, side, log);
        }
        // Destiny Bond: the fallen defender takes its attacker down too, unless that attacker is
        // already gone (a Struggle recoil, say, that finished itself off on the very same hit).
        if state.sides[other].active_pokemon().volatiles.contains_key("DESTINY_BOND")
            && !state.sides[side].active_pokemon().fainted()
        {
            let attacker = state.sides[side].active_mut();
            let all_of_it = attacker.hp;
            attacker.take_damage(all_of_it);
            let nickname = attacker.nickname.clone();
            log.push(Event::Fainted { side: side as i32, pokemon: nickname });
        }
    }
    // Once for the whole move, against the total: a multi-hit drain heals on the sum, not per blow.
    recoil_and_drain(state, side, the_move, total_dealt, log);
    // `ON_ACTION_RESOLVE`, emitted from exactly this point in the Python: after recoil/drain, once
    // for the whole move rather than once per hit. Skipped if the attacker already left mid-move,
    // same reason as Moxie above: Life Orb's own handler is a bus subscription too.
    //
    // Life Orb and the balloon's pop are the only handlers here at `EventPriority.ITEM`, so they
    // run in registration order: the attacker's Life Orb, and both actives' balloons.
    for holder in crate::hooks::by_registration(state) {
        if holder == side && attacker_switched_mid_move {
            continue;
        }
        if holder == side {
            crate::hooks::life_orb_recoil(state, side, total_dealt, log);
        }
        crate::hooks::check_air_balloon(state, holder, other, log);
    }
    // Un-pin: whoever Red Card actually sent out is who the rest of the turn (and the game) sees.
    state.sides[side].active = attacker_active_after_hits;
    Ok(())
}

/// `_apply_recoil_and_drain`, which runs once for the whole move rather than once per hit — so on a
/// multi-hit move the percentages are taken against the total, not against each blow.
///
/// Struggle's quarter comes first and returns: it is unconditional, where recoil proper is stopped
/// by Rock Head and drain is turned into damage by Liquid Ooze. All three of those abilities are
/// still refused, so none of them can be on the field; each belongs in its condition when it lands.
fn recoil_and_drain(state: &mut State, side: usize, the_move: &Move, total_dealt: i32, log: &mut Log) {
    let Some(Effect::DamageEffect { struggle_recoil, recoil_percent, drain_percent, .. }) =
        the_move.effects.iter().find(|e| matches!(e, Effect::DamageEffect { .. }))
    else {
        return;
    };
    let attacker_is_guarded = crate::inline::ignores_indirect_damage(state.sides[side].active_pokemon());
    if *struggle_recoil {
        // Logged as the amount it asked for rather than the amount that landed.
        let attacker = state.sides[side].active_mut();
        let recoil = std::cmp::max(1, attacker.totals.hp / 4);
        attacker.take_damage(recoil);
        let nickname = attacker.nickname.clone();
        log.push(Event::RecoilDamage { side: side as i32, pokemon: nickname, amount: recoil });
        return;
    }
    if let Some(percent) = recoil_percent {
        let stopped = attacker_is_guarded || state.sides[side].active_pokemon().ability == "ROCK_HEAD";
        if total_dealt > 0 && !stopped {
            let recoil = std::cmp::max(1, (total_dealt as f64 * percent) as i32);
            let attacker = state.sides[side].active_mut();
            attacker.take_damage(recoil);
            let nickname = attacker.nickname.clone();
            log.push(Event::RecoilDamage { side: side as i32, pokemon: nickname, amount: recoil });
        }
    }
    if let Some(percent) = drain_percent {
        if total_dealt > 0 {
            let heal = std::cmp::max(1, (total_dealt as f64 * percent) as i32);
            // Liquid Ooze turns the drink into a wound. Magic Guard does not stop it: the Python
            // applies the backfire unconditionally, and this engine agrees with that one.
            if state.sides[1 - side].active_pokemon().ability == "LIQUID_OOZE" {
                let attacker = state.sides[side].active_mut();
                attacker.take_damage(heal);
                let nickname = attacker.nickname.clone();
                log.push(Event::DrainBackfired {
                    side: side as i32,
                    pokemon: nickname,
                    ability: "LIQUID_OOZE".into(),
                    amount: heal,
                });
            } else {
                let attacker = state.sides[side].active_mut();
                attacker.hp = std::cmp::min(attacker.totals.hp, attacker.hp + heal);
                let nickname = attacker.nickname.clone();
                log.push(Event::Drained { side: side as i32, pokemon: nickname, amount: heal });
            }
        }
    }
}

/// `_apply_status`, which draws its probability *first and always* — even at 1.0, which is why a
/// guaranteed status still costs a place on the tape.
fn apply_status(
    state: &mut State,
    side: usize,
    status_name: &str,
    probability: f64,
    to_self: bool,
    tape: &mut Tape,
    log: &mut Log,
) -> Result<(), Refusal> {
    if tape.probability()? >= probability {
        return Ok(());
    }
    let target_side = if to_self { side } else { 1 - side };
    // `_apply_status_routed`: only the opponent-facing branch ever names an inflictor — a status
    // move never reflects Synchronize off its own user.
    let inflictor_side = if to_self { None } else { Some(side) };
    match Status::parse(status_name) {
        Some(status) => apply_main_status(state, target_side, status, inflictor_side, tape, log),
        None if status_name == "LOCKED_MOVE" => start_rampage(state, target_side, tape),
        None if status_name == "ENCORE" => {
            start_encore(state, target_side, log);
            Ok(())
        }
        None if status_name == "DISABLE" => {
            start_disable(state, target_side, log);
            Ok(())
        }
        None if status_name == "SUBSTITUTE" => {
            start_substitute(state, target_side, log);
            Ok(())
        }
        None if PORTED_VOLATILES.contains(&status_name) => {
            apply_volatile(state, target_side, status_name, tape, log)
        }
        None => Err(Refusal::Unported(format!("{status_name} is a volatile, which is not ported"))),
    }
}

/// `elif effect.status is ExtraStatus.LOCKED_MOVE`: no ability/type immunity, no log line, and a
/// duration draw only on the turn that starts the rampage — a turn that continues one reaches this
/// with `LOCKED_MOVE` already present and does nothing at all, silently, same as the Python.
fn start_rampage(state: &mut State, target_side: usize, tape: &mut Tape) -> Result<(), Refusal> {
    if state.sides[target_side].active_pokemon().volatiles.contains_key("LOCKED_MOVE") {
        return Ok(());
    }
    let turns = tape.integer(2, 4)?;
    let target = state.sides[target_side].active_mut();
    target.volatiles.insert("LOCKED_MOVE".to_string(), turns);
    target.locked_slot = target.last_move_slot;
    Ok(())
}

/// `_mental_herb_cure`: Taunt, Encore or Disable, cured the instant any of them lands — a no-op,
/// self-guarded, for every other volatile so callers can reach for it unconditionally the way the
/// Python's own `_apply_volatile` does.
fn mental_herb_cure(state: &mut State, target_side: usize, volatile: &str, log: &mut Log) -> bool {
    if !matches!(volatile, "TAUNT" | "ENCORE" | "DISABLE")
        || state.sides[target_side].active_pokemon().item != "MENTAL_HERB"
    {
        return false;
    }
    let target = state.sides[target_side].active_mut();
    target.last_consumed_item = target.item.clone();
    target.item = "NONE".to_string();
    target.item_consumed = true;
    target.volatiles.remove(volatile);
    let nickname = target.nickname.clone();
    log.push(Event::StatusCleared { side: target_side as i32, pokemon: nickname, clearance: "berry".to_string() });
    true
}

/// `_start_encore`: fails silently (no draw, no log — the move then falls through to the empty-log
/// `MoveFailed`) with nothing to encore or an encore already running.
fn start_encore(state: &mut State, target_side: usize, log: &mut Log) {
    let target = state.sides[target_side].active_pokemon();
    if target.last_move_slot.is_none() || target.volatiles.contains_key("ENCORE") {
        return;
    }
    let target = state.sides[target_side].active_mut();
    target.encored_slot = target.last_move_slot;
    target.volatiles.insert("ENCORE".to_string(), 3);
    let nickname = target.nickname.clone();
    log.push(Event::VolatileInflicted { side: target_side as i32, pokemon: nickname, volatile: "ENCORE".into() });
    if mental_herb_cure(state, target_side, "ENCORE", log) {
        state.sides[target_side].active_mut().encored_slot = None;
    }
}

/// `_start_disable`: fails silently with nothing to disable or one already in effect. Logs
/// `DisableApplied` naming the move it silenced, not the generic `VolatileInflicted`.
pub fn start_disable(state: &mut State, target_side: usize, log: &mut Log) {
    let target = state.sides[target_side].active_pokemon();
    if target.last_move_slot.is_none() || target.disabled_slot.is_some() {
        return;
    }
    let slot = target.last_move_slot.expect("checked above");
    let disabled_move = target.moves[slot].clone();
    let target = state.sides[target_side].active_mut();
    target.disabled_slot = Some(slot);
    target.volatiles.insert("DISABLE".to_string(), 5);
    let nickname = target.nickname.clone();
    log.push(Event::DisableApplied { side: target_side as i32, pokemon: nickname, the_move: disabled_move });
    if mental_herb_cure(state, target_side, "DISABLE", log) {
        state.sides[target_side].active_mut().disabled_slot = None;
    }
}

/// `_make_substitute`: a quarter of max HP, paid up front, with its own two failure logs rather
/// than the empty-log `MoveFailed` — one already up, or too little HP left to afford it.
fn start_substitute(state: &mut State, target_side: usize, log: &mut Log) {
    let target = state.sides[target_side].active_pokemon();
    let cost = target.totals.hp / 4;
    if target.volatiles.contains_key("SUBSTITUTE") {
        let nickname = target.nickname.clone();
        log.push(Event::SubstituteAlready { side: target_side as i32, pokemon: nickname });
        return;
    }
    if cost >= target.hp {
        log.push(Event::SubstituteTooWeak);
        return;
    }
    let target = state.sides[target_side].active_mut();
    target.take_damage(cost);
    target.volatiles.insert("SUBSTITUTE".to_string(), cost);
    let nickname = target.nickname.clone();
    log.push(Event::VolatileInflicted { side: target_side as i32, pokemon: nickname, volatile: "SUBSTITUTE".into() });
}

/// `_apply_volatile`, for the two this engine knows.
///
/// A volatile already present is not re-applied and — this is the part that matters — takes no
/// duration draw either, because the Python's `elif effect.status not in target.volatiles` skips
/// the whole branch. Inner Focus and Own Tempo would block these outright; both are still refused.
fn apply_volatile(
    state: &mut State,
    target_side: usize,
    volatile: &str,
    tape: &mut Tape,
    log: &mut Log,
) -> Result<(), Refusal> {
    let target = state.sides[target_side].active_pokemon();
    if target.volatiles.contains_key(volatile) || crate::inline::ability_blocks_volatile(target, volatile) {
        // An ability that refuses a volatile does so silently, and — the part that matters — it
        // does so *before* the duration draw, so Inner Focus costs the tape nothing.
        return Ok(());
    }
    // `_VOLATILE_TYPE_IMMUNITY`: a Grass type cannot be seeded, and says so.
    if volatile == "LEECH_SEED" && target.types.iter().flatten().any(|t| t == "GRASS") {
        let nickname = target.nickname.clone();
        log.push(Event::DoesNotAffect { side: target_side as i32, pokemon: nickname });
        return Ok(());
    }
    // Nightmare needs a sleeping target and Yawn an unstatused one. Neither logs; the move then
    // falls through to the empty-log "But it failed!".
    if volatile == "NIGHTMARE" && target.status != Status::Sleep {
        return Ok(());
    }
    if volatile == "YAWN" && target.status != Status::None {
        return Ok(());
    }
    // `_initial_volatile_duration`: a span whose ends are adjacent is a constant and costs no draw,
    // which is why a flinch never touches the tape and a confusion always does.
    let turns = match volatile {
        "CONFUSION" => tape.integer(2, 6)?,
        "PARTIALLY_TRAPPED" => tape.integer(4, 6)?,
        "YAWN" => 2,
        "TAUNT" => 3,
        _ => 1,
    };
    let pokemon = state.sides[target_side].active_mut();
    pokemon.volatiles.insert(volatile.to_string(), turns);
    let nickname = pokemon.nickname.clone();
    log.push(Event::VolatileInflicted {
        side: target_side as i32,
        pokemon: nickname,
        volatile: volatile.to_string(),
    });
    if volatile == "FLINCH" && state.sides[target_side].active_pokemon().ability == "STEADFAST" {
        apply_stage_changes_from(state, target_side, &[("SPEED".to_string(), 1)], "steadfast", false, log);
    }
    mental_herb_cure(state, target_side, volatile, log);
    Ok(())
}

/// `_apply_main_status`: everything after whatever roll decided the status should be attempted.
///
/// Shared, because a move's secondary and an ability like Static reach it by different routes and
/// must land identically once they get there. The sleep-clause check the Python makes here is a
/// no-op for this format — `_CLAUSED_STATUSES` is deliberately empty for Anything Goes — so it is
/// not reproduced; if a format ever wants one, it belongs right here.
pub fn apply_main_status(
    state: &mut State,
    target_side: usize,
    status: Status,
    inflictor_side: Option<usize>,
    tape: &mut Tape,
    log: &mut Log,
) -> Result<(), Refusal> {
    // A move knows what the weather is; an ability or a held orb does not.
    let weather = state.field.weather.clone();
    apply_main_status_from(state, target_side, status, Some(&weather), inflictor_side, tape, log)
}

/// The same, told whether the caller knows the weather and who (if anyone) is inflicting it.
///
/// `weather: None` is not a shortcut — it is the Python's own behaviour. Only the move path passes
/// `field` to `_apply_main_status`; the contact abilities and the Toxic and Flame Orbs all call it
/// without one, and `_ability_immune_to_status` reads `field is not None` before it will let Leaf
/// Guard block anything. So a Toxic Orb poisons its holder in blazing sun and a Leaf Guard cannot
/// stop it, while Sleep Powder in the same sun fails.
///
/// `inflictor_side` is the same story one level further: only an ordinary status move passes one
/// at all (`Some(side)`, the attacker — never a target's own self-inflicted status, which is why
/// `apply_status` only sets it `if !to_self`). A contact ability's own retaliation status
/// (Static, Effect Spore, Poison Touch...) passes `None` in the Python too, which is why a
/// Synchronize holder paralysed by touching a Static Pokemon does not reflect it back, and why
/// Poison Touch cannot poison a Steel type through Corrosion — this engine has no reason to
/// behave differently, so every call site but the ordinary move path stays `None`.
pub fn apply_main_status_from(
    state: &mut State,
    target_side: usize,
    status: Status,
    weather: Option<&str>,
    inflictor_side: Option<usize>,
    tape: &mut Tape,
    log: &mut Log,
) -> Result<(), Refusal> {
    let corrosive = inflictor_side.is_some_and(|s| state.sides[s].active_pokemon().ability == "CORROSION");
    {
        let target = state.sides[target_side].active_pokemon();
        // Type immunity is silent in the Python. Corrosion lets a poison-family status through a
        // Poison or Steel type's own immunity — the only clause `_STATUS_TYPE_IMMUNITY` has for
        // either status — everything else stays exactly as immune as it always was.
        let poison_family = matches!(status, Status::Poison | Status::Toxic);
        if !(poison_family && corrosive)
            && target.types.iter().flatten().any(|t| status.immune_types().contains(&t.as_str()))
        {
            return Ok(());
        }
        // An ability immunity announces itself — as a plain `DoesNotAffect`, which says nothing
        // about *which* ability refused it. That is the Python's line, so it is this one's.
        if crate::inline::ability_blocks_status(target, status, weather.unwrap_or("NONE")) {
            let nickname = target.nickname.clone();
            log.push(Event::DoesNotAffect { side: target_side as i32, pokemon: nickname });
            return Ok(());
        }
        if target.status != Status::None {
            let already = target.status.name().to_string();
            log.push(Event::StatusAlready {
                side: target_side as i32,
                pokemon: target.nickname.clone(),
                status: already,
            });
            return Ok(());
        }
    }
    // Sleep rolls its duration as it lands; toxic starts its counter at zero. The status is set
    // before the draw in the Python, which does not matter here but is why the order reads oddly.
    let turns = match status {
        Status::Sleep => tape.integer(2, 5)?,
        _ => 0,
    };
    let target = state.sides[target_side].active_mut();
    target.status = status;
    target.status_turns = turns;
    let nickname = target.nickname.clone();
    log.push(Event::StatusInflicted {
        side: target_side as i32,
        pokemon: nickname,
        status: status.name().to_string(),
    });
    // `_reflect_synchronize`: the newly-statused Pokemon's own Synchronize, not the inflictor's,
    // mirrors a burn/paralysis/poison/toxic straight back — only if the inflictor is not already
    // statused, and without letting the reflection re-trigger anything (`inflictor_side: None`).
    if matches!(status, Status::Burn | Status::Paralysis | Status::Poison | Status::Toxic) {
        if let Some(inflictor_side) = inflictor_side {
            let reflects = state.sides[target_side].active_pokemon().ability == "SYNCHRONIZE";
            let inflictor_healthy = state.sides[inflictor_side].active_pokemon().status == Status::None;
            if reflects && inflictor_healthy {
                apply_main_status_from(state, inflictor_side, status, weather, None, tape, log)?;
            }
        }
    }
    // Poison Puppeteer: the *inflictor's* ability, not the target's, confuses whatever it just
    // poisoned. `apply_volatile`'s own "already has it" guard is exactly the Python's own
    // `ExtraStatus.CONFUSION not in target.volatiles` check, so nothing extra is needed here.
    if matches!(status, Status::Poison | Status::Toxic) {
        if let Some(inflictor_side) = inflictor_side {
            if state.sides[inflictor_side].active_pokemon().ability == "POISON_PUPPETEER" {
                apply_volatile(state, target_side, "CONFUSION", tape, log)?;
            }
        }
    }
    // Lum Berry (any status) / Chesto Berry (sleep only) cure themselves off the instant the
    // status lands — after Synchronize's reflect and Poison Puppeteer's confusion, matching the
    // Python's own order in `_apply_main_status`.
    let cures = match state.sides[target_side].active_pokemon().item.as_str() {
        "LUM_BERRY" => true,
        "CHESTO_BERRY" => status == Status::Sleep,
        _ => false,
    };
    if cures {
        let target = state.sides[target_side].active_mut();
        target.last_consumed_item = target.item.clone();
        target.item = "NONE".to_string();
        target.item_consumed = true;
        target.status = Status::None;
        target.status_turns = 0;
        let nickname = target.nickname.clone();
        log.push(Event::StatusCleared { side: target_side as i32, pokemon: nickname, clearance: "berry".to_string() });
    }
    Ok(())
}

fn apply_stages(
    state: &mut State,
    side: usize,
    stages: &[(String, i32)],
    probability: f64,
    target: &str,
    tape: &mut Tape,
    log: &mut Log,
) -> Result<bool, Refusal> {
    if tape.probability()? >= probability {
        return Ok(false);
    }
    let target_side = if target == "SELF" { side } else { 1 - side };
    apply_stage_changes_from(state, target_side, stages, "move", target_side != side, log);
    Ok(true)
}

/// `mechanics.stages.apply_stage_changes`, for the cases this engine can reach.
///
/// One entry is logged per stat *whether or not the stage moved* — something already at +6 still
/// reports a requested +1 with a delta of 0, and the comparator would notice its absence. What is
/// missing is everything gated on an ability or item that is still refused: Contrary and Simple
/// rewriting the request, Clear Body and friends intercepting an opponent's drop, Defiant
/// retaliating, a White Herb undoing it.
pub fn apply_stage_changes(state: &mut State, target_side: usize, stages: &[(String, i32)], source: &str, log: &mut Log) {
    apply_stage_changes_from(state, target_side, stages, source, false, log)
}

/// `_STAGED_STATS`: every `Stats` member but HP — the ones a stage can apply to at all.
const _STAGED_STATS: [&str; 7] =
    ["ATTACK", "DEFENCE", "SP_ATTACK", "SP_DEFENCE", "SPEED", "ACCURACY", "EVASION"];

/// `apply_stage_changes`, with the Python's `inflicted_by_opponent` — the flag that decides whether
/// a drop can be intercepted at all.
///
/// One entry is logged per stat *whether or not the stage moved*: something already at +6 still
/// reports a requested +1 with a delta of 0, and the comparator would notice its absence.
pub fn apply_stage_changes_from(
    state: &mut State,
    target_side: usize,
    stages: &[(String, i32)],
    source: &str,
    inflicted_by_opponent: bool,
    log: &mut Log,
) {
    let mut wanted: Vec<(String, i32)> = stages.to_vec();
    let ability = state.sides[target_side].active_pokemon().ability.clone();
    // Contrary reverses the request and Simple doubles it — both ways, drops included — before
    // anything else looks at it.
    match ability.as_str() {
        "CONTRARY" => wanted.iter_mut().for_each(|(_, change)| *change = -*change),
        "SIMPLE" => wanted.iter_mut().for_each(|(_, change)| *change *= 2),
        _ => {}
    }
    if inflicted_by_opponent && wanted.iter().any(|(_, change)| *change < 0) {
        match intercept_drops(state, target_side, &wanted, source, log) {
            None => return,
            Some(surviving) => wanted = surviving,
        }
    }
    let mut dropped = 0;
    let pokemon = state.sides[target_side].active_mut();
    let nickname = pokemon.nickname.clone();
    for (stat, requested) in &wanted {
        let before = pokemon.stage(stat);
        let after = (before + requested).clamp(-6, 6);
        pokemon.stages.insert(stat.clone(), after);
        let delta = after - before;
        log.push(Event::StatStageChanged {
            side: target_side as i32,
            pokemon: nickname.clone(),
            stat: stat.clone(),
            delta,
            requested: *requested,
            source: source.into(),
        });
        if delta < 0 && inflicted_by_opponent {
            dropped += 1;
        }
    }
    if dropped > 0 {
        retaliate_drops(state, target_side, dropped, log);
        // Adrenaline Orb: Intimidate specifically, not any other opponent-inflicted drop — checked
        // after retaliation, in the Python's own order, and only reached at all when the drop
        // actually went through (a Clear Amulet/Clear Body block returns before `dropped` is ever
        // counted).
        if source == "intimidate" && state.sides[target_side].active_pokemon().item == "ADRENALINE_ORB" {
            let pokemon = state.sides[target_side].active_mut();
            pokemon.last_consumed_item = pokemon.item.clone();
            pokemon.item = "NONE".to_string();
            pokemon.item_consumed = true;
            apply_stage_changes(state, target_side, &[("SPEED".to_string(), 1)], "seed", log);
        }
        // Eject Pack: armed here, drained by `resolve_eject_pack` once the action that triggered
        // this finishes resolving — not switched out on the spot, since the Python's own
        // `_resolve_eject_packs` runs once per completed action, after this function returns.
        if state.sides[target_side].active_pokemon().item == "EJECT_PACK" {
            state.sides[target_side].active_mut().eject_pending = true;
        }
    }
    // White Herb: unconditional of whether *this* call dropped anything — any stat still sitting
    // negative, from any earlier call, is reset the moment any stage change resolves at all.
    let holder = state.sides[target_side].active_pokemon();
    if holder.item == "WHITE_HERB" {
        let lowered: Vec<String> =
            _STAGED_STATS.iter().filter(|stat| holder.stage(stat) < 0).map(|s| s.to_string()).collect();
        if !lowered.is_empty() {
            let pokemon = state.sides[target_side].active_mut();
            for stat in &lowered {
                pokemon.stages.insert(stat.clone(), 0);
            }
            pokemon.last_consumed_item = pokemon.item.clone();
            pokemon.item = "NONE".to_string();
            pokemon.item_consumed = true;
            let nickname = pokemon.nickname.clone();
            log.push(Event::WhiteHerbRestored { side: target_side as i32, pokemon: nickname });
        }
    }
}

/// `_intercept_drops`. `None` means a blocker ate the whole change; `Some` is what still applies,
/// which for the single-stat guards is the request with their one stat's drop removed.
///
/// Mirror Armor and Guard Dog are absent: both answer back at the Pokemon that caused the drop,
/// which needs the inflictor threaded through every call site, and both are still refused.
fn intercept_drops(
    state: &mut State,
    target_side: usize,
    stages: &[(String, i32)],
    source: &str,
    log: &mut Log,
) -> Option<Vec<(String, i32)>> {
    let ability = state.sides[target_side].active_pokemon().ability.clone();
    let nickname = state.sides[target_side].active_pokemon().nickname.clone();
    if matches!(ability.as_str(), "CLEAR_BODY" | "FULL_METAL_BODY" | "WHITE_SMOKE") {
        log.push(Event::StatDropBlocked { side: target_side as i32, pokemon: nickname, ability });
        return None;
    }
    if state.sides[target_side].active_pokemon().item == "CLEAR_AMULET" {
        log.push(Event::StatDropBlockedByItem { side: target_side as i32, pokemon: nickname, item: "CLEAR_AMULET".to_string() });
        return None;
    }
    let protected = match ability.as_str() {
        "KEEN_EYE" => "ACCURACY",
        "HYPER_CUTTER" => "ATTACK",
        "BIG_PECKS" => "DEFENCE",
        _ => return Some(stages.to_vec()),
    };
    // Only a *drop* to the guarded stat triggers it, and only that stat is stripped: Tickle lowers
    // Attack and Defence, and Hyper Cutter saves one of them.
    if !stages.iter().any(|(stat, change)| stat == protected && *change < 0) {
        return Some(stages.to_vec());
    }
    let _ = source;
    log.push(Event::StatDropBlocked { side: target_side as i32, pokemon: nickname, ability });
    Some(stages.iter().filter(|(stat, _)| stat != protected).cloned().collect())
}

/// `_retaliate_drops`: Defiant and Competitive answer an opponent's drop with +2 per stat lowered.
fn retaliate_drops(state: &mut State, target_side: usize, dropped: i32, log: &mut Log) {
    let ability = state.sides[target_side].active_pokemon().ability.clone();
    let (stat, source) = match ability.as_str() {
        "DEFIANT" => ("ATTACK", "defiant"),
        "COMPETITIVE" => ("SP_ATTACK", "competitive"),
        _ => return,
    };
    let requested = 2 * dropped;
    let pokemon = state.sides[target_side].active_mut();
    let before = pokemon.stage(stat);
    let after = (before + requested).clamp(-6, 6);
    pokemon.stages.insert(stat.to_string(), after);
    let delta = after - before;
    if delta > 0 {
        let nickname = pokemon.nickname.clone();
        log.push(Event::StatStageChanged {
            side: target_side as i32,
            pokemon: nickname,
            stat: stat.to_string(),
            delta,
            requested,
            source: source.into(),
        });
    }
}

/// Status is on the struct but nothing sets it yet; kept so the digest comparison has a field to
/// disagree about the moment statuses are ported.
pub fn _status_is_modelled(status: Status) -> bool {
    matches!(status, Status::None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn switches_resolve_before_moves() {
        assert!(category_of(&Action::Switch { to: 1 }) < category_of(&Action::Move { slot: 0, z_move: false }));
    }

    #[test]
    fn priority_brackets_map_to_the_pythons_numbers() {
        let mut the_move = Move { priority: "QUICK_ATTACK".into(), ..Move::default() };
        assert_eq!(priority_of(&the_move).unwrap(), 1);
        the_move.priority = "NORMAL".into();
        assert_eq!(priority_of(&the_move).unwrap(), 0);
        the_move.priority = "TRICK_ROOM".into();
        assert_eq!(priority_of(&the_move).unwrap(), -7);
    }

    #[test]
    fn an_unknown_priority_bracket_is_refused_rather_than_guessed() {
        // The first version of this table invented plausible names and mapped everything else to
        // zero. A bracket nobody has mapped has to stop the run, not quietly become normal speed.
        let the_move = Move { priority: "NOT_A_BRACKET".into(), ..Move::default() };
        assert!(matches!(priority_of(&the_move), Err(Refusal::Unported(_))));
    }
}
