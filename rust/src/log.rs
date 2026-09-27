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
    /// A charge move's first turn: it commits, boosts if it does that, and stops there.
    ChargingUp {
        side: i32,
        pokemon: String,
        #[serde(rename = "move")]
        the_move: String,
    },
    /// Disable landing: which move it silenced.
    DisableApplied {
        side: i32,
        pokemon: String,
        #[serde(rename = "move")]
        the_move: String,
    },
    /// A disabled move chosen anyway: refused before any PP is spent or `MoveUsed` is logged.
    DisabledBlocked {
        side: i32,
        pokemon: String,
        #[serde(rename = "move")]
        the_move: String,
    },
    /// A status move turned aside by Taunt.
    TauntBlocked {
        side: i32,
        pokemon: String,
        #[serde(rename = "move")]
        the_move: String,
    },
    /// The move resolved but nothing came of it — every effect it had was skipped. The Python adds
    /// this by checking whether the log grew, so it is a statement about the log rather than about
    /// the move, and it is ported the same way.
    MoveFailed,
    /// The weather or terrain changing, fading, or a side raising a screen.
    WeatherChanged {
        weather: String,
    },
    WeatherFaded {
        weather: String,
    },
    TerrainChanged {
        terrain: String,
    },
    TerrainFaded {
        terrain: String,
    },
    ScreenSet {
        side: i32,
        screen: String,
    },
    ScreenFaded {
        side: i32,
        screen: String,
    },
    TailwindSet {
        side: i32,
    },
    TailwindFaded {
        side: i32,
    },
    HazardSet {
        side: i32,
        hazard: String,
    },
    HazardsCleared {
        side: i32,
        hazard: String,
    },
    HazardDamage {
        side: i32,
        pokemon: String,
        hazard: String,
        amount: i32,
    },
    HazardStatus {
        side: i32,
        pokemon: String,
        status: String,
    },
    HazardAbsorbed {
        side: i32,
        pokemon: String,
    },
    /// A move healing its user.
    Healed {
        side: i32,
        pokemon: String,
        amount: i32,
    },
    /// How many times a multi-hit move landed, reported once after the blows.
    MultiHitSummary {
        hits: i32,
    },
    /// A total immunity announced before the hit: an ability refusing a status, a volatile that
    /// cannot take hold. It deliberately does not say what refused it.
    DoesNotAffect {
        side: i32,
        pokemon: String,
    },
    /// An ability healing its holder at the end of the turn.
    AbilityHealed {
        side: i32,
        pokemon: String,
        ability: String,
        amount: i32,
    },
    /// The weather or terrain a Pokemon brings onto the field.
    WeatherSetByAbility {
        side: i32,
        pokemon: String,
        ability: String,
    },
    TerrainSetByAbility {
        side: i32,
        pokemon: String,
        ability: String,
    },
    /// Clear Body and its relatives refusing an opponent's stat drop.
    StatDropBlocked {
        side: i32,
        pokemon: String,
        ability: String,
    },
    /// Liquid Ooze turning a drink into a wound.
    DrainBackfired {
        side: i32,
        pokemon: String,
        ability: String,
        amount: i32,
    },
    /// A draining move giving its user back a share of what it dealt.
    Drained {
        side: i32,
        pokemon: String,
        amount: i32,
    },
    /// A pivot announcing that its user is on its way out.
    SelfSwitchPending {
        side: i32,
        pokemon: String,
    },
    /// A Protect turning a move aside, and an Endure surviving one.
    Protected {
        side: i32,
        pokemon: String,
    },
    SurvivedAtOneHp {
        side: i32,
        pokemon: String,
        cause: String,
    },
    /// Leech Seed's drink, and a trap squeezing or letting go.
    LeechSeedSap {
        side: i32,
        pokemon: String,
        amount: i32,
    },
    TrapSqueezed {
        side: i32,
        pokemon: String,
        amount: i32,
    },
    TrapReleased {
        side: i32,
        pokemon: String,
    },
    /// A volatile taking hold — a flinch, a confusion.
    VolatileInflicted {
        side: i32,
        pokemon: String,
        volatile: String,
    },
    /// A confused Pokemon hitting itself instead of acting.
    ConfusionSelfHit {
        side: i32,
        pokemon: String,
        amount: i32,
    },
    /// Rough Skin, Iron Barbs, Aftermath: an ability taking a bite out of whoever touched it.
    AbilityChipDamage {
        side: i32,
        pokemon: String,
        ability: String,
        amount: i32,
    },
    /// Rocky Helmet's share of the same idea.
    ItemChipDamage {
        side: i32,
        pokemon: String,
        item: String,
        amount: i32,
    },
    /// A berry eaten for health. The Python logs no amount here, only that it happened.
    ItemHealed {
        side: i32,
        pokemon: String,
        item: String,
    },
    /// A resist berry halving a super-effective hit, and being eaten for it.
    BerryWeakened {
        side: i32,
        pokemon: String,
        item: String,
    },
    /// Crash damage, which the Python logs under the same entry as recoil.
    RecoilDamage {
        side: i32,
        pokemon: String,
        amount: i32,
    },
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
    CantAct {
        side: i32,
        pokemon: String,
        reason: String,
    },
    StatusInflicted {
        side: i32,
        pokemon: String,
        status: String,
    },
    StatusAlready {
        side: i32,
        pokemon: String,
        status: String,
    },
    StatusCleared {
        side: i32,
        pokemon: String,
        clearance: String,
    },
    StatStageChanged {
        side: i32,
        pokemon: String,
        stat: String,
        delta: i32,
        requested: i32,
        source: String,
    },
    ResidualDamage {
        side: i32,
        pokemon: String,
        source: String,
        amount: i32,
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
