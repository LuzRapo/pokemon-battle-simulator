"""Immutable log entries carrying facts, never display strings — rendering happens in mechanics.log.

Entries snapshot side (0-based) and pokemon (nickname) at event time so rendering needs no battle state.
"""

from dataclasses import dataclass
from typing import Literal

from battle_sim.utils import (
    Ability,
    ExtraStatus,
    Hazards,
    Item,
    Outcome,
    PseudoWeather,
    Stats,
    Status,
    Terrain,
    Type,
    Weather,
)

type StatChangeSource = Literal[
    "move",
    "intimidate",
    "sticky_web",
    "speed_boost",
    "motor_drive",
    "lightning_rod",
    "storm_drain",
    "moxie",
    "weak_armor",
    "stamina",
    "berserk",
    "justified",
    "thermal_exchange",
    "steadfast",
    "beast_boost",
    "download",
    "dauntless_shield",
    "intrepid_sword",
    "defiant",
    "competitive",
    "weakness_policy",
    "seed",
    "sap_sipper",
    "well_baked_body",
    "soul_heart",
    "chilling_neigh",
    "as_one_glastrier",
]
type CantActReason = Literal["frozen", "asleep", "flinch", "paralysis", "confused", "recharge", "loafing"]
type StatusClearance = Literal[
    "thawed",
    "woke",
    "confusion_ended",
    "taunt_ended",
    "freed_from_leech_seed",
    "encore_ended",
    "disable_ended",
    "slow_start_ended",
    "berry",
    "natural_cure",
    "hydration",
    "shed_skin",
    "refreshed",
]
type SurvivalCause = Literal["focus_sash", "sturdy", "endure", "nine_lives"]
type ResidualSource = Literal["sandstorm", "burn", "poison", "toxic", "nightmare", "salt_cure", "curse"]
type EffectivenessLevel = Literal["super", "resisted"]


# -- Switching --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Switched:
    side: int
    withdrew: str
    sent_out: str


@dataclass(frozen=True, slots=True)
class SelfSwitchPending:
    side: int
    pokemon: str


# -- Move execution ----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MoveUsed:
    """`move` always names the slot — the move that fills it and that `BattleObserver` learns the
    user knows. `unleashed_as` is set only when this use was a Z-move: the Z-move's own name, which
    fills no slot of its own and so is never what `move` holds."""

    side: int
    pokemon: str
    move: str
    unleashed_as: str | None = None


@dataclass(frozen=True, slots=True)
class MoveMissed:
    pass


@dataclass(frozen=True, slots=True)
class MoveFailed:
    pass


@dataclass(frozen=True, slots=True)
class DisableApplied:
    side: int
    pokemon: str
    move: str


@dataclass(frozen=True, slots=True)
class DisabledBlocked:
    side: int
    pokemon: str
    move: str


@dataclass(frozen=True, slots=True)
class TauntBlocked:
    side: int
    pokemon: str
    move: str


@dataclass(frozen=True, slots=True)
class DoesNotAffect:
    """Prankster-into-Dark style total immunity announced before the hit."""

    side: int
    pokemon: str


@dataclass(frozen=True, slots=True)
class StatusClauseBlocked:
    """Sleep Clause (or whichever status this build claps a ceiling on): the side already has as
    many Pokemon under this status as the format allows, so this one more just doesn't take."""

    side: int
    pokemon: str
    status: Status


@dataclass(frozen=True, slots=True)
class NoEffect:
    """Type-chart 0x immunity discovered on application."""

    side: int
    pokemon: str


@dataclass(frozen=True, slots=True)
class Effectiveness:
    level: EffectivenessLevel


@dataclass(frozen=True, slots=True)
class CriticalHit:
    """A hit that rolled a critical, announced just before the damage it explains.

    Carries no side, because it is a remark about the hit rather than something that happened to
    somebody — which is also how the games print it. It sits immediately before its `DamageDealt`.
    """


@dataclass(frozen=True, slots=True)
class DamageDealt:
    side: int
    pokemon: str
    amount: int


@dataclass(frozen=True, slots=True)
class MultiHitSummary:
    hits: int


@dataclass(frozen=True, slots=True)
class RecoilDamage:
    side: int
    pokemon: str
    amount: int


@dataclass(frozen=True, slots=True)
class Drained:
    side: int
    pokemon: str
    amount: int


@dataclass(frozen=True, slots=True)
class Healed:
    side: int
    pokemon: str
    amount: int


@dataclass(frozen=True, slots=True)
class Fainted:
    side: int
    pokemon: str


@dataclass(frozen=True, slots=True)
class SurvivedAtOneHp:
    side: int
    pokemon: str
    cause: SurvivalCause


@dataclass(frozen=True, slots=True)
class NineLivesRestored:
    """One of the butler's lives spent: back to full, status burned away, and how many are left."""

    side: int
    pokemon: str
    healed: int
    remaining: int


@dataclass(frozen=True, slots=True)
class ConfusionSelfHit:
    side: int
    pokemon: str
    amount: int


# -- Acting impediments ------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CantAct:
    side: int
    pokemon: str
    reason: CantActReason


@dataclass(frozen=True, slots=True)
class StatusCleared:
    side: int
    pokemon: str
    clearance: StatusClearance


# -- Statuses and stat stages -------------------------------------------------


@dataclass(frozen=True, slots=True)
class StatusInflicted:
    side: int
    pokemon: str
    status: Status


@dataclass(frozen=True, slots=True)
class StatusAlready:
    side: int
    pokemon: str
    status: Status


@dataclass(frozen=True, slots=True)
class VolatileInflicted:
    side: int
    pokemon: str
    volatile: ExtraStatus


@dataclass(frozen=True, slots=True)
class StatStageChanged:
    side: int
    pokemon: str
    stat: Stats
    delta: int
    requested: int
    source: StatChangeSource = "move"


# -- Abilities and items reacting in battle ------------------------------------


@dataclass(frozen=True, slots=True)
class WeatherSetByAbility:
    side: int
    pokemon: str
    ability: Ability


@dataclass(frozen=True, slots=True)
class AvoidedWithLevitate:
    side: int
    pokemon: str


@dataclass(frozen=True, slots=True)
class FlashFireActivated:
    side: int
    pokemon: str


@dataclass(frozen=True, slots=True)
class FlashFireAbsorbed:
    side: int
    pokemon: str


@dataclass(frozen=True, slots=True)
class AbsorbHealed:
    side: int
    pokemon: str
    ability: Ability
    amount: int


@dataclass(frozen=True, slots=True)
class AbsorbBlocked:
    side: int
    pokemon: str
    ability: Ability


@dataclass(frozen=True, slots=True)
class AirBalloonPopped:
    side: int
    pokemon: str


@dataclass(frozen=True, slots=True)
class FloatedOnAirBalloon:
    """A Ground move that could not reach its target because the balloon was holding it up."""

    side: int
    pokemon: str


@dataclass(frozen=True, slots=True)
class AirBalloonRevealed:
    """The balloon announcing itself on the way in, as it does in the games.

    Not decoration: it is the only reason the other trainer can know not to reach for a Ground move.
    """

    side: int
    pokemon: str


@dataclass(frozen=True, slots=True)
class ItemChipDamage:
    """Rocky Helmet, Life Orb, Black Sludge chip — same wording family."""

    side: int
    pokemon: str
    item: Item
    amount: int


@dataclass(frozen=True, slots=True)
class AbilityChipDamage:
    side: int
    pokemon: str
    ability: Ability
    amount: int


@dataclass(frozen=True, slots=True)
class StatDropBlocked:
    side: int
    pokemon: str
    ability: Ability


@dataclass(frozen=True, slots=True)
class StatDropBlockedByItem:
    side: int
    pokemon: str
    item: Item


@dataclass(frozen=True, slots=True)
class WhiteHerbRestored:
    side: int
    pokemon: str


@dataclass(frozen=True, slots=True)
class StatusMoveBlocked:
    """Good as Gold: the holder is immune to opponents' status moves."""

    side: int
    pokemon: str
    ability: Ability


@dataclass(frozen=True, slots=True)
class ParadoxActivated:
    """Protosynthesis / Quark Drive boosting the holder's strongest stat."""

    side: int
    pokemon: str
    ability: Ability
    stat: Stats
    from_booster: bool


@dataclass(frozen=True, slots=True)
class TerrainSetByAbility:
    side: int
    pokemon: str
    ability: Ability


@dataclass(frozen=True, slots=True)
class TypeChanged:
    """Libero / Protean: the user becomes the type of the move it is using."""

    side: int
    pokemon: str
    new_type: Type


@dataclass(frozen=True, slots=True)
class AllStatsReset:
    """Haze wiped every stat change on the field."""


@dataclass(frozen=True, slots=True)
class StatChangesSwept:
    """Nine Lives: the Pokemon facing him loses everything it had built up when he rises."""

    side: int
    pokemon: str


@dataclass(frozen=True, slots=True)
class CourtChanged:
    """Court Change swapped both sides' field effects."""


@dataclass(frozen=True, slots=True)
class Revived:
    side: int
    pokemon: str


@dataclass(frozen=True, slots=True)
class WishMade:
    side: int
    pokemon: str


@dataclass(frozen=True, slots=True)
class FutureAttackQueued:
    side: int
    pokemon: str
    move: str


@dataclass(frozen=True, slots=True)
class FutureAttackLands:
    side: int
    pokemon: str
    move: str


@dataclass(frozen=True, slots=True)
class Transformed:
    side: int
    pokemon: str
    into: str


@dataclass(frozen=True, slots=True)
class ItemRemoved:
    """Knock Off."""

    side: int
    pokemon: str
    item: Item


@dataclass(frozen=True, slots=True)
class ItemDevoured:
    """Nine Lives: an item Tricked onto him, eaten on the spot."""

    side: int
    pokemon: str
    item: Item


@dataclass(frozen=True, slots=True)
class ItemRestored:
    """Nine Lives: an item an opponent took comes back with its holder."""

    side: int
    pokemon: str
    item: Item


@dataclass(frozen=True, slots=True)
class ItemStolen:
    """Pickpocket / Magician: the thief's side, with what it took."""

    side: int
    pokemon: str
    item: Item


@dataclass(frozen=True, slots=True)
class ItemsSwapped:
    """Trick / Switcheroo."""

    side: int
    pokemon: str


@dataclass(frozen=True, slots=True)
class AbilityCopied:
    """Trace."""

    side: int
    pokemon: str
    ability: Ability


@dataclass(frozen=True, slots=True)
class AbilityChanged:
    """Mummy, Entrainment, Worry Seed, Simple Beam: this Pokemon now has a different ability."""

    side: int
    pokemon: str
    ability: Ability


@dataclass(frozen=True, slots=True)
class AbilityUnchanged:
    """An ability that refused to be moved. Announced so the failure is legible rather than silent."""

    side: int
    pokemon: str
    ability: Ability


@dataclass(frozen=True, slots=True)
class AbilitiesSwapped:
    """Skill Swap."""

    side: int
    pokemon: str


@dataclass(frozen=True, slots=True)
class FormeChanged:
    side: int
    pokemon: str
    forme: str


@dataclass(frozen=True, slots=True)
class MoveBounced:
    """Magic Bounce reflected a status move back at its user."""

    side: int
    pokemon: str


@dataclass(frozen=True, slots=True)
class PpRestored:
    """Leppa Berry."""

    side: int
    pokemon: str
    move: str


@dataclass(frozen=True, slots=True)
class ChargingUp:
    side: int
    pokemon: str
    move: str


@dataclass(frozen=True, slots=True)
class AbilityHealed:
    side: int
    pokemon: str
    ability: Ability
    amount: int


@dataclass(frozen=True, slots=True)
class BerryWeakened:
    side: int
    pokemon: str
    item: Item


@dataclass(frozen=True, slots=True)
class ItemHealed:
    """Leftovers / Black Sludge end-of-turn recovery."""

    side: int
    pokemon: str
    item: Item


# -- Hazards, screens, side conditions -----------------------------------------


@dataclass(frozen=True, slots=True)
class HazardSet:
    side: int
    hazard: Hazards


@dataclass(frozen=True, slots=True)
class HazardsCleared:
    side: int
    hazard: Hazards


@dataclass(frozen=True, slots=True)
class HazardDamage:
    side: int
    pokemon: str
    hazard: Hazards
    amount: int


@dataclass(frozen=True, slots=True)
class HazardStatus:
    """Toxic Spikes poisoning on entry."""

    side: int
    pokemon: str
    status: Status


@dataclass(frozen=True, slots=True)
class HazardAbsorbed:
    """A grounded Poison-type soaking up Toxic Spikes."""

    side: int
    pokemon: str


@dataclass(frozen=True, slots=True)
class ScreenSet:
    side: int
    screen: Hazards


@dataclass(frozen=True, slots=True)
class ScreenFaded:
    side: int
    screen: Hazards


@dataclass(frozen=True, slots=True)
class TailwindSet:
    side: int


@dataclass(frozen=True, slots=True)
class TailwindFaded:
    side: int


# -- Field -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WeatherChanged:
    weather: Weather


@dataclass(frozen=True, slots=True)
class WeatherFaded:
    weather: Weather


@dataclass(frozen=True, slots=True)
class TerrainChanged:
    terrain: Terrain


@dataclass(frozen=True, slots=True)
class TerrainFaded:
    terrain: Terrain


@dataclass(frozen=True, slots=True)
class PseudoWeatherStarted:
    kind: PseudoWeather


@dataclass(frozen=True, slots=True)
class PseudoWeatherEnded:
    kind: PseudoWeather


# -- Residuals and outcome ------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ResidualDamage:
    side: int
    pokemon: str
    source: ResidualSource
    amount: int


@dataclass(frozen=True, slots=True)
class Protected:
    side: int
    pokemon: str


@dataclass(frozen=True, slots=True)
class SubstituteTookHit:
    side: int
    pokemon: str


@dataclass(frozen=True, slots=True)
class SubstituteBroke:
    side: int
    pokemon: str


@dataclass(frozen=True, slots=True)
class SubstituteTooWeak:
    pass


@dataclass(frozen=True, slots=True)
class SubstituteAlready:
    side: int
    pokemon: str


@dataclass(frozen=True, slots=True)
class LeechSeedSap:
    side: int
    pokemon: str
    amount: int


@dataclass(frozen=True, slots=True)
class TrapSqueezed:
    """Wrap, Magma Storm and friends: the end-of-turn squeeze on whoever is caught."""

    side: int
    pokemon: str
    amount: int


@dataclass(frozen=True, slots=True)
class TrapReleased:
    """The grip finally lets go, which is also the turn the victim can switch again."""

    side: int
    pokemon: str


@dataclass(frozen=True, slots=True)
class BattleEnded:
    outcome: Outcome


type LogEntry = (
    Switched
    | SelfSwitchPending
    | MoveUsed
    | MoveMissed
    | MoveFailed
    | TauntBlocked
    | DisableApplied
    | DisabledBlocked
    | DoesNotAffect
    | StatusClauseBlocked
    | NoEffect
    | Effectiveness
    | CriticalHit
    | DamageDealt
    | MultiHitSummary
    | RecoilDamage
    | Drained
    | Healed
    | Fainted
    | SurvivedAtOneHp
    | NineLivesRestored
    | ConfusionSelfHit
    | CantAct
    | StatusCleared
    | StatusInflicted
    | StatusAlready
    | VolatileInflicted
    | StatStageChanged
    | WeatherSetByAbility
    | AvoidedWithLevitate
    | FlashFireActivated
    | FlashFireAbsorbed
    | AbsorbHealed
    | AbsorbBlocked
    | AirBalloonPopped
    | FloatedOnAirBalloon
    | AirBalloonRevealed
    | ItemChipDamage
    | ItemHealed
    | AbilityChipDamage
    | AbilityHealed
    | StatDropBlocked
    | StatDropBlockedByItem
    | WhiteHerbRestored
    | StatusMoveBlocked
    | ParadoxActivated
    | TerrainSetByAbility
    | TypeChanged
    | AllStatsReset
    | StatChangesSwept
    | CourtChanged
    | Revived
    | WishMade
    | FutureAttackQueued
    | FutureAttackLands
    | Transformed
    | ChargingUp
    | ItemRemoved
    | ItemDevoured
    | ItemRestored
    | ItemStolen
    | ItemsSwapped
    | AbilityCopied
    | AbilityChanged
    | AbilityUnchanged
    | AbilitiesSwapped
    | FormeChanged
    | MoveBounced
    | PpRestored
    | BerryWeakened
    | HazardSet
    | HazardsCleared
    | HazardDamage
    | HazardStatus
    | HazardAbsorbed
    | ScreenSet
    | ScreenFaded
    | TailwindSet
    | TailwindFaded
    | WeatherChanged
    | WeatherFaded
    | TerrainChanged
    | TerrainFaded
    | PseudoWeatherStarted
    | PseudoWeatherEnded
    | ResidualDamage
    | LeechSeedSap
    | TrapSqueezed
    | TrapReleased
    | Protected
    | SubstituteTookHit
    | SubstituteBroke
    | SubstituteTooWeak
    | SubstituteAlready
    | BattleEnded
)
