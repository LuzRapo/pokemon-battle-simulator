"""BattleLog accumulates structured entries; render_text is the only place display strings are formed."""

from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum

from battle_sim.models.log_events import (
    AbilitiesSwapped,
    AbilityChipDamage,
    AbilityCopied,
    AbilityHealed,
    AbsorbBlocked,
    AbsorbHealed,
    AirBalloonPopped,
    AllStatsReset,
    AvoidedWithLevitate,
    BattleEnded,
    BerryWeakened,
    CantAct,
    ChargingUp,
    ConfusionSelfHit,
    CourtChanged,
    DamageDealt,
    DisableApplied,
    DisabledBlocked,
    DoesNotAffect,
    Drained,
    Effectiveness,
    Fainted,
    FlashFireAbsorbed,
    FlashFireActivated,
    FormeChanged,
    FutureAttackLands,
    FutureAttackQueued,
    HazardAbsorbed,
    HazardDamage,
    HazardsCleared,
    HazardSet,
    HazardStatus,
    Healed,
    ItemChipDamage,
    ItemHealed,
    ItemRemoved,
    ItemsSwapped,
    ItemStolen,
    LeechSeedSap,
    LogEntry,
    MoveBounced,
    MoveFailed,
    MoveMissed,
    MoveUsed,
    MultiHitSummary,
    NoEffect,
    ParadoxActivated,
    PpRestored,
    Protected,
    PseudoWeatherEnded,
    PseudoWeatherStarted,
    RecoilDamage,
    ResidualDamage,
    Revived,
    ScreenFaded,
    ScreenSet,
    SelfSwitchPending,
    StatDropBlocked,
    StatDropBlockedByItem,
    StatStageChanged,
    StatusAlready,
    StatusClauseBlocked,
    StatusCleared,
    StatusInflicted,
    StatusMoveBlocked,
    SubstituteAlready,
    SubstituteBroke,
    SubstituteTookHit,
    SubstituteTooWeak,
    SurvivedAtOneHp,
    Switched,
    TailwindFaded,
    TailwindSet,
    TauntBlocked,
    TerrainChanged,
    TerrainFaded,
    TerrainSetByAbility,
    Transformed,
    TypeChanged,
    VolatileInflicted,
    WeatherChanged,
    WeatherFaded,
    WeatherSetByAbility,
    WhiteHerbRestored,
    WishMade,
    ZMoveUnleashed,
)
from battle_sim.utils import ExtraStatus, Outcome, Status


@dataclass(slots=True)
class BattleLog:
    entries: list[LogEntry] = field(default_factory=list)

    def add(self, entry: LogEntry) -> None:
        self.entries.append(entry)

    def rendered(self) -> list[str]:
        return [render_text(entry) for entry in self.entries]

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[LogEntry]:
        yield from self.entries


_STATUS_INFLICTED_VERB: dict[Status, str] = {
    Status.BURN: "was burned",
    Status.POISON: "was poisoned",
    Status.TOXIC: "was badly poisoned",
    Status.PARALYSIS: "was paralyzed",
    Status.SLEEP: "fell asleep",
    Status.FREEZE: "was frozen solid",
}
_STATUS_ADJECTIVE: dict[Status, str] = {
    Status.BURN: "burned",
    Status.POISON: "poisoned",
    Status.TOXIC: "badly poisoned",
    Status.PARALYSIS: "paralyzed",
    Status.SLEEP: "asleep",
    Status.FREEZE: "frozen",
}
_VOLATILE_INFLICTED_VERB: dict[ExtraStatus, str] = {
    ExtraStatus.CONFUSION: "became confused",
    ExtraStatus.ENCORE: "must do an encore",
    ExtraStatus.FOCUS_ENERGY: "is getting pumped",
    ExtraStatus.YAWN: "grew drowsy",
    ExtraStatus.FLINCH: "flinched",
    ExtraStatus.IDENTIFIED: "was identified",
    ExtraStatus.MIRACLE_EYE: "is in Miracle Eye's sight",
    ExtraStatus.LEECH_SEED: "was seeded",
    ExtraStatus.NIGHTMARE: "fell into a nightmare",
    ExtraStatus.PROTECT: "protected itself",
    ExtraStatus.SUBSTITUTE: "put in a substitute",
    ExtraStatus.TAUNT: "fell for the taunt",
    ExtraStatus.SALT_CURE: "is being salted",
    ExtraStatus.CURSE: "was cursed",
    ExtraStatus.PERISH: "will faint in three turns",
    ExtraStatus.DESTINY_BOND: "is hoping to take its attacker down with it",
    ExtraStatus.ENDURE: "braced itself",
}
_CANT_ACT_TEXT: dict[str, str] = {
    "frozen": "is frozen solid!",
    "asleep": "is fast asleep.",
    "flinch": "flinched and couldn't move!",
    "paralysis": "is paralyzed and couldn't move!",
    "confused": "is confused…",
    "recharge": "must recharge!",
    "loafing": "is loafing around!",
}
_FUTURE_ATTACK_QUEUED_TEXT: dict[str, str] = {
    "Future Sight": "foresaw an attack!",
    "Doom Desire": "chose Doom Desire as its destiny!",
}
_STATUS_CLEARED_TEXT: dict[str, str] = {
    "thawed": "thawed out!",
    "woke": "woke up!",
    "confusion_ended": "snapped out of confusion!",
    "taunt_ended": "shook off the taunt!",
    "freed_from_leech_seed": "was freed from Leech Seed!",
    "encore_ended": "'s encore ended!",
    "disable_ended": "'s move is no longer disabled!",
    "berry": "'s berry cured its status!",
    "refreshed": "shook off its status!",
    "natural_cure": "was cured on the way out!",
    "hydration": "was cured by Hydration!",
}
_RESIDUAL_TEXT: dict[str, str] = {
    "sandstorm": "was hurt by the sandstorm!",
    "burn": "was hurt by its burn!",
    "poison": "was hurt by poison!",
    "toxic": "was hurt by toxic!",
    "nightmare": "is locked in a nightmare!",
    "salt_cure": "is hurt by Salt Cure!",
    "curse": "is afflicted by the curse!",
}
_OUTCOME_TEXT: dict[Outcome, str] = {
    Outcome.P1_WIN: "Player 1 wins!",
    Outcome.P2_WIN: "Player 2 wins!",
    Outcome.DRAW: "It's a draw.",
}


def _pretty(kind: Enum) -> str:
    return kind.name.replace("_", " ").title()


def _label(side: int, pokemon: str) -> str:
    return f"P{side + 1}'s {pokemon}"


def render_text(entry: LogEntry) -> str:  # noqa: C901 — exhaustive match over every entry type is this function's job
    match entry:
        case Switched(side, withdrew, sent_out):
            return f"P{side + 1} withdrew {withdrew} and sent out {sent_out}!"
        case SelfSwitchPending(side, pokemon):
            return f"{_label(side, pokemon)} is switching out!"
        case ZMoveUnleashed(side, pokemon, move):
            return f"{_label(side, pokemon)} unleashed {move}!"
        case MoveUsed(side, pokemon, move):
            return f"{_label(side, pokemon)} used {move}!"
        case MoveMissed():
            return "But it missed!"
        case MoveFailed():
            return "But it failed!"
        case DisableApplied(side, pokemon, move):
            return f"{_label(side, pokemon)}'s {move} was disabled!"
        case DisabledBlocked(side, pokemon, move):
            return f"{_label(side, pokemon)}'s {move} is disabled!"
        case TauntBlocked(side, pokemon, move):
            return f"{_label(side, pokemon)} can't use {move} after Taunt!"
        case DoesNotAffect(side, pokemon):
            return f"It doesn't affect {_label(side, pokemon)}…"
        case NoEffect(side, pokemon):
            return f"It had no effect on {_label(side, pokemon)}."
        case Effectiveness(level):
            return "It's super effective!" if level == "super" else "It's not very effective…"
        case DamageDealt(side, pokemon, amount):
            return f"{_label(side, pokemon)} took {amount} damage!"
        case MultiHitSummary(hits):
            return f"Hit {hits} time{'s' if hits != 1 else ''}!"
        case RecoilDamage(side, pokemon, amount):
            return f"{_label(side, pokemon)} took {amount} recoil damage!"
        case Drained(side, pokemon, amount):
            return f"{_label(side, pokemon)} drained {amount} HP!"
        case Healed(side, pokemon, amount):
            return f"{_label(side, pokemon)} regained health! ({amount} HP)"
        case Fainted(side, pokemon):
            return f"{_label(side, pokemon)} fainted!"
        case SurvivedAtOneHp(side, pokemon, cause):
            if cause == "focus_sash":
                return f"{_label(side, pokemon)} hung on with its Focus Sash!"
            if cause == "endure":
                return f"{_label(side, pokemon)} endured the hit!"
            return f"{_label(side, pokemon)} held on with Sturdy!"
        case ConfusionSelfHit(side, pokemon, amount):
            return f"{_label(side, pokemon)} is confused! It hurt itself in confusion! ({amount} HP)"
        case CantAct(side, pokemon, reason):
            return f"{_label(side, pokemon)} {_CANT_ACT_TEXT[reason]}"
        case StatusCleared(side, pokemon, clearance):
            text = _STATUS_CLEARED_TEXT[clearance]
            joiner = "" if text.startswith("'") else " "
            return f"{_label(side, pokemon)}{joiner}{text}"
        case StatusInflicted(side, pokemon, status):
            return f"{_label(side, pokemon)} {_STATUS_INFLICTED_VERB[status]}!"
        case StatusAlready(side, pokemon, status):
            return f"{_label(side, pokemon)} is already {_STATUS_ADJECTIVE[status]}."
        case StatusClauseBlocked(side, pokemon, status):
            adjective = _STATUS_ADJECTIVE[status]
            return f"{_label(side, pokemon)} is unaffected — too many of that side are already {adjective}!"
        case VolatileInflicted(side, pokemon, volatile):
            return f"{_label(side, pokemon)} {_VOLATILE_INFLICTED_VERB[volatile]}!"
        case StatStageChanged(side, pokemon, stat, delta, requested, source):
            label = _label(side, pokemon)
            if source == "intimidate":
                return f"{label}'s Attack fell from Intimidate!"
            if source == "sticky_web":
                return f"{label}'s Speed fell by {-delta} from Sticky Web!"
            if source == "speed_boost":
                return f"{label}'s Speed rose via Speed Boost!"
            if source == "motor_drive":
                return f"{label}'s Motor Drive raised its Speed!"
            if source == "moxie":
                return f"{label}'s Moxie raised its Attack!"
            stat_label = stat.name.replace("_", " ").title()
            suffix = "" if source == "move" else f" via {source.replace('_', ' ').title()}"
            if delta == 0:
                direction = "higher" if requested > 0 else "lower"
                return f"{label}'s {stat_label} won't go any {direction}!"
            if delta > 0:
                return f"{label}'s {stat_label} rose by {delta}{suffix}!"
            return f"{label}'s {stat_label} fell by {-delta}{suffix}!"
        case WeatherSetByAbility(side, pokemon, ability):
            return f"{_label(side, pokemon)}'s {_pretty(ability)} set the weather!"
        case AvoidedWithLevitate(side, pokemon):
            return f"{_label(side, pokemon)} avoided the attack with Levitate!"
        case FlashFireActivated(side, pokemon):
            return f"{_label(side, pokemon)}'s Flash Fire activated!"
        case FlashFireAbsorbed(side, pokemon):
            return f"{_label(side, pokemon)}'s Flash Fire absorbed the attack!"
        case AbsorbHealed(side, pokemon, ability, amount):
            return f"{_label(side, pokemon)} restored HP via {_pretty(ability)}! ({amount} HP)"
        case AbsorbBlocked(side, pokemon, ability):
            return f"{_label(side, pokemon)}'s {_pretty(ability)} blocked the attack!"
        case AirBalloonPopped(side, pokemon):
            return f"{_label(side, pokemon)}'s Air Balloon popped!"
        case ItemChipDamage(side, pokemon, item, amount):
            return f"{_label(side, pokemon)} was hurt by {_pretty(item)}! ({amount} HP)"
        case AbilityChipDamage(side, pokemon, ability, amount):
            return f"{_label(side, pokemon)} was hurt by {_pretty(ability)}! ({amount} HP)"
        case StatDropBlocked(side, pokemon, ability):
            return f"{_label(side, pokemon)}'s {_pretty(ability)} prevents stat loss!"
        case StatDropBlockedByItem(side, pokemon, item):
            return f"{_label(side, pokemon)}'s {_pretty(item)} prevents stat loss!"
        case WhiteHerbRestored(side, pokemon):
            return f"{_label(side, pokemon)} restored its stats using its White Herb!"
        case StatusMoveBlocked(side, pokemon, ability):
            return f"{_label(side, pokemon)}'s {_pretty(ability)} blocked the status move!"
        case ParadoxActivated(side, pokemon, ability, stat, from_booster):
            stat_label = stat.name.replace("_", " ").title()
            trigger = "its Booster Energy" if from_booster else "the conditions"
            return f"{_label(side, pokemon)}'s {_pretty(ability)} boosted its {stat_label} using {trigger}!"
        case TerrainSetByAbility(side, pokemon, ability):
            return f"{_label(side, pokemon)}'s {_pretty(ability)} set the terrain!"
        case TypeChanged(side, pokemon, new_type):
            return f"{_label(side, pokemon)} became the {_pretty(new_type)} type!"
        case AllStatsReset():
            return "All stat changes were eliminated!"
        case CourtChanged():
            return "Court Change swapped the battlefield effects!"
        case Revived(side, pokemon):
            return f"{_label(side, pokemon)} was revived and is ready to fight again!"
        case WishMade(side, pokemon):
            return f"{_label(side, pokemon)} made a wish!"
        case FutureAttackQueued(side, pokemon, move):
            return f"{_label(side, pokemon)} {_FUTURE_ATTACK_QUEUED_TEXT.get(move, 'foresaw an attack!')}"
        case FutureAttackLands(side, pokemon, move):
            return f"{_label(side, pokemon)} took the {move} attack!"
        case Transformed(side, pokemon, into):
            return f"{_label(side, pokemon)} transformed into {into}!"
        case ItemRemoved(side, pokemon, item):
            return f"{_label(side, pokemon)} lost its {_pretty(item)}!"
        case ItemStolen(side, pokemon, item):
            return f"{_label(side, pokemon)} stole the target's {_pretty(item)}!"
        case ItemsSwapped(side, pokemon):
            return f"{_label(side, pokemon)} switched items with its target!"
        case AbilityCopied(side, pokemon, ability):
            return f"{_label(side, pokemon)} traced {_pretty(ability)}!"
        case AbilitiesSwapped(side, pokemon):
            return f"{_label(side, pokemon)} swapped abilities with its target!"
        case FormeChanged(side, pokemon, forme):
            return f"{_label(side, pokemon)} changed into {forme}!"
        case MoveBounced(side, pokemon):
            return f"{_label(side, pokemon)} bounced the move back!"
        case PpRestored(side, pokemon, move):
            return f"{_label(side, pokemon)} restored {move}'s PP with its Leppa Berry!"
        case ChargingUp(side, pokemon, move):
            return f"{_label(side, pokemon)} is charging up {move}!"
        case AbilityHealed(side, pokemon, ability, amount):
            return f"{_label(side, pokemon)} restored HP via {_pretty(ability)}! ({amount} HP)"
        case BerryWeakened(side, pokemon, item):
            return f"The {_pretty(item)} weakened the damage to {_label(side, pokemon)}!"
        case ItemHealed(side, pokemon, item):
            return f"{_label(side, pokemon)} restored a little HP using {_pretty(item)}!"
        case HazardsCleared(side, hazard):
            return f"{_pretty(hazard)} disappeared from around P{side + 1}'s side!"
        case HazardSet(side, hazard):
            return f"{_pretty(hazard)} was scattered on P{side + 1}'s side!"
        case HazardDamage(side, pokemon, hazard, amount):
            return f"{_label(side, pokemon)} was hurt by {_pretty(hazard)}! ({amount} HP)"
        case HazardStatus(side, pokemon, status):
            adverb = "badly poisoned" if status is Status.TOXIC else "poisoned"
            return f"{_label(side, pokemon)} was {adverb} by Toxic Spikes!"
        case HazardAbsorbed(side, pokemon):
            return f"{_label(side, pokemon)} absorbed the Toxic Spikes!"
        case ScreenSet(side, screen):
            return f"{_pretty(screen)} raised on P{side + 1}'s side!"
        case ScreenFaded(side, screen):
            return f"P{side + 1}'s {screen.name} wore off."
        case TailwindSet(side):
            return f"The Tailwind blew from behind P{side + 1}'s team!"
        case TailwindFaded(side):
            return f"P{side + 1}'s Tailwind faded."
        case WeatherChanged(weather):
            return f"The weather changed to {_pretty(weather)}!"
        case WeatherFaded(weather):
            return f"{_pretty(weather)} faded."
        case TerrainChanged(terrain):
            return f"{_pretty(terrain)} Terrain set!"
        case TerrainFaded(terrain):
            return f"{_pretty(terrain)} Terrain faded."
        case PseudoWeatherStarted(kind):
            return f"{_pretty(kind)} began!"
        case PseudoWeatherEnded(kind):
            return f"{_pretty(kind)} ended."
        case ResidualDamage(side, pokemon, source, amount):
            return f"{_label(side, pokemon)} {_RESIDUAL_TEXT[source]} ({amount} HP)"
        case Protected(side, pokemon):
            return f"{_label(side, pokemon)} protected itself!"
        case SubstituteTookHit(side, pokemon):
            return f"The substitute took damage for {_label(side, pokemon)}!"
        case SubstituteBroke(side, pokemon):
            return f"{_label(side, pokemon)}'s substitute faded!"
        case SubstituteTooWeak():
            return "It was too weak to make a substitute!"
        case SubstituteAlready(side, pokemon):
            return f"{_label(side, pokemon)} already has a substitute!"
        case LeechSeedSap(side, pokemon, amount):
            return f"{_label(side, pokemon)}'s health is sapped by Leech Seed! ({amount} HP)"
        case BattleEnded(outcome):
            return _OUTCOME_TEXT[outcome]
