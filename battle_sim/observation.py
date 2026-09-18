import random
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

from battle_sim.database.loader import get_move, get_species, normalize_id
from battle_sim.maths.rng import RNG
from battle_sim.mechanics.battle import BattleState, SideState
from battle_sim.mechanics.log import BattleLog
from battle_sim.models.log_events import (
    AbilitiesSwapped,
    AbilityChipDamage,
    AbilityCopied,
    AbilityHealed,
    AbsorbBlocked,
    AbsorbHealed,
    AirBalloonPopped,
    AvoidedWithLevitate,
    BerryWeakened,
    FlashFireAbsorbed,
    FlashFireActivated,
    ItemChipDamage,
    ItemHealed,
    ItemRemoved,
    ItemsSwapped,
    ItemStolen,
    LogEntry,
    MoveBounced,
    MoveUsed,
    ParadoxActivated,
    PpRestored,
    StatDropBlocked,
    StatDropBlockedByItem,
    StatStageChanged,
    StatusMoveBlocked,
    Switched,
    TerrainSetByAbility,
    WeatherSetByAbility,
    WhiteHerbRestored,
)
from battle_sim.models.moves import MoveSlot
from battle_sim.models.pokemon import BelievedSet, Pokemon
from battle_sim.models.spec import PokemonSpec
from battle_sim.teams import ability_from_showdown, build_pokemon
from battle_sim.utils import CHOICE_ITEMS, Ability, Item

type BeliefSampler = Callable[[random.Random], BattleState]

_SOURCE_ABILITIES: dict[str, Ability] = {
    "speed_boost": Ability.SPEED_BOOST,
    "motor_drive": Ability.MOTOR_DRIVE,
    "moxie": Ability.MOXIE,
    "weak_armor": Ability.WEAK_ARMOR,
    "stamina": Ability.STAMINA,
    "berserk": Ability.BERSERK,
    "justified": Ability.JUSTIFIED,
    "thermal_exchange": Ability.THERMAL_EXCHANGE,
    "download": Ability.DOWNLOAD,
    "dauntless_shield": Ability.DAUNTLESS_SHIELD,
    "intrepid_sword": Ability.INTREPID_SWORD,
    "defiant": Ability.DEFIANT,
    "competitive": Ability.COMPETITIVE,
    "soul_heart": Ability.SOUL_HEART,
    "chilling_neigh": Ability.CHILLING_NEIGH,
    "as_one_glastrier": Ability.AS_ONE_GLASTRIER,
    "steadfast": Ability.STEADFAST,
    "beast_boost": Ability.BEAST_BOOST,
    "sap_sipper": Ability.SAP_SIPPER,
    "well_baked_body": Ability.WELL_BAKED_BODY,
}
_CONSUMED_SOURCES = frozenset({"weakness_policy", "seed"})
_BELIEF_CANDIDATES = 12  # posterior sets kept per pokemon; the rest of the tail cannot move an expectation


@dataclass
class Knowledge:
    """Everything one player has learned about one opposing pokemon's hidden set."""

    moves: set[str] = field(default_factory=set)
    ability: Ability | None = None  # None: not yet revealed
    item: Item | None = None  # None: not yet revealed
    item_gone: bool = False  # knocked off, consumed, or stolen: known to hold nothing
    version: int = 0  # bumped on every learned fact: believed pokemon rebuild only on change
    pp_used: dict[str, int] = field(default_factory=dict)  # observed uses per move: a PP floor (no Pressure)
    last_move: str | None = None  # last move used since entering: the choice-lock candidate


class SetPrior:
    """Species -> candidate sets ranked by metagame frequency: the shared 'knowing the meta'.

    Both players draw beliefs from the same prior, exactly as two ladder players share
    the same usage statistics.
    """

    def __init__(self, candidates: dict[str, list[tuple[PokemonSpec, int]]]):
        self._candidates = candidates  # per species: (set, metagame count), most common first

    @classmethod
    def from_teams(cls, teams: Iterable[Sequence[PokemonSpec]]) -> "SetPrior":
        counts: dict[str, dict[str, tuple[PokemonSpec, int]]] = {}
        for team in teams:
            for spec in team:
                canonical = spec.model_copy(update={"moves": [get_move(name).name for name in spec.moves]})
                variants = counts.setdefault(get_species(spec.species).name, {})
                key = canonical.model_dump_json(exclude={"nickname"})
                seen, count = variants.get(key, (canonical, 0))
                variants[key] = (seen, count + 1)
        return cls(
            {
                species: sorted(variants.values(), key=lambda pair: (-pair[1], pair[0].moves))
                for species, variants in counts.items()
            }
        )

    def preview(self, species: str, level: int) -> PokemonSpec:
        """The set a player assumes at team preview: the species' most common one."""
        return self.believe(species, Knowledge()).model_copy(update={"level": level})

    def believe(self, species: str, knowledge: Knowledge) -> PokemonSpec:
        """The most plausible full set: best reveal-consistency, ties broken by frequency."""
        best, _ = self._consistent(species, knowledge)[0]
        return self._reconcile(best, knowledge)

    def sample(self, species: str, knowledge: Knowledge, rng: random.Random) -> PokemonSpec:
        """One frequency-weighted draw from the maximally reveal-consistent sets."""
        consistent = self._consistent(species, knowledge)
        drawn = rng.choices([spec for spec, _ in consistent], weights=[count for _, count in consistent])[0]
        return self._reconcile(drawn, knowledge)

    def posterior(self, species: str, knowledge: Knowledge) -> list[tuple[float, PokemonSpec]]:
        """Every reveal-consistent set with its normalised posterior weight, heaviest first.

        The belief a threat estimate should average over, rather than the single modal set
        `believe` returns for simulation.
        """
        consistent = self._consistent(species, knowledge)
        total = sum(count for _, count in consistent)
        return [(count / total, self._reconcile(spec, knowledge)) for spec, count in consistent]

    def _consistent(self, species: str, knowledge: Knowledge) -> list[tuple[PokemonSpec, int]]:
        """Candidates matching every reveal, in frequency order; the closest ones if none match all.

        Hard filtering is what makes the weights a posterior: a candidate contradicted by a revealed
        move is not merely ranked lower, it is impossible. The relaxed fallback covers sets the prior
        genuinely cannot represent — an illegal-looking reveal from Transform, or a species whose
        corpus sets all predate the move — where an empty belief would be worse than a loose one.
        """
        candidates = self._candidates[get_species(species).name]  # a missing species means the prior can't cover
        matching = [(spec, count) for spec, count in candidates if self._matches(spec, knowledge)]
        if matching:
            return matching
        best = max(self._consistency(candidate, knowledge) for candidate, _ in candidates)
        return [(spec, count) for spec, count in candidates if self._consistency(spec, knowledge) == best]

    def _matches(self, candidate: PokemonSpec, knowledge: Knowledge) -> bool:
        """Whether a candidate set is consistent with every revealed fact."""
        if not knowledge.moves.issubset(candidate.moves):
            return False
        if knowledge.ability is not None and candidate.ability is not knowledge.ability:
            return False
        if knowledge.item_gone:
            return True  # a gone item rules nothing out: what was knocked off is what we never saw
        return knowledge.item is None or candidate.item is knowledge.item

    def _consistency(self, candidate: PokemonSpec, knowledge: Knowledge) -> int:
        score = sum(1 for move in knowledge.moves if move in candidate.moves)
        if knowledge.ability is not None and candidate.ability is knowledge.ability:
            score += 1
        if knowledge.item is not None and not knowledge.item_gone and candidate.item is knowledge.item:
            score += 1
        return score

    def _reconcile(self, candidate: PokemonSpec, knowledge: Knowledge) -> PokemonSpec:
        """Force every revealed fact into the believed set; the prior only fills the gaps."""
        moves = list(candidate.moves)
        missing = sorted(move for move in knowledge.moves if move not in moves)
        while len(moves) < 4 and missing:
            moves.append(missing.pop())
        replaceable = [i for i, move in enumerate(moves) if move not in knowledge.moves]
        # Transform can reveal more than four moves; the overflow stays unmodelled.
        for index, move in zip(reversed(replaceable), missing, strict=False):
            moves[index] = move
        ability = candidate.ability if knowledge.ability is None else knowledge.ability
        item = candidate.item if knowledge.item is None else knowledge.item
        if knowledge.item_gone:
            item = Item.NONE
        return candidate.model_copy(update={"moves": moves, "ability": ability, "item": item})


def believed_sets(
    prior: SetPrior, species: str, knowledge: Knowledge, candidates: int = _BELIEF_CANDIDATES
) -> tuple[BelievedSet, ...]:
    """The posterior over one opposing pokemon's set, trimmed to the candidates worth averaging over.

    The tail past `candidates` is renormalised away rather than kept: threat is an expectation
    evaluated at every search leaf, and a candidate holding a percent of the mass cannot move it
    enough to pay for its damage calculations.
    """
    weighted = prior.posterior(species, knowledge)[:candidates]
    total = sum(weight for weight, _ in weighted)
    return tuple(
        BelievedSet(
            weight=weight / total,
            moves=tuple(get_move(name) for name in spec.moves),
            item=Item.NONE if knowledge.item_gone else spec.item,
            ability=spec.ability,
        )
        for weight, spec in weighted
    )


@dataclass
class _View:
    version: int
    state: BattleState
    believed: list[Pokemon]  # the opponent's believed team, index-aligned with the true one
    mon_versions: list[int]  # each believed pokemon's Knowledge.version at build time


class BattleObserver:
    """Accumulates reveals from battle logs and serves each side the battle as it sees it."""

    def __init__(self, state: BattleState, prior: SetPrior, belief_candidates: tuple[int, int] | None = None):
        """`belief_candidates` caps each viewer's set posterior independently.

        Per viewer because that is what makes a belief model measurable: two sides differing only
        in how many candidate sets they average over can be run against each other in one battle,
        where same-seed side pairs cancel team and roll variance. A cap of 1 is the degenerate
        model that commits to the single likeliest set.
        """
        self._state = state
        self._prior = prior
        self._belief_candidates = belief_candidates or (_BELIEF_CANDIDATES, _BELIEF_CANDIDATES)
        for side in state.sides:
            nicknames = [mon.nickname for mon in side.team]
            if len(set(nicknames)) != len(nicknames):
                raise ValueError(f"Reveal tracking needs unique nicknames per side, got {nicknames}.")
        self._knowledge = tuple({mon.nickname: Knowledge() for mon in side.team} for side in state.sides)
        # Prior lookups stay keyed by the previewed species; formes change mid-battle (Palafin-Hero).
        self._preview_species = tuple({mon.nickname: mon.name for mon in side.team} for side in state.sides)
        self._active = [side.active_pokemon.nickname for side in state.sides]
        self._versions = [0, 0]
        self._views: list[_View | None] = [None, None]
        self._retired: list[Pokemon] = []  # replaced believed pokemon, kept alive so their id()s stay unique

    def ingest(self, log: BattleLog) -> None:
        for entry in log:
            self._observe(entry)

    def view(self, viewer: int) -> BattleState:
        """The believed BattleState for one side: own side true, opponent side believed."""
        opponent = 1 - viewer
        current = self._views[viewer]
        if current is None or current.version != self._versions[opponent] or self._formes_changed(current, opponent):
            current = self._rebuild(viewer)
            self._views[viewer] = current
        self._sync(current.state, current.believed, opponent)
        return current.state

    def sample_view(self, viewer: int, rng: random.Random) -> BattleState:
        """A believed BattleState with the opponent's unrevealed sets sampled from the prior.

        Fresh objects every call — sampling is a per-decision hedge, not a cached belief —
        and every sampled pokemon is pinned in the graveyard so id()-keyed caches stay sound.
        """
        opponent = 1 - viewer
        true_side = self._state.sides[opponent]
        believed = []
        for mon in true_side.team:
            knowledge = self._knowledge[opponent][mon.nickname]
            preview = self._preview_species[opponent][mon.nickname]
            spec = self._prior.sample(preview, knowledge, rng)
            believed.append(build_pokemon(spec.model_copy(update=self._believed_update(mon, preview))))
        self._retired.extend(believed)
        state = self._assemble(viewer, believed)
        self._sync(state, believed, opponent)
        return state

    def _believed_update(self, mon: Pokemon, preview: str) -> dict[str, object]:
        """The fields a believed Pokemon takes from what is publicly visible rather than guessed.

        The forme a Pokemon is standing in is public, and for the formes reached mid-battle the
        ability comes with it: a Sableye that Mega Evolves is visibly a Sableye-Mega, and every
        Sableye-Mega has Magic Bounce. Without this the belief kept the ability sampled for the base
        species, so the search would happily throw status at a Magic Bounce it could see was there.
        """
        update: dict[str, object] = {"species": mon.name, "nickname": mon.nickname, "level": mon.level}
        if normalize_id(mon.name) != normalize_id(preview):
            update["ability"] = ability_from_showdown(get_species(normalize_id(mon.name)).regular_abilities[0])
        return update

    def _formes_changed(self, view: _View, opponent: int) -> bool:
        """Forme changes are public but bump no knowledge version; catch them by name."""
        return any(
            believed.name != true.name
            for believed, true in zip(view.believed, self._state.sides[opponent].team, strict=True)
        )

    # -- Reveal extraction ------------------------------------------------------------

    def _observe(self, entry: LogEntry) -> None:
        match entry:
            case Switched(side=side, withdrew=withdrew, sent_out=sent_out):
                self._active[side] = sent_out
                self._knowledge[side][withdrew].last_move = None  # any lock leaves with the pokemon
            case MoveUsed(side=side, pokemon=pokemon, move=move) if move != "Struggle":
                self._learn_move(side, pokemon, move)
                knowledge = self._knowledge[side][pokemon]
                knowledge.pp_used[move] = knowledge.pp_used.get(move, 0) + 1
                knowledge.last_move = move
            case StatStageChanged(side=side, pokemon=pokemon, source=source):
                self._observe_stat_source(side, pokemon, source)
            case _:
                self._observe_item(entry)
                self._observe_ability(entry)

    def _observe_stat_source(self, side: int, pokemon: str, source: str) -> None:
        if source == "intimidate":
            self._learn_ability(1 - side, self._active[1 - side], Ability.INTIMIDATE)
        elif source in _SOURCE_ABILITIES:
            self._learn_ability(side, pokemon, _SOURCE_ABILITIES[source])
        elif source in _CONSUMED_SOURCES:
            self._learn_item_gone(side, pokemon)

    def _observe_item(self, entry: LogEntry) -> None:
        match entry:
            case ItemChipDamage(side=side, pokemon=pokemon, item=item):
                if item is Item.ROCKY_HELMET:  # the chip lands on the attacker; the helmet is the defender's
                    self._learn_item(1 - side, self._active[1 - side], item)
                else:
                    self._learn_item(side, pokemon, item)
            case (
                ItemHealed(side=side, pokemon=pokemon, item=item)
                | StatDropBlockedByItem(side=side, pokemon=pokemon, item=item)
            ):
                self._learn_item(side, pokemon, item)
            case ItemStolen(side=side, pokemon=pokemon, item=item):
                self._learn_item(side, pokemon, item)
                self._learn_item_gone(1 - side, self._active[1 - side])
            case ItemsSwapped(side=side):
                for swapped in (side, 1 - side):  # who got what is left unknown
                    self._forget_item(swapped, self._active[swapped])
            case (
                ItemRemoved(side=side, pokemon=pokemon)
                | AirBalloonPopped(side=side, pokemon=pokemon)
                | BerryWeakened(side=side, pokemon=pokemon)
                | WhiteHerbRestored(side=side, pokemon=pokemon)
                | PpRestored(side=side, pokemon=pokemon)
            ):
                self._learn_item_gone(side, pokemon)
            case _:
                pass

    def _observe_ability(self, entry: LogEntry) -> None:
        match entry:
            case (
                WeatherSetByAbility(side=side, pokemon=pokemon, ability=ability)
                | TerrainSetByAbility(side=side, pokemon=pokemon, ability=ability)
                | AbsorbHealed(side=side, pokemon=pokemon, ability=ability)
                | AbsorbBlocked(side=side, pokemon=pokemon, ability=ability)
                | AbilityChipDamage(side=side, pokemon=pokemon, ability=ability)
                | AbilityHealed(side=side, pokemon=pokemon, ability=ability)
                | StatDropBlocked(side=side, pokemon=pokemon, ability=ability)
                | StatusMoveBlocked(side=side, pokemon=pokemon, ability=ability)
                | ParadoxActivated(side=side, pokemon=pokemon, ability=ability)
            ):
                self._learn_ability(side, pokemon, ability)
            case AbilityCopied(side=side, pokemon=pokemon, ability=ability):
                self._learn_ability(side, pokemon, ability)  # Trace: the copier now has it...
                self._learn_ability(1 - side, self._active[1 - side], ability)  # ...because the target showed it
            case AbilitiesSwapped(side=side):
                for swapped in (side, 1 - side):
                    self._forget_ability(swapped, self._active[swapped])
            case MoveBounced(side=side, pokemon=pokemon):
                # Being bounced is as public as a Pokemon gets about its ability, and without this
                # the search would keep feeding status to it turn after turn.
                self._learn_ability(side, pokemon, Ability.MAGIC_BOUNCE)
            case AvoidedWithLevitate(side=side, pokemon=pokemon):
                self._learn_ability(side, pokemon, Ability.LEVITATE)
            case FlashFireActivated(side=side, pokemon=pokemon) | FlashFireAbsorbed(side=side, pokemon=pokemon):
                self._learn_ability(side, pokemon, Ability.FLASH_FIRE)
            case _:
                pass

    def _learn_move(self, side: int, nickname: str, move: str) -> None:
        knowledge = self._knowledge[side][nickname]
        if move not in knowledge.moves:
            knowledge.moves.add(move)
            self._bump(side, knowledge)

    def _learn_ability(self, side: int, nickname: str, ability: Ability) -> None:
        knowledge = self._knowledge[side][nickname]
        if knowledge.ability is not ability:
            knowledge.ability = ability
            self._bump(side, knowledge)

    def _forget_ability(self, side: int, nickname: str) -> None:
        knowledge = self._knowledge[side][nickname]
        if knowledge.ability is not None:
            knowledge.ability = None
            self._bump(side, knowledge)

    def _learn_item(self, side: int, nickname: str, item: Item) -> None:
        knowledge = self._knowledge[side][nickname]
        if knowledge.item is not item or knowledge.item_gone:
            knowledge.item = item
            knowledge.item_gone = False
            self._bump(side, knowledge)

    def _learn_item_gone(self, side: int, nickname: str) -> None:
        knowledge = self._knowledge[side][nickname]
        if not knowledge.item_gone:
            knowledge.item_gone = True
            self._bump(side, knowledge)

    def _forget_item(self, side: int, nickname: str) -> None:
        knowledge = self._knowledge[side][nickname]
        if knowledge.item is not None or knowledge.item_gone:
            knowledge.item = None
            knowledge.item_gone = False
            self._bump(side, knowledge)

    def _bump(self, side: int, knowledge: Knowledge) -> None:
        knowledge.version += 1
        self._versions[side] += 1

    # -- Believed-state construction ----------------------------------------------------

    def _rebuild(self, viewer: int) -> _View:
        opponent = 1 - viewer
        true_side = self._state.sides[opponent]
        previous = self._views[viewer]
        believed, mon_versions = [], []
        for index, mon in enumerate(true_side.team):
            knowledge = self._knowledge[opponent][mon.nickname]
            if (
                previous is not None
                and previous.mon_versions[index] == knowledge.version
                and previous.believed[index].name == mon.name
            ):
                believed.append(previous.believed[index])  # unchanged belief: keep the object and its caches
            else:
                if previous is not None:
                    self._retired.append(previous.believed[index])  # pin: cache keys are id()s, never reused
                preview = self._preview_species[opponent][mon.nickname]
                spec = self._prior.believe(preview, knowledge)
                # The current forme is public; the set behind it is still the prior's guess.
                guess = build_pokemon(spec.model_copy(update=self._believed_update(mon, preview)))
                guess.believed_sets = believed_sets(self._prior, preview, knowledge, self._belief_candidates[viewer])
                believed.append(guess)
            mon_versions.append(knowledge.version)
        state = self._assemble(viewer, believed)
        return _View(version=self._versions[opponent], state=state, believed=believed, mon_versions=mon_versions)

    def _assemble(self, viewer: int, believed: list[Pokemon]) -> BattleState:
        opponent = 1 - viewer
        believed_side = SideState(team=believed, active=list(self._state.sides[opponent].active))
        sides = (self._state.sides[0], believed_side) if viewer == 0 else (believed_side, self._state.sides[1])
        return BattleState(sides=sides, field=self._state.field, rng=RNG(seed=0), turn=self._state.turn)

    def _sync(self, state: BattleState, believed: list[Pokemon], opponent: int) -> None:
        true_side = self._state.sides[opponent]
        believed_side = state.sides[opponent]
        believed_side.active = list(true_side.active)
        believed_side.hazards = dict(true_side.hazards)
        believed_side.screens = dict(true_side.screens)
        believed_side.tailwind_turns = true_side.tailwind_turns
        believed_side.needs_switch = true_side.needs_switch
        state.turn = self._state.turn
        for believed_mon, true in zip(believed, true_side.team, strict=True):
            _sync_public(believed_mon, true)
            _apply_inferences(believed_mon, self._knowledge[opponent][true.nickname])


def _apply_inferences(believed: Pokemon, knowledge: Knowledge) -> None:
    """PP floors and choice locks: believed bookkeeping the battle display never shows."""
    believed.choice_locked_move = None
    for slot in MoveSlot:
        move = believed.moves[slot]
        if move is None:
            continue
        used = knowledge.pp_used.get(move.name)  # genuinely optional: most moves are never seen
        if used is not None:
            believed.pp[slot] = max(0, move.pp - used)
        if knowledge.last_move == move.name and believed.item in CHOICE_ITEMS:
            believed.choice_locked_move = slot


def _sync_public(believed: Pokemon, true: Pokemon) -> None:
    """Mirror everything the battle display shows onto the believed pokemon."""
    believed.live_stats.HP = _percent_hp(true, believed)
    believed.status = true.status
    believed.status_turns = true.status_turns
    believed.stat_stages = true.stat_stages.model_copy()
    believed.volatiles = dict(true.volatiles)
    believed.flash_fire_active = true.flash_fire_active
    believed.protect_streak = true.protect_streak
    believed.times_hit = true.times_hit
    believed.paradox_boost = true.paradox_boost
    believed.paradox_from_booster = true.paradox_from_booster


def _percent_hp(true: Pokemon, believed: Pokemon) -> int:
    """Showdown shows opponent HP as a whole percent; translate it onto the believed max."""
    if true.is_fainted():
        return 0
    percent = round(100 * true.live_stats.HP / true.stat_totals.HP)
    return max(1, round(percent / 100 * believed.stat_totals.HP))
