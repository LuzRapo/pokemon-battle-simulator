"""Mid-battle species swaps: Mega Evolution, Primal Reversion, and the forme-changing abilities.

A swap replaces the species-derived half of a Pokemon — name, base stats, types, ability, weight —
and leaves everything the trainer chose alone: moves, item, EVs/IVs/nature/level. Base HP is
identical across every Gen-7 mega/primal forme and its base species (verified for all 49 pairs), so
current HP carries straight over instead of being rescaled.

Stance Change, Schooling, Shields Down and Zen Mode are decided here and *applied* by the engine, at
the same controlled points Mega Evolution uses. The split is deliberate: swapping a forme means
rebinding the holder's ability handlers, and doing that from inside an event handler would mutate
the bus while it is walking it. So these functions only answer "which forme should this be?" and
never touch the battle.

Both sides of each pair share a base HP stat, so a swap never rescales current HP and a Pokemon
sitting exactly on a threshold cannot oscillate between formes.

The stone -> forme table is derived from the vendored pokedex's own `requiredItem` field rather than
hand-maintained: that field is what makes a forme reachable, so deriving from it cannot drift out of
step with the species data. Those formes are all flagged `isNonstandard: "Past"` (Mega Evolution
does not exist in Gen 9), which `scope.py` deliberately keeps in scope — it excludes only
`{"CAP", "Custom"}`. `in_scope_species` still bars them as *starting* species; they are only ever
reached mid-battle through the stone, exactly as Palafin-Hero is reached through its ability.
"""

from collections.abc import Sequence
from functools import cache

from battle_sim.database.loader import get_all_species, get_species, normalize_id
from battle_sim.mechanics.effects import EffectRegistry, rewire_active
from battle_sim.mechanics.events import EventBus
from battle_sim.mechanics.log import BattleLog
from battle_sim.models.log_events import FormeChanged
from battle_sim.models.moves import Move
from battle_sim.models.pokemon import Pokemon
from battle_sim.teams import ability_from_showdown, item_from_showdown
from battle_sim.utils import Ability, Category, Item
from battle_sim.zmoves import crystal_type

_MEGA_FORME_PREFIXES = ("Mega", "Primal")
# Checked against the *end* of the name, not against what follows the base species. A forme of a
# forme does not put its suffix where the old check looked: "Meowstic-M-Mega" minus "Meowstic" is
# "M-Mega", which starts with neither word, so Meowstic, Tatsugiri and Magearna-Original were
# rejected before anything else got a chance to map them.
_MEGA_FORME_SUFFIXES = ("-Mega", "-Mega-X", "-Mega-Y", "-Mega-Z", "-Primal")


def _is_transformed_forme(name: str) -> bool:
    return name.endswith(_MEGA_FORME_SUFFIXES)


def _is_playable(forme: str) -> bool:
    """Whether we can actually field this forme, rather than merely name it.

    Two reasons a forme is left out, both of them "this would need real work, not a table entry":

    Eleven of the Legends Z-A megas carry abilities this engine does not model -- Piercing Drill,
    Dragonize, Mega Sol, Commander and so on. An unmapped ability resolves to `Ability.NONE`, so the
    forme would transform and then quietly play without the thing that makes it worth transforming
    into.

    A few also change base HP, which no Gen 6 or 7 mega did and which several things here assume
    never happens. They also tend to need a source forme nobody can obtain -- Floette-Mega wants
    Floette-Eternal -- so the reward for unpicking that assumption is close to nothing.

    Held back rather than half-implemented; `unplayable_formes()` reports which, so the gap is a
    list somebody can work through instead of a silent wrong answer mid-battle.
    """
    species = get_all_species().get(normalize_id(forme))
    if species is None or not species.regular_abilities:
        return False
    if ability_from_showdown(species.regular_abilities[0]) is Ability.NONE:
        return False
    source = get_all_species().get(normalize_id(_source_forme(species.name, species.base_species or "")))
    return source is None or source.base_stats.HP == species.base_stats.HP


def unplayable_formes() -> dict[str, str]:
    """Transformed formes left out of the tables, and why."""
    out = {}
    for species in get_all_species().values():
        if not _is_transformed_forme(species.name) or _is_playable(species.name):
            continue
        ability = species.regular_abilities[0] if species.regular_abilities else ""
        unmapped = not ability or ability_from_showdown(ability) is Ability.NONE
        out[species.name] = ability if unmapped else "changes base HP"
    return out


# Ultra Burst. The vendored entry gates on `requiredItem` like a Mega Stone does, but its
# `baseSpecies` is plain "Necrozma", and plain Necrozma is exactly what cannot Ultra Burst -- only
# the two fused formes can. Reading it out of the data would therefore let the wrong Pokemon
# transform and stop the right ones, so this pairing is written down instead.
_ULTRA_BURST: dict[tuple[str, Item], str] = {
    ("necrozmaduskmane", Item.ULTRANECROZIUM_Z): "Necrozma-Ultra",
    ("necrozmadawnwings", Item.ULTRANECROZIUM_Z): "Necrozma-Ultra",
}

AEGISLASH_SHIELD, AEGISLASH_BLADE = "Aegislash", "Aegislash-Blade"
MIMIKYU, MIMIKYU_BUSTED = "Mimikyu", "Mimikyu-Busted"
ZYGARDE_COMPLETE = "Zygarde-Complete"
POWER_CONSTRUCT_THRESHOLD = 0.5  # the cells arrive when it drops to half or below
# (ability, the forme once the threshold is crossed, the forme above it, the fraction of max HP the
# threshold sits at). Schooling reads the other way round from the rest — the strong forme is the one
# held *above* the line — which is why the pair is stored rather than a single "transformed" name.
_HP_FORMES: tuple[tuple[Ability, str, str, float], ...] = (
    (Ability.SCHOOLING, "Wishiwashi", "Wishiwashi-School", 0.25),
    (Ability.SHIELDS_DOWN, "Minior", "Minior-Meteor", 0.5),
    (Ability.ZEN_MODE, "Darmanitan-Zen", "Darmanitan", 0.5),
)
SCHOOLING_MIN_LEVEL = 20  # below it a Wishiwashi cannot school at all, however healthy
# The abilities that drive a swap, and so the ones that have to survive one — see `swap_forme`.
FORME_ABILITIES = frozenset(
    {
        Ability.STANCE_CHANGE,
        Ability.DISGUISE,
        Ability.ZEN_MODE,
        Ability.SCHOOLING,
        Ability.SHIELDS_DOWN,
        Ability.POWER_CONSTRUCT,
    }
)


def stance_forme(pokemon: Pokemon, move: Move) -> str | None:
    """Which forme Stance Change puts Aegislash in for this move, or None if it stays as it is.

    Attacking draws the blade and King's Shield puts it away; every other status move leaves the
    stance alone, which is what makes Aegislash's turn a real decision rather than a coin flip.
    """
    if pokemon.ability is not Ability.STANCE_CHANGE:
        return None
    if move.name == "King's Shield":
        wanted = AEGISLASH_SHIELD
    elif move.category is not Category.STATUS:
        wanted = AEGISLASH_BLADE
    else:
        return None
    return wanted if wanted != pokemon.name else None


def hp_forme(pokemon: Pokemon) -> str | None:
    """Which forme this Pokemon's HP-triggered ability wants it in, or None if it is already there.

    Checked often and cheaply rather than hooked to a damage event: a Pokemon can cross the line to
    a residual, a recoil, a hazard or a hit, and reading the HP it currently has covers all of them.
    """
    if pokemon.ability is Ability.POWER_CONSTRUCT:
        return _power_construct_forme(pokemon)
    for ability, below, above, threshold in _HP_FORMES:
        if pokemon.ability is not ability:
            continue
        if pokemon.is_fainted():
            return None  # nothing changes forme on the way out
        if ability is Ability.SCHOOLING and pokemon.level < SCHOOLING_MIN_LEVEL:
            wanted = below
        else:
            wanted = below if threshold * pokemon.stat_totals.HP >= pokemon.live_stats.HP else above
        return wanted if wanted != pokemon.name else None
    return None


def _power_construct_forme(pokemon: Pokemon) -> str | None:
    """Zygarde's cells swarming in at half HP — and never leaving again.

    This cannot live in `_HP_FORMES` because every entry there is a threshold a Pokemon crosses in
    both directions: heal a schooling Wishiwashi and it schools again. Power Construct is one-way.
    Once the cells arrive there is no forme to go back to, so healing above half must not undo it —
    which is also why the halves are not symmetrical here: there is only a "below", never an "above".
    """
    if pokemon.is_fainted() or pokemon.name == ZYGARDE_COMPLETE:
        return None
    if pokemon.live_stats.HP > POWER_CONSTRUCT_THRESHOLD * pokemon.stat_totals.HP:
        return None
    return ZYGARDE_COMPLETE


def swap_forme(
    bus: EventBus, registry: EffectRegistry, pokemon: Pokemon, forme: str, side_index: int, log: BattleLog
) -> None:
    """Move a Pokemon onto another forme and rebind what that changes, as Mega Evolution does.

    Takes the bus and registry rather than the battle so that `engine.moves` can reach it without
    importing `engine.turn`. Never call it from inside an event handler: `rewire_active` rewrites the
    bus's handler tables, and doing that while the bus is walking them loses handlers mid-turn.

    The ability that caused the swap survives it. `apply_forme` otherwise takes the new forme's first
    listed ability, which is right for a Mega but wrong here: Zen Mode is Darmanitan's *hidden*
    ability, so reverting out of Zen would hand it Sheer Force and it could never enter Zen again.
    """
    keeper = pokemon.ability if pokemon.ability in FORME_ABILITIES else None
    was_max, was_live = pokemon.stat_totals.HP, pokemon.live_stats.HP
    apply_forme(pokemon, forme)
    if keeper is not None:
        pokemon.ability = keeper
    if keeper is Ability.POWER_CONSTRUCT:
        # `apply_forme` keeps the same *fraction* of a maximum, which is right for every other forme
        # here because they all share a base HP stat. Zygarde-Complete doubles it, and the cells that
        # turn up are extra HP rather than a rescaling of the HP already there: current HP rises by
        # exactly what the maximum rose by, so half a Zygarde becomes a comfortably-above-half
        # Complete. Rescaling instead would arrive at half of a much larger bar, which is the
        # difference between Power Construct being a second wind and being cosmetic.
        pokemon.live_stats.HP = min(pokemon.stat_totals.HP, was_live + (pokemon.stat_totals.HP - was_max))
    rewire_active(bus, registry, pokemon)
    log.add(FormeChanged(side=side_index, pokemon=pokemon.nickname, forme=pokemon.name))


def disguise_broken(pokemon: Pokemon) -> bool:
    """Whether Mimikyu's disguise has already taken its one hit — the busted forme is the record of
    it, so no extra state has to be carried or reset on switch."""
    return pokemon.name == MIMIKYU_BUSTED


def _source_forme(name: str, base_species: str) -> str:
    """Which forme actually holds the stone and turns into `name`.

    Usually the base species, but not always: Meowstic-M and Meowstic-F both list `baseSpecies:
    Meowstic` and both share one Meowsticite, as do Tatsugiri's three and Magearna-Original. Keyed
    on the base alone those pairings collide and only the last one survives -- six Mega Evolutions
    silently lost to a dictionary key. When stripping the Mega suffix leaves a real species, that
    species is the one that transforms.
    """
    for suffix in ("-Mega-X", "-Mega-Y", "-Mega-Z", "-Mega"):
        if name.endswith(suffix):
            candidate = name[: -len(suffix)]
            return candidate if normalize_id(candidate) in get_all_species() else base_species
    return base_species


@cache
def _forme_by_base_and_item() -> dict[tuple[str, Item], str]:
    """(source forme key, held item) -> the forme name that pairing reaches."""
    table: dict[tuple[str, Item], str] = {}
    for species in get_all_species().values():
        if species.required_item is None or species.base_species is None:
            continue
        if not _is_transformed_forme(species.name) or not _is_playable(species.name):
            continue
        item = item_from_showdown(species.required_item)
        if item is not None:
            table[normalize_id(_source_forme(species.name, species.base_species)), item] = species.name
    return table


@cache
def _forme_by_base_and_move() -> dict[tuple[str, str], str]:
    """(base species key, move key) -> the forme that knowing that move reaches.

    One entry in Gen 7, and it is the most-used Pokemon in Anything Goes: Mega Rayquaza carries no
    Mega Stone and is gated on knowing Dragon Ascent instead. The item-driven table skips any forme
    with no `requiredItem`, so it was excluded outright — Rayquaza simply never Mega Evolved.
    """
    table: dict[tuple[str, str], str] = {}
    for species in get_all_species().values():
        if species.required_move is None or species.base_species is None:
            continue
        if not _is_transformed_forme(species.name) or not _is_playable(species.name):
            continue
        table[normalize_id(species.base_species), normalize_id(species.required_move)] = species.name
    return table


def ultra_bursts(species: str, item: Item) -> bool:
    """Whether this pairing Ultra Bursts rather than Mega Evolving.

    They are separate mechanics: a side may do both in one battle, so they cannot share a flag.
    """
    return (normalize_id(species), item) in _ULTRA_BURST


def mega_forme(species: str, item: Item, known_moves: Sequence[str] = ()) -> str | None:
    """The forme this species reaches right now, or None if nothing about it megas.

    Held item first, then the move-gated pairs. `known_moves` may be left out by callers that only
    care about stones — every Gen 7 move-gated forme is Rayquaza's.
    """
    key = normalize_id(species)
    burst = _ULTRA_BURST.get((key, item))
    if burst is not None:
        return burst
    by_item = _forme_by_base_and_item().get((key, item))
    if by_item is not None:
        return by_item
    # Holding a Z-crystal locks Rayquaza out of Mega Evolution, which is the whole reason a Z-move
    # Rayquaza is a different Pokemon from a Mega one rather than strictly worse. `crystal_type`
    # answers for the generic crystals, which are the ones such a set would ever hold.
    if crystal_type(item) is not None:
        return None
    moves = _forme_by_base_and_move()
    return next((forme for move in known_moves if (forme := moves.get((key, normalize_id(move)))) is not None), None)


@cache
def mega_stones() -> frozenset[Item]:
    """Every item that reaches a Mega Evolution or Primal Reversion forme."""
    return frozenset(item for _, item in _forme_by_base_and_item())


def apply_forme(pokemon: Pokemon, forme: str) -> None:
    """Swap a live Pokemon onto another forme's species data, in place.

    Current HP keeps its *fraction* of the maximum. No Gen-7 mega or primal forme changes base HP,
    so for those this is exactly a no-op on HP; it matters only for the handful of other formes that
    do (Zygarde-Complete's Power Construct doubles it).

    Ability event handlers are NOT rewired here — a caller holding a live battle must follow this
    with `rewire_active`, while a reconstructed state (which never bound handlers) must not.
    """
    species = get_species(forme)
    fraction = pokemon.live_stats.HP / pokemon.stat_totals.HP
    pokemon.name = species.name
    pokemon.base_stats = species.base_stats
    pokemon.types = species.types
    pokemon.ability = ability_from_showdown(species.regular_abilities[0])
    pokemon.weight_kg = species.weight_kg
    pokemon.refresh_stats()
    pokemon.live_stats.HP = max(1, round(fraction * pokemon.stat_totals.HP)) if fraction > 0 else 0
