"""Mid-battle species swaps: Mega Evolution, Primal Reversion and the forme-changing abilities."""

from collections.abc import Sequence
from functools import cache

from battle_sim.database.loader import get_all_species, get_all_z_moves, get_species, normalize_id
from battle_sim.mechanics.effects import EffectRegistry, rewire_active
from battle_sim.mechanics.events import EventBus
from battle_sim.mechanics.log import BattleLog
from battle_sim.models.log_events import FormeChanged
from battle_sim.models.moves import Move
from battle_sim.models.pokemon import Pokemon
from battle_sim.teams import ability_from_showdown, item_from_showdown, item_showdown_name
from battle_sim.utils import Ability, Category, Item
from battle_sim.zmoves import crystal_type

_MEGA_FORME_PREFIXES = ("Mega", "Primal")
# Checked against the *end* of the name, not against what follows the base species.
_MEGA_FORME_SUFFIXES = ("-Mega", "-Mega-X", "-Mega-Y", "-Mega-Z", "-Primal")


def _is_transformed_forme(name: str) -> bool:
    return name.endswith(_MEGA_FORME_SUFFIXES)


def _is_playable(forme: str) -> bool:
    """Whether we can actually field this forme, rather than merely name it."""
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


# Ultra Burst is written down, because the vendored data would let plain Necrozma transform.
_ULTRA_BURST: dict[tuple[str, Item], str] = {
    ("necrozmaduskmane", Item.ULTRANECROZIUM_Z): "Necrozma-Ultra",
    ("necrozmadawnwings", Item.ULTRANECROZIUM_Z): "Necrozma-Ultra",
}

AEGISLASH_SHIELD, AEGISLASH_BLADE = "Aegislash", "Aegislash-Blade"
MIMIKYU, MIMIKYU_BUSTED = "Mimikyu", "Mimikyu-Busted"
ZYGARDE_COMPLETE = "Zygarde-Complete"
POWER_CONSTRUCT_THRESHOLD = 0.5  # the cells arrive when it drops to half or below
# (ability, forme past the threshold, forme above it, threshold as a fraction of max HP).
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
    """Which forme Stance Change puts Aegislash in for this move, or None if it stays as it is."""
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
    """Which forme this Pokemon's HP-triggered ability wants it in, or None if it is already there."""
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
    """Zygarde's cells swarming in at half HP — and never leaving again."""
    if pokemon.is_fainted() or pokemon.name == ZYGARDE_COMPLETE:
        return None
    if pokemon.live_stats.HP > POWER_CONSTRUCT_THRESHOLD * pokemon.stat_totals.HP:
        return None
    return ZYGARDE_COMPLETE


def swap_forme(
    bus: EventBus, registry: EffectRegistry, pokemon: Pokemon, forme: str, side_index: int, log: BattleLog
) -> None:
    """Move a Pokemon onto another forme and rebind what that changes, as Mega Evolution does."""
    keeper = pokemon.ability if pokemon.ability in FORME_ABILITIES else None
    was_max, was_live = pokemon.stat_totals.HP, pokemon.live_stats.HP
    apply_forme(pokemon, forme)
    if keeper is not None:
        pokemon.ability = keeper
    if keeper is Ability.POWER_CONSTRUCT:
        # `apply_forme` keeps the same fraction of max HP, right for every forme that shares its base HP.
        pokemon.live_stats.HP = min(pokemon.stat_totals.HP, was_live + (pokemon.stat_totals.HP - was_max))
    rewire_active(bus, registry, pokemon)
    log.add(FormeChanged(side=side_index, pokemon=pokemon.nickname, forme=pokemon.name))


def disguise_broken(pokemon: Pokemon) -> bool:
    """Whether Mimikyu's disguise has already taken its one hit, as recorded by the busted forme."""
    return pokemon.name == MIMIKYU_BUSTED


def _source_forme(name: str, base_species: str) -> str:
    """Which forme actually holds the stone and turns into `name`."""
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
    """(base species key, move key) -> the forme that knowing that move reaches."""
    table: dict[tuple[str, str], str] = {}
    for species in get_all_species().values():
        if species.required_move is None or species.base_species is None:
            continue
        if not _is_transformed_forme(species.name) or not _is_playable(species.name):
            continue
        table[normalize_id(species.base_species), normalize_id(species.required_move)] = species.name
    return table


def ultra_bursts(species: str, item: Item) -> bool:
    """Whether this pairing Ultra Bursts rather than Mega Evolving."""
    return (normalize_id(species), item) in _ULTRA_BURST


def mega_forme(species: str, item: Item, known_moves: Sequence[str] = ()) -> str | None:
    """The forme this species reaches right now, or None if nothing about it megas."""
    key = normalize_id(species)
    burst = _ULTRA_BURST.get((key, item))
    if burst is not None:
        return burst
    by_item = _forme_by_base_and_item().get((key, item))
    if by_item is not None:
        return by_item
    # Holding a Z-crystal locks Rayquaza out of Mega Evolution.
    if crystal_type(item) is not None:
        return None
    moves = _forme_by_base_and_move()
    return next((forme for move in known_moves if (forme := moves.get((key, normalize_id(move)))) is not None), None)


@cache
def mega_stones() -> frozenset[Item]:
    """Every item that reaches a Mega Evolution or Primal Reversion forme."""
    return frozenset(item for _, item in _forme_by_base_and_item())


def is_primal(forme: str) -> bool:
    """Whether reaching this forme is Primal Reversion rather than Mega Evolution."""
    return forme.endswith("-Primal")


@cache
def primal_orbs() -> frozenset[Item]:
    """The subset of `mega_stones` that trigger Primal Reversion — the Red and Blue Orbs."""
    return frozenset(item for (_, item), forme in _forme_by_base_and_item().items() if is_primal(forme))


def forme_item_kind(item: Item) -> str | None:
    """ "Primal orb" or "Mega Stone" for an item that transforms its holder, else None."""
    if item in primal_orbs():
        return "Primal orb"
    if item in mega_stones():
        return "Mega Stone"
    return None


def apply_forme(pokemon: Pokemon, forme: str) -> None:
    """Swap a live Pokemon onto another forme's species data, in place."""
    species = get_species(forme)
    fraction = pokemon.live_stats.HP / pokemon.stat_totals.HP
    pokemon.name = species.name
    pokemon.base_stats = species.base_stats
    pokemon.types = species.types
    pokemon.ability = ability_from_showdown(species.regular_abilities[0])
    pokemon.weight_kg = species.weight_kg
    pokemon.refresh_stats()
    pokemon.live_stats.HP = max(1, round(fraction * pokemon.stat_totals.HP)) if fraction > 0 else 0


# Arceus, whose plates the dex does not record as forme requirements, and the two names one item goes by.
_PLATE_HOLDER = "arceus"
_GRISEOUS = frozenset({Item.GRISEOUS_CORE, Item.GRISEOUS_ORB})


@cache
def _items_fused_to() -> dict[str, frozenset[Item]]:
    """Base species key -> the items that are part of what it *is* rather than what it holds."""
    fused: dict[str, set[Item]] = {}
    for species in get_all_species().values():
        if species.required_item is None or species.base_species is None:
            continue
        item = item_from_showdown(species.required_item)
        if item is not None:
            fused.setdefault(normalize_id(species.base_species), set()).add(item)
    return {key: frozenset(items) for key, items in fused.items()}


def is_fused_to(species: str, item: Item) -> bool:
    """Whether this item is welded to this Pokemon and cannot be taken off it."""
    if crystal_type(item) is not None or normalize_id(item_showdown_name(item)) in get_all_z_moves():
        return True
    # Keyed by base species, so a Mega Evolved Pokemon's stone is still welded on.
    entry = get_all_species().get(normalize_id(species))
    key = normalize_id(entry.base_species or species) if entry is not None else normalize_id(species)
    if key == _PLATE_HOLDER and item.name.endswith("_PLATE"):
        return True
    fused = _items_fused_to().get(key, frozenset())
    return item in fused or (item in _GRISEOUS and bool(fused & _GRISEOUS))
