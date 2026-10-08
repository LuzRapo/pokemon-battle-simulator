"""Random legal `PokemonSpec` generation for any Gen-7-in-scope species."""

import json
import random
from collections.abc import Callable
from functools import cache
from pathlib import Path

from battle_sim.database.loader import get_all_moves, get_move, get_species, normalize_id
from battle_sim.database.raw import RawRandbatsRole, RawRandbatsSpecies
from battle_sim.database.scope import gen7_movepool, in_scope_species
from battle_sim.models.spec import PokemonSpec
from battle_sim.models.species import BaseSpecies
from battle_sim.models.stats import EVs, IVs
from battle_sim.teams import _ABILITY_BY_SHOWDOWN_NAME, _ITEM_BY_SHOWDOWN_NAME
from battle_sim.utils import Ability, Category, Item, Nature, Stats

DEFAULT_LEVEL = 50

_EV_TOTAL_CAP = 510
_EV_STAT_CAP = 252
_MAX_MOVES = 4
_EV_STATS = (Stats.HP, Stats.ATTACK, Stats.DEFENCE, Stats.SP_ATTACK, Stats.SP_DEFENCE, Stats.SPEED)
_VENDOR_DIR = Path(__file__).parent / "database" / "_vendor"

_FAST_SPEED_THRESHOLD = 95  # base Speed at/above this reads as a real speed tier, not a wall's leftover stat
_BULKY_DOMINANCE = 1.15  # average defensive stat must clearly exceed the best offensive stat to read as a wall
_BULKY_ITEMS = (Item.LEFTOVERS, Item.SITRUS_BERRY, Item.ROCKY_HELMET, Item.HEAVY_DUTY_BOOTS, Item.ASSAULT_VEST)
_FAST_OFFENSE_ITEMS = (Item.LIFE_ORB, Item.CHOICE_SCARF)
_SLOW_OFFENSE_ITEMS = (Item.LIFE_ORB, Item.EXPERT_BELT)
_GUARANTEED_ATTACKS = 2  # a heuristic set gets at least this many attacks off the species' own better stat


@cache
def _randbats() -> dict[str, RawRandbatsSpecies]:
    raw_data = json.loads((_VENDOR_DIR / "randbats_gen7.json").read_text())
    return {normalize_id(key): RawRandbatsSpecies.model_validate(entry) for key, entry in raw_data.items()}


def random_set(species: str, rng: random.Random, level: int | None = None) -> PokemonSpec:
    """A fresh, randomly-EV'd/IV'd/natured legal set for `species` (a Gen-7-in-scope name)."""
    key = normalize_id(species)
    if key not in in_scope_species():
        raise KeyError(f"{species!r} is not a Gen-7-legal battling species.")
    entry = _randbats().get(key)
    ability, item, moves = _from_role(key, entry, rng) if entry is not None else _from_heuristic(key, rng)
    return PokemonSpec(
        species=get_species(key).name,
        level=DEFAULT_LEVEL if level is None else level,
        ability=ability,
        item=item,
        nature=rng.choice(list(Nature)),
        effort_values=_random_evs(rng),
        individual_values=_random_ivs(rng),
        moves=moves,
    )


def learnpool_set(
    species: str,
    rng: random.Random,
    move_ok: Callable[[str], bool],
    ability_ok: Callable[[Ability], bool],
    level: int | None = None,
) -> PokemonSpec:
    """A set drawn from the species' own Gen 7 learnpool, holding nothing."""
    key = normalize_id(species)
    if key not in in_scope_species():
        raise KeyError(f"{species!r} is not a Gen-7-legal battling species.")
    entry = get_species(key)
    # None of its abilities playable, so it plays with none rather than being left out.
    abilities = [ability for ability in legal_abilities(species) if ability_ok(ability)] or [Ability.NONE]
    pool = [name for name in sorted(gen7_movepool(key) & get_all_moves().keys()) if move_ok(get_move(name).name)]
    if not pool:
        raise ValueError(f"{entry.name} has no playable moves")
    moves = _pick_heuristic_moves(entry, _with_effects(pool), rng)
    return PokemonSpec(
        species=entry.name,
        level=DEFAULT_LEVEL if level is None else level,
        ability=rng.choice(abilities),
        item=Item.NONE,
        nature=rng.choice(list(Nature)),
        effort_values=_random_evs(rng),
        individual_values=_random_ivs(rng),
        moves=[get_move(name).name for name in moves],
    )


def _from_role(key: str, entry: RawRandbatsSpecies, rng: random.Random) -> tuple[Ability, Item, list[str]]:
    role: RawRandbatsRole = rng.choice(list(entry.roles.values()))
    ability = _pick_ability(role.abilities, rng)
    if ability is Ability.NONE:
        # If this role's pick is not wired, try the species' other real abilities before giving up.
        species = get_species(key)
        real_names = [n for n in (*species.regular_abilities, species.hidden_ability) if n is not None]
        ability = _pick_ability(real_names, rng)
    item = _pick_item(role.items or entry.items, rng)
    # randbats can list a move our current-gen vendored data tags unobtainable.
    known_moves = get_all_moves()
    pool = _with_effects([name for name in role.moves if normalize_id(name) in known_moves])
    chosen = rng.sample(pool, min(_MAX_MOVES, len(pool)))
    return ability, item, [get_move(name).name for name in chosen]


def _with_effects(pool: list[str]) -> list[str]:
    """The moves in `pool` that actually do something when used."""
    known = get_all_moves()

    def does_something(name: str) -> bool:
        move = known[normalize_id(name)]
        return bool(move.effects) or move.force_switch or move.self_switch or move.recharges

    usable = [name for name in pool if does_something(name)]
    return usable or pool


def legal_abilities(species: str) -> tuple[Ability, ...]:
    """Every ability this species can have, as the engine models them, ordinary ones first."""
    entry = get_species(normalize_id(species))
    names = [name for name in (*entry.regular_abilities, entry.hidden_ability) if name is not None]
    mapped = [_ABILITY_BY_SHOWDOWN_NAME[normalize_id(n)] for n in names if normalize_id(n) in _ABILITY_BY_SHOWDOWN_NAME]
    return tuple(dict.fromkeys(mapped))


def _from_heuristic(key: str, rng: random.Random) -> tuple[Ability, Item, list[str]]:
    """No randbats role exists for this species, so draw moves biased toward being able to fight."""
    species = get_species(key)
    ability_names = [name for name in (*species.regular_abilities, species.hidden_ability) if name is not None]
    ability = _pick_ability(ability_names, rng)
    pool = _with_effects(sorted(gen7_movepool(key) & get_all_moves().keys()))
    chosen = _pick_heuristic_moves(species, pool, rng)
    item = _infer_item(species, pool, rng)
    return ability, item, [get_move(name).name for name in chosen]


def _pick_heuristic_moves(species: BaseSpecies, pool: list[str], rng: random.Random) -> list[str]:
    """A uniform draw over the whole movepool can hand a species four status moves and nothing to hit back with."""
    category = Category.PHYSICAL if species.base_stats.ATTACK >= species.base_stats.SP_ATTACK else Category.SPECIAL
    types = {t for t in species.types if t is not None}

    def attacks(names: list[str]) -> list[str]:
        return [name for name in names if get_move(name).category is category]

    stab_attacks = attacks([name for name in pool if get_move(name).type in types])
    candidates = (
        stab_attacks or attacks(pool) or [name for name in pool if get_move(name).category is not Category.STATUS]
    )
    guaranteed = rng.sample(candidates, min(_GUARANTEED_ATTACKS, len(candidates)))

    remaining = [name for name in pool if name not in guaranteed]
    filler = rng.sample(remaining, min(_MAX_MOVES - len(guaranteed), len(remaining)))
    chosen = guaranteed + filler
    rng.shuffle(chosen)  # so the guaranteed attacks don't always land in the first move slots
    return chosen


_GENESECT_DRIVES: dict[str, Item] = {
    "Genesect-Douse": Item.DOUSE_DRIVE,
    "Genesect-Shock": Item.SHOCK_DRIVE,
    "Genesect-Burn": Item.BURN_DRIVE,
    "Genesect-Chill": Item.CHILL_DRIVE,
}


def _infer_item(species: BaseSpecies, movepool: list[str], rng: random.Random) -> Item:
    """No usage data exists for this species, so read a role off its base stats and movepool."""
    if species.name in _GENESECT_DRIVES:
        return _GENESECT_DRIVES[species.name]
    if not species.fully_evolved:
        return Item.EVIOLITE
    stats = species.base_stats
    physical_moves = sum(1 for name in movepool if get_move(name).category is Category.PHYSICAL)
    special_moves = sum(1 for name in movepool if get_move(name).category is Category.SPECIAL)
    leans_physical = (stats.ATTACK, physical_moves) >= (stats.SP_ATTACK, special_moves)
    offense = max(stats.ATTACK, stats.SP_ATTACK)
    bulk_average = (stats.HP + stats.DEFENCE + stats.SP_DEFENCE) / 3
    choice_item = Item.CHOICE_BAND if leans_physical else Item.CHOICE_SPECS
    pool: tuple[Item, ...]
    if bulk_average > offense * _BULKY_DOMINANCE:
        pool = _BULKY_ITEMS
    elif stats.SPEED >= _FAST_SPEED_THRESHOLD:
        pool = (choice_item, *_FAST_OFFENSE_ITEMS)
    else:
        pool = (choice_item, *_SLOW_OFFENSE_ITEMS)
    return rng.choice(pool)


def _pick_ability(names: list[str], rng: random.Random) -> Ability:
    mapped = [_ABILITY_BY_SHOWDOWN_NAME[normalize_id(n)] for n in names if normalize_id(n) in _ABILITY_BY_SHOWDOWN_NAME]
    return rng.choice(mapped) if mapped else Ability.NONE


def _pick_item(names: list[str], rng: random.Random) -> Item:
    mapped = [_ITEM_BY_SHOWDOWN_NAME[normalize_id(n)] for n in names if normalize_id(n) in _ITEM_BY_SHOWDOWN_NAME]
    return rng.choice(mapped) if mapped else Item.NONE


def _random_evs(rng: random.Random) -> EVs:
    stats = list(_EV_STATS)
    rng.shuffle(stats)
    remaining = _EV_TOTAL_CAP
    values: dict[str, int] = {}
    for stat in stats:
        value = rng.randint(0, min(_EV_STAT_CAP, remaining))
        values[stat.name] = value
        remaining -= value
    return EVs(**values)


def _random_ivs(rng: random.Random) -> IVs:
    return IVs(**{stat.name: rng.randint(0, 31) for stat in _EV_STATS})
