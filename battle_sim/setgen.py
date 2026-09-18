"""Random legal `PokemonSpec` generation for any Gen-7-in-scope species.

Species with a vendored Gen-7 Random Battle role (`database/_vendor/randbats_gen7.json`, 524 of
the 950 in-scope species) draw ability/item/moves from that role's real archetype; the rest fall
back to a bare-legality draw from the species' own ability pool and `scope.gen7_movepool`, with
the item inferred from the species' base stat spread and movepool (see `_infer_item`).

EVs, IVs, and nature are always independently randomized within their legal bounds regardless of
source, and level defaults to 50 — Sam's design call (2026-08-22): in the Discord bot, a trainer
is "caught" by naming a silhouetted species, so no set should be a fixed, memorizable spread.
"""

import json
import random
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


def _from_role(key: str, entry: RawRandbatsSpecies, rng: random.Random) -> tuple[Ability, Item, list[str]]:
    role: RawRandbatsRole = rng.choice(list(entry.roles.values()))
    ability = _pick_ability(role.abilities, rng)
    if ability is Ability.NONE:
        # This role's own pick (e.g. Zygarde's Power Construct, a forme-change mechanic we don't
        # model) doesn't map to anything wired — try the species' other real abilities before
        # giving up on it entirely (e.g. Mewtwo's role offers only Unnerve, but Pressure works).
        species = get_species(key)
        real_names = [n for n in (*species.regular_abilities, species.hidden_ability) if n is not None]
        ability = _pick_ability(real_names, rng)
    item = _pick_item(role.items or entry.items, rng)
    # randbats is a snapshot in its own right: a move it lists can be tagged unobtainable in
    # our (later, current-gen) vendored moves.json, e.g. Rayquaza's signature V-create.
    known_moves = get_all_moves()
    pool = _with_effects([name for name in role.moves if normalize_id(name) in known_moves])
    chosen = rng.sample(pool, min(_MAX_MOVES, len(pool)))
    return ability, item, [get_move(name).name for name in chosen]


def _with_effects(pool: list[str]) -> list[str]:
    """The moves in `pool` that actually do something when used.

    Some Gen-7 moves have no mechanics in this engine — the doubles-only ones (Follow Me, Wide Guard,
    ally targets) plus a tail of exotica like Mimic and Psych Up — and they load with no effects at
    all, so using one spends the turn on nothing. A set built around three real moves and one of
    those is a Pokemon fighting with three, which is a rating artefact rather than a fact about the
    species. The whole pool falling away is the one case worth keeping: better an odd set than none.
    """
    known = get_all_moves()

    def does_something(name: str) -> bool:
        move = known[normalize_id(name)]
        return bool(move.effects) or move.force_switch or move.self_switch or move.recharges

    usable = [name for name in pool if does_something(name)]
    return usable or pool


def legal_abilities(species: str) -> tuple[Ability, ...]:
    """Every ability this species can have, as the engine models them, ordinary ones first.

    Regular before hidden and in Showdown's own order, so a caller that just wants "an ability this
    species could plausibly have" can take the first and get the common one rather than the rare
    one. Anything the engine does not implement is left out, so this can come back empty — which
    means the same thing `Ability.NONE` does, and callers should treat it that way.
    """
    entry = get_species(normalize_id(species))
    names = [name for name in (*entry.regular_abilities, entry.hidden_ability) if name is not None]
    mapped = [_ABILITY_BY_SHOWDOWN_NAME[normalize_id(n)] for n in names if normalize_id(n) in _ABILITY_BY_SHOWDOWN_NAME]
    return tuple(dict.fromkeys(mapped))


def _from_heuristic(key: str, rng: random.Random) -> tuple[Ability, Item, list[str]]:
    """No randbats role exists for this species: any of its real abilities, moves biased toward
    actually being able to fight (see `_pick_heuristic_moves`)."""
    species = get_species(key)
    ability_names = [name for name in (*species.regular_abilities, species.hidden_ability) if name is not None]
    ability = _pick_ability(ability_names, rng)
    pool = _with_effects(sorted(gen7_movepool(key) & get_all_moves().keys()))
    chosen = _pick_heuristic_moves(species, pool, rng)
    item = _infer_item(species, pool, rng)
    return ability, item, [get_move(name).name for name in chosen]


def _pick_heuristic_moves(species: BaseSpecies, pool: list[str], rng: random.Random) -> list[str]:
    """A uniform draw over the whole movepool can hand a species four status moves and nothing to
    hit back with. Guarantee `_GUARANTEED_ATTACKS` attacks off whichever of Attack/Sp. Atk is
    higher, preferring the species' own type (STAB) — then fill the rest of the set at random as
    before, so there's no other bias toward what a "good" set looks like.
    """
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
    """No usage data exists for this species: read a role off its base stats and movepool instead.

    Not-fully-evolved species get Eviolite outright — it's close to a free +stat boost for them
    and there's no meaningful alternative to weigh it against. A Genesect Drive forme gets its
    matching Drive outright too — it's the only item that makes its signature move Techno Blast
    anything but Normal-type (see engine.moves._move_type_override), so there's no real
    alternative to weigh there either. Fully evolved species get a random item from whichever of
    {bulky wall, fast offense, slow offense/wallbreaker} pool matches their stat spread; ties
    between a physical and special leaning are broken by which the movepool actually leans
    toward, since a mon's base Attack/Sp. Atk split doesn't always match its kit.
    """
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
