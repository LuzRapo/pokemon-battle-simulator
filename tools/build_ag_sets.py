"""Mine a couple of real sets per Gen 7 Anything Goes species from the downloaded replay logs."""

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from battle_sim.database.loader import get_all_z_moves, get_move, get_species, normalize_id
from battle_sim.differential import encode_spec
from battle_sim.models.moves import DamageEffect, MoveEffect
from battle_sim.models.spec import PokemonSpec
from battle_sim.models.stats import EVs
from battle_sim.teams import ability_from_showdown, build_pokemon, item_from_showdown
from battle_sim.utils import Ability, Category, Item, Nature
from tools.build_ag_teams import LEVEL, REPLAYS, STATS_ALIAS, parse_moveset_stats

LOGS = REPLAYS / "logs"
MAX_SETS = 2
MIN_APPEARANCES = 30  # below this there is too little to mine a set from, and the species is fringe
MIN_SET_SUPPORT = 5  # full reveals behind a set before it counts as real
SECOND_SET_SHARE = 0.2  # a second set needs this fraction of the first's support

# Formes a Pokemon changes into mid-battle; a log shows the new name on any later switch-in.
_IN_BATTLE_FORME = re.compile(r"-(Mega(-[XY])?|Primal|Complete|Ultra|Blade|Busted|School|Zen|Pirouette|Ash)$")
_FORCED_ITEMS: dict[str, Item] = {
    "Groudon": Item.RED_ORB,
    "Kyogre": Item.BLUE_ORB,
    "Giratina-Origin": Item.GRISEOUS_ORB,
}
_PLATE_BY_TYPE = {
    "Fighting": "Fist Plate",
    "Flying": "Sky Plate",
    "Poison": "Toxic Plate",
    "Ground": "Earth Plate",
    "Rock": "Stone Plate",
    "Bug": "Insect Plate",
    "Ghost": "Spooky Plate",
    "Steel": "Iron Plate",
    "Fire": "Flame Plate",
    "Water": "Splash Plate",
    "Grass": "Meadow Plate",
    "Electric": "Zap Plate",
    "Psychic": "Mind Plate",
    "Ice": "Icicle Plate",
    "Dragon": "Draco Plate",
    "Dark": "Dread Plate",
    "Fairy": "Pixie Plate",
}
_DRIVE_BY_FORME = {"Burn": "Burn Drive", "Chill": "Chill Drive", "Douse": "Douse Drive", "Shock": "Shock Drive"}
# An item a Pokemon gains mid-battle is not the one it brought.
_GAINED = ("Trick", "Switcheroo", "Thief", "Covet", "Magician", "Pickpocket", "Bestow", "Symbiosis")
# Species a log cannot describe.
_FIXED_SETS: dict[str, PokemonSpec] = {
    "Ditto": PokemonSpec(
        species="Ditto",
        level=LEVEL,
        ability=Ability.IMPOSTER,
        item=Item.CHOICE_SCARF,
        nature=Nature.RELAXED,
        effort_values=EVs(HP=248, DEFENCE=252, SP_DEFENCE=8),
        moves=["Transform"],
    ),
}


@dataclass
class Appearance:
    species: str
    moves: set[str] = field(default_factory=set)
    items: set[str] = field(default_factory=set)
    abilities: set[str] = field(default_factory=set)
    z_moves: set[str] = field(default_factory=set)
    primal: bool = False


def _power(effect: MoveEffect) -> int | None:
    return effect.power if isinstance(effect, DamageEffect) else None


def _who(token: str) -> tuple[str, str]:
    side, nick = token.split(": ", 1)
    return side[:2], nick


def _base(species: str) -> str:
    return _IN_BATTLE_FORME.sub("", species)


def read_log(text: str) -> list[Appearance]:  # noqa: C901 — a flat dispatch over every log line kind
    """Every Pokemon that switched in, with whatever the game revealed about it."""
    active: dict[tuple[str, str], Appearance] = {}
    seen: dict[tuple[str, str, str], Appearance] = {}
    z_pending: set[tuple[str, str]] = set()
    for line in text.splitlines():
        parts = line.split("|")
        if len(parts) < 3 or ": " not in parts[2]:
            continue
        kind, who = parts[1], _who(parts[2])
        if kind in ("switch", "drag") and len(parts) > 3:
            species = parts[3].split(",")[0]
            base = _base(species)
            if base == "Necrozma":  # Ultra Burst names neither Necrozma it came from
                base = next((k[2] for k in seen if k[:2] == who and k[2].startswith("Necrozma-")), species)
            active[who] = seen.setdefault((*who, base), Appearance(base))
            continue
        pokemon = active.get(who)
        if pokemon is None:
            continue
        tags = parts[4:]
        if kind == "move" and len(parts) > 3:
            if who in z_pending:
                z_pending.discard(who)
                pokemon.z_moves.add(parts[3])
            elif parts[3] != "Struggle" and not any(t.startswith("[from]") and "lockedmove" not in t for t in tags):
                pokemon.moves.add(parts[3])
        elif kind == "-zpower":
            z_pending.add(who)
        elif kind == "-mega" and len(parts) > 4 and parts[4]:
            pokemon.items.add(parts[4])
        elif kind == "-primal":
            pokemon.primal = True
        elif kind in ("-enditem", "-item") and len(parts) > 3:
            if not any(g in t for t in tags for g in _GAINED):
                pokemon.items.add(parts[3])
        elif kind == "-ability" and len(parts) > 3:
            if not any(t.startswith("[from]") for t in tags):
                pokemon.abilities.add(parts[3])
        elif kind == "-activate" and len(parts) > 3:
            # An ability or item announcing itself on an "|-activate|" line.
            found = re.match(r"(ability|item): (.+)", parts[3])
            if found:
                (pokemon.items if found.group(1) == "item" else pokemon.abilities).add(found.group(2))
        elif kind == "detailschange" and len(parts) > 3 and parts[3].split(",")[0].endswith("-Complete"):
            pokemon.abilities.add("Power Construct")
        # "[from] item:" or "[from] ability:" credits whoever "[of]" names, else the line's subject.
        for tag in parts[3:]:
            found = re.match(r"\[from\] (item|ability): (.+)", tag)
            if not found:
                continue
            of = next((t[len("[of] ") :] for t in parts[3:] if t.startswith("[of] ")), None)
            owner = active.get(_who(of)) if of and ": " in of else pokemon
            if owner is not None:
                (owner.items if found.group(1) == "item" else owner.abilities).add(found.group(2))
    return list(seen.values())


def _crystal_for(z_move: str) -> str | None:
    """The Z-Crystal a Z-move name implies: its own for a signature move, its type's for the rest."""
    for crystal, move in get_all_z_moves().items():
        if move.name == z_move:
            return crystal
    if z_move.startswith("Z-"):  # a status move's Z-effect names the base move instead
        try:
            base_type = get_move(z_move[2:]).type
        except KeyError:
            return None
        for crystal, move in get_all_z_moves().items():
            if move.type == base_type and move.effects and _power(move.effects[0]) == 1:
                return crystal
    return None


def _known_move(name: str) -> bool:
    try:
        get_move(name)
    except KeyError:
        return False
    return True


def _hidden_power(stats_moves: list[str]) -> str | None:
    return next((m for m in stats_moves if m.startswith("Hidden Power ") and _known_move(m)), None)


def _resolve_moves(moves: set[str], stats_moves: list[str]) -> set[str] | None:
    """A log shows Hidden Power untyped; name its type from the statistics, or give up on the set."""
    resolved = set()
    for move in moves:
        if move == "Hidden Power":
            typed = _hidden_power(stats_moves)
            if typed is None:
                return None
            move = typed
        if not _known_move(move):
            return None
        resolved.add(move)
    return resolved


def mine_move_sets(appearances: list[Appearance], stats_moves: list[str]) -> list[tuple[set[str], int]]:
    """Up to `MAX_SETS` four-move sets, each with the number of full reveals behind it."""
    full = Counter(frozenset(a.moves) for a in appearances if len(a.moves) == 4)
    groups: list[tuple[set[str], int]] = []
    for combo, count in full.most_common():
        for index, (founder, support) in enumerate(groups):
            if len(founder & combo) >= 3:
                groups[index] = (founder, support + count)
                break
        else:
            groups.append((set(combo), count))
    groups.sort(key=lambda g: -g[1])
    chosen: list[tuple[set[str], int]] = []
    for founder, support in groups:
        if support < MIN_SET_SUPPORT or (chosen and support < SECOND_SET_SHARE * chosen[0][1]):
            break
        resolved = _resolve_moves(founder, stats_moves)
        if resolved is not None and len(resolved) == 4:
            chosen.append((resolved, support))
        if len(chosen) == MAX_SETS:
            break
    if chosen:
        return chosen
    grown = _grow_set(appearances, stats_moves)
    return [(grown, 0)] if grown else []


def _grow_set(appearances: list[Appearance], stats_moves: list[str]) -> set[str] | None:
    """One set built move by move: each next move the one most often seen alongside those chosen."""
    reveals = [_resolve_moves(a.moves, stats_moves) or set() for a in appearances]
    chosen: set[str] = set()
    while len(chosen) < 4:
        with_chosen = Counter(m for r in reveals if chosen <= r for m in r - chosen)
        if not with_chosen:
            with_chosen = Counter(m for m in stats_moves if m not in chosen and _known_move(m))
            with_chosen.update(dict.fromkeys(stats_moves, 0))
        best = next((m for m, _ in with_chosen.most_common() if m not in chosen and _known_move(m)), None)
        if best is None:
            return None
        chosen.add(best)
    return chosen


def _supported_item(name: str) -> Item | None:
    item = item_from_showdown(name)
    return item if item is not None and item is not Item.NONE else None


def choose_item(species: str, moves: set[str], appearances: list[Appearance], stats_items: list[str]) -> Item:
    if species in _FORCED_ITEMS:
        return _FORCED_ITEMS[species]
    if species.startswith("Arceus-"):
        return item_from_showdown(_PLATE_BY_TYPE[species.split("-", 1)[1]]) or Item.NONE
    if species.startswith("Silvally-"):
        return item_from_showdown(f"{species.split('-', 1)[1]} Memory") or Item.NONE
    if species.startswith("Genesect-"):
        return item_from_showdown(_DRIVE_BY_FORME[species.split("-", 1)[1]]) or Item.NONE
    consistent = [a for a in appearances if len(a.moves) >= 2 and a.moves <= moves | {"Hidden Power"}]
    seen: Counter[str] = Counter()
    for a in consistent:
        seen.update(a.items)
        seen.update(c for c in (_crystal_for(z) for z in a.z_moves) if c)
    for name, count in seen.most_common():
        item = _supported_item(name)
        if count >= 3 and item is not None:
            return item
    for name in stats_items:
        item = _supported_item(name)
        if item is not None:
            return item
    return Item.NONE


def choose_ability(
    species: str, appearances: list[Appearance], stats_abilities: list[str], own_stats_abilities: list[str]
) -> Ability:
    """`own_stats_abilities` are the base form's own statistics, which count as legal."""
    entry = get_species(normalize_id(species))
    legal = {normalize_id(a) for a in [*entry.regular_abilities, entry.hidden_ability, *own_stats_abilities] if a}
    seen = Counter(a for p in appearances for a in p.abilities if normalize_id(a) in legal)
    for name, _ in seen.most_common():
        if ability_from_showdown(name) is not Ability.NONE:
            return ability_from_showdown(name)
    for name in [*stats_abilities, *[a for a in [*entry.regular_abilities, entry.hidden_ability] if a]]:
        if normalize_id(name) in legal and ability_from_showdown(name) is not Ability.NONE:
            return ability_from_showdown(name)
    return Ability.NONE


def choose_spread(moves: set[str], spreads: list[str]) -> tuple[Nature, EVs]:
    physical = sum(1 for m in moves if get_move(m).category is Category.PHYSICAL)
    special = sum(1 for m in moves if get_move(m).category is Category.SPECIAL)
    wanted = 1 if physical > special else 3 if special > physical else None

    def parse(raw: str) -> tuple[str, list[int]] | None:
        nature, _, numbers = raw.partition(":")
        values = [int(v) for v in numbers.split("/") if v.isdigit()]
        return (
            (nature, values)
            if len(values) == 6 and nature in Nature.__members__.keys() | {n.capitalize() for n in Nature.__members__}
            else None
        )

    parsed = [p for p in (parse(s) for s in spreads) if p]
    pick = next((p for p in parsed if wanted is None or p[1][wanted] > 0), parsed[0] if parsed else None)
    if pick is None:
        return Nature.SERIOUS, EVs()
    nature, values = pick
    keys = ("HP", "ATTACK", "DEFENCE", "SP_ATTACK", "SP_DEFENCE", "SPEED")
    return Nature[nature.upper()], EVs(**dict(zip(keys, values, strict=True)))


def species_sets(
    species: str, appearances: list[Appearance], stats: dict[str, dict[str, list[str]]], dropped: list[str]
) -> list[dict[str, object]]:
    """Up to `MAX_SETS` playable sets for one species, each with the full reveals behind it."""
    if species in _FIXED_SETS:
        return [{"support": 0, "set": encode_spec(_FIXED_SETS[species])}]
    try:
        get_species(normalize_id(species))
    except KeyError:
        dropped.append(f"{species}: not in the species data")
        return []
    empty: dict[str, list[str]] = {"abilities": [], "items": [], "spreads": [], "moves": []}
    entry = stats.get(STATS_ALIAS.get(species, species)) or stats.get(species) or empty
    # A Mega's statistics carry the Mega's ability; the base form's say what people chose.
    own_abilities = (stats.get(species) or {}).get("abilities", [])
    ability = choose_ability(species, appearances, [*own_abilities, *entry["abilities"]], own_abilities)
    sets: list[dict[str, object]] = []
    for moves, support in mine_move_sets(appearances, entry["moves"]):
        nature, evs = choose_spread(moves, entry["spreads"])
        spec = PokemonSpec(
            species=species,
            level=LEVEL,
            ability=ability,
            item=choose_item(species, moves, appearances, entry["items"]),
            nature=nature,
            effort_values=evs,
            moves=sorted(moves),
        )
        try:
            build_pokemon(spec)
        except (KeyError, ValueError) as why:
            dropped.append(f"{species} {sorted(moves)}: {why}")
            continue
        sets.append({"support": support, "set": encode_spec(spec)})
    return sets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("battle_sim/data/ag_sets.json"))
    args = parser.parse_args()

    stats = parse_moveset_stats(REPLAYS / "moveset_stats.txt")
    by_species: dict[str, list[Appearance]] = defaultdict(list)
    games = sorted(LOGS.glob("*.log"))
    for path in games:
        for appearance in read_log(path.read_text(errors="replace")):
            by_species[appearance.species].append(appearance)

    result: dict[str, list[dict[str, object]]] = {}
    dropped: list[str] = []
    for species, appearances in sorted(by_species.items(), key=lambda kv: -len(kv[1])):
        if len(appearances) < MIN_APPEARANCES:
            continue
        sets = species_sets(species, appearances, stats, dropped)
        if sets:
            result[species] = sets
        else:
            dropped.append(f"{species}: no set could be mined")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    total = sum(len(s) for s in result.values())
    logger.info(f"{len(games)} games -> {len(result)} species, {total} sets written to {args.out}")
    for line in dropped:
        logger.info(f"  dropped {line}")


if __name__ == "__main__":
    main()
