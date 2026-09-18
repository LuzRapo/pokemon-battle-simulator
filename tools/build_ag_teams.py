"""Reconstruct the most-played Gen 7 Anything Goes teams as a Showdown teams file.

Two sources, because neither is enough alone. The replays say which six Pokemon actually go
*together* — 4,979 high-ladder games, and the same compositions recur, so these are real teams
people laddered with rather than something assembled from a usage list. What the replays cannot say
is how each one was built: a log reveals an item only when something knocks it off, and never
reveals an EV spread or a nature at all. So the sets come from Smogon's moveset statistics for the
same format (283,096 battles), which carry the most common ability, item, spread and moves.

That makes each team a faithful *composition* with a statistically typical build, not a copy of one
player's exact team. Good enough to answer "which of these does our AI pilot best", which is the
question this exists to serve.

    uv run python -m tools.build_ag_teams --teams 30 --out ag_teams.txt
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

REPLAYS = Path("data/ag_replays")
# Where a Pokemon is brought as one thing and fights as another, the statistics live under the
# fighting name — Groudon-Primal carries the Red Orb that makes it Primal in the first place.
STATS_ALIAS = {
    "Groudon": "Groudon-Primal",
    "Kyogre": "Kyogre-Primal",
    "Rayquaza": "Rayquaza-Mega",
    "Gengar": "Gengar-Mega",
    "Mewtwo": "Mewtwo-Mega-Y",
    "Sableye": "Sableye-Mega",
    "Tyranitar": "Tyranitar-Mega",
    "Salamence": "Salamence-Mega",
    "Metagross": "Metagross-Mega",
    "Lucario": "Lucario-Mega",
    "Kangaskhan": "Kangaskhan-Mega",
}
LEVEL = 50  # every trainer in this bot battles at 50; what matters is that it is the same for all


def parse_moveset_stats(path: Path) -> dict[str, dict[str, list[str]]]:
    """species -> {abilities, items, spreads, moves}, each ordered by usage."""
    entries: dict[str, dict[str, list[str]]] = {}
    sections = {"Abilities", "Items", "Spreads", "Moves", "Teammates", "Checks and Counters"}
    current: str | None = None
    section: str | None = None
    for line in path.read_text().splitlines():
        # A data row carries a percentage. Tested first, because a species name matches the plain
        # "| text |" shape just as a row like "| Other  4.446% |" does — checking names first made
        # every percentage into its own species and left the real ones empty.
        row = re.match(r"^ \| (.+?) +([\d.]+)% *\| *$", line)
        if row:
            if current is not None and section in {"abilities", "items", "spreads", "moves"}:
                value = row.group(1).strip()
                if value not in {"Other", "Nothing"}:
                    entries[current][section].append(value)
            continue
        plain = re.match(r"^ \| (.+?) +\| *$", line)
        if not plain:
            continue
        label = plain.group(1).strip()
        if label in sections:
            section = label.lower()
        elif not label.startswith(("Raw count", "Avg. weight", "Viability Ceiling")):
            current, section = label, None
            entries.setdefault(current, {"abilities": [], "items": [], "spreads": [], "moves": []})
    return {k: v for k, v in entries.items() if any(v.values())}


def top_teams(count: int, exclude: frozenset[str] = frozenset()) -> list[tuple[tuple[str, ...], int]]:
    """The most-played compositions, skipping any that field an excluded species.

    Excluding is done before the count is taken, so asking for twenty gets twenty usable teams
    rather than twenty minus whatever was filtered out.
    """
    sides = json.loads((REPLAYS / "sides.json").read_text())
    compositions = Counter(tuple(sorted(s["team"])) for s in sides if len(s["team"]) == 6)
    keep = [(comp, n) for comp, n in compositions.most_common() if not exclude.intersection(comp)]
    return keep[:count]


def _first_supported(candidates: list[str], recognised) -> tuple[str | None, str | None]:
    """The most-used value this engine actually knows, and what it displaced.

    Some real builds lean on things not modelled here — Smeargle's Moody, Shed Shell — and a set
    naming one is dropped whole by `load_teams`, which would silently delete the team from the
    tournament rather than fail loudly. Falling through to the next most-used option keeps the team
    in, and returning the displaced name keeps that honest rather than invisible.
    """
    for index, candidate in enumerate(candidates):
        if recognised(candidate):
            return candidate, (candidates[0] if index else None)
    return None, (candidates[0] if candidates else None)


def build_set(species: str, stats: dict[str, dict[str, list[str]]], swaps: list[str]) -> str | None:
    """One Showdown block for a species, from its most common real build."""
    from battle_sim.teams import ability_from_showdown, item_from_showdown
    from battle_sim.utils import Ability

    entry = stats.get(STATS_ALIAS.get(species, species)) or stats.get(species)
    if entry is None or not entry["moves"]:
        return None
    item, dropped_item = _first_supported(entry["items"], lambda name: item_from_showdown(name) is not None)
    ability, dropped_ability = _first_supported(
        entry["abilities"], lambda name: ability_from_showdown(name) is not Ability.NONE
    )
    for displaced, replacement, kind in ((dropped_item, item, "item"), (dropped_ability, ability, "ability")):
        if displaced:
            swaps.append(f"{species}: {kind} {displaced} -> {replacement or 'none'}")
    nature, evs = "Serious", ""
    if entry["spreads"]:
        nature, _, spread = entry["spreads"][0].partition(":")
        order = ("HP", "Atk", "Def", "SpA", "SpD", "Spe")
        evs = " / ".join(f"{v} {n}" for n, v in zip(order, spread.split("/"), strict=False) if v.isdigit() and int(v))
    lines = [f"{species} @ {item}" if item else species]
    if ability:
        lines.append(f"Ability: {ability}")
    lines.append(f"Level: {LEVEL}")
    if evs:
        lines.append(f"EVs: {evs}")
    lines.append(f"{nature} Nature")
    lines += [f"- {move}" for move in entry["moves"][:4]]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teams", type=int, default=30)
    parser.add_argument("--out", type=Path, default=Path("ag_teams.txt"))
    parser.add_argument("--exclude", nargs="*", default=[], help="species whose teams to skip entirely")
    args = parser.parse_args()

    stats = parse_moveset_stats(REPLAYS / "moveset_stats.txt")
    print(f"{len(stats)} species in the moveset statistics")
    chunks, skipped, swaps = [], [], []
    excluded = frozenset(args.exclude)
    if excluded:
        print(f"skipping any team fielding: {', '.join(sorted(excluded))}")
    for rank, (composition, seen) in enumerate(top_teams(args.teams, excluded), start=1):
        blocks = [build_set(species, stats, swaps) for species in composition]
        if any(block is None for block in blocks):
            skipped.append((composition, [s for s, b in zip(composition, blocks, strict=True) if b is None]))
            continue
        # `load_trainers` matches "=== Name (Version) ===", so the parenthetical has to come last.
        headline = "/".join(species.replace(" ", "-") for species in composition[:3])
        name = f"AG{rank:02d} {headline} ({seen} games)"
        chunks.append(f"=== {name} ===\n\n" + "\n\n".join(b for b in blocks if b))
    args.out.write_text("\n\n".join(chunks) + "\n")
    print(f"wrote {len(chunks)} teams to {args.out}")
    for composition, missing in skipped:
        print(f"  skipped (no stats for {', '.join(missing)}): {', '.join(composition)}")
    if swaps:
        print(f"\n{len(set(swaps))} substitutions, where the most-used build named something unmodelled:")
        for swap in sorted(set(swaps)):
            print(f"  {swap}")


if __name__ == "__main__":
    main()
