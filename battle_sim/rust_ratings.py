"""`species_rating` on the Rust engine: rate every species by what it adds to a team that wins."""

import argparse
import json
import random
import signal
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import FrameType
from typing import Any

from loguru import logger

from battle_sim.database.loader import get_move, get_species, normalize_id
from battle_sim.database.scope import in_scope_species
from battle_sim.differential import encode_spec
from battle_sim.models.spec import PokemonSpec
from battle_sim.rust_bridge import database, load
from battle_sim.setgen import learnpool_set
from battle_sim.species_rating import Pairing, completed_pairings, deal_pairings
from battle_sim.utils import Item

LOG_VERSION = 1
PILOT = "rust-matchup"
DEFAULT_OUT = Path("ratings/rust-battles.jsonl")
DEFAULT_GENOME = Path(__file__).resolve().parent.parent / "champions" / "random-a-myopic-tuned.json"
DEFAULT_BATCH = 512  # pairings per round: enough to keep every core busy, few enough to stop promptly
MAX_TURNS = 1000  # `run_battle`'s own cap

# Future Sight and Doom Desire can land after their user switches out, which Rust refuses.
UNDEALT_MOVES = frozenset({"Future Sight", "Doom Desire"})

_stopping = False


@dataclass(frozen=True)
class Route:
    """How a transformed forme is reached: this species, holding this item or knowing this move."""

    source: str
    item: Item | None = None
    move: str | None = None


@cache
def transformations() -> dict[str, tuple[Route, ...]]:
    """Every Mega Evolution, Primal Reversion and Ultra Burst forme, with the ways to reach it."""
    from battle_sim.formes import _ULTRA_BURST, _forme_by_base_and_item, _forme_by_base_and_move

    routes: dict[str, list[Route]] = {}
    for (key, item), forme in (*_forme_by_base_and_item().items(), *_ULTRA_BURST.items()):
        routes.setdefault(forme, []).append(Route(get_species(key).name, item=item))
    for (key, move), forme in _forme_by_base_and_move().items():
        routes.setdefault(forme, []).append(Route(get_species(key).name, move=get_move(move).name))
    return {forme: tuple(found) for forme, found in sorted(routes.items())}


def kind(name: str) -> str | None:
    """Which once-per-battle allowance this entry spends, if it transforms at all."""
    if name not in transformations():
        return None
    if name.endswith("-Primal"):
        return "primal"
    return "ultra" if name == "Necrozma-Ultra" else "mega"


def _base(name: str) -> str:
    source = transformations()[name][0].source if name in transformations() else name
    entry = get_species(normalize_id(source))
    return entry.base_species or entry.name


def clash(team: Sequence[str], name: str) -> bool:
    """A team spends each transformation allowance once and never fields a forme beside its own species."""
    allowance = kind(name)
    if allowance is not None and any(kind(other) == allowance for other in team):
        return True
    transforms = [other for other in team if kind(other) is not None]
    if allowance is not None:
        return any(_base(other) == _base(name) for other in team)
    return any(_base(other) == _base(name) for other in transforms)


def deal(pool: Sequence[str], count: int, rng: random.Random, start: int = 0) -> Iterator[Pairing]:
    return deal_pairings(pool, count, rng, start, clash=clash)


def _transforming_moves(species: str) -> frozenset[str]:
    """Moves that would turn this species into a forme it is not being rated as (Dragon Ascent)."""
    return frozenset(
        route.move
        for routes in transformations().values()
        for route in routes
        if route.move is not None and route.source == species
    )


def playable_set(name: str, rng: random.Random) -> PokemonSpec:
    """A learnpool set holding only what the Rust engine plays, plus whatever transforms a forme."""
    db = database()
    routes = transformations().get(name)
    route = rng.choice(routes) if routes else None
    species = route.source if route is not None else name
    kept_out = UNDEALT_MOVES | ({route.move} if route is not None and route.move else _transforming_moves(name))

    def move_ok(move: str) -> bool:
        return move not in kept_out and db.move_playable(move)

    spec = learnpool_set(species, rng, move_ok, lambda ability: db.ability_playable(ability.name))
    if route is None:
        return spec
    moves = list(spec.moves)
    if route.move is not None and route.move not in moves:
        moves[rng.randrange(len(moves))] = route.move
    return spec.model_copy(update={"nickname": name, "item": route.item or spec.item, "moves": moves})


def _plays(name: str) -> str | None:
    """Why the Rust engine cannot play this entry, if it cannot: a short battle is the test."""
    try:
        spec = playable_set(name, random.Random(0))
    except (KeyError, ValueError) as error:
        return str(error)
    foe = playable_set("Snorlax", random.Random(0))
    team, other = json.dumps([encode_spec(spec)]), json.dumps([encode_spec(foe)])
    refusal: str | None = load().play_matchup_battles(database(), [(team, other, 0)], None, 20, 1)[0][4]
    return refusal


def species_pool() -> tuple[list[str], dict[str, str]]:
    """Every rateable entry by display name, and every one left out with the reason."""
    kept: list[str] = []
    dropped: dict[str, str] = {}
    for key in sorted(in_scope_species()):
        entry = get_species(key)
        if entry.name in transformations():
            continue  # rated below, as a transformed forme
        if "-Totem" in entry.name:
            dropped[entry.name] = "a Totem forme: the same Pokemon as its ordinary forme"
            continue
        if entry.required_item is not None:
            dropped[entry.name] = f"needs {entry.required_item} to exist, and nobody holds items"
            continue
        try:
            playable_set(entry.name, random.Random(0))
        except ValueError as error:
            dropped[entry.name] = str(error)
            continue
        kept.append(entry.name)
    for forme in transformations():
        why = _plays(forme)
        if why is None:
            kept.append(forme)
        else:
            dropped[forme] = why
    return sorted(kept), dropped


def build_teams(pairing: Pairing) -> tuple[list[PokemonSpec], list[PokemonSpec]]:
    rng = random.Random(pairing.seed)
    return [playable_set(name, rng) for name in pairing.team_a], [playable_set(name, rng) for name in pairing.team_b]


def play_batch(pairings: Sequence[Pairing], genome: str | None, threads: int) -> list[dict[str, Any]]:
    """Both battles of every pairing, as log records, in pairing order."""
    jobs: list[tuple[str, str, int]] = []
    sets: list[list[list[tuple[str, list[str]]]]] = []
    for pairing in pairings:
        team_a, team_b = build_teams(pairing)
        # What each Pokemon actually carried, so a species' ratings can be broken down by set later.
        sets.append([[(spec.ability.name, list(spec.moves)) for spec in team] for team in (team_a, team_b)])
        a = json.dumps([encode_spec(spec) for spec in team_a])
        b = json.dumps([encode_spec(spec) for spec in team_b])
        jobs += [(a, b, pairing.seed), (b, a, pairing.seed)]
    results = load().play_matchup_battles(database(), jobs, genome, MAX_TURNS, threads)
    records: list[dict[str, Any]] = []
    for number, (outcome, turns, first, second, error) in enumerate(results):
        pairing, side = pairings[number // 2], number % 2
        record: dict[str, Any] = {
            "v": LOG_VERSION,
            "index": pairing.index,
            "seed": pairing.seed,
            "side": side,  # which team led; the pair covers both, so side bias cancels
            "a": list(pairing.team_a),
            "b": list(pairing.team_b),
            "sets": sets[number // 2],
        }
        if error is not None:
            records.append(record | {"error": error[:200]})
            continue
        # Always from team A's point of view, whichever side it played, as `species_rating` writes it.
        own, theirs = (first, second) if side == 0 else (second, first)
        records.append(
            record | {"margin": round(0.5 + (own - theirs) / 12, 6), "turns": turns, "outcome": outcome, "pilot": PILOT}
        )
    return records


def _manifest(path: Path, seed: int, genome: Path, pool: Sequence[str], dropped: dict[str, str]) -> None:
    payload = {
        "v": LOG_VERSION,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pilot": PILOT,
        "genome": genome.name,
        "seed": seed,
        "sets": "learnpool, no items, no ability where none is playable (setgen.learnpool_set)",
        "moves_never_dealt": sorted(UNDEALT_MOVES),
        "species": len(pool),
        "left_out": dropped,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _stop(signum: int, frame: FrameType | None) -> None:
    global _stopping
    _stopping = True
    logger.info("stop requested — finishing the batch in hand, then writing out")


def run(out: Path, hours: float, seed: int, genome: Path, threads: int, batch: int, limit: int | None = None) -> int:
    """Play batches of pairings until the clock (or `limit` pairings) runs out."""
    pool, dropped = species_pool()
    logger.info(f"{len(pool)} species in the pool, {len(dropped)} left out (see the manifest)")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = completed_pairings(out)
    if done:
        logger.info(f"resuming {out}: {done} pairings already logged")
    _manifest(out.with_suffix(".manifest.json"), seed, genome, pool, dropped)
    genes = genome.read_text()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    total = limit if limit is not None else 10**12
    pairings = deal(pool, total, random.Random(seed), start=done)
    deadline = time.monotonic() + hours * 3600
    started = time.monotonic()
    written = failed = 0
    with out.open("a") as handle:
        while not _stopping and time.monotonic() < deadline:
            chunk = [pairing for _, pairing in zip(range(batch), pairings, strict=False)]
            if not chunk:
                break
            records = play_batch(chunk, genes, threads)
            handle.write("".join(json.dumps(record) + "\n" for record in records))
            handle.flush()
            written += len(records)
            failed += sum("error" in record for record in records)
            elapsed = time.monotonic() - started
            rate = written / elapsed
            logger.info(f"{written} battles ({rate:.0f}/s, {failed} failed), through pairing {chunk[-1].index}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--hours", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--genome", type=Path, default=DEFAULT_GENOME, help="the MatchupPlayer weights both sides use")
    parser.add_argument("--threads", type=int, default=3, help="cores to play on (this server keeps one for the bot)")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--limit", type=int, default=None, help="stop after this many pairings")
    args = parser.parse_args()
    written = run(args.out, args.hours, args.seed, args.genome, args.threads, args.batch, args.limit)
    logger.info(f"done: {written} battles appended to {args.out}")


if __name__ == "__main__":
    main()
