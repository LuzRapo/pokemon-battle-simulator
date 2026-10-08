"""Rate every species by how much it contributes to a team that wins."""

import argparse
import json
import os
import random
import signal
import subprocess
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import Any

from loguru import logger

from battle_sim.database.loader import get_species
from battle_sim.database.scope import in_scope_species
from battle_sim.evolution import Team, battle_margin, load_weights
from battle_sim.matchup import MatchupWeights
from battle_sim.runner import run_battle
from battle_sim.search import SearchPlayer, SearchProfile
from battle_sim.setgen import random_set

LOG_VERSION = 1
TEAM_SIZE = 6
DEFAULT_BUDGET = 120  # what the ratings are measured at; 30 falls back to the myopic scorer
DEFAULT_WORKERS = 3  # one core left for whatever else the machine is running
DEFAULT_OUT = Path("ratings/battles.jsonl")
_CHAMPION = Path("champions/random-a-myopic-tuned.json")
_stopping = False


@dataclass(frozen=True)
class Pairing:
    """One matchup, played twice on the same seed with the sides swapped."""

    index: int
    seed: int
    team_a: tuple[str, ...]
    team_b: tuple[str, ...]


def _species_pool() -> list[str]:
    """Every rateable species, under its display name."""
    return sorted(get_species(key).name for key in in_scope_species())


def deal_pairings(
    pool: Sequence[str],
    count: int,
    rng: random.Random,
    start: int = 0,
    clash: Callable[[Sequence[str], str], bool] | None = None,
) -> Iterator[Pairing]:
    """`count` pairings, dealing species from a reshuffled bag so appearances stay even."""
    bag: list[str] = []

    def deal(size: int) -> tuple[str, ...]:
        """One team, with no species repeated on it."""
        nonlocal bag
        drawn: list[str] = []
        held: list[str] = []
        while len(drawn) < size:
            if not bag:
                bag = list(pool)
                rng.shuffle(bag)
            name = bag.pop()
            refused = name in drawn or (clash is not None and clash(drawn, name))
            (held if refused else drawn).append(name)
        bag.extend(held)
        return tuple(drawn)

    for index in range(count):
        pairing = Pairing(index=index, seed=rng.randrange(2**32), team_a=deal(TEAM_SIZE), team_b=deal(TEAM_SIZE))
        if index >= start:
            yield pairing


def _build_teams(pairing: Pairing, rng: random.Random) -> tuple[Team, Team]:
    return (
        tuple(random_set(name, rng) for name in pairing.team_a),
        tuple(random_set(name, rng) for name in pairing.team_b),
    )


def play_pairing(job: tuple[Pairing, int]) -> list[dict[str, Any]]:
    """Both battles of one pairing, as log records."""
    pairing, budget = job
    genome = _worker_genome()
    profile = SearchProfile(budget=budget)
    rng = random.Random(pairing.seed)
    records: list[dict[str, Any]] = []
    try:
        team_a, team_b = _build_teams(pairing, rng)
    except Exception as error:  # noqa: BLE001 - one species whose set cannot be built must not end the run
        logger.warning(f"pairing {pairing.index} could not be built: {error}")
        return [_failure(pairing, 0, str(error))]

    for side in (0, 1):
        first, second = (team_a, team_b) if side == 0 else (team_b, team_a)
        try:
            result = run_battle(
                first,
                second,
                SearchPlayer(genome, profile=profile),
                SearchPlayer(genome, profile=profile),
                seed=pairing.seed,
            )
        except Exception as error:  # noqa: BLE001 - one broken battle must not end a run of millions
            logger.warning(f"pairing {pairing.index} side {side} failed: {error}")
            records.append(_failure(pairing, side, str(error)))
            continue
        records.append(
            {
                "v": LOG_VERSION,
                "index": pairing.index,
                "seed": pairing.seed,
                "side": side,  # which team led; the pair covers both, so side bias cancels
                "a": list(pairing.team_a),
                "b": list(pairing.team_b),
                # Always from team A's point of view, so every row reads the same way to the fit.
                "margin": round(battle_margin(result, 0 if side == 0 else 1), 6),
                "turns": result.turns,
                "budget": budget,
            }
        )
    return records


def _failure(pairing: Pairing, side: int, error: str) -> dict[str, Any]:
    return {
        "v": LOG_VERSION,
        "index": pairing.index,
        "seed": pairing.seed,
        "side": side,
        "a": list(pairing.team_a),
        "b": list(pairing.team_b),
        "error": error[:200],
    }


_GENOME: MatchupWeights | None = None


def _worker_genome() -> MatchupWeights:
    """Loaded once per worker process rather than per battle."""
    global _GENOME
    if _GENOME is None:
        _GENOME = load_weights(_CHAMPION)
    return _GENOME


def completed_pairings(path: Path) -> int:
    """How far a previous run got: one past the highest pairing index already written."""
    if not path.exists():
        return 0
    highest = -1
    with path.open() as handle:
        for line in handle:
            try:
                highest = max(highest, int(json.loads(line)["index"]))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue  # truncated or malformed: skip it, the pairing is simply replayed
    return highest + 1


def _manifest(path: Path, budget: int, workers: int, seed: int, pool: Sequence[str]) -> None:
    """Alongside the log, what produced it."""
    payload = {
        "v": LOG_VERSION,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "budget": budget,
        "workers": workers,
        "seed": seed,
        "species": len(pool),
        "champion": _CHAMPION.name,
        "engine_commit": _engine_commit(),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _engine_commit() -> str:
    try:
        out = subprocess.run(  # noqa: S603 — fixed argv, no shell, no user input
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            cwd=Path(__file__).parent.parent,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _stop(signum: int, frame: FrameType | None) -> None:
    global _stopping
    _stopping = True
    logger.info("stop requested — finishing the pairings already in flight, then writing out")


def run(out: Path, hours: float, budget: int, workers: int, seed: int, limit: int | None = None) -> int:
    """Play pairings until the clock runs out, writing each one down as it lands."""
    pool = _species_pool()
    out.parent.mkdir(parents=True, exist_ok=True)
    done = completed_pairings(out)
    total = limit if limit is not None else 10**9
    if done:
        logger.info(f"resuming {out}: {done} pairings already logged")
    _manifest(out.with_suffix(".manifest.json"), budget, workers, seed, pool)

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    deadline = time.monotonic() + hours * 3600
    pairings = deal_pairings(pool, total, random.Random(seed), start=done)
    written = 0
    started = time.monotonic()
    # One writer, in the parent, with work submitted a few pairings at a time.
    with out.open("a") as handle, ProcessPoolExecutor(max_workers=workers) as executor:
        queued: set[Future[list[dict[str, Any]]]] = set()
        remaining = iter(pairings)
        draining = False
        while True:
            while not draining and len(queued) < workers * 2:
                pairing = next(remaining, None)
                if pairing is None:
                    draining = True
                    break
                queued.add(executor.submit(play_pairing, (pairing, budget)))
            if not queued:
                break
            finished, queued = wait(queued, return_when=FIRST_COMPLETED)
            for future in finished:
                records = future.result()
                for record in records:
                    handle.write(json.dumps(record) + "\n")
                handle.flush()
                os.fsync(handle.fileno())  # a night's work should survive the machine losing power
                written += len(records)
                if written % 100 < len(records):
                    _report(written, started, done)
            if _stopping or time.monotonic() > deadline:
                break
    _report(written, started, done)
    return written


def _report(written: int, started: float, resumed_from: int) -> None:
    elapsed = max(1e-6, time.monotonic() - started)
    logger.info(
        f"{written} battles written this session ({written / elapsed * 3600:.0f}/hour), "
        f"resumed from pairing {resumed_from}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--hours", type=float, default=12.0)
    parser.add_argument("--search-budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--limit", type=int, default=None, help="stop after this many pairings (for smoke tests)")
    args = parser.parse_args()
    written = run(args.out, args.hours, args.search_budget, args.workers, args.seed, args.limit)
    logger.info(f"done: {written} battles appended to {args.out}")


if __name__ == "__main__":
    main()
