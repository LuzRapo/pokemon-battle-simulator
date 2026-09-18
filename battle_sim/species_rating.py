"""Rate every species by how much it contributes to a team that wins.

`uv run python -m battle_sim.species_rating --hours 12`

The old species ratings came from sequential elo over one-on-one battles, which had two problems: a
Pokemon was measured with no team to lean on, and elo's running update throws away most of what each
game says. This runs 6v6 battles between randomly assembled teams and writes every result to a log;
`species_fit` then reads that log and attributes the results to the species involved in one batch
regression. A wall is credited for the win its team got rather than punished for having no offence.

Three decisions worth knowing about:

*Same-seed mirrored pairs.* Each pairing is played twice, once from each side, on the same seed. The
difference between those two battles is the side and nothing else, so most of the luck cancels.

*A dealing bag, not random draws.* Species are dealt from a shuffled bag rather than sampled
independently, so appearances stay near-even instead of Poisson-scattered. With a fixed budget, even
coverage is what buys precision.

*A fresh set every appearance.* `random_set` is re-rolled per battle, so a species is measured across
its whole spread of sets rather than one lucky roll.

The log is the point of the exercise: battles are expensive and fitting is cheap, so everything is
written down and any model can be fitted later without re-running anything. It is append-only JSONL,
written and fsynced by the parent process alone — workers only return results — so no two writers
ever interleave, and a run that dies keeps everything up to the last completed pairing. Restarting
with the same `--out` resumes where it stopped.
"""

import argparse
import json
import os
import random
import signal
import subprocess
import time
from collections.abc import Iterator, Sequence
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
    """Every rateable species, under its display name.

    `in_scope_species` deals in normalised keys ("marowakalola"), but the ratings file this feeds is
    keyed by display name ("Marowak-Alola") and so is everything that reads it. Converting here means
    the log is legible and the fit's output drops straight in, rather than needing a rename pass that
    someone would eventually forget to run.
    """
    return sorted(get_species(key).name for key in in_scope_species())


def deal_pairings(pool: Sequence[str], count: int, rng: random.Random, start: int = 0) -> Iterator[Pairing]:
    """`count` pairings, dealing species from a reshuffled bag so appearances stay even.

    Deterministic in `rng`, and `start` skips forward without changing what any later pairing is —
    which is what lets a resumed run carry on with the sequence the first one was partway through.
    """
    bag: list[str] = []

    def deal(size: int) -> tuple[str, ...]:
        """One team, with no species repeated on it.

        A repeat is possible whenever a team spans a bag refill, and the engine refuses a side whose
        nicknames are not unique — so left alone this would have failed a battle here and there all
        night. A duplicate goes back in the bag rather than being dropped, so coverage stays even.
        """
        nonlocal bag
        drawn: list[str] = []
        held: list[str] = []
        while len(drawn) < size:
            if not bag:
                bag = list(pool)
                rng.shuffle(bag)
            name = bag.pop()
            (held if name in drawn else drawn).append(name)
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
    """Both battles of one pairing, as log records. Runs in a worker process.

    Never raises: a battle that falls over is written down as a failure and the run carries on. Two
    lost battles are noise; a run that stops at 3am because of one is half a night.
    """
    pairing, budget = job
    genome = _worker_genome()
    profile = SearchProfile(budget=budget)
    rng = random.Random(pairing.seed)
    records: list[dict[str, Any]] = []
    try:
        team_a, team_b = _build_teams(pairing, rng)
    except Exception as error:  # a species whose set cannot be generated should not end the run
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
        except Exception as error:  # one broken battle, not one broken run
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
                # Always from team A's point of view, whichever side it played, so every row reads
                # the same way to the fit.
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
    """How far a previous run got: one past the highest pairing index already written.

    Reads the whole log rather than trusting a counter, and tolerates a final truncated line — a run
    killed mid-write leaves one, and refusing to resume over it would be the log costing us the data
    it exists to protect.
    """
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
    """Alongside the log, what produced it. A rating file nobody can trace is a rating file nobody
    can defend."""
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
    # One writer, in the parent: workers hand back records and this is the only thing that touches
    # the file, so there is no interleaving to reason about and no lock to get wrong.
    #
    # Work is submitted a few pairings at a time rather than through `Executor.map`, which submits
    # its whole iterable up front — on an open-ended run that is a billion queued futures and an
    # out-of-memory kill before the first battle finishes.
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
