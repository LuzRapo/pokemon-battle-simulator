"""`species_rating`, on the Rust engine: rate every species by what it adds to a team that wins.

`uv run python -m battle_sim.rust_ratings --hours 2`, then
`uv run python -m battle_sim.species_fit --log ratings/rust-battles.jsonl --out rust_elo.json`.

The same experiment as `species_rating`, run hundreds of times faster:
- random 6v6 teams dealt from a bag, so every species appears about equally often;
- each pairing played twice on one seed with the sides swapped, so most of the luck cancels;
- the same append-only JSONL log, which `species_fit` reads unchanged.

What differs is what that speed bought, and what it cost:

* **Sets come from the learnpool, not from Random Battles roles.** Four moves the species can
  learn (two attacks off its better attacking stat, its own types preferred, then two at random),
  any of its abilities, and no item — `setgen.learnpool_set`. Each appearance is a fresh roll, so a
  species is rated across its whole movepool rather than its best build.
* **The pilot is `MatchupPlayer` ported to Rust (`rust/src/matchup.rs`), not `SearchPlayer`.** It
  scores every action with the genome's 28 matchup features and plays the best, a move at a time,
  with both sides' sets known. It is the Python search's own evaluator without the search on top:
  weaker, but thousands of times cheaper, and `tests/test_rust_matchup.py` checks it scores every
  action exactly as the Python does.
* **Only what the Rust engine plays.** Moves it has not ported are never dealt, nor are Future
  Sight and Doom Desire (see `UNDEALT_MOVES`). A species none of whose abilities it plays (Truant,
  Stance Change, and the abilities neither engine models) plays with no ability, as `random_set`
  always played the unmodelled ones. Formes that need a held item to exist (Silvally's memories,
  Genesect's drives, ...) are left out, as there are no items; the manifest lists them.

Each batch of pairings is played in parallel in Rust with the GIL released, then written down
before the next starts; stopping (Ctrl-C) finishes the batch in hand. Restarting with the same
`--out` resumes where it stopped.
"""

import argparse
import json
import random
import signal
import time
from collections.abc import Sequence
from pathlib import Path
from types import FrameType
from typing import Any

from loguru import logger

from battle_sim.database.loader import get_species
from battle_sim.database.scope import in_scope_species
from battle_sim.differential import encode_spec
from battle_sim.models.spec import PokemonSpec
from battle_sim.rust_bridge import database, load
from battle_sim.setgen import learnpool_set
from battle_sim.species_rating import Pairing, completed_pairings, deal_pairings

LOG_VERSION = 1
PILOT = "rust-matchup"
DEFAULT_OUT = Path("ratings/rust-battles.jsonl")
DEFAULT_GENOME = Path(__file__).resolve().parent.parent / "champions" / "random-a-myopic-tuned.json"
DEFAULT_BATCH = 512  # pairings per round: enough to keep every core busy, few enough to stop promptly
MAX_TURNS = 1000  # `run_battle`'s own cap

# Playable, but not in every situation they create: the Rust engine refuses a Future Sight or Doom
# Desire that lands after its user has switched out. Dealt, they would make a battle fail now and
# then, and dropping those battles would quietly drop results that depend on who carries the move.
UNDEALT_MOVES = frozenset({"Future Sight", "Doom Desire"})

_stopping = False


def playable_set(species: str, rng: random.Random) -> PokemonSpec:
    """A learnpool set holding only what the Rust engine plays."""
    db = database()

    def move_ok(move: str) -> bool:
        return move not in UNDEALT_MOVES and db.move_playable(move)

    return learnpool_set(species, rng, move_ok, lambda ability: db.ability_playable(ability.name))


def species_pool() -> tuple[list[str], dict[str, str]]:
    """Every rateable species by display name, and every one left out with the reason."""
    kept: list[str] = []
    dropped: dict[str, str] = {}
    for key in sorted(in_scope_species()):
        entry = get_species(key)
        if entry.required_item is not None:
            dropped[entry.name] = f"needs {entry.required_item} to exist, and nobody holds items"
            continue
        try:
            playable_set(entry.name, random.Random(0))
        except ValueError as error:
            dropped[entry.name] = str(error)
            continue
        kept.append(entry.name)
    return sorted(kept), dropped


def build_teams(pairing: Pairing) -> tuple[list[PokemonSpec], list[PokemonSpec]]:
    rng = random.Random(pairing.seed)
    return [playable_set(name, rng) for name in pairing.team_a], [playable_set(name, rng) for name in pairing.team_b]


def play_batch(pairings: Sequence[Pairing], genome: str | None, threads: int) -> list[dict[str, Any]]:
    """Both battles of every pairing, as log records, in pairing order."""
    jobs: list[tuple[str, str, int]] = []
    sets: list[list[list[Any]]] = []
    for pairing in pairings:
        team_a, team_b = build_teams(pairing)
        # What each Pokemon actually carried, so a species' ratings can be broken down by set later.
        sets.append([[[spec.ability.name, list(spec.moves)] for spec in team] for team in (team_a, team_b)])
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
    pairings = deal_pairings(pool, total, random.Random(seed), start=done)
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
