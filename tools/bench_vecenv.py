"""How fast `VecEnv` plays mirror battles under a uniformly random policy — the engine's ceiling.

Every step pays for the observation encoding, the masks and the numpy hand-over, exactly as
training will; only the network is missing. Training throughput is this number divided by however
long the network takes on a batch.

    uv run python tools/bench_vecenv.py --envs 256 --threads 3 --seconds 20
"""

import argparse
import json
import random
import time

import numpy as np

from battle_sim.ag_sets import mirror_team
from battle_sim.differential import encode_spec
from battle_sim.rust_bridge import database, load


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--envs", type=int, default=256)
    parser.add_argument("--threads", type=int, default=0, help="0: one per core")
    parser.add_argument("--seconds", type=float, default=20.0)
    args = parser.parse_args()

    teams = random.Random(0)
    pool = [json.dumps([encode_spec(spec) for spec in mirror_team(teams)]) for _ in range(512)]
    env = load().VecEnv(database(), args.envs, max_turns=300, threads=args.threads)
    rng = np.random.default_rng(0)
    seed = 0
    decisions = battles = turns = 0
    start = time.perf_counter()
    while time.perf_counter() - start < args.seconds:
        for index in env.empty():
            team = pool[seed % len(pool)]
            env.reset(index, [team, team], seed)
            seed += 1
        rows, sides, _, _, _, _, mask = env.observe()
        # A uniform choice among each row's legal actions: the largest of uniform noise, masked.
        choice = np.where(mask, rng.random(mask.shape), -1.0).argmax(axis=1)
        env.act(rows, sides, choice)
        decisions += len(rows)
        for _, ending, _, played in env.collect():
            battles += 1
            turns += played
            if ending.startswith("failed"):
                raise SystemExit(ending)
    elapsed = time.perf_counter() - start
    print(
        f"{args.envs} envs, {args.threads or 'all'} threads: {decisions / elapsed:,.0f} decisions/s, "
        f"{battles / elapsed:,.1f} battles/s, {turns / max(battles, 1):.1f} turns per battle"
    )


if __name__ == "__main__":
    main()
