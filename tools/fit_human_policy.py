"""Fit a coarse human policy from the Gen 7 Anything Goes replays.

Every measurement in this project is our AI against itself, which cannot see a blindspot both sides
share -- neither punishes the other for under-switching, because neither switches. A benchmark
opponent that plays like the ladder does breaks that symmetry.

What is learnable from a log is *which kind* of move a human chose, not why: the logs never reveal an
EV spread, and reconstructing every position exactly enough to featurise it would mean replaying five
thousand games through the engine in lockstep. So this fits the honest thing -- the distribution over
decision kinds, conditioned on how far into the game it is, which is the axis that visibly matters
(a quarter of all hazards go down on turn one).

    uv run python -m tools.fit_human_policy --out battle_sim/data/human_policy.json
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from tools.play_profile import CATEGORIES, categorise

LOGS = Path("data/ag_replays/logs")
# Opening, middlegame, and the long tail. Hazards and leads live almost entirely in the first.
BUCKETS = ((1, 3), (4, 8), (9, 10_000))


def bucket_of(turn: int) -> str:
    for low, high in BUCKETS:
        if low <= turn <= high:
            return f"{low}-{high}" if high < 10_000 else f"{low}+"
    return "1-3"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=5001)
    parser.add_argument("--out", type=Path, default=Path("battle_sim/data/human_policy.json"))
    args = parser.parse_args()

    counts: dict[str, Counter] = defaultdict(Counter)
    games = 0
    for path in sorted(LOGS.glob("*.log"))[: args.limit]:
        fainted: set[str] = set()
        turn = 1
        for line in path.read_text(errors="replace").splitlines():
            parts = line.split("|")
            if len(parts) < 3:
                continue
            tag = parts[1]
            if tag == "turn":
                turn = int(parts[2])
            elif tag == "faint":
                fainted.add(parts[2].split(":")[0])
            elif tag == "move":
                counts[bucket_of(turn)][categorise(parts[3])] += 1
            elif tag in {"switch", "drag"}:
                side = parts[2].split(":")[0]
                # A replacement after a faint was not a choice, and neither was being dragged out.
                if tag == "drag" or side in fainted:
                    fainted.discard(side)
                else:
                    counts[bucket_of(turn)]["switch"] += 1
        games += 1

    policy = {
        bucket: {name: counts[bucket][name] / sum(counts[bucket].values()) for name in CATEGORIES}
        for bucket in counts
        if sum(counts[bucket].values())
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"games": games, "buckets": policy}, indent=2, sort_keys=True) + "\n")
    print(f"{games} replays -> {args.out}\n")
    print(f"{'turns':<8}" + "".join(f"{name:>10}" for name in CATEGORIES))
    for bucket in sorted(policy, key=lambda b: int(b.split("-")[0].rstrip("+"))):
        print(f"{bucket:<8}" + "".join(f"{policy[bucket][name]:>9.1%} " for name in CATEGORIES))


if __name__ == "__main__":
    main()
