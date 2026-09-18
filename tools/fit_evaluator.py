"""What is a position actually worth? Fit the evaluator's coefficients to 5,000 games of outcomes.

Every coefficient in `evaluate_position` is a guess. `timer_value` prices any status as one flat
constant; `hazard_value` counted heads until today; `residual_pressure` is 6.0 because that scale
makes it commensurate with material, not because anything measured a landed Toxic. The replays
settle it: what a position was worth is what the side holding it went on to score.

Each turn of each game becomes one training row -- the state differential from one side's view, and
whether that side won -- and a logistic regression reads off how much each term is worth *relative
to material*, which is the unit the evaluator already speaks. A coefficient ratio of 1.0 means "worth
one Pokemon's health".

    uv run python -m tools.fit_evaluator --stall-only
"""

import argparse
import json
import math
import random
from pathlib import Path

from battle_sim.replay_state import SideSnapshot, apply, tick_toxic

LOGS = Path("data/ag_replays/logs")
FEATURES = ("material", "alive", "toxic_pressure", "lingering", "hazard_layers", "active_boosts")
# What a stall side is made of; a game counts as stall if either side fields three of them.
STALL_CORE = {"Giratina", "Chansey", "Skarmory", "Sableye", "Tentacruel", "Blissey", "Ferrothorn", "Toxapex", "Lugia"}


def row(mine: SideSnapshot, theirs: SideSnapshot) -> list[float]:
    """The differential this side sees. Same sign convention as `evaluate_position`: bigger is better."""
    return [
        mine.material() - theirs.material(),
        float(mine.alive() - theirs.alive()),
        theirs.toxic_pressure() - mine.toxic_pressure(),  # a clock on *them* is good for us
        float(theirs.lingering() - mine.lingering()),
        float(theirs.hazard_layers() - mine.hazard_layers()),
        float(mine.active_boosts() - theirs.active_boosts()),
    ]


def parse(path: Path, stall_only: bool) -> list[tuple[list[float], int]]:
    """One row per turn per side, labelled with whether that side went on to win."""
    text = path.read_text(errors="replace")
    if "|win|" not in text:
        return []
    players: dict[str, str] = {}
    sides = {"p1": SideSnapshot(), "p2": SideSnapshot()}
    seen_species: set[str] = set()
    pending: list[tuple[str, list[float]]] = []
    winner_name = ""
    for line in text.splitlines():
        parts = line.split("|")
        if len(parts) < 2:
            continue
        tag = parts[1]
        if tag == "player" and len(parts) > 3:
            players[parts[2]] = parts[3]
        elif tag == "win":
            winner_name = parts[2].strip()
        elif tag in {"switch", "drag", "replace"} and len(parts) > 3:
            seen_species.add(parts[3].split(",")[0].strip())
        if tag == "turn":
            tick_toxic(sides)
            for key in ("p1", "p2"):
                pending.append((key, row(sides[key], sides["p2" if key == "p1" else "p1"])))
            continue
        apply(line, sides)
    if not winner_name or (stall_only and len(STALL_CORE & seen_species) < 3):
        return []
    won = {key: players.get(key, "") == winner_name for key in ("p1", "p2")}
    if not any(won.values()):
        return []  # forfeit or a name we could not match; no usable label
    return [(features, int(won[key])) for key, features in pending]


def fit(rows: list[tuple[list[float], int]], passes: int = 60, rate: float = 0.05, width: int = 0) -> list[float]:
    """Plain logistic regression by gradient descent; no dependency worth adding for six weights."""
    weights = [0.0] * (width or len(FEATURES))
    order = list(range(len(rows)))
    rng = random.Random(0)
    for step in range(passes):
        rng.shuffle(order)
        lr = rate / (1 + step / 10)
        for index in order:
            features, label = rows[index]
            z = sum(w * f for w, f in zip(weights, features, strict=True))
            error = label - 1 / (1 + math.exp(-max(-30.0, min(30.0, z))))
            for i, f in enumerate(features):
                weights[i] += lr * error * f
    return weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=5001)
    parser.add_argument("--stall-only", action="store_true", help="only games where a stall core appeared")
    parser.add_argument("--out", type=Path, default=Path("evaluator_fit.json"))
    parser.add_argument("--holdout", type=float, default=0.25, help="share of *games* kept back for testing")
    parser.add_argument("--drop", nargs="*", default=[], help="features to leave out; use to break collinear pairs")
    args = parser.parse_args()

    # Split by *game*, never by row: two turns of one battle share almost all their state, so a
    # row-wise split leaks the answer into the test set and every accuracy comes out flattering.
    train: list[tuple[list[float], int]] = []
    test: list[tuple[list[float], int]] = []
    games = 0
    splitter = random.Random(1)
    for path in sorted(LOGS.glob("*.log"))[: args.limit]:
        got = parse(path, args.stall_only)
        if not got:
            continue
        games += 1
        (test if splitter.random() < args.holdout else train).extend(got)
    if not train or not test:
        raise SystemExit("no usable rows")
    label = " (stall only)" if args.stall_only else ""
    print(f"{len(train)} train / {len(test)} held-out turn-rows from {games} games{label}\n")

    keep = [i for i, name in enumerate(FEATURES) if name not in args.drop]
    if args.drop:
        train = [([f[i] for i in keep], lab) for f, lab in train]
        test = [([f[i] for i in keep], lab) for f, lab in test]
        print(f"dropped: {', '.join(args.drop)}")
    names = [FEATURES[i] for i in keep]
    weights = fit(train, width=len(keep))
    material = weights[names.index("material")]
    print(f"{'feature':<18} {'weight':>9} {'in Pokemon':>12}")
    for name, w in zip(names, weights, strict=True):
        print(f"{name:<18} {w:>9.4f} {w / material:>12.3f}")

    def accuracy(rows: list[tuple[list[float], int]]) -> float:
        hits = sum(
            (sum(w * f for w, f in zip(weights, features, strict=True)) > 0) == bool(lab) for features, lab in rows
        )
        return hits / len(rows)

    print(f"\npredicts the eventual winner: {accuracy(train):.1%} on train, {accuracy(test):.1%} held out")
    args.out.write_text(
        json.dumps(
            {
                "features": names,
                "weights": weights,
                "per_material": [w / material for w in weights],
                "rows": len(train),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
