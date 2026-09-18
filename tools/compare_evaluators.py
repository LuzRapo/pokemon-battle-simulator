"""Where the fitted evaluator and the live one disagree, which one is right?

A fitted coefficient being different from a hand-set one proves nothing on its own -- the two can
disagree everywhere and still rank real positions identically, in which case rewriting the weights
buys nothing. What matters is the subset of positions where they actually call *different winners*,
and which of them the game agreed with.

Both are scored on the same log-derived features, so the comparison is weights against weights with
nothing else moving. The live evaluator's coefficients are translated into this feature space in
`live_weights` below, which is the one place any judgement enters.

    uv run python -m tools.compare_evaluators
"""

import argparse
import json
import random
from pathlib import Path

from battle_sim.search import PositionWeights
from tools.fit_evaluator import FEATURES, LOGS, fit, parse

# Which features the live `evaluate_position` actually has an opinion about. `alive` has no
# counterpart at all, and the boost terms (`sweep_threat`, `setup_potential`) are not linear in a
# stage count, so both are scored at zero rather than guessed at.
SCORED = ("material", "alive", "toxic_pressure", "hazard_layers", "active_boosts")


def live_weights(weights: PositionWeights) -> list[float]:
    """The live evaluator's coefficients, expressed against these features.

    `_material` is already a sum of HP fractions and carries an implicit weight of one, so it is the
    unit here exactly as it is in the fit. The other two are read off `evaluate_position`:

      * `residual_pressure * residual_pressure(theirs)`, and `analysis.residual_pressure` divides its
        per-member total by six -- so a coefficient of 6.0 lands as 1.0 per unit of clock.
      * `hazard_value * hazard_pressure(theirs)`, where `hazard_pressure` is layers/4 scaled by the
        bench share, roughly 5/6 of a full team. So 0.594 lands near 0.12 per layer.
    """
    per_layer = weights.hazard_value / 4 * 5 / 6
    return [1.0, 0.0, weights.residual_pressure / 6, per_layer, 0.0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=5001)
    parser.add_argument("--holdout", type=float, default=0.25)
    parser.add_argument("--stall-only", action="store_true")
    args = parser.parse_args()

    keep = [FEATURES.index(name) for name in SCORED]
    train: list[tuple[list[float], int]] = []
    test: list[tuple[list[float], int]] = []
    splitter = random.Random(1)
    for path in sorted(LOGS.glob("*.log"))[: args.limit]:
        got = parse(path, args.stall_only)
        if not got:
            continue
        rows = [([f[i] for i in keep], label) for f, label in got]
        (test if splitter.random() < args.holdout else train).extend(rows)
    fitted = fit(train, width=len(SCORED))
    material = fitted[SCORED.index("material")]
    fitted = [w / material for w in fitted]  # into units of one Pokemon, like the live side
    live = live_weights(PositionWeights())

    print(f"{len(test)} held-out turn-rows\n")
    print(f"{'feature':<18} {'live':>8} {'fitted':>8}")
    for name, a, b in zip(SCORED, live, fitted, strict=True):
        print(f"{name:<18} {a:>8.3f} {b:>8.3f}")

    def score(weights: list[float], features: list[float]) -> float:
        return sum(w * f for w, f in zip(weights, features, strict=True))

    agree_right = agree_wrong = 0
    live_right = fitted_right = 0
    for features, label in test:
        live_call, fitted_call = score(live, features) > 0, score(fitted, features) > 0
        if live_call == fitted_call:
            if live_call == bool(label):
                agree_right += 1
            else:
                agree_wrong += 1
        elif live_call == bool(label):
            live_right += 1
        else:
            fitted_right += 1

    disputed = live_right + fitted_right
    agreed = agree_right + agree_wrong
    print(
        f"\nthey agree on {agreed / len(test):.1%} of positions, and are right on {agree_right / agreed:.1%} of those"
    )
    print(f"they disagree on {disputed / len(test):.1%} ({disputed} positions):")
    print(f"  the live evaluator called it   {live_right:>6}  ({live_right / disputed:.1%})")
    print(f"  the fitted evaluator called it {fitted_right:>6}  ({fitted_right / disputed:.1%})")
    overall_live = (agree_right + live_right) / len(test)
    overall_fitted = (agree_right + fitted_right) / len(test)
    print(f"\noverall: live {overall_live:.1%}, fitted {overall_fitted:.1%}")
    Path("evaluator_compare.json").write_text(
        json.dumps(
            {
                "features": SCORED,
                "live": live,
                "fitted": fitted,
                "disputed": disputed,
                "live_right": live_right,
                "fitted_right": fitted_right,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
