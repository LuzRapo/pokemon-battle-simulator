"""Where the fitted evaluator and the live one disagree, which one is right?"""

import argparse
import json
import random
from pathlib import Path

from loguru import logger

from battle_sim.search import PositionWeights
from tools.fit_evaluator import FEATURES, LOGS, fit, parse

# Which features the live `evaluate_position` actually has an opinion about.
SCORED = ("material", "alive", "toxic_pressure", "hazard_layers", "active_boosts")


def live_weights(weights: PositionWeights) -> list[float]:
    """The live evaluator's coefficients, expressed against these features."""
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

    logger.info(f"{len(test)} held-out turn-rows\n")
    logger.info(f"{'feature':<18} {'live':>8} {'fitted':>8}")
    for name, a, b in zip(SCORED, live, fitted, strict=True):
        logger.info(f"{name:<18} {a:>8.3f} {b:>8.3f}")

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
    logger.info(
        f"\nthey agree on {agreed / len(test):.1%} of positions, and are right on {agree_right / agreed:.1%} of those"
    )
    logger.info(f"they disagree on {disputed / len(test):.1%} ({disputed} positions):")
    logger.info(f"  the live evaluator called it   {live_right:>6}  ({live_right / disputed:.1%})")
    logger.info(f"  the fitted evaluator called it {fitted_right:>6}  ({fitted_right / disputed:.1%})")
    overall_live = (agree_right + live_right) / len(test)
    overall_fitted = (agree_right + fitted_right) / len(test)
    logger.info(f"\noverall: live {overall_live:.1%}, fitted {overall_fitted:.1%}")
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
