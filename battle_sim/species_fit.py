"""Turn a `species_rating` battle log into a rating per species.

`uv run python -m battle_sim.species_fit --log ratings/battles.jsonl --out pokemon_elo.json`

Every battle says one thing: these six beat those six by this margin. Attributing that to the twelve
species involved is a regression, not a running average — each species' coefficient is fitted with
every other species it played beside and against held constant, which is exactly what "how much does
this Pokemon contribute to a team" means and exactly what a one-on-one ladder cannot measure.

Fitting is separate from running on purpose. Battles cost hours and this costs seconds, so the log
is the asset: refit it with a different model, or against a later night's larger log, for free.

Ridge rather than plain least squares. A species seen a handful of times would otherwise take an
extreme coefficient on almost no evidence; the penalty pulls it back toward the field, which is the
honest reading of "we hardly saw this one".
"""

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

# Tuned rather than guessed. Fitting two independent halves of the log and correlating them measures
# how much of a rating is signal rather than noise, and 30 maximised it. The previous 1.0 was
# arbitrary and cost about two points of reliability at 12k battles (0.785 -> 0.805) and half a point
# at 20k (0.867 -> 0.872) — the more battles there are, the less the penalty has to do. 10 and 60 sit
# just behind it, so this is a broad optimum and not a knife edge worth re-tuning often.
RIDGE = 30.0
ELO_SCALE = 400 / np.log(10)  # logit units -> elo-like points, so the numbers read as they used to
ELO_CENTRE = 1500.0


@dataclass(frozen=True)
class Fit:
    names: list[str]
    coefficients: np.ndarray  # logit units, one per species in `names`
    appearances: Counter[str]
    battles: int

    def elo(self) -> dict[str, float]:
        """The coefficients on the scale the old ratings used, so the two can be compared at a glance."""
        centred = self.coefficients - self.coefficients.mean()
        return {name: ELO_CENTRE + ELO_SCALE * value for name, value in zip(self.names, centred, strict=True)}


def read_log(path: Path) -> list[dict[str, Any]]:
    """Every complete record in the log. A truncated last line (a run killed mid-write) is skipped
    rather than fatal, and so are the failure records the runner writes for battles that fell over."""
    rows: list[dict[str, Any]] = []
    skipped = 0
    with path.open() as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if "margin" in record and "a" in record and "b" in record:
                rows.append(record)
            else:
                skipped += 1
    if skipped:
        logger.info(f"skipped {skipped} incomplete or failed records")
    return rows


def fit(rows: list[dict[str, Any]], ridge: float = RIDGE) -> Fit:
    """Least squares with a ridge penalty, solved on the normal equations.

    The design matrix is one row per battle, +1 for each species on team A and -1 for each on team B,
    so a coefficient is "what this species does to the margin from whichever side it is on". It is
    far too sparse to build densely — 950 columns against tens of thousands of rows — but the normal
    equations only ever need the 12x12 block each battle touches, so they accumulate row by row.
    """
    names = sorted({name for row in rows for name in (*row["a"], *row["b"])})
    index = {name: i for i, name in enumerate(names)}
    size = len(names)
    gram = np.zeros((size, size))
    moment = np.zeros(size)
    appearances: Counter[str] = Counter()

    for row in rows:
        columns = np.array([index[name] for name in (*row["a"], *row["b"])])
        signs = np.r_[np.ones(len(row["a"])), -np.ones(len(row["b"]))]
        gram[np.ix_(columns, columns)] += np.outer(signs, signs)
        moment[columns] += signs * (float(row["margin"]) - 0.5)  # centred: 0.5 is an even battle
        appearances.update((*row["a"], *row["b"]))

    coefficients = np.linalg.solve(gram + ridge * np.eye(size), moment)
    return Fit(names=names, coefficients=coefficients, appearances=appearances, battles=len(rows))


def write_ratings(result: Fit, path: Path, log: Path) -> None:
    payload = {
        "source": str(log),
        "battles": result.battles,
        "species": len(result.names),
        "method": "ridge regression on 6v6 margins (battle_sim.species_fit)",
        "ratings": {name: round(value, 1) for name, value in sorted(result.elo().items())},
        "appearances": dict(sorted(result.appearances.items())),
    }
    path.write_text(json.dumps(payload, indent=1) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=Path("ratings/battles.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("ratings/pokemon_elo.json"))
    parser.add_argument("--ridge", type=float, default=RIDGE)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    rows = read_log(args.log)
    if not rows:
        raise SystemExit(f"no usable battles in {args.log}")
    result = fit(rows, args.ridge)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_ratings(result, args.out, args.log)

    seen = result.appearances
    logger.info(
        f"{result.battles} battles, {len(result.names)} species, "
        f"{min(seen.values())}-{max(seen.values())} appearances each "
        f"(median {sorted(seen.values())[len(seen) // 2]})"
    )
    ranked = sorted(result.elo().items(), key=lambda kv: -kv[1])
    print(f"\nstrongest {args.top}:")
    for name, value in ranked[: args.top]:
        print(f"  {value:7.0f}  {name:24} ({seen[name]} appearances)")
    print(f"\nweakest {args.top}:")
    for name, value in ranked[-args.top :]:
        print(f"  {value:7.0f}  {name:24} ({seen[name]} appearances)")


if __name__ == "__main__":
    main()
