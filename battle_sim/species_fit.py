"""Turn a `species_rating` battle log into a rating per species."""

import argparse
import json
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

# Tuned by split-half reliability rather than guessed.
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


def iter_log(path: Path) -> Iterator[dict[str, Any]]:
    """Every complete record in the log, streamed."""
    skipped = 0
    with path.open() as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if "margin" in record and "a" in record and "b" in record:
                yield record
            else:
                skipped += 1
    if skipped:
        logger.info(f"skipped {skipped} incomplete or failed records")


def read_log(path: Path) -> list[dict[str, Any]]:
    return list(iter_log(path))


def species_in(rows: Iterable[dict[str, Any]]) -> list[str]:
    return sorted({name for row in rows for name in (*row["a"], *row["b"])})


def fit(rows: Iterable[dict[str, Any]], ridge: float = RIDGE, names: Sequence[str] | None = None) -> Fit:
    """Least squares with a ridge penalty, solved on the normal equations."""
    if names is None:
        rows = list(rows)
        names = species_in(rows)
    names = list(names)
    index = {name: i for i, name in enumerate(names)}
    size = len(names)
    gram = np.zeros((size, size))
    moment = np.zeros(size)
    appearances: Counter[str] = Counter()

    battles = 0
    for row in rows:
        battles += 1
        columns = np.array([index[name] for name in (*row["a"], *row["b"])])
        signs = np.r_[np.ones(len(row["a"])), -np.ones(len(row["b"]))]
        gram[np.ix_(columns, columns)] += np.outer(signs, signs)
        moment[columns] += signs * (float(row["margin"]) - 0.5)  # centred: 0.5 is an even battle
        appearances.update((*row["a"], *row["b"]))

    coefficients = np.linalg.solve(gram + ridge * np.eye(size), moment)
    return Fit(names=names, coefficients=coefficients, appearances=appearances, battles=battles)


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

    # Two passes over the file rather than one over a list: a night's log does not fit in memory.
    names = species_in(iter_log(args.log))
    if not names:
        raise SystemExit(f"no usable battles in {args.log}")
    result = fit(iter_log(args.log), args.ridge, names)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_ratings(result, args.out, args.log)

    seen = result.appearances
    logger.info(
        f"{result.battles} battles, {len(result.names)} species, "
        f"{min(seen.values())}-{max(seen.values())} appearances each "
        f"(median {sorted(seen.values())[len(seen) // 2]})"
    )
    ranked = sorted(result.elo().items(), key=lambda kv: -kv[1])
    logger.info(f"\nstrongest {args.top}:")
    for name, value in ranked[: args.top]:
        logger.info(f"  {value:7.0f}  {name:24} ({seen[name]} appearances)")
    logger.info(f"\nweakest {args.top}:")
    for name, value in ranked[-args.top :]:
        logger.info(f"  {value:7.0f}  {name:24} ({seen[name]} appearances)")


if __name__ == "__main__":
    main()
