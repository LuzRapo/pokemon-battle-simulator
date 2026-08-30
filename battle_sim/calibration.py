"""Diagnostic: is our incoming-threat estimate calibrated against what the attacker can really do?

Every switch decision, pivot ranking and exchange edge is priced off one number — how hard the
opposing active is about to hit. We only *believe* their set, so that number is an estimate, and
its error propagates into every decision built on it.

For sampled positions drawn from a team pool, this compares each estimator against the truth the
believer cannot see: the best expected hit the attacker's *actual* set can land. Reported per
reveal count, so an estimator that starts wide and sharpens is distinguishable from one that is
wrong throughout.

A note on what this does *not* say, because the project's earlier reading of it was wrong. The
0.599-predicted-vs-0.402-actual gap measured against replay logs is not evidence of bias in this
sense: realized damage per turn averages in the turns an attacker spends on status, setup, pivots
and resisted hits, so it is a much lower quantity than the best hit a set can land, and no threat
estimator meant to price a worst case should match it. Both estimators here come out near
unbiased against the honest target. What separates them is per-position error.

    uv run python -m battle_sim.calibration --teams-dir sample_teams --positions 400
"""

import argparse
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from battle_sim.analysis import best_expected_damage, posterior_threat
from battle_sim.evolution import Team, load_teams
from battle_sim.maths.rng import RNG
from battle_sim.mechanics.battle import BattleState, SideState
from battle_sim.mechanics.log import BattleLog
from battle_sim.models.log_events import MoveUsed
from battle_sim.observation import BattleObserver, SetPrior
from battle_sim.runner import build_side

_MAX_REVEALS = 3  # a fourth revealed move leaves nothing to believe


@dataclass(frozen=True)
class Sample:
    """One position's three readings of the same incoming hit, as fractions of the defender's HP."""

    reveals: int
    believed_best: float  # max over the modal believed set: the estimator we are replacing
    posterior: float  # expectation over the set posterior: the estimator we are adopting
    truth: float  # max over the attacker's real set: what they can actually land


@dataclass(frozen=True)
class Calibration:
    """One reveal count's readings: mean levels, and the error that actually separates estimators.

    Mean bias is the weaker test — an estimator can be unbiased on average while being wrong in
    both directions position by position, and a search consumes the per-position number, not the
    average. `believed_error`/`posterior_error` are mean absolute error against truth.
    """

    reveals: int
    positions: int
    believed_best: float
    posterior: float
    truth: float
    believed_error: float
    posterior_error: float

    @property
    def believed_bias(self) -> float:
        return self.believed_best / self.truth

    @property
    def posterior_bias(self) -> float:
        return self.posterior / self.truth

    @property
    def error_reduction(self) -> float:
        """Fraction of the old estimator's per-position error the posterior removes."""
        return (self.believed_error - self.posterior_error) / self.believed_error


def sample_positions(teams: Sequence[Team], prior: SetPrior, count: int, seed: int) -> list[Sample]:
    """Believed-vs-true threat readings for random lead pairs at each reveal count."""
    rng = random.Random(seed)
    samples: list[Sample] = []
    for _ in range(count):
        team_a, team_b = rng.choice(teams), rng.choice(teams)
        state = BattleState(
            sides=(build_side(team_a, range(len(team_a))), build_side(team_b, range(len(team_b)))),
            rng=RNG(seed=0),
        )
        truth_attacker = state.sides[1].active_pokemon
        defender = state.sides[0].active_pokemon
        real_moves = [move.name for move in truth_attacker.moves.to_list()]
        rng.shuffle(real_moves)
        observer = BattleObserver(state, prior)
        for reveals in range(_MAX_REVEALS + 1):
            if reveals:
                observer.ingest(_reveal(state.sides[1], real_moves[reveals - 1]))
            view = observer.view(0)
            believed_attacker = view.sides[1].active_pokemon
            maximum = defender.stat_totals.HP
            samples.append(
                Sample(
                    reveals=reveals,
                    believed_best=best_expected_damage(believed_attacker, defender, view) / maximum,
                    posterior=posterior_threat(believed_attacker, defender, view) / maximum,
                    truth=best_expected_damage(truth_attacker, defender, state) / maximum,
                )
            )
    return samples


def _reveal(side: SideState, move_name: str) -> BattleLog:
    """The log a defender would have seen: their active using one of its real moves."""
    return BattleLog(entries=[MoveUsed(side=1, pokemon=side.active_pokemon.nickname, move=move_name)])


def summarise(samples: Sequence[Sample]) -> list[Calibration]:
    by_reveals: dict[int, list[Sample]] = {}
    for sample in samples:
        by_reveals.setdefault(sample.reveals, []).append(sample)
    return [
        Calibration(
            reveals=reveals,
            positions=len(group),
            believed_best=sum(s.believed_best for s in group) / len(group),
            posterior=sum(s.posterior for s in group) / len(group),
            truth=sum(s.truth for s in group) / len(group),
            believed_error=sum(abs(s.believed_best - s.truth) for s in group) / len(group),
            posterior_error=sum(abs(s.posterior - s.truth) for s in group) / len(group),
        )
        for reveals, group in sorted(by_reveals.items())
    ]


def report(rows: Sequence[Calibration]) -> str:
    columns = ("reveals", "n", "truth", "bias: old", "bias: new", "error: old", "error: new", "reduction")
    header = f"{columns[0]:>7}  {columns[1]:>5}  " + "  ".join(f"{name:>10}" for name in columns[2:])
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{row.reveals:>7}  {row.positions:>5}  {row.truth:>10.3f}  {row.believed_bias:>10.3f}  "
            f"{row.posterior_bias:>10.3f}  {row.believed_error:>10.4f}  {row.posterior_error:>10.4f}  "
            f"{row.error_reduction:>9.1%}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teams-dir", type=Path, default=Path("sample_teams"))
    parser.add_argument("--positions", type=int, default=400, help="lead pairs sampled; each gives 4 reveal counts")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    teams = load_teams(args.teams_dir)
    logger.info(f"{len(teams)} teams from {args.teams_dir}")
    prior = SetPrior.from_teams(teams)
    samples = sample_positions(teams, prior, args.positions, args.seed)
    print(report(summarise(samples)))


if __name__ == "__main__":
    main()
