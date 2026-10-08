import argparse
import math
import os
import random
import statistics
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from battle_sim.evolution import Matchup, Progress, fitness, load_teams, load_weights, sample_matchups
from battle_sim.matchup import MatchupPlayer, MatchupWeights
from battle_sim.observation import SetPrior
from battle_sim.runner import Player
from battle_sim.search import SearchProfile

type SidePair = list[Matchup]


@dataclass(frozen=True)
class BudgetCell:
    genome: str
    budget: int
    margin: float
    ci: float  # 95% half-width over pair means
    battles: int
    seconds: float


def pair_margins(
    weights: MatchupWeights,
    search: SearchProfile | None,
    pairs: Sequence[SidePair],
    prior: SetPrior,
    workers: int,
    tick: Callable[[], None],
) -> list[float]:
    """One margin per side pair: fitness over exactly that pair's two same-seed battles."""
    opponents: list[Player] = [MatchupPlayer(weights)]
    if workers == 1:
        margins = []
        for pair in pairs:
            margins.append(fitness(weights, pair, opponents, prior, search))
            tick()
        return margins
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fitness, weights, pair, opponents, prior, search) for pair in pairs]
        for _ in as_completed(futures):
            tick()
        return [future.result() for future in futures]


def summarize(margins: Sequence[float]) -> tuple[float, float]:
    return statistics.mean(margins), 1.96 * statistics.stdev(margins) / math.sqrt(len(margins))


def _row(cell: BudgetCell) -> str:
    rate = cell.battles / cell.seconds
    return (
        f"{cell.genome:<24} {cell.budget:>6}  {cell.margin:.3f} ± {cell.ci:.3f}"
        f"  {cell.battles:>7}  {cell.seconds / 60:>6.1f}m  {rate:>5.1f}/s"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Margin of SearchPlayer vs its own myopic genome, per node budget.")
    parser.add_argument("genomes", nargs="*", type=Path, help="champion weight JSONs (stock is always included)")
    parser.add_argument("--teams-dir", type=Path, default=Path("sample_teams"))
    parser.add_argument(
        "--budgets", type=int, nargs="+", default=[30, 400], help="stock profile: 30 depth 1, 400 depth 2, 3500 depth 3"
    )
    parser.add_argument("--matchups", type=int, default=600, help="battles per cell, played as same-seed side pairs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--root-k", type=int, default=5, help="actions per side kept at the search root")
    parser.add_argument("--top-k", type=int, default=3, help="opponent replies kept per node")
    parser.add_argument("--chance-samples", type=int, default=2, help="roll realizations averaged per root cell")
    parser.add_argument("--determinizations", type=int, default=1, help="sampled belief worlds hedged per decision")
    args = parser.parse_args()

    teams = load_teams(args.teams_dir)
    prior = SetPrior.from_teams(teams)
    genomes = [("stock", MatchupWeights())] + [(path.stem, load_weights(path)) for path in args.genomes]
    matchups = sample_matchups(teams, args.matchups, random.Random(args.seed))
    pairs = [matchups[i : i + 2] for i in range(0, len(matchups), 2)]
    logger.info(
        f"loaded {len(teams)} teams; {len(genomes)} genome(s) x {len(args.budgets)} budget(s),"
        f" {len(pairs)} side pairs per cell"
    )
    header = f"{'genome':<24} {'budget':>6}  {'margin':<13}  {'battles':>7}  {'time':>7}  {'rate':>6}"
    print(header)
    progress = Progress(total=len(genomes) * len(args.budgets) * len(pairs))
    cells = []
    for label, weights in genomes:
        for budget in args.budgets:
            progress.note(f"{label} @ budget {budget}")
            search = (
                SearchProfile(
                    budget=budget,
                    root_k=args.root_k,
                    top_k=args.top_k,
                    chance_samples=args.chance_samples,
                    determinizations=args.determinizations,
                )
                if budget
                else None
            )
            started = time.monotonic()
            margin, ci = summarize(pair_margins(weights, search, pairs, prior, args.workers, progress.tick))
            cell = BudgetCell(label, budget, margin, ci, battles=len(matchups), seconds=time.monotonic() - started)
            cells.append(cell)
            print(f"\n{_row(cell)}")
    print(f"\n{header}")
    for cell in cells:
        print(_row(cell))


if __name__ == "__main__":
    main()
