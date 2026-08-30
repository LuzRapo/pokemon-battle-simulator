import argparse
import os
import random
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

from loguru import logger

from battle_sim.budget_curve import summarize
from battle_sim.evolution import Progress, Team, battle_margin, load_teams, load_weights
from battle_sim.matchup import MatchupPlayer, MatchupWeights
from battle_sim.observation import SetPrior
from battle_sim.players import BasicPlayer
from battle_sim.runner import Player, run_battle
from battle_sim.search import SearchPlayer, SearchProfile, gated_mixture, row_values

type PairSpec = tuple[Team, Team, int]  # (team a, team b, seed) — one same-seed side pair


@dataclass(frozen=True)
class Pairing:
    a: str
    b: str
    margin: float  # a's mean margin vs b
    ci: float


def build_players(genomes: Sequence[tuple[str, MatchupWeights]], profile: SearchProfile) -> list[tuple[str, Player]]:
    players: list[tuple[str, Player]] = [("Basic", BasicPlayer())]
    for label, weights in genomes:
        players.append((label, MatchupPlayer(weights)))
        players.append((f"{label}+S", SearchPlayer(weights, profile=profile)))
    return players


def sample_pairs(teams: Sequence[Team], count: int, rng: random.Random) -> list[PairSpec]:
    if count % 2:
        raise ValueError(f"Battle count must be even to form side pairs, got {count}.")
    return [(rng.choice(teams), rng.choice(teams), rng.randrange(2**32)) for _ in range(count // 2)]


def duel_margin(a: Player, b: Player, spec: PairSpec, prior: SetPrior) -> float:
    """A's mean margin over one side pair: both orientations of the same teams and seed."""
    team_a, team_b, seed = spec
    first = run_battle(team_a, team_b, a, b, seed=seed, prior=prior)
    second = run_battle(team_a, team_b, b, a, seed=seed, prior=prior)
    return (battle_margin(first, candidate_index=0) + battle_margin(second, candidate_index=1)) / 2


def duel(
    a: Player,
    b: Player,
    pairs: Sequence[PairSpec],
    prior: SetPrior,
    workers: int,
    tick: Callable[[], None],
) -> tuple[float, float]:
    if workers == 1:
        margins = []
        for spec in pairs:
            margins.append(duel_margin(a, b, spec, prior))
            tick()
        return summarize(margins)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(duel_margin, a, b, spec, prior) for spec in pairs]
        for _ in as_completed(futures):
            tick()
        return summarize([future.result() for future in futures])


def _matrix(labels: Sequence[str], pairings: Sequence[Pairing]) -> str:
    margins = {(p.a, p.b): p.margin for p in pairings} | {(p.b, p.a): 1 - p.margin for p in pairings}
    width = max(len(label) for label in labels) + 2
    lines = [" " * width + "".join(f"{label:>{width}}" for label in labels)]
    for row in labels:
        cells = "".join(f"{'-':>{width}}" if row == col else f"{margins[row, col]:>{width}.3f}" for col in labels)
        lines.append(f"{row:>{width}}{cells}")
    return "\n".join(lines)


def _ranking(labels: Sequence[str], pairings: Sequence[Pairing]) -> str:
    margins = {(p.a, p.b): p.margin for p in pairings} | {(p.b, p.a): 1 - p.margin for p in pairings}
    means = {row: sum(margins[row, col] for col in labels if col != row) / (len(labels) - 1) for row in labels}
    lines = [
        f"  {rank + 1}. {label:<14} {mean:.3f}"
        for rank, (label, mean) in enumerate(sorted(means.items(), key=lambda item: item[1], reverse=True))
    ]
    return "\n".join(lines)


def _nash_ranking(labels: Sequence[str], pairings: Sequence[Pairing]) -> str:
    """Strength against the field's *equilibrium* mixture rather than its flat average.

    Mean-vs-field is the statistic non-transitivity breaks: it rewards beating whoever happens to be
    in the pool, so padding the field with agents that one entrant farms inflates that entrant. This
    matrix has real cycles (a > b > c > a), so a flat mean has no fixed meaning. Scoring against the
    equilibrium mixture instead asks the question that survives cycles — how does this agent do
    against the toughest distribution of opponents the pool can field — and it reuses the same
    zero-sum solver the search already runs on payoff matrices.
    """
    margins = {(p.a, p.b): p.margin for p in pairings} | {(p.b, p.a): 1 - p.margin for p in pairings}
    matrix = [[0.0 if a == b else margins[a, b] - 0.5 for b in labels] for a in labels]
    values = row_values(matrix)
    mixture = gated_mixture(matrix)
    ranked = sorted(zip(labels, values, mixture, strict=True), key=lambda row: row[1], reverse=True)
    lines = [
        f"  {rank + 1}. {label:<14} {value:+.4f}" + (f"   equilibrium weight {weight:.2f}" if weight > 0 else "")
        for rank, (label, value, weight) in enumerate(ranked)
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Round-robin: every genome, myopic and searching, vs every other.")
    parser.add_argument("champions", nargs="*", type=Path, help="champion weight JSONs (stock is always included)")
    parser.add_argument("--teams-dir", type=Path, default=Path("sample_teams"))
    parser.add_argument("--matchups", type=int, default=200, help="battles per pairing, as same-seed side pairs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--search-budget", type=int, default=30, help="profile for the +S players")
    parser.add_argument("--root-k", type=int, default=5)
    parser.add_argument("--chance-samples", type=int, default=2)
    parser.add_argument("--determinizations", type=int, default=1)
    args = parser.parse_args()

    teams = load_teams(args.teams_dir)
    prior = SetPrior.from_teams(teams)
    profile = SearchProfile(
        budget=args.search_budget,
        root_k=args.root_k,
        chance_samples=args.chance_samples,
        determinizations=args.determinizations,
    )
    genomes = [("stock", MatchupWeights())] + [(path.stem, load_weights(path)) for path in args.champions]
    players = build_players(genomes, profile)
    pairs = sample_pairs(teams, args.matchups, random.Random(args.seed))
    matchups = list(combinations(range(len(players)), 2))
    logger.info(f"{len(players)} players, {len(matchups)} pairings, {len(pairs)} side pairs each")
    progress = Progress(total=len(matchups) * len(pairs))
    pairings = []
    for i, j in matchups:
        (label_a, player_a), (label_b, player_b) = players[i], players[j]
        progress.note(f"{label_a} vs {label_b}")
        started = time.monotonic()
        margin, ci = duel(player_a, player_b, pairs, prior, args.workers, progress.tick)
        pairings.append(Pairing(label_a, label_b, margin, ci))
        rate = len(pairs) * 2 / (time.monotonic() - started)
        print(f"\n{label_a:>12} vs {label_b:<12} {margin:.3f} ± {ci:.3f}  ({rate:.1f}/s)")
    labels = [label for label, _ in players]
    print(f"\nstrength vs the field's equilibrium mixture (cycle-proof):\n{_nash_ranking(labels, pairings)}")
    print(f"\nmean margin vs the field (flat average, distorted by cycles):\n{_ranking(labels, pairings)}")
    print(f"\nmargin matrix (row vs column):\n{_matrix(labels, pairings)}")


if __name__ == "__main__":
    main()
