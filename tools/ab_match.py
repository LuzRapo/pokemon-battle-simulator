"""Head-to-head between two evaluator configurations, on matched teams and matched seeds.

Everything added to the scorer this session was verified one decision at a time -- does it pick
Toxic here, does it set rocks there -- and never once by whether it wins more games. This is that
gate. Each pairing is played twice with the sides swapped and the same seed, so team strength and
roll luck cancel between the two halves and what is left is the configuration.

    uv run python -m tools.ab_match --games 2 --workers 3
"""

import argparse
import json
import math
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from itertools import combinations
from pathlib import Path

from battle_sim.evolution import Progress, load_weights
from battle_sim.observation import SetPrior
from battle_sim.runner import run_battle
from battle_sim.search import PositionWeights, SearchPlayer, SearchProfile
from battle_sim.trainer_db_tournament import Trainer, load_trainers
from battle_sim.utils import Outcome

# What this session added to `evaluate_position`, and what it looked like before.
# The hazard coefficient, measured against 4,970 replays, versus the genome value it replaced.
AFTER = {"hazard_value": 3.0}
BEFORE = {"hazard_value": 0.594}


def play(
    a: Trainer, b: Trainer, weights, profile: SearchProfile, prior: SetPrior, seed: int, games: int
) -> tuple[int, int, int]:
    """(wins for AFTER, wins for BEFORE, draws) over `games` sides-swapped pairs."""
    base = PositionWeights.from_matchup(weights)
    after, before = replace(base, **AFTER), replace(base, **BEFORE)
    wins = losses = draws = 0
    for game in range(games):
        for after_is_p1 in (True, False):
            p1 = SearchPlayer(weights, profile=profile, position_weights=after if after_is_p1 else before)
            p2 = SearchPlayer(weights, profile=profile, position_weights=before if after_is_p1 else after)
            result = run_battle(a.team, b.team, p1, p2, seed=seed * 10 + game, prior=prior)
            if result.outcome is None or result.outcome is Outcome.DRAW:
                draws += 1
            elif (result.outcome is Outcome.P1_WIN) == after_is_p1:
                wins += 1
            else:
                losses += 1
    return wins, losses, draws


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teams-file", type=Path, default=Path("ag_teams.txt"))
    parser.add_argument("--search-budget", type=int, default=120)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--games", type=int, default=2, help="sides-swapped pairs per team pairing")
    parser.add_argument("--pairings", type=int, default=0, help="0 for every pairing; otherwise a random sample")
    parser.add_argument("--out", type=Path, default=Path("ab_match.json"))
    args = parser.parse_args()

    trainers = load_trainers(args.teams_file)
    prior = SetPrior.from_teams([t.team for t in trainers])
    weights = load_weights(Path("champions/random-a-myopic-tuned.json"))
    profile = SearchProfile(budget=args.search_budget)

    pairs = list(combinations(trainers, 2))
    if args.pairings:
        pairs = random.Random(0).sample(pairs, min(args.pairings, len(pairs)))
    progress = Progress(total=len(pairs))
    wins = losses = draws = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(play, a, b, weights, profile, prior, idx, args.games) for idx, (a, b) in enumerate(pairs)
        ]
        for future in as_completed(futures):
            w, l, d = future.result()
            wins, losses, draws = wins + w, losses + l, draws + d
            progress.tick()
    print()

    decided = wins + losses
    rate = wins / decided if decided else 0.0
    # Standard error on a proportion; each game is one Bernoulli trial.
    stderr = math.sqrt(rate * (1 - rate) / decided) if decided else 0.0
    print(f"AFTER  {AFTER}")
    print(f"BEFORE {BEFORE}\n")
    print(f"{wins}-{losses}-{draws} over {wins + losses + draws} games")
    print(f"AFTER wins {rate:.1%} of decided games  (+/- {stderr:.1%}, one standard error)")
    verdict = (
        "no measurable difference"
        if abs(rate - 0.5) < 2 * stderr
        else ("AFTER is better" if rate > 0.5 else "AFTER is WORSE")
    )
    print(f"verdict at two standard errors: {verdict}")
    args.out.write_text(
        json.dumps(
            {"after": AFTER, "before": BEFORE, "wins": wins, "losses": losses, "draws": draws, "rate": rate}, indent=2
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
