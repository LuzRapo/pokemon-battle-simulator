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

# The default comparison, kept from the session that built this: the hazard coefficient measured
# against 4,970 replays, versus the genome value it replaced. Any other gene can be named on the
# command line -- `--gene setup_concession --after 0.6` -- which is what makes this a gate rather
# than a script about one experiment.
AFTER = {"hazard_value": 3.0}
BEFORE = {"hazard_value": 0.594}


def play(
    a: Trainer,
    b: Trainer,
    profile: SearchProfile,
    prior: SetPrior,
    seed: int,
    games: int,
    after: tuple,
    before: tuple,
) -> tuple[int, int, int]:
    """(wins for AFTER, wins for BEFORE, draws) over `games` sides-swapped pairs.

    Each configuration is a (genome, position weights) pair, because a gene lives in one or the
    other: `MatchupWeights` ranks the actions inside one choice set, `PositionWeights` values the
    leaf. Varying the wrong struct is a no-op that looks exactly like "no measurable difference".
    """
    wins = losses = draws = 0
    for game in range(games):
        for after_is_p1 in (True, False):
            first, second = (after, before) if after_is_p1 else (before, after)
            p1 = SearchPlayer(first[0], profile=profile, position_weights=first[1])
            p2 = SearchPlayer(second[0], profile=profile, position_weights=second[1])
            result = run_battle(a.team, b.team, p1, p2, seed=seed * 10 + game, prior=prior)
            if result.outcome is None or result.outcome is Outcome.DRAW:
                draws += 1
            elif (result.outcome is Outcome.P1_WIN) == after_is_p1:
                wins += 1
            else:
                losses += 1
    return wins, losses, draws


def _configure(weights, genes: dict[str, float]) -> tuple:
    """One side of the comparison: the genome and the leaf weights, with `genes` applied to whichever
    of the two actually carries them."""
    genome = replace(weights, **{k: v for k, v in genes.items() if hasattr(weights, k)})
    position = PositionWeights.from_matchup(genome)
    return genome, replace(position, **{k: v for k, v in genes.items() if hasattr(position, k)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teams-file", type=Path, default=Path("ag_teams.txt"))
    parser.add_argument("--search-budget", type=int, default=120)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--games", type=int, default=2, help="sides-swapped pairs per team pairing")
    parser.add_argument("--pairings", type=int, default=0, help="0 for every pairing; otherwise a random sample")
    parser.add_argument("--out", type=Path, default=Path("ab_match.json"))
    parser.add_argument("--weights", type=Path, default=Path("champions/random-a-myopic-tuned.json"))
    parser.add_argument("--gene", help="one gene to vary; defaults to the built-in hazard comparison")
    parser.add_argument("--after", type=float, help="the value to test")
    parser.add_argument("--before", type=float, help="what to test it against (default: the genome's own)")
    args = parser.parse_args()

    trainers = load_trainers(args.teams_file)
    prior = SetPrior.from_teams([t.team for t in trainers])
    weights = load_weights(args.weights)
    profile = SearchProfile(budget=args.search_budget)
    after_genes, before_genes = AFTER, BEFORE
    if args.gene:
        if args.after is None:
            parser.error("--gene needs --after")
        # Default the control to whatever the genome already says, so "before" is the live AI rather
        # than a second number somebody has to remember to look up.
        holder = weights if hasattr(weights, args.gene) else PositionWeights.from_matchup(weights)
        if not hasattr(holder, args.gene):
            parser.error(f"no gene called {args.gene!r} on either the genome or the position weights")
        after_genes = {args.gene: args.after}
        before_genes = {args.gene: args.before if args.before is not None else getattr(holder, args.gene)}

    after_config, before_config = _configure(weights, after_genes), _configure(weights, before_genes)
    if after_config == before_config:
        parser.error("after and before are the same configuration; nothing to measure")

    pairs = list(combinations(trainers, 2))
    if args.pairings:
        pairs = random.Random(0).sample(pairs, min(args.pairings, len(pairs)))
    progress = Progress(total=len(pairs))
    wins = losses = draws = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(play, a, b, profile, prior, idx, args.games, after_config, before_config)
            for idx, (a, b) in enumerate(pairs)
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
    print(f"AFTER  {after_genes}")
    print(f"BEFORE {before_genes}\n")
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
            {
                "after": after_genes,
                "before": before_genes,
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "rate": rate,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
