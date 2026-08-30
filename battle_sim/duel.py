"""Head-to-head A/B between two configurations, at the resolution needed to actually decide.

Measuring a change against a shared opponent pool wastes most of its statistical power: the
pool-margin design landed at +/-0.05 where the same compute spent on same-seed side pairs lands
near +/-0.011. Every pair runs the identical teams and the identical roll stream twice with the
sides swapped, so team strength and luck cancel exactly and what is left is the configurations.

Both belief models live in one battle, one per side, because a battle has a single observer
serving both players — so a belief change is dueled by giving side A a posterior over `--a-belief`
candidate sets and side B one over `--b-belief`, rather than by running two separate populations.

    uv run python -m battle_sim.duel --pairs 240 --a-belief 12 --b-belief 1
    uv run python -m battle_sim.duel --pairs 240 --a-budget 512 --b-budget 30
"""

import argparse
import math
import os
import random
import statistics
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from battle_sim.evolution import Progress, Team, load_teams, load_weights
from battle_sim.matchup import MatchupPlayer, MatchupWeights
from battle_sim.observation import SetPrior
from battle_sim.runner import Player, run_battle
from battle_sim.search import SearchPlayer, SearchProfile

_MARGIN_SPAN = 12  # survivor differential runs -6..+6; a margin is 0.5 + difference / span


@dataclass(frozen=True)
class Side:
    """One duellist: a genome, how far it searches, and how many candidate sets it believes."""

    weights: MatchupWeights
    budget: int
    belief: int
    determinizations: int = 1
    switch_tax: float = 0.15
    exploit_p: float = 0.0
    predict_temperature: float = 1.0
    genome_prior: float = 0.5
    opponent_model: MatchupWeights | None = None

    def __str__(self) -> str:
        search = "myopic" if self.budget == 0 else f"budget {self.budget}"
        return f"{search}, {self.belief} believed sets, {self.determinizations} worlds, exploit {self.exploit_p:.2f}"

    def player(self) -> Player:
        if self.budget == 0:
            return MatchupPlayer(self.weights)
        profile = SearchProfile(
            budget=self.budget,
            determinizations=self.determinizations,
            switch_tax=self.switch_tax,
            exploit_p=self.exploit_p,
            predict_temperature=self.predict_temperature,
            genome_prior=self.genome_prior,
        )
        return SearchPlayer(self.weights, profile=profile, opponent_model=self.opponent_model)


@dataclass(frozen=True)
class Pairing:
    """One team pairing played twice, with A on each side of the identical roll stream."""

    team_a: Team
    team_b: Team
    seed: int


@dataclass(frozen=True)
class Verdict:
    pairs: int
    margin: float
    ci: float  # 95% half-width over pair means

    @property
    def sigma(self) -> float:
        """Standard deviations between the measured margin and break-even."""
        return abs(self.margin - 0.5) / (self.ci / 1.96) if self.ci else 0.0


def sample_pairings(teams: Sequence[Team], count: int, rng: random.Random) -> list[Pairing]:
    return [Pairing(rng.choice(teams), rng.choice(teams), rng.randrange(2**32)) for _ in range(count)]


def pair_margin(side_a: Side, side_b: Side, pairing: Pairing, prior: SetPrior) -> float:
    """A's mean margin over the pairing's two battles, playing each side of it once."""
    margins = []
    for a_is_first in (True, False):
        first, second = (side_a, side_b) if a_is_first else (side_b, side_a)
        result = run_battle(
            pairing.team_a,
            pairing.team_b,
            first.player(),
            second.player(),
            seed=pairing.seed,
            prior=prior,
            belief_candidates=(first.belief, second.belief),
        )
        alive_first, alive_second = result.survivors
        difference = alive_first - alive_second if a_is_first else alive_second - alive_first
        margins.append(0.5 + difference / _MARGIN_SPAN)
    return statistics.mean(margins)


def duel(
    side_a: Side, side_b: Side, pairings: Sequence[Pairing], prior: SetPrior, workers: int, progress: Progress
) -> Verdict:
    if workers == 1:
        margins = []
        for pairing in pairings:
            margins.append(pair_margin(side_a, side_b, pairing, prior))
            progress.tick()
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(pair_margin, side_a, side_b, pairing, prior) for pairing in pairings]
            for _ in as_completed(futures):
                progress.tick()
            margins = [future.result() for future in futures]
    return Verdict(
        pairs=len(margins),
        margin=statistics.mean(margins),
        ci=1.96 * statistics.stdev(margins) / math.sqrt(len(margins)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teams-dir", type=Path, default=Path("random_teams"))
    parser.add_argument("--pairs", type=int, default=240, help="team pairings; each plays 2 battles")
    parser.add_argument("--a-champion", type=Path, default=None)
    parser.add_argument("--b-champion", type=Path, default=None)
    parser.add_argument("--a-budget", type=int, default=0, help="0 plays the genome myopically, without search")
    parser.add_argument("--b-budget", type=int, default=0)
    parser.add_argument("--a-belief", type=int, default=12, help="candidate sets averaged into A's threat reads")
    parser.add_argument("--b-belief", type=int, default=12)
    parser.add_argument("--a-worlds", type=int, default=1, help="belief worlds A's search hedges each decision over")
    parser.add_argument("--b-worlds", type=int, default=1)
    parser.add_argument("--a-tax", type=float, default=0.15, help="what a switch must earn for A")
    parser.add_argument("--b-tax", type=float, default=0.15)
    tiebreak_help = "how much of the genome ranking A blends into its searched payoffs"
    parser.add_argument("--a-tiebreak", type=float, default=0.5, help=tiebreak_help)
    parser.add_argument("--b-tiebreak", type=float, default=0.5)
    parser.add_argument("--a-exploit", type=float, default=0.0, help="A's weight on the predicted opponent policy")
    parser.add_argument("--b-exploit", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0, help="sharpness of both sides' opponent prediction")
    parser.add_argument("--opponent-model", type=Path, default=None, help="genome both sides predict the other with")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    teams = load_teams(args.teams_dir)
    prior = SetPrior.from_teams(teams)
    model = None if args.opponent_model is None else load_weights(args.opponent_model)
    side_a = Side(
        weights=_weights(args.a_champion),
        budget=args.a_budget,
        belief=args.a_belief,
        determinizations=args.a_worlds,
        switch_tax=args.a_tax,
        exploit_p=args.a_exploit,
        predict_temperature=args.temperature,
        genome_prior=args.a_tiebreak,
        opponent_model=model,
    )
    side_b = Side(
        weights=_weights(args.b_champion),
        budget=args.b_budget,
        belief=args.b_belief,
        determinizations=args.b_worlds,
        switch_tax=args.b_tax,
        exploit_p=args.b_exploit,
        predict_temperature=args.temperature,
        genome_prior=args.b_tiebreak,
        opponent_model=model,
    )
    logger.info(f"{len(teams)} teams | A {side_a} | B {side_b} | {args.workers} workers")

    pairings = sample_pairings(teams, args.pairs, random.Random(args.seed))
    progress = Progress(total=len(pairings))
    verdict = duel(side_a, side_b, pairings, prior, args.workers, progress)
    print()
    print(f"A margin vs B: {verdict.margin:.4f} +/- {verdict.ci:.4f} over {verdict.pairs} pairs")
    print(f"that is {verdict.sigma:.1f} sigma from break-even")


def _weights(path: Path | None) -> MatchupWeights:
    return MatchupWeights() if path is None else load_weights(path)


if __name__ == "__main__":
    main()
