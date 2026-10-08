"""How many turns ahead does the search actually see, at a given budget?"""

import argparse
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from loguru import logger

from battle_sim.evolution import load_weights
from battle_sim.matchup import MatchupWeights
from battle_sim.mechanics.battle import BattleState
from battle_sim.models.actions import Action
from battle_sim.observation import SetPrior
from battle_sim.runner import run_battle
from battle_sim.search import SearchPlayer, SearchProfile
from battle_sim.trainer_db_tournament import load_trainers


class _DepthProbe(SearchPlayer):
    """Records the deepest iteration that completed for each decision."""

    def __init__(self, weights: MatchupWeights, profile: SearchProfile, log: list[int]) -> None:
        super().__init__(weights, profile=profile)
        self._log = log
        self._deepest = 0

    def _view_strategy(
        self,
        view: BattleState,
        side_index: int,
        my_actions: list[Action],
        their_actions: list[Action],
        depth: int,
        scores: dict[int, float],
    ) -> list[float]:
        result = super()._view_strategy(view, side_index, my_actions, their_actions, depth, scores)
        # Reached only if the iteration did not raise _BudgetExhausted, so this depth completed.
        self._deepest = max(self._deepest, depth)
        return result

    def choose_action(self, state: BattleState, side_index: int, actions: Sequence[Action]) -> Action:
        self._deepest = 0
        action = super().choose_action(state, side_index, actions)
        self._log.append(self._deepest)
        return action


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teams-file", type=Path, default=Path("ag_teams.txt"))
    parser.add_argument("--budgets", type=int, nargs="+", default=[120, 1000])
    parser.add_argument("--a", default="AG01", help="team tag to pilot")
    parser.add_argument("--b", default="AG12", help="opposing team tag")
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument(
        "--max-turns",
        type=int,
        default=1000,
        help="stop each battle early; depth per decision does not need the game to finish",
    )
    args = parser.parse_args()

    trainers = {t.name.split()[0]: t for t in load_trainers(args.teams_file)}
    a, b = trainers[args.a], trainers[args.b]
    weights = load_weights(Path("champions/random-a-myopic-tuned.json"))
    prior = SetPrior.from_teams([t.team for t in trainers.values()])

    for budget in args.budgets:
        profile = SearchProfile(budget=budget)
        log_a: list[int] = []
        log_b: list[int] = []
        turns = []
        started = time.monotonic()
        for game in range(args.games):
            player_a = _DepthProbe(weights, profile=profile, log=log_a)
            player_b = _DepthProbe(weights, profile=profile, log=log_b)
            result = run_battle(a.team, b.team, player_a, player_b, seed=game, prior=prior, max_turns=args.max_turns)
            turns.append(result.turns)
        elapsed = time.monotonic() - started
        both = log_a + log_b
        spread = Counter(both)
        share = "  ".join(f"depth {d}: {spread[d] / len(both):.0%}" for d in sorted(spread))
        logger.info(
            f"budget {budget:>6}  {len(both):>4} decisions  mean depth {sum(both) / len(both):.2f}  "
            f"{elapsed:>6.1f}s  {share}"
        )


if __name__ == "__main__":
    main()
