"""How a trained network fares against the bot's own AI, in the Python engine the bot runs."""

import argparse
import math
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from battle_sim.ag_sets import mirror_team
from battle_sim.evolution import load_weights
from battle_sim.matchup import MatchupPlayer
from battle_sim.players import RandomPlayer
from battle_sim.rl.net_player import NetPlayer
from battle_sim.runner import Player, run_battle
from battle_sim.search import SearchPlayer, SearchProfile
from battle_sim.utils import Outcome

CHAMPION = Path(__file__).resolve().parents[2] / "sir-meowfred" / "pokemon" / "champion.json"
TIE_BAND = 0.10  # sir-meowfred's `battles.TIE_BAND`


@dataclass(frozen=True)
class Opponent:
    kind: str  # "search", "matchup" or "random"
    budget: int
    champion: Path

    def player(self, seed: int) -> Player:
        if self.kind == "random":
            return RandomPlayer(seed)
        genome = load_weights(self.champion)
        if self.kind == "matchup":
            return MatchupPlayer(genome)
        profile = SearchProfile(budget=self.budget, exploit_p=1.0, predict_temperature=0.3, tie_band=TIE_BAND)
        return SearchPlayer(genome, profile=profile, opponent_model=genome, seed=seed)


def score(outcome: Outcome | None, net_side: int) -> float:
    """1 a net win, 0 a loss, a half for a draw or the turn cap."""
    if outcome is Outcome.DRAW or outcome is None:
        return 0.5
    return 1.0 if (outcome is Outcome.P1_WIN) == (net_side == 0) else 0.0


def play_pair(model: Path, opponent: Opponent, pair: int, max_turns: int) -> tuple[float, float]:
    """One team and seed, the net on each side in turn."""
    team = mirror_team(random.Random(f"evaluate:{pair}"))
    results = []
    for net_side in (0, 1):
        net = NetPlayer(model, greedy=True, seed=pair)
        other = opponent.player(pair)
        players = (net, other) if net_side == 0 else (other, net)
        battle = run_battle(team, team, players[0], players[1], seed=pair, max_turns=max_turns)
        results.append(score(battle.outcome, net_side))
    return results[0], results[1]


def evaluate(model: Path, opponent: Opponent, pairs: int, workers: int, max_turns: int = 300) -> tuple[float, float]:
    """The net's score over `pairs` pairings, and its 95% interval's half-width."""
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(play_pair, model, opponent, pair, max_turns) for pair in range(pairs)]
        per_pair = [sum(future.result()) / 2 for future in futures]
    mean = sum(per_pair) / len(per_pair)
    spread = math.sqrt(sum((x - mean) ** 2 for x in per_pair) / max(len(per_pair) - 1, 1))
    return mean, 1.96 * spread / math.sqrt(len(per_pair))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path, help="policy.onnx, or a .pt checkpoint to export first")
    parser.add_argument("--opponent", choices=["search", "matchup", "random"], default="search")
    parser.add_argument(
        "--budget", type=int, default=1000, help="the search opponent's budget: the bot plays at 1000, rates at 30"
    )
    parser.add_argument("--pairs", type=int, default=50)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--champion", type=Path, default=CHAMPION)
    args = parser.parse_args()

    model: Path = args.model
    if model.suffix == ".pt":
        from training.export import export, load_checkpoint

        onnx = model.with_suffix(".onnx")
        export(load_checkpoint(model), onnx)
        model = onnx
    opponent = Opponent(args.opponent, args.budget, args.champion)
    mean, half_width = evaluate(model, opponent, args.pairs, args.workers)
    against = f"{args.opponent} (budget {args.budget})" if args.opponent == "search" else args.opponent
    logger.info(f"{model.name} vs {against}: {mean:.1%} ± {half_width:.1%} over {args.pairs} pairs")


if __name__ == "__main__":
    main()
