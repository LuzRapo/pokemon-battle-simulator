"""Elo ratings for every trainer in `trainers_db.txt`, from the same round-robin as the tournament."""

import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

from loguru import logger

from battle_sim.evolution import Progress, load_weights
from battle_sim.matchup import MatchupWeights
from battle_sim.observation import SetPrior
from battle_sim.runner import run_battle
from battle_sim.search import SearchPlayer, SearchProfile
from battle_sim.trainer_db_tournament import Trainer, load_trainers
from battle_sim.utils import Outcome

_GAMES_PER_SERIES = 3
_WINS_TO_CLINCH = 2
_DEFAULT_RATING = 1500.0
_K_FACTOR = 32.0
_CONVERGENCE_THRESHOLD = 0.5
_MAX_PASSES = 20000
_ANCHOR = "__anchor__"  # fixed-rating phantom opponent; see _anchor_games
_ANCHOR_GAMES_PER_PLAYER = 2  # virtual draws vs the anchor: keeps a perfect record from diverging


@dataclass(frozen=True)
class GameResult:
    a: str
    b: str
    winner: str | None  # None: draw (including a turn-cap timeout)
    margin: int = 0  # Pokemon the winner had left; 0 for a draw. See `_margin_multiplier`.


def play_series_games(
    a: Trainer,
    b: Trainer,
    weights: MatchupWeights,
    profile: SearchProfile,
    prior: SetPrior,
    series_seed: int,
    per_series: int = _GAMES_PER_SERIES,
    clinch: int = _WINS_TO_CLINCH,
    pilots: dict[str, str] | None = None,
) -> list[GameResult]:
    """Play every game out, since the ranking is fitted per game rather than per series."""
    games: list[GameResult] = []
    a_wins = b_wins = 0
    for game in range(1, per_series + 1):
        if a_wins >= clinch or b_wins >= clinch:
            break
        seed = series_seed * 10 + game
        player_a = _pilot_for(a.name, pilots, weights, profile, seed)
        player_b = _pilot_for(b.name, pilots, weights, profile, seed + 1)
        a_is_p1 = game % 2 == 1  # alternate who's "P1" each game so neither side always moves first
        if a_is_p1:
            result = run_battle(a.team, b.team, player_a, player_b, seed=seed, prior=prior)
        else:
            result = run_battle(b.team, a.team, player_b, player_a, seed=seed, prior=prior)
        # `survivors` is indexed by battle side, which alternates; the winner's count is the margin.
        won_as_p1 = result.outcome is Outcome.P1_WIN
        margin = result.survivors[0] if won_as_p1 else result.survivors[1]
        if result.outcome is None or result.outcome is Outcome.DRAW:
            games.append(GameResult(a=a.name, b=b.name, winner=None))
        elif won_as_p1 == a_is_p1:
            games.append(GameResult(a=a.name, b=b.name, winner=a.name, margin=margin))
            a_wins += 1
        else:
            games.append(GameResult(a=a.name, b=b.name, winner=b.name, margin=margin))
            b_wins += 1
    return games


def _pilot_for(
    name: str, pilots: dict[str, str] | None, weights: MatchupWeights, profile: SearchProfile, seed: int
) -> SearchPlayer:
    """The player driving this trainer. A named override lets one team be rated as it will be played."""
    if pilots and pilots.get(name) == "stall":
        from battle_sim.stall_player import StallPlayer

        return StallPlayer(weights, profile=profile, seed=seed)
    return SearchPlayer(weights, profile=profile, seed=seed)


_NEUTRAL_MARGIN = 3.0  # a win with half a team left moves ratings exactly as the old binary Elo did


def _margin_multiplier(margin: int, rating_diff: float) -> float:
    """How much harder a 6-0 should move ratings than a 6-5, without letting favourites run away."""
    if margin <= 0:
        return 1.0
    scaled = math.log1p(margin) / math.log1p(_NEUTRAL_MARGIN)
    return scaled * 2.2 / (2.2 + 0.001 * max(0.0, rating_diff))


def _game_weight(game: GameResult, rating_a: float, rating_b: float, use_margin: bool) -> float:
    """How much this one game is allowed to move the ratings."""
    if not use_margin or game.winner is None:
        return 1.0
    winner_lead = (rating_a - rating_b) if game.winner == game.a else (rating_b - rating_a)
    return _margin_multiplier(game.margin, winner_lead)


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))


def anchor_games(names: list[str], count: int = _ANCHOR_GAMES_PER_PLAYER) -> list[GameResult]:
    """A few virtual draws per player against a fixed-1500 phantom opponent."""
    return [GameResult(a=name, b=_ANCHOR, winner=None) for name in names for _ in range(count)]


def compute_elo(
    names: list[str],
    games: list[GameResult],
    k: float = _K_FACTOR,
    max_passes: int = _MAX_PASSES,
    threshold: float = _CONVERGENCE_THRESHOLD,
    fixed: dict[str, float] | None = None,
    use_margin: bool = False,
) -> tuple[dict[str, float], int]:
    """Converged Elo: replay the games with a shrinking step until no rating moves more than `threshold`."""
    if fixed and not set(names).isdisjoint(fixed):
        raise ValueError(f"names given as both floating and fixed: {sorted(set(names) & set(fixed))}")
    ratings = dict.fromkeys(names, _DEFAULT_RATING)
    ratings[_ANCHOR] = _DEFAULT_RATING
    if fixed:
        ratings.update(fixed)
    all_games = games + anchor_games(names)
    passes_run = 0
    for pass_number in range(1, max_passes + 1):
        passes_run = pass_number
        step = k / pass_number
        max_delta = 0.0
        for game in all_games:
            rating_a, rating_b = ratings[game.a], ratings[game.b]
            expected_a = expected_score(rating_a, rating_b)
            if game.winner is None:
                score_a = 0.5
            elif game.winner == game.a:
                score_a = 1.0
            else:
                score_a = 0.0
            delta = step * _game_weight(game, rating_a, rating_b, use_margin) * (score_a - expected_a)
            a_fixed = fixed is not None and game.a in fixed
            b_fixed = game.b == _ANCHOR or (fixed is not None and game.b in fixed)
            if not a_fixed:
                ratings[game.a] = rating_a + delta
            if not b_fixed:
                ratings[game.b] = rating_b - delta
            max_delta = max(max_delta, abs(delta) if not (a_fixed and b_fixed) else 0.0)
        if max_delta < threshold:
            break
    del ratings[_ANCHOR]
    return ratings, passes_run


def render_ratings(trainers: list[Trainer], ratings: dict[str, float], games_played: dict[str, int]) -> str:
    ordered = sorted(trainers, key=lambda t: ratings[t.name], reverse=True)
    header = f"{'#':>3} {'Trainer':<45} {'Gen':<6} {'Elo':>7} {'Games':>6}"
    lines = [header, "-" * len(header)]
    for rank, t in enumerate(ordered, start=1):
        gen = t.generation.split(" — ")[0].replace("Generation ", "Gen ")
        lines.append(f"{rank:>3} {t.name:<45} {gen:<6} {ratings[t.name]:>7.1f} {games_played[t.name]:>6}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Elo ratings from the trainer_db.txt round-robin.")
    parser.add_argument("--teams-file", type=Path, default=Path("trainers_db.txt"))
    parser.add_argument("--champion", type=Path, default=Path("champions/random-a-myopic-tuned.json"))
    parser.add_argument("--search-budget", type=int, default=30)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--games", type=int, default=_GAMES_PER_SERIES, help="games per pairing")
    parser.add_argument("--clinch", type=int, default=_WINS_TO_CLINCH, help=">= --games plays them all out")
    parser.add_argument("--k-factor", type=float, default=_K_FACTOR)
    parser.add_argument("--margin", action="store_true", help="weight updates by surviving Pokemon")
    parser.add_argument("--stall-pilot", nargs="*", default=[], help="trainers to drive with StallPlayer")
    parser.add_argument("--out", type=Path, default=Path("trainer_elo.txt"))
    parser.add_argument("--json-out", type=Path, default=Path("trainer_elo.json"))
    parser.add_argument(
        "--games-log",
        type=Path,
        default=None,
        help="JSONL of individual results, appended as pairings finish (default: alongside --json-out)",
    )
    args = parser.parse_args()

    trainers = load_trainers(args.teams_file)
    prior = SetPrior.from_teams([t.team for t in trainers])
    weights = load_weights(args.champion)
    profile = SearchProfile(budget=args.search_budget)

    pilots = dict.fromkeys(args.stall_pilot, "stall")
    missing = [name for name in pilots if name not in {t.name for t in trainers}]
    if missing:
        raise SystemExit(f"--stall-pilot names not on the roster: {missing}")
    if pilots:
        logger.info(f"stall specialist driving: {', '.join(pilots)}")
    pairs = list(combinations(trainers, 2))
    progress = Progress(total=len(pairs))
    all_games: list[GameResult] = []
    # Log each pairing as it finishes, so a long run leaves something behind before it ends.
    games_log = args.games_log or args.json_out.with_suffix(".games.jsonl")
    with ProcessPoolExecutor(max_workers=args.workers) as executor, games_log.open("w") as log:
        futures = {
            executor.submit(
                play_series_games,
                a,
                b,
                weights,
                profile,
                prior,
                idx + args.seed * len(pairs),
                args.games,
                args.clinch,
                pilots,
            ): idx
            for idx, (a, b) in enumerate(pairs)
        }
        by_idx: dict[int, list[GameResult]] = {}
        for future in as_completed(futures):
            idx = futures[future]
            by_idx[idx] = future.result()
            for game in by_idx[idx]:
                log.write(json.dumps({"pair": idx, "a": game.a, "b": game.b, "winner": game.winner}) + "\n")
            log.flush()
            progress.tick()
    logger.info("")
    for idx in range(len(pairs)):
        all_games.extend(by_idx[idx])

    games_played = dict.fromkeys((t.name for t in trainers), 0)
    for game in all_games:
        games_played[game.a] += 1
        games_played[game.b] += 1

    ratings, passes_run = compute_elo(
        [t.name for t in trainers],
        all_games,
        k=args.k_factor,
        max_passes=_MAX_PASSES,
        threshold=_CONVERGENCE_THRESHOLD,
        use_margin=args.margin,
    )
    logger.info(f"converged after {passes_run} passes over {len(all_games)} games")

    report = render_ratings(trainers, ratings, games_played)
    logger.info(report)
    args.out.write_text(
        f"Trainer Elo — {len(trainers)} trainers, {len(all_games)} individual games, "
        f"champion={args.champion.stem}, search-budget={args.search_budget}, k={args.k_factor}, "
        f"converged in {passes_run} passes\n\n{report}\n"
    )
    args.json_out.write_text(
        json.dumps(
            {
                "champion": args.champion.stem,
                "search_budget": args.search_budget,
                "k_factor": args.k_factor,
                "default_rating": _DEFAULT_RATING,
                "passes_to_converge": passes_run,
                "ratings": {name: round(rating, 2) for name, rating in ratings.items()},
                "games_played": games_played,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
