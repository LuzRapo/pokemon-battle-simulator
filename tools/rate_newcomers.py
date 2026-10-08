"""Rate a handful of new trainers against the existing ladder, without replaying the ladder."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from loguru import logger

from battle_sim.evolution import Progress, load_weights
from battle_sim.matchup import MatchupWeights
from battle_sim.observation import SetPrior
from battle_sim.runner import Player, run_battle
from battle_sim.search import SearchPlayer, SearchProfile
from battle_sim.stall_player import StallPlayer
from battle_sim.trainer_db_tournament import Trainer, load_trainers
from battle_sim.trainer_elo import GameResult, compute_elo
from battle_sim.utils import Outcome

BOT = Path.home() / "Development" / "sir-meowfred" / "pokemon"
PILOTS = ("search", "stall")


def _pilot(kind: str, weights: MatchupWeights, profile: SearchProfile, seed: int) -> Player:
    return (StallPlayer if kind == "stall" else SearchPlayer)(weights, profile=profile, seed=seed)


def play_piloted(
    a: Trainer,
    label: str,
    kind: str,
    b: Trainer,
    weights: MatchupWeights,
    profile: SearchProfile,
    prior: SetPrior,
    seed: int,
    games: int,
) -> list[GameResult]:
    """`a` driven by `kind` against `b` driven by the ordinary search, sides alternating."""
    out: list[GameResult] = []
    for game in range(games):
        a_is_p1 = game % 2 == 0
        mine = _pilot(kind, weights, profile, seed * 10 + game)
        theirs = SearchPlayer(weights, profile=profile)
        if a_is_p1:
            result = run_battle(a.team, b.team, mine, theirs, seed=seed * 10 + game, prior=prior)
        else:
            result = run_battle(b.team, a.team, theirs, mine, seed=seed * 10 + game, prior=prior)
        if result.outcome is None or result.outcome is Outcome.DRAW:
            out.append(GameResult(a=label, b=b.name, winner=None))
        elif (result.outcome is Outcome.P1_WIN) == a_is_p1:
            out.append(GameResult(a=label, b=b.name, winner=label))
        else:
            out.append(GameResult(a=label, b=b.name, winner=b.name))
    return out


def spread(pool: list[tuple[Trainer, float]], count: int) -> list[tuple[Trainer, float]]:
    """`count` opponents spanning the rated range, strongest first and evenly spaced."""
    ranked = sorted(pool, key=lambda pair: -pair[1])
    if count >= len(ranked):
        return ranked
    step = (len(ranked) - 1) / (count - 1)
    return [ranked[round(i * step)] for i in range(count)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster", type=Path, default=BOT / "trainers_db.txt")
    parser.add_argument("--ratings", type=Path, default=BOT / "trainer_elo.json")
    parser.add_argument("--new", nargs="+", default=None, help="names to rate; default: everyone unrated")
    parser.add_argument("--opponents", type=int, default=10)
    parser.add_argument("--games", type=int, default=3)
    parser.add_argument("--search-budget", type=int, default=120)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--top-only", type=int, default=45, help="draw opponents from this many strongest")
    parser.add_argument("--pilots", nargs="*", default=[], help="rate each name once per pilot, e.g. search stall")
    parser.add_argument("--exclude", nargs="*", default=[], help="names never to use as a fixed anchor")
    args = parser.parse_args()

    trainers = {t.name: t for t in load_trainers(args.roster)}
    known: dict[str, float] = json.loads(args.ratings.read_text())["ratings"]
    newcomers = args.new or [name for name in trainers if name not in known]
    missing = [n for n in newcomers if n not in trainers]
    if missing:
        raise SystemExit(f"not on the roster: {missing}")

    # The butler is excluded, since his shown rating is pinned and his team has changed.
    excluded = set(newcomers) | set(args.exclude) | {"Sir Meowfred — Resident Cat Butler"}
    rated = [(t, known[name]) for name, t in trainers.items() if name in known and name not in excluded]
    rated = sorted(rated, key=lambda pair: -pair[1])[: args.top_only]
    opponents = spread(rated, args.opponents)
    logger.info(f"rating {len(newcomers)} newcomers against {len(opponents)} fixed opponents")
    logger.info(f"  {', '.join(f'{t.name.split(chr(8212))[0].strip()} ({r:.0f})' for t, r in opponents)}\n")

    weights = load_weights(Path("champions/random-a-myopic-tuned.json"))
    prior = SetPrior.from_teams([t.team for t in trainers.values()])
    profile = SearchProfile(budget=args.search_budget)
    # One floating entrant per (name, pilot).
    pilots = args.pilots or ["search"]
    labelled = [(name, kind, f"{name} [{kind}]" if args.pilots else name) for name in newcomers for kind in pilots]
    jobs = [(trainers[name], label, kind, opponent) for name, kind, label in labelled for opponent, _ in opponents]
    progress = Progress(total=len(jobs))
    games: list[GameResult] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(play_piloted, a, label, kind, b, weights, profile, prior, idx, args.games)
            for idx, (a, label, kind, b) in enumerate(jobs)
        ]
        for future in as_completed(futures):
            games.extend(future.result())
            progress.tick()
    logger.info("")

    floating = [label for _, _, label in labelled]
    fixed = {opponent.name: rating for opponent, rating in opponents}
    ratings, passes = compute_elo(floating, games, fixed=fixed)
    logger.info(f"converged in {passes} passes over {len(games)} games\n")
    record: dict[str, list[int]] = {label: [0, 0, 0] for label in floating}
    for game in games:
        for name in (game.a, game.b):
            if name in record:
                record[name][0 if game.winner == name else (2 if game.winner is None else 1)] += 1
    for name in sorted(floating, key=lambda n: -ratings[n]):
        won, lost, drew = record[name]
        was = known.get(name.split(" [")[0])
        moved = f"  (placed at {was:.0f})" if was is not None else ""
        shown = name.split("—")[0].strip() + ("" if " [" not in name else " [" + name.split(" [")[1])
        logger.info(f"  {shown:<28} {ratings[name]:>7.1f}   {won}-{lost}-{drew}{moved}")


if __name__ == "__main__":
    main()
