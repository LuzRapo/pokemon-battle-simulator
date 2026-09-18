"""One Elo per (team, pilot) pair, so a team can be rated twice: once as our search drives it and
once as the human-policy model does.

The round-robin rates a *trainer*, which silently assumes the pilot is a constant. It is not. Our
search finished a stall team at 11-19 and a hyper-offence team at 29-1, while on the real ladder the
same two compositions go 50.7% and 43.1% -- the ordering does not just differ in size, it inverts.
That is only legible if the pilot is part of what gets rated.

Entering each team twice, under both pilots, into one pool makes the comparison direct: if a team
rates higher in the weaker pilot's hands, its own driver is costing it more than the pilot's missing
lookahead is.

Ratings are within-pool. Half the entrants are human-policy-piloted and weaker than the live ladder,
so these numbers order teams against each other and are not on the roster's scale.

    uv run python -m tools.pilot_elo --games 4
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

from battle_sim.evolution import Progress, load_weights
from battle_sim.human_policy import HumanPolicyPlayer
from battle_sim.matchup import MatchupWeights
from battle_sim.observation import SetPrior
from battle_sim.runner import Player, run_battle
from battle_sim.search import SearchPlayer, SearchProfile
from battle_sim.trainer_db_tournament import Trainer, load_trainers
from battle_sim.trainer_elo import GameResult, compute_elo
from battle_sim.utils import Outcome

BOT = Path.home() / "Development" / "sir-meowfred" / "pokemon"
DEFAULT_TEAMS = [
    "Benjamin Netanyahu — Elite Four",
    "Greta Thunberg — Elite Four",
    "Ben Shapiro — Elite Four",
    "Donald Trump — Elite Four",
    "Hxrl — Legendary Trainer",
    "Sir Meowfred — Resident Cat Butler",
]


@dataclass(frozen=True)
class Entrant:
    team: Trainer
    pilot: str  # "search" | "human"

    @property
    def label(self) -> str:
        return f"{self.team.name.split('—')[0].strip()} [{self.pilot}]"


def _player(pilot: str, weights: MatchupWeights, profile: SearchProfile, seed: int) -> Player:
    if pilot == "search":
        return SearchPlayer(weights, profile=profile, seed=seed)
    return HumanPolicyPlayer(weights, seed=seed)


def play(  # noqa: PLR0913
    a: Entrant,
    b: Entrant,
    weights: MatchupWeights,
    profile: SearchProfile,
    prior: SetPrior,
    seed: int,
    games: int,
) -> list[GameResult]:
    """`games` games with the sides alternating, so neither entrant always moves first."""
    out: list[GameResult] = []
    for game in range(games):
        a_is_p1 = game % 2 == 0
        pa = _player(a.pilot, weights, profile, seed * 10 + game)
        pb = _player(b.pilot, weights, profile, seed * 10 + game + 5)
        if a_is_p1:
            result = run_battle(a.team.team, b.team.team, pa, pb, seed=seed * 10 + game, prior=prior)
        else:
            result = run_battle(b.team.team, a.team.team, pb, pa, seed=seed * 10 + game, prior=prior)
        if result.outcome is None or result.outcome is Outcome.DRAW:
            out.append(GameResult(a=a.label, b=b.label, winner=None))
        elif (result.outcome is Outcome.P1_WIN) == a_is_p1:
            out.append(GameResult(a=a.label, b=b.label, winner=a.label))
        else:
            out.append(GameResult(a=a.label, b=b.label, winner=b.label))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster", type=Path, default=BOT / "trainers_db.txt")
    parser.add_argument("--teams", nargs="+", default=DEFAULT_TEAMS)
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--search-budget", type=int, default=120)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--out", type=Path, default=Path("pilot_elo.json"))
    args = parser.parse_args()

    trainers = {t.name: t for t in load_trainers(args.roster)}
    entrants = [Entrant(trainers[name], pilot) for name in args.teams for pilot in ("search", "human")]
    weights = load_weights(Path("champions/random-a-myopic-tuned.json"))
    prior = SetPrior.from_teams([t.team for t in trainers.values()])
    profile = SearchProfile(budget=args.search_budget)

    pairs = list(combinations(entrants, 2))
    print(
        f"{len(entrants)} entrants ({len(args.teams)} teams x 2 pilots), {len(pairs)} pairings, {args.games} games each"
    )
    progress = Progress(total=len(pairs))
    games: list[GameResult] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(play, a, b, weights, profile, prior, idx, args.games) for idx, (a, b) in enumerate(pairs)
        ]
        for future in as_completed(futures):
            games.extend(future.result())
            progress.tick()
    print()

    ratings, passes = compute_elo([e.label for e in entrants], games)
    print(f"converged in {passes} passes over {len(games)} games\n")
    print(f"{'team':<22} {'our search':>12} {'human policy':>14}   {'search edge':>12}")
    for name in args.teams:
        short = name.split("—")[0].strip()
        search, human = ratings[f"{short} [search]"], ratings[f"{short} [human]"]
        flag = "   <-- the human policy drives it better" if human > search else ""
        print(f"{short:<22} {search:>12.1f} {human:>14.1f}   {search - human:>+12.1f}{flag}")
    args.out.write_text(json.dumps({"ratings": ratings, "games": len(games)}, indent=2) + "\n")


if __name__ == "__main__":
    main()
