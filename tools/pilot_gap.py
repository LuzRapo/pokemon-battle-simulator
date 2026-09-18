"""Is the stall team weak, or is our AI bad at driving it?

Two ways to be bad at stall look identical in a win-loss record, and the round-robin cannot tell
them apart: the Elite Four's stall team went 11-19 while its hyper-offence team went 29-1, but that
is equally consistent with "stall is a worse team" and with "our search cannot pilot stall".

Separating them needs the *same team* in two different hands. Our search and the human-policy model
each drive both teams against the same fixed opponents, and what matters is not either pilot's
absolute score -- the human policy has no lookahead at all and is weaker everywhere -- but how much
each one *drops* when handed stall instead of offence. A pilot-independent gap means the team is
weak; a gap that appears only for our search means the piloting is.

    uv run python -m tools.pilot_gap --games 3
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from battle_sim.evolution import Progress, load_weights
from battle_sim.human_policy import HumanPolicyPlayer
from battle_sim.observation import SetPrior
from battle_sim.runner import run_battle
from battle_sim.search import SearchPlayer, SearchProfile
from battle_sim.stall_player import StallPlayer
from battle_sim.trainer_db_tournament import Trainer, load_trainers
from battle_sim.utils import Outcome

BOT = Path.home() / "Development" / "sir-meowfred" / "pokemon"
STALL = "Benjamin Netanyahu — Elite Four"
OFFENCE = "Donald Trump — Elite Four"


def _pilot(kind: str, weights, profile: SearchProfile, seed: int):  # type: ignore[no-untyped-def]
    if kind == "search":
        return SearchPlayer(weights, profile=profile, seed=seed)
    if kind == "stall":
        return StallPlayer(weights, profile=profile, seed=seed)
    return HumanPolicyPlayer(weights, seed=seed)


def run_set(
    team: Trainer, opponent: Trainer, kind: str, weights, profile: SearchProfile, prior: SetPrior, seed: int, games: int
) -> tuple[int, int]:
    """(wins, decided) for `team` driven by `kind`, sides alternating."""
    wins = decided = 0
    for game in range(games):
        ours_is_p1 = game % 2 == 0
        mine = _pilot(kind, weights, profile, seed * 10 + game)
        theirs = SearchPlayer(weights, profile=profile)  # the opposition is always our search
        if ours_is_p1:
            result = run_battle(team.team, opponent.team, mine, theirs, seed=seed * 10 + game, prior=prior)
        else:
            result = run_battle(opponent.team, team.team, theirs, mine, seed=seed * 10 + game, prior=prior)
        if result.outcome is None or result.outcome is Outcome.DRAW:
            continue
        decided += 1
        wins += (result.outcome is Outcome.P1_WIN) == ours_is_p1
    return wins, decided


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster", type=Path, default=BOT / "trainers_db.txt")
    parser.add_argument("--ratings", type=Path, default=BOT / "trainer_elo.json")
    parser.add_argument("--opponents", type=int, default=8)
    parser.add_argument("--games", type=int, default=3)
    parser.add_argument("--search-budget", type=int, default=120)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    trainers = {t.name: t for t in load_trainers(args.roster)}
    known: dict[str, float] = json.loads(args.ratings.read_text())["ratings"]
    skip = {STALL, OFFENCE, "Sir Meowfred — Resident Cat Butler"}
    rated = sorted(((t, known[n]) for n, t in trainers.items() if n in known and n not in skip), key=lambda p: -p[1])[
        :45
    ]
    step = (len(rated) - 1) / (args.opponents - 1)
    opponents = [rated[round(i * step)][0] for i in range(args.opponents)]
    print(f"{len(opponents)} shared opponents, {args.games} games each, both teams, both pilots\n")

    weights = load_weights(Path("champions/random-a-myopic-tuned.json"))
    prior = SetPrior.from_teams([t.team for t in trainers.values()])
    profile = SearchProfile(budget=args.search_budget)
    jobs = [
        (label, kind, trainers[label], opponent, kind_index * 1000 + i)
        for label in (STALL, OFFENCE)
        for kind_index, kind in enumerate(("search", "human", "stall"))
        for i, opponent in enumerate(opponents)
    ]
    progress = Progress(total=len(jobs))
    tally: dict[tuple[str, str], list[int]] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_set, team, opponent, kind, weights, profile, prior, seed, args.games): (label, kind)
            for label, kind, team, opponent, seed in jobs
        }
        for future in as_completed(futures):
            key = futures[future]
            wins, decided = future.result()
            row = tally.setdefault(key, [0, 0])
            row[0] += wins
            row[1] += decided
            progress.tick()
    print()

    print(f"{'pilot':<18} {'stall team':>16} {'offence team':>16}   {'drop':>8}")
    for kind, shown in (("search", "our search"), ("human", "human policy"), ("stall", "stall specialist")):
        rates = {}
        for label in (STALL, OFFENCE):
            wins, decided = tally.get((label, kind), [0, 0])
            rates[label] = wins / decided if decided else 0.0
        drop = rates[OFFENCE] - rates[STALL]
        stall_w, stall_n = tally.get((STALL, kind), [0, 0])
        off_w, off_n = tally.get((OFFENCE, kind), [0, 0])
        print(
            f"{shown:<18} {rates[STALL]:>9.1%} ({stall_w}/{stall_n}) {rates[OFFENCE]:>9.1%} ({off_w}/{off_n})"
            f"   {drop:>+7.1%}"
        )
    print("\nA drop that appears for one pilot and not the other is a piloting problem, not a team problem.")


if __name__ == "__main__":
    main()
