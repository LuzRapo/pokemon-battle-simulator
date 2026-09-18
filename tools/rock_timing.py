"""When do people actually set hazards, and does the answer change with ladder rating?

Knowing that humans spend more turns on hazards than our AI does not say *which* turns. If strong
players put rocks down in a narrow, recognisable window -- on the lead, before the first trade,
off a free turn -- then that is a rule the scorer can be judged against, and a per-turn rate is
too coarse to see it.

Reads `|turn|` markers for timing, the `|player|` lines for rating, and treats a game as "rocks up"
from the first successful Stealth Rock in it.

    uv run python -m tools.rock_timing --top 1800
"""

import argparse
import random
import statistics
from collections import Counter
from pathlib import Path

from battle_sim.evolution import load_weights
from battle_sim.observation import SetPrior
from battle_sim.runner import run_battle
from battle_sim.search import SearchPlayer, SearchProfile
from battle_sim.trainer_db_tournament import load_trainers

LOGS = Path("data/ag_replays/logs")
SETTERS = {"Stealth Rock", "Spikes", "Toxic Spikes", "Sticky Web"}


def scan(path: Path) -> tuple[int | None, int | None, int, bool] | None:
    """(rating, turn rocks went up, total turns, set by the lead) for one replay."""
    ratings: list[int] = []
    turn = 0
    rocks_turn: int | None = None
    lead_set = False
    leads: dict[str, str] = {}
    started = False
    for line in path.read_text(errors="replace").splitlines():
        parts = line.split("|")
        if len(parts) < 2:
            continue
        tag = parts[1]
        if tag == "start":  # a bare `|start` has no third field, so this is checked before length
            started = True
            continue
        if len(parts) < 3:
            continue
        if tag == "player" and len(parts) > 5 and parts[5].strip().isdigit():
            ratings.append(int(parts[5]))
        elif tag == "turn":
            turn = int(parts[2])
        elif tag == "switch" and started:
            slot = parts[2].split(":")[0]
            leads.setdefault(slot, parts[2])  # first thing each side sends out
        elif tag == "move" and parts[3] == "Stealth Rock" and rocks_turn is None:
            rocks_turn = turn
            slot = parts[2].split(":")[0]
            lead_set = leads.get(slot) == parts[2]
    if not started:
        return None
    return (min(ratings) if ratings else None, rocks_turn, turn, lead_set)


class _RockWatcher(SearchPlayer):
    """Notes the turn this side first commits to Stealth Rock, if it ever does."""

    def __init__(self, *args, log: dict, key: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._log = log
        self._key = key

    def choose_action(self, state, side_index, actions):  # type: ignore[no-untyped-def]
        action = super().choose_action(state, side_index, actions)
        if action.move is not None and self._key not in self._log:
            move = state.sides[side_index].active_pokemon.moves[action.move]
            if move is not None and move.name == "Stealth Rock":
                self._log[self._key] = state.turn
        return action


def ai_timing(teams_file: Path, games: int, budget: int) -> tuple[float, list[int]]:
    """Share of games our AI gets rocks up in, and the turns it does it on."""
    trainers = load_trainers(teams_file)
    weights = load_weights(Path("champions/random-a-myopic-tuned.json"))
    prior = SetPrior.from_teams([t.team for t in trainers])
    profile = SearchProfile(budget=budget)
    rng = random.Random(0)
    sides_with_rocks, turns, sides_total = 0, [], 0
    for game in range(games):
        a, b = rng.sample(trainers, 2)
        log: dict[int, int] = {}
        run_battle(
            a.team,
            b.team,
            _RockWatcher(weights, profile=profile, log=log, key=0),
            _RockWatcher(weights, profile=profile, log=log, key=1),
            seed=game,
            prior=prior,
        )
        # Only sides that actually carry the move can be judged on whether they used it.
        for key, team in ((0, a.team), (1, b.team)):
            if any("Stealth Rock" in [str(m) for m in spec.moves] for spec in team):
                sides_total += 1
                if key in log:
                    sides_with_rocks += 1
                    turns.append(log[key])
    return (sides_with_rocks / sides_total if sides_total else 0.0), turns


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=1800, help="rating floor for the 'strong' split")
    parser.add_argument("--limit", type=int, default=5001)
    parser.add_argument("--ai-games", type=int, default=0, help="also play this many games and compare")
    parser.add_argument("--teams-file", type=Path, default=Path("ag_teams.txt"))
    parser.add_argument("--budget", type=int, default=120)
    args = parser.parse_args()

    rows = [r for path in sorted(LOGS.glob("*.log"))[: args.limit] if (r := scan(path)) is not None]
    print(f"{len(rows)} replays\n")

    for label, subset in (
        ("all games", rows),
        (f"both players >= {args.top}", [r for r in rows if r[0] is not None and r[0] >= args.top]),
        (f"both players < {args.top}", [r for r in rows if r[0] is not None and r[0] < args.top]),
    ):
        if not subset:
            continue
        withrocks = [r for r in subset if r[1] is not None]
        turns = [r[1] for r in withrocks]
        share = len(withrocks) / len(subset)
        lead = sum(1 for r in withrocks if r[3]) / len(withrocks) if withrocks else 0.0
        print(f"{label}  ({len(subset)} games)")
        print(f"  rocks go up in            {share:>6.1%} of games")
        if turns:
            print(f"  median turn they go up    {statistics.median(turns):>6.0f}")
            spread = Counter(min(t, 11) for t in turns)
            for bucket in range(1, 12):
                n = spread.get(bucket, 0)
                label_t = "11+" if bucket == 11 else str(bucket)
                bar = "#" * round(40 * n / len(turns))
                print(f"    turn {label_t:>3}  {n / len(turns):>5.1%} {bar}")
            print(f"  set by that side's lead   {lead:>6.1%}")
        print()
    if args.ai_games:
        _report_ai(args)


def _report_ai(args) -> None:  # type: ignore[no-untyped-def]
    share, turns = ai_timing(args.teams_file, args.ai_games, args.budget)
    print(f"our AI  ({args.ai_games} games, only sides actually carrying Stealth Rock)")
    print(f"  rocks go up in            {share:>6.1%} of those sides' games")
    if turns:
        print(f"  median turn they go up    {statistics.median(turns):>6.0f}")


if __name__ == "__main__":
    main()
