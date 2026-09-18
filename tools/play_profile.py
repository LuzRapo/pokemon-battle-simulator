"""What humans do with these teams each turn, against what our AI does with them.

The Anything Goes round-robin said which composition our AI drives best, but not *how* it drives
them, and the replay corpus is five thousand games of people driving the same compositions. Both
sides of that comparison can be reduced to the same thing -- the mix of decisions taken per turn --
and a gap in that mix is a concrete, checkable claim about how the AI plays.

Human turns come from the logs. Our AI's come from playing the same teams against each other and
recording every choice. Categories are deliberately coarse (switch / setup / status / hazard /
recovery / phaze / attack), because that is the resolution the logs actually support: they never
reveal an EV spread, so anything finer would be inventing detail.

    uv run python -m tools.play_profile --games 40
"""

import argparse
import random
from collections import Counter
from pathlib import Path

from battle_sim.evolution import load_weights
from battle_sim.observation import SetPrior
from battle_sim.runner import run_battle
from battle_sim.search import SearchPlayer, SearchProfile
from battle_sim.trainer_db_tournament import load_trainers

REPLAYS = Path("data/ag_replays")

SETUP = {
    "Dragon Dance",
    "Swords Dance",
    "Calm Mind",
    "Nasty Plot",
    "Quiver Dance",
    "Bulk Up",
    "Shell Smash",
    "Rock Polish",
    "Agility",
    "Growth",
    "Work Up",
    "Coil",
    "Hone Claws",
    "Geomancy",
    "Tail Glow",
}
STATUS = {"Toxic", "Will-O-Wisp", "Thunder Wave", "Spore", "Sleep Powder", "Glare", "Hypnosis", "Yawn"}
HAZARD = {"Stealth Rock", "Spikes", "Toxic Spikes", "Sticky Web", "Defog", "Rapid Spin"}
RECOVERY = {"Recover", "Roost", "Soft-Boiled", "Slack Off", "Morning Sun", "Moonlight", "Synthesis", "Rest", "Wish"}
PHAZE = {"Whirlwind", "Roar", "Dragon Tail", "Circle Throw", "Haze", "Perish Song"}
CATEGORIES = ("switch", "setup", "status", "hazard", "recovery", "phaze", "attack")


def categorise(move: str | None) -> str:
    """A move's role, or `switch` when no move was used."""
    if move is None:
        return "switch"
    for name, bucket in (("setup", SETUP), ("status", STATUS), ("hazard", HAZARD), ("recovery", RECOVERY)):
        if move in bucket:
            return name
    return "phaze" if move in PHAZE else "attack"


def human_profile(limit: int) -> tuple[Counter, int]:
    """Decision mix across replay logs, and how many games it covers.

    A replacement sent in after a faint is not a decision to switch -- the player had no other
    option -- so those are dropped rather than counted as switching. Everything else on a turn is
    one side's one choice.
    """
    picks: Counter = Counter()
    games = 0
    for path in sorted(REPLAYS.glob("**/*.log"))[:limit]:
        fainted: set[str] = set()
        for line in path.read_text(errors="replace").splitlines():
            parts = line.split("|")
            if len(parts) < 3:
                continue
            tag = parts[1]
            if tag == "faint":
                fainted.add(parts[2].split(":")[0])
            elif tag == "move":
                picks[categorise(parts[3])] += 1
            elif tag in {"switch", "drag"}:
                side = parts[2].split(":")[0]
                if tag == "drag" or side in fainted:  # forced: dragged out, or replacing a corpse
                    fainted.discard(side)
                else:
                    picks["switch"] += 1
        games += 1
    return picks, games


class _Profiler(SearchPlayer):
    """Records the category of every action it chooses."""

    def __init__(self, *args, picks: Counter, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._picks = picks

    def choose_action(self, state, side_index, actions):  # type: ignore[no-untyped-def]
        action = super().choose_action(state, side_index, actions)
        side = state.sides[side_index]
        if side.needs_switch or side.active_pokemon.is_fainted():
            return action  # forced replacement, same exclusion the logs get
        move = None if action.move is None else side.active_pokemon.moves[action.move]
        self._picks[categorise(None if move is None else move.name)] += 1
        return action


def ai_profile(teams_file: Path, games: int, budget: int) -> tuple[Counter, int]:
    trainers = load_trainers(teams_file)
    weights = load_weights(Path("champions/random-a-myopic-tuned.json"))
    prior = SetPrior.from_teams([t.team for t in trainers])
    profile = SearchProfile(budget=budget)
    picks: Counter = Counter()
    rng = random.Random(0)
    for game in range(games):
        a, b = rng.sample(trainers, 2)
        run_battle(
            a.team,
            b.team,
            _Profiler(weights, profile=profile, picks=picks),
            _Profiler(weights, profile=profile, picks=picks),
            seed=game,
            prior=prior,
        )
    return picks, games


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teams-file", type=Path, default=Path("ag_teams.txt"))
    parser.add_argument("--replays", type=int, default=1500)
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--budget", type=int, default=120)
    args = parser.parse_args()

    human, human_games = human_profile(args.replays)
    ai, ai_games = ai_profile(args.teams_file, args.games, args.budget)
    human_total, ai_total = sum(human.values()), sum(ai.values())
    print(f"humans: {human_total:>6} decisions over {human_games} replays")
    print(f"our AI: {ai_total:>6} decisions over {ai_games} games at budget {args.budget}\n")
    print(f"{'category':<10} {'human':>8} {'AI':>8}   {'gap':>7}")
    print("-" * 38)
    for name in CATEGORIES:
        h, a = human[name] / human_total, ai[name] / ai_total
        print(f"{name:<10} {h:>7.1%} {a:>8.1%}   {a - h:>+7.1%}")


if __name__ == "__main__":
    main()
