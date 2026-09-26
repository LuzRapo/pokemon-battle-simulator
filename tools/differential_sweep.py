"""Play many battles through both engines and report where they part company.

The pytest suite runs a couple of dozen seeds so it stays fast enough to run on every change. This
runs hundreds or thousands, which is how the rarer disagreements get found — crash damage showed up
once in fifty battles, and the fainted-target secondary once in twelve. It is the loop the rest of
the port is done in: widen the slice by one feature, sweep, fix what it names, sweep again.

    uv run python tools/differential_sweep.py --battles 500
    uv run python tools/differential_sweep.py --battles 500 --slice status --switches

Three outcomes, kept strictly apart:

  agreed     both engines played the same battle, turn for turn
  unported   the Rust engine refused, naming something it has not learned (exit 2)
  DIVERGED   the two produced different battles, or asked the tape for different things (exit 3)

Only the third is a bug. Counting a divergence as a refusal is the one failure this whole apparatus
exists to prevent, so they travel on different exit codes rather than on a parsed message.
"""

import argparse
import json
import random
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from battle_sim.differential import Scenario, compare, record  # noqa: E402
from battle_sim.engine import legal_actions  # noqa: E402
from battle_sim.models.actions import ActionType  # noqa: E402
from tests.test_rust_battles import BINARY, DATA, PLAIN_MOVES, STATUS_MOVES, _team  # noqa: E402

UNPORTED, DIVERGED = 2, 3
SLICES = {"plain": PLAIN_MOVES, "status": STATUS_MOVES}


@dataclass
class Result:
    agreed: int = 0
    unported: Counter = None  # type: ignore[assignment]
    diverged: list[tuple[int, str]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.unported = Counter()
        self.diverged = []


def _teams_rng(which: str, seed: int) -> random.Random:
    """The generator that picks the teams and the choices, seeded reproducibly.

    A string seed, deliberately: `hash()` of a str is salted per process, so seeding from one meant
    `--explain 81` built a different battle from the `seed 81` the sweep had just reported. The
    whole value of writing down a seed is that somebody can come back to it later.
    """
    return random.Random(f"{which}:{seed}")


def _chooser(rng: random.Random, switches: bool):  # type: ignore[no-untyped-def]
    def choose(state, side_index):  # type: ignore[no-untyped-def]
        options = legal_actions(state, side_index)
        moves = [a for a in options if a.action is ActionType.USE_MOVE]
        # A switch every so often, because a switch resets stat stages, volatiles and the toxic
        # counter — three things that only ever disagree once something has been switched out.
        if switches and moves and rng.random() < 0.15 and len(options) > len(moves):
            return rng.choice([a for a in options if a.action is not ActionType.USE_MOVE])
        return rng.choice(moves or options)

    return choose


def sweep(battles: int, which: str, team_size: int, switches: bool, max_turns: int, quiet: bool) -> Result:
    pool = SLICES[which]
    result = Result()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "scenario.json"
        for seed in range(battles):
            rng = _teams_rng(which, seed)
            teams = (_team(rng, size=team_size, pool=pool), _team(rng, size=team_size, pool=pool))
            scenario, expected = record(teams, _chooser(rng, switches), seed=seed, max_turns=max_turns)
            path.write_text(scenario.to_json())
            run = subprocess.run([str(BINARY), str(path), str(DATA)], capture_output=True, text=True)

            if run.returncode == UNPORTED:
                result.unported[_reason(run.stderr)] += 1
            elif run.returncode == DIVERGED:
                result.diverged.append((seed, run.stderr.strip()))
                _keep(scenario, seed, which, quiet)
            elif run.returncode != 0:
                result.diverged.append((seed, f"replay crashed ({run.returncode}): {run.stderr.strip()}"))
            else:
                divergence = compare(expected, json.loads(run.stdout))
                if divergence is None:
                    result.agreed += 1
                else:
                    result.diverged.append((seed, str(divergence)))
                    _keep(scenario, seed, which, quiet)
    return result


def _reason(stderr: str) -> str:
    """The refusal with the move name stripped, so the tally groups by cause rather than by move."""
    words = stderr.strip().split()
    return " ".join(words[1:]) if len(words) > 1 else stderr.strip()


def _show_board(state: dict) -> None:
    """Who was standing going into the failing turn, since most divergences are about one of them."""
    for number, side in enumerate(state["sides"]):
        for mon in side["team"]:
            if mon["fainted"]:
                continue
            stages = {stat: value for stat, value in mon["stages"].items() if value}
            print(
                f"  side {number}: {mon['nickname']} {mon['species']} hp={mon['hp']} "
                f"status={mon['status']}/{mon['status_turns']} stages={stages}"
            )


def _keep(scenario: Scenario, seed: int, which: str, quiet: bool) -> None:
    """A failing scenario is the whole reproduction, so it is written out rather than described."""
    if quiet:
        return
    out = Path("rust/failures")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{which}-{seed}.json").write_text(scenario.to_json())


def explain(battles: int, which: str, team_size: int, switches: bool, max_turns: int, seed: int) -> int:
    """Re-run one seed and show the first turn the two engines describe differently.

    The tape message says *where* the engines parted, which is almost never *why*: by the time one
    of them asks for the wrong kind of draw it has usually been out of step for several turns. So
    this replays growing prefixes, finds the earliest turn whose events differ, and prints both
    accounts of it side by side. That turn is the bug; everything after it is consequence.
    """
    rng = _teams_rng(which, seed)
    teams = (_team(rng, size=team_size, pool=SLICES[which]), _team(rng, size=team_size, pool=SLICES[which]))
    scenario, expected = record(teams, _chooser(rng, switches), seed=seed, max_turns=max_turns)

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "scenario.json"
        theirs: list[dict] = []
        for length in range(1, len(scenario.actions) + 1):
            prefix = Scenario(teams=scenario.teams, actions=scenario.actions[:length], tape=scenario.tape, seed=seed)
            path.write_text(prefix.to_json())
            run = subprocess.run([str(BINARY), str(path), str(DATA)], capture_output=True, text=True)
            if run.returncode != 0:
                print(f"the Rust engine stopped at turn {length - 1} (exit {run.returncode}): {run.stderr.strip()}")
                break
            theirs = json.loads(run.stdout)

    # The full comparator, not an event diff: a turn can agree entry for entry and still leave the
    # two engines holding different states — a sleep counter, a toxic counter, a stat stage nobody
    # logged. Those are the ones that surface three turns later as a tape mismatch, so the state
    # digest has to be part of finding the *first* disagreement rather than the first visible one.
    divergence = compare(expected[: len(theirs)], theirs)
    if divergence is not None:
        index = divergence.turn - 1  # Divergence numbers turns from one
        print(f"\nfirst disagreement, turn {index}: {expected[index]['actions']}\n  {divergence}")
        for label, events in (("python", expected[index]["events"]), ("rust  ", theirs[index]["events"])):
            for event in events:
                print(f"  {label}: {json.dumps(event, sort_keys=True)}")
        return 1
    if len(theirs) < len(expected):
        index = len(theirs)
        turn = expected[index]
        before = expected[index - 1]["drawn"] if index else 0
        print(f"\nagreed through turn {index - 1}; the Python's next turn was {turn['actions']}")
        print(f"  it took draws {before}..{turn['drawn'] - 1}: {scenario.tape[before:turn['drawn']]}")
        for event in turn["events"]:
            print(f"  python: {json.dumps(event, sort_keys=True)}")
        if index:
            _show_board(expected[index - 1]["state"])
        return 1
    print("no disagreement found for this seed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--battles", type=int, default=200)
    parser.add_argument("--slice", dest="which", choices=sorted(SLICES), default="status")
    parser.add_argument("--team-size", type=int, default=3)
    parser.add_argument("--max-turns", type=int, default=60)
    parser.add_argument("--switches", action="store_true", help="let either side switch sometimes")
    parser.add_argument("--quiet", action="store_true", help="do not write failing scenarios to rust/failures/")
    parser.add_argument("--explain", type=int, metavar="SEED", help="show the first turn one seed disagrees on")
    args = parser.parse_args()

    if not BINARY.exists():
        print(f"no binary at {BINARY}; run `cargo build --release` in rust/", file=sys.stderr)
        return 1

    if args.explain is not None:
        return explain(args.battles, args.which, args.team_size, args.switches, args.max_turns, args.explain)

    result = sweep(args.battles, args.which, args.team_size, args.switches, args.max_turns, args.quiet)
    total = args.battles
    refused = sum(result.unported.values())
    print(f"{result.agreed}/{total} agreed  |  {refused} unported  |  {len(result.diverged)} DIVERGED")
    if result.unported:
        print("\nrefused, by cause:")
        for reason, count in result.unported.most_common(12):
            print(f"  {count:5d}  {reason}")
    if result.diverged:
        print("\ndivergences (scenarios written to rust/failures/):")
        for seed, why in result.diverged[:15]:
            print(f"  seed {seed}: {why.splitlines()[0]}")
    return 1 if result.diverged else 0


if __name__ == "__main__":
    raise SystemExit(main())
