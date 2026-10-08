"""Play many battles through both engines and report where they part company."""

import argparse
import json
import random
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from battle_sim.differential import Chooser, Scenario, compare, record
from battle_sim.engine import legal_actions
from battle_sim.mechanics.battle import BattleState
from battle_sim.models.actions import Action, ActionType
from tests.test_rust_battles import BINARY, DATA, PLAIN_MOVES, STATUS_MOVES, _team

UNPORTED, DIVERGED = 2, 3
SLICES = {"plain": PLAIN_MOVES, "status": STATUS_MOVES}


@dataclass
class Result:
    agreed: int = 0
    unported: Counter[str] = field(default_factory=Counter)
    diverged: list[tuple[int, str]] = field(default_factory=list)


def _teams_rng(which: str, seed: int) -> random.Random:
    """The generator that picks the teams and the choices, seeded reproducibly."""
    return random.Random(f"{which}:{seed}")


def _chooser(rng: random.Random, switches: bool) -> Chooser:
    def choose(state: BattleState, side_index: int) -> Action:
        options = legal_actions(state, side_index)
        moves = [a for a in options if a.action is ActionType.USE_MOVE]
        # A switch every so often, since switching resets stages, volatiles and the toxic counter.
        if switches and moves and rng.random() < 0.15 and len(options) > len(moves):
            return rng.choice([a for a in options if a.action is not ActionType.USE_MOVE])
        return rng.choice(moves or options)

    return choose


def sweep(
    battles: int, which: str, team_size: int, switches: bool, max_turns: int, quiet: bool, abilities: bool
) -> Result:
    pool = SLICES[which]
    result = Result()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "scenario.json"
        for seed in range(battles):
            rng = _teams_rng(which, seed)
            teams = (
                _team(rng, size=team_size, pool=pool, abilities=abilities, items=abilities),
                _team(rng, size=team_size, pool=pool, abilities=abilities, items=abilities),
            )
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


def _show_board(state: dict[str, Any]) -> None:
    """Who was standing going into the failing turn, since most divergences are about one of them."""
    for number, side in enumerate(state["sides"]):
        for mon in side["team"]:
            if mon["fainted"]:
                continue
            stages = {stat: value for stat, value in mon["stages"].items() if value}
            logger.info(
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


def explain(
    battles: int, which: str, team_size: int, switches: bool, max_turns: int, seed: int, abilities: bool
) -> int:
    """Re-run one seed and show the first turn the two engines describe differently."""
    rng = _teams_rng(which, seed)
    teams = (
        _team(rng, size=team_size, pool=SLICES[which], abilities=abilities, items=abilities),
        _team(rng, size=team_size, pool=SLICES[which], abilities=abilities, items=abilities),
    )
    scenario, expected = record(teams, _chooser(rng, switches), seed=seed, max_turns=max_turns)

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "scenario.json"
        theirs: list[dict[str, Any]] = []
        for length in range(1, len(scenario.actions) + 1):
            prefix = Scenario(teams=scenario.teams, actions=scenario.actions[:length], tape=scenario.tape, seed=seed)
            path.write_text(prefix.to_json())
            run = subprocess.run([str(BINARY), str(path), str(DATA)], capture_output=True, text=True)
            if run.returncode != 0:
                logger.info(
                    f"the Rust engine stopped at turn {length - 1} (exit {run.returncode}): {run.stderr.strip()}"
                )
                break
            theirs = json.loads(run.stdout)

    # The full comparator, not an event diff, so unlogged state differences surface too.
    divergence = compare(expected[: len(theirs)], theirs)
    if divergence is not None:
        index = divergence.turn - 1  # Divergence numbers turns from one
        logger.info(f"\nfirst disagreement, turn {index}: {expected[index]['actions']}\n  {divergence}")
        for label, events in (("python", expected[index]["events"]), ("rust  ", theirs[index]["events"])):
            for event in events:
                logger.info(f"  {label}: {json.dumps(event, sort_keys=True)}")
        return 1
    if len(theirs) < len(expected):
        index = len(theirs)
        turn = expected[index]
        before = expected[index - 1]["drawn"] if index else 0
        logger.info(f"\nagreed through turn {index - 1}; the Python's next turn was {turn['actions']}")
        logger.info(f"  it took draws {before}..{turn['drawn'] - 1}: {scenario.tape[before : turn['drawn']]}")
        for event in turn["events"]:
            logger.info(f"  python: {json.dumps(event, sort_keys=True)}")
        if index:
            _show_board(expected[index - 1]["state"])
        return 1
    logger.info("no disagreement found for this seed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--battles", type=int, default=200)
    parser.add_argument("--slice", dest="which", choices=sorted(SLICES), default="status")
    parser.add_argument("--team-size", type=int, default=3)
    parser.add_argument("--max-turns", type=int, default=60)
    parser.add_argument("--switches", action="store_true", help="let either side switch sometimes")
    parser.add_argument(
        "--abilities", action="store_true", help="give Pokemon the abilities and items the engine has ported"
    )
    parser.add_argument("--quiet", action="store_true", help="do not write failing scenarios to rust/failures/")
    parser.add_argument("--explain", type=int, metavar="SEED", help="show the first turn one seed disagrees on")
    args = parser.parse_args()

    if not BINARY.exists():
        logger.error(f"no binary at {BINARY}; run `cargo build --release` in rust/")
        return 1

    if args.explain is not None:
        return explain(
            args.battles, args.which, args.team_size, args.switches, args.max_turns, args.explain, args.abilities
        )

    result = sweep(args.battles, args.which, args.team_size, args.switches, args.max_turns, args.quiet, args.abilities)
    total = args.battles
    refused = sum(result.unported.values())
    logger.info(f"{result.agreed}/{total} agreed  |  {refused} unported  |  {len(result.diverged)} DIVERGED")
    if result.unported:
        logger.info("\nrefused, by cause:")
        for reason, count in result.unported.most_common(12):
            logger.info(f"  {count:5d}  {reason}")
    if result.diverged:
        logger.info("\ndivergences (scenarios written to rust/failures/):")
        for seed, why in result.diverged[:15]:
            logger.info(f"  seed {seed}: {why.splitlines()[0]}")
    return 1 if result.diverged else 0


if __name__ == "__main__":
    raise SystemExit(main())
