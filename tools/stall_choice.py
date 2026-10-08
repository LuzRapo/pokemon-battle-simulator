"""Does more search depth make the AI *play* stall, or only think about it longer?"""

import argparse
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from loguru import logger

from battle_sim.engine.status_apply import status_cannot_land
from battle_sim.evolution import load_weights
from battle_sim.matchup import MatchupWeights
from battle_sim.mechanics.battle import BattleState
from battle_sim.models.actions import Action
from battle_sim.observation import SetPrior
from battle_sim.runner import run_battle
from battle_sim.search import (
    PositionWeights,
    SearchPlayer,
    SearchProfile,
    clone_for_search,
    remap_action,
)
from battle_sim.trainer_db_tournament import load_trainers
from battle_sim.utils import Status

# What the stall side is trying to do, as opposed to hitting something.
STALL_MOVES = {"Toxic", "Recover", "Roost", "Whirlwind", "Defog", "Soft-Boiled", "Will-O-Wisp", "Protect"}


@dataclass(frozen=True)
class StallPosition:
    state: BattleState
    side_index: int
    actions: list[Action]


class _Recorder(SearchPlayer):
    """Plays normally, keeping the positions worth re-asking later."""

    def __init__(self, weights: MatchupWeights, profile: SearchProfile, keep: list[StallPosition], side: int) -> None:
        super().__init__(weights, profile=profile)
        self._keep = keep
        self._side = side

    def choose_action(self, state: BattleState, side_index: int, actions: Sequence[Action]) -> Action:
        if side_index == self._side and _is_stall_decision(state, side_index, actions):
            # Snapshot the state, since a view shares Pokemon with the live battle.
            frozen = clone_for_search(state, seed=0)
            self._keep.append(
                StallPosition(frozen, side_index, [remap_action(a, state, frozen, side_index) for a in actions])
            )
        return super().choose_action(state, side_index, actions)


def _move_name(state: BattleState, side_index: int, action: Action) -> str | None:
    """The move an action names, or None for a switch. Slots are indices, so this needs the mon."""
    if action.move is None:
        return None
    move = state.sides[side_index].active_pokemon.moves[action.move]
    return None if move is None else move.name


def _is_stall_decision(state: BattleState, side_index: int, actions: Sequence[Action]) -> bool:
    """A real choice between poisoning something and hitting it."""
    target = state.sides[1 - side_index].active_pokemon
    if target.is_fainted() or target.status is not Status.NONE:
        return False
    if target.live_stats.HP < target.stat_totals.HP * 0.8:  # nearly dead: hitting it is just correct
        return False
    # A Steel type cannot be poisoned, so declining Toxic there says nothing.
    if status_cannot_land(Status.TOXIC, target, state.sides[side_index].active_pokemon, state.field):
        return False
    names = {_move_name(state, side_index, a) for a in actions}
    return "Toxic" in names and len(actions) > 1


def _label(state: BattleState, side_index: int, action: Action) -> str:
    move = _move_name(state, side_index, action)
    if move is None:
        return "switch"
    return move if move in STALL_MOVES else "attack"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teams-file", type=Path, default=Path("ag_teams.txt"))
    parser.add_argument("--stall", default="AG01", help="team tag whose decisions are examined")
    parser.add_argument("--against", default="AG12")
    parser.add_argument("--budgets", type=int, nargs="+", default=[120, 1000, 8000])
    parser.add_argument("--positions", type=int, default=12)
    parser.add_argument("--gather-budget", type=int, default=120)
    parser.add_argument(
        "--residual-weights",
        type=float,
        nargs="+",
        default=[None],
        help="override PositionWeights.residual_pressure; 0 restores the pre-horizon evaluator",
    )
    args = parser.parse_args()

    trainers = {t.name.split()[0]: t for t in load_trainers(args.teams_file)}
    a, b = trainers[args.stall], trainers[args.against]
    weights = load_weights(Path("champions/random-a-myopic-tuned.json"))
    prior = SetPrior.from_teams([t.team for t in trainers.values()])

    keep: list[StallPosition] = []
    profile = SearchProfile(budget=args.gather_budget)
    for game in range(8):
        if len(keep) >= args.positions:
            break
        run_battle(
            a.team,
            b.team,
            _Recorder(weights, profile=profile, keep=keep, side=0),
            SearchPlayer(weights, profile=profile),
            seed=game,
            prior=prior,
        )
    positions = keep[: args.positions]
    logger.info(f"{len(positions)} positions where {args.stall} may poison a healthy, un-statused target\n")

    for residual in args.residual_weights:
        base = PositionWeights.from_matchup(weights)
        position = base if residual is None else replace(base, residual_pressure=residual)
        for budget in args.budgets:
            started = time.monotonic()
            picks: Counter[str] = Counter()
            for kept in positions:
                player = SearchPlayer(weights, profile=SearchProfile(budget=budget), position_weights=position)
                choice = player.choose_action(kept.state, kept.side_index, kept.actions)
                picks[_label(kept.state, kept.side_index, choice)] += 1
            toxic = picks["Toxic"] / len(positions)
            stall = sum(n for k, n in picks.items() if k in STALL_MOVES) / len(positions)
            elapsed = time.monotonic() - started
            spread = "  ".join(f"{k}: {n}" for k, n in picks.most_common())
            logger.info(
                f"residual {position.residual_pressure:>4.1f}  budget {budget:>6}   Toxic {toxic:>4.0%}   "
                f"any stall move {stall:>4.0%}   {elapsed:>6.1f}s   {spread}"
            )


if __name__ == "__main__":
    main()
