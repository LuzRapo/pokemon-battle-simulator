from battle_sim.mechanics.battle import BattleState
from battle_sim.mechanics.log import BattleLog
from battle_sim.models.log_events import BattleEnded
from battle_sim.utils import Outcome


def _update_outcome(state: BattleState, log: BattleLog | None = None) -> None:
    if state.outcome is not None:
        return
    side0_out = all(pokemon.is_fainted() for pokemon in state.sides[0].team)
    side1_out = all(pokemon.is_fainted() for pokemon in state.sides[1].team)
    if side0_out and side1_out:
        state.outcome = Outcome.DRAW
    elif side0_out:
        state.outcome = Outcome.P2_WIN
    elif side1_out:
        state.outcome = Outcome.P1_WIN
    if state.outcome is not None and log is not None:
        log.add(BattleEnded(outcome=state.outcome))
