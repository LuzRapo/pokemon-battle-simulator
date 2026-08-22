from collections.abc import Sequence
from dataclasses import dataclass

from battle_sim.matchup import MatchupPlayer
from battle_sim.mechanics.battle import BattleState
from battle_sim.models.actions import Action
from battle_sim.search import SearchPlayer

GRADES = ((0.05, "Best"), (0.25, "Great"), (0.6, "Good"), (1.2, "Inaccuracy"), (2.0, "Mistake"))
_BLUNDER = "Blunder"
_BRILLIANT = "Brilliant"
_MYOPIC_OBVIOUS = 2  # a Best move already in the myopic top two is merely Best, not Brilliant


@dataclass(frozen=True)
class Verdict:
    label: str
    loss: float  # expected pokemon given up vs the best offered action
    best: Action  # what the annotator would have played


def grade_label(loss: float) -> str:
    return next((name for ceiling, name in GRADES if loss <= ceiling), _BLUNDER)


def grade_choice(
    annotator: SearchPlayer, view: BattleState, side_index: int, options: Sequence[Action], chosen: Action
) -> Verdict:
    """Grade one decision among the offered options, as seen from the chooser's believed state."""
    values = annotator.action_values(view, side_index, options)
    best_index = max(range(len(values)), key=lambda i: values[i])
    loss = values[best_index] - values[options.index(chosen)]
    label = grade_label(loss)
    if label == "Best" and _is_brilliant(annotator, view, side_index, options, chosen):
        label = _BRILLIANT
    return Verdict(label=label, loss=loss, best=options[best_index])


def _is_brilliant(
    annotator: SearchPlayer, view: BattleState, side_index: int, options: Sequence[Action], chosen: Action
) -> bool:
    myopic = MatchupPlayer(annotator.weights).score_actions(view, side_index, list(options))
    ranked = sorted(myopic, key=lambda pair: pair[0], reverse=True)
    return all(action != chosen for _, action in ranked[:_MYOPIC_OBVIOUS])
