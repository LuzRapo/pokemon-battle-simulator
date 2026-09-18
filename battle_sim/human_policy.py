"""A benchmark opponent that plays with the ladder's habits rather than our own.

Every evaluation in this project is our AI against itself, and self-play cannot see a blindspot both
sides share: neither punishes the other for under-switching, because neither switches. Measured over
5,001 Gen 7 Anything Goes replays, humans and our AI decide turns very differently -- humans switch
on 29.5% of turns to our 16.6%, spend 6.5% on hazards to our 3.3%, and attack on 45.2% to our 55.2%.
An opponent drawn from the human distribution puts that difference on the board.

What this is: a *stylistic* model. It samples which **kind** of decision to make from the observed
human distribution for that stage of the game, then plays the best action of that kind according to
the genome. What it is not: a model of human judgement. The logs never reveal an EV spread and only
say what was chosen, not why, so nothing here claims to know when a human would switch -- only how
often, and into what shape of position. Treat it as a sparring partner with human habits, not as a
strong player, and never as a ceiling to aim at.
"""

import json
import random
from collections.abc import Sequence
from pathlib import Path

from battle_sim.matchup import MatchupPlayer, MatchupWeights, set_hazard
from battle_sim.mechanics.battle import BattleState
from battle_sim.models.actions import Action
from battle_sim.models.spec import PokemonSpec

POLICY = Path(__file__).parent / "data" / "human_policy.json"
_LEAD_WITH_A_SETTER = 0.5  # measured: half of all first hazards are set by that side's lead

# Mirrors tools.play_profile, which is where the distribution was counted; kept here so the engine
# does not import from tools.
SETUP = {
    "Dragon Dance", "Swords Dance", "Calm Mind", "Nasty Plot", "Quiver Dance", "Bulk Up", "Shell Smash",
    "Rock Polish", "Agility", "Growth", "Work Up", "Coil", "Hone Claws", "Geomancy", "Tail Glow",
}  # fmt: skip
STATUS = {"Toxic", "Will-O-Wisp", "Thunder Wave", "Spore", "Sleep Powder", "Glare", "Hypnosis", "Yawn"}
HAZARD = {"Stealth Rock", "Spikes", "Toxic Spikes", "Sticky Web", "Defog", "Rapid Spin"}
RECOVERY = {"Recover", "Roost", "Soft-Boiled", "Slack Off", "Morning Sun", "Moonlight", "Synthesis", "Rest", "Wish"}
PHAZE = {"Whirlwind", "Roar", "Dragon Tail", "Circle Throw", "Haze", "Perish Song"}


def categorise(move: str | None) -> str:
    """A move's role, or `switch` when no move was used."""
    if move is None:
        return "switch"
    for name, bucket in (("setup", SETUP), ("status", STATUS), ("hazard", HAZARD), ("recovery", RECOVERY)):
        if move in bucket:
            return name
    return "phaze" if move in PHAZE else "attack"


class HumanPolicyPlayer:
    """Samples the *kind* of decision from the human distribution, then plays it well."""

    def __init__(self, weights: MatchupWeights | None = None, seed: int = 0, policy_path: Path = POLICY) -> None:
        self._genome = MatchupPlayer(weights) if weights is not None else MatchupPlayer()
        self._rng = random.Random(seed)
        self._seed = seed
        raw = json.loads(policy_path.read_text())["buckets"]
        self._buckets: dict[str, dict[str, float]] = {
            name: {k: float(v) for k, v in share.items()} for name, share in raw.items()
        }

    def _distribution(self, turn: int) -> dict[str, float]:
        for name, share in self._buckets.items():
            low, _, high = name.rstrip("+").partition("-")
            upper = int(high) if high else 10_000
            if int(low) <= turn <= upper:
                return share
        return next(iter(self._buckets.values()))

    def choose_order(self, own: Sequence[PokemonSpec], opponent: Sequence[PokemonSpec]) -> Sequence[int]:
        """Half the time, lead something that can put hazards down -- which is what the ladder does."""
        self._rng = random.Random(self._seed)
        setters = [i for i, spec in enumerate(own) if any(_sets_hazards(name) for name in spec.moves)]
        if setters and self._rng.random() < _LEAD_WITH_A_SETTER:
            lead = self._rng.choice(setters)
        else:
            lead = self._rng.randrange(len(own))
        return [lead, *[i for i in range(len(own)) if i != lead]]

    def choose_action(self, state: BattleState, side_index: int, actions: Sequence[Action]) -> Action:
        if len(actions) == 1:
            return actions[0]
        scored = self._genome.score_actions(state, side_index, actions)
        side = state.sides[side_index]
        if side.needs_switch or side.active_pokemon.is_fainted():
            return max(scored, key=lambda pair: pair[0])[1]  # forced replacement: no style to model
        by_category: dict[str, list[tuple[float, Action]]] = {}
        for score, action in scored:
            move = None if action.move is None else side.active_pokemon.moves[action.move]
            by_category.setdefault(categorise(None if move is None else move.name), []).append((score, action))
        shares = self._distribution(state.turn)
        # Restricted to what is actually on offer and renormalised, so a Pokemon with no recovery
        # does not silently forfeit that share of its turns to nothing.
        available = {name: shares.get(name, 0.0) for name in by_category}
        total = sum(available.values())
        if total <= 0:
            return max(scored, key=lambda pair: pair[0])[1]
        names = sorted(available)
        chosen = self._rng.choices(names, weights=[available[n] for n in names], k=1)[0]
        return max(by_category[chosen], key=lambda pair: pair[0])[1]


def _sets_hazards(move_name: str) -> bool:
    from battle_sim.database.loader import get_move

    try:
        return set_hazard(get_move(move_name)) is not None
    except KeyError:
        return False
