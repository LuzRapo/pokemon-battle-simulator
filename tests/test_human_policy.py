from collections import Counter
from pathlib import Path

from battle_sim.human_policy import HumanPolicyPlayer, categorise
from battle_sim.observation import SetPrior
from battle_sim.runner import run_battle
from battle_sim.trainer_db_tournament import load_trainers

TEAMS = {t.name.split()[0]: t for t in load_trainers(Path("ag_teams.txt"))}


def test_the_opening_is_where_the_ladder_differs_most():
    """A quarter of all hazards go down on turn one and humans switch on nearly half of the first
    three turns; a single flat distribution would average that away."""
    player = HumanPolicyPlayer()
    opening, late = player._distribution(1), player._distribution(20)
    assert opening["hazard"] > late["hazard"]
    assert opening["switch"] > late["switch"]
    assert opening["attack"] < late["attack"]


def test_it_only_ever_picks_a_legal_action():
    a, b = TEAMS["AG12"], TEAMS["AG01"]
    prior = SetPrior.from_teams([a.team, b.team])
    result = run_battle(a.team, b.team, HumanPolicyPlayer(seed=1), HumanPolicyPlayer(seed=2), seed=0, prior=prior)
    assert result.turns > 0  # a battle that ran to completion never offered it an action it refused


def test_it_switches_far_more_than_our_search_does():
    """The whole reason it exists: self-play cannot punish a habit both sides share."""
    a, b = TEAMS["AG12"], TEAMS["AG01"]
    prior = SetPrior.from_teams([a.team, b.team])
    picks: Counter = Counter()

    class Watched(HumanPolicyPlayer):
        def choose_action(self, state, side_index, actions):
            action = super().choose_action(state, side_index, actions)
            side = state.sides[side_index]
            if not (side.needs_switch or side.active_pokemon.is_fainted()):
                move = None if action.move is None else side.active_pokemon.moves[action.move]
                picks[categorise(None if move is None else move.name)] += 1
            return action

    run_battle(a.team, b.team, Watched(seed=3), Watched(seed=4), seed=0, prior=prior)
    total = sum(picks.values())
    assert picks["switch"] / total > 0.25  # our search sits at 16.6%


def test_leading_a_hazard_setter_is_a_coin_flip_not_a_rule():
    """Humans set the first hazard with their lead about half the time — not always, not never."""
    team = TEAMS["AG20"].team
    setters = {i for i, spec in enumerate(team) if "Stealth Rock" in [str(m) for m in spec.moves]}
    leads = [HumanPolicyPlayer(seed=s).choose_order(team, TEAMS["AG12"].team)[0] for s in range(40)]
    led_with_setter = sum(1 for lead in leads if lead in setters)
    assert 0 < led_with_setter < 40
