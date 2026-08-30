from collections.abc import Sequence

import pytest

from battle_sim.engine import legal_actions
from battle_sim.maths.rng import RNG
from battle_sim.mechanics.battle import BattleState, SideState
from battle_sim.models.actions import Action, ActionType
from battle_sim.models.moves import MoveSlot
from battle_sim.models.pokemon import Pokemon
from battle_sim.models.spec import PokemonSpec
from battle_sim.runner import BattleResult, run_battle
from battle_sim.teams import build_pokemon
from battle_sim.utils import ExtraStatus, Item, Outcome

TEAM_A = [
    PokemonSpec(species="Garchomp", moves=["Earthquake", "Dragon Claw", "Swords Dance", "Fire Fang"]),
    PokemonSpec(species="Gengar", moves=["Shadow Ball", "Will-O-Wisp", "Taunt", "Icy Wind"]),
    PokemonSpec(species="Tyranitar", moves=["Stone Edge", "Crunch", "Stealth Rock", "Earthquake"]),
]
TEAM_B = [
    PokemonSpec(species="Staraptor", moves=["Brave Bird", "Double-Edge", "Quick Attack", "U-turn"]),
    PokemonSpec(species="Metagross", moves=["Meteor Mash", "Earthquake", "Bullet Punch", "Ice Punch"]),
    PokemonSpec(species="Gyarados", moves=["Waterfall", "Earthquake", "Ice Fang", "Stone Edge"]),
]


class FirstLegal:
    def choose_order(self, own: Sequence[PokemonSpec], opponent: Sequence[PokemonSpec]) -> Sequence[int]:
        return list(range(len(own)))

    def choose_action(self, state: BattleState, side_index: int, actions: Sequence[Action]) -> Action:
        return actions[0]


class ReversedOrder(FirstLegal):
    def choose_order(self, own: Sequence[PokemonSpec], opponent: Sequence[PokemonSpec]) -> Sequence[int]:
        return list(reversed(range(len(own))))


class BadOrder(FirstLegal):
    def choose_order(self, own: Sequence[PokemonSpec], opponent: Sequence[PokemonSpec]) -> Sequence[int]:
        return [0, 0, 1]


class PivotOutThenFirstLegal(FirstLegal):
    """U-turn the instant it's legal; every other decision (including who replaces it) is FirstLegal's."""

    def choose_action(self, state: BattleState, side_index: int, actions: Sequence[Action]) -> Action:
        active = state.sides[side_index].active_pokemon
        for candidate in actions:
            if candidate.action is ActionType.USE_MOVE and candidate.move is not None:
                move = active.moves[candidate.move]
                if move is not None and move.name == "U-turn":
                    return candidate
        return super().choose_action(state, side_index, actions)


def test_run_battle_lets_a_faster_pivot_protect_its_user():
    """End-to-end through the real driver: U-turn's user must never take the opponent's same-turn
    hit — the mon that switches in does, exactly as it would in a real game."""
    pivot_team = [
        PokemonSpec(species="Staraptor", moves=["U-turn", "Brave Bird", "Quick Attack", "Double-Edge"]),
        PokemonSpec(species="Blissey", moves=["Seismic Toss", "Soft-Boiled", "Thunder Wave", "Toxic"]),
    ]
    slow_attacker = [PokemonSpec(species="Chansey", moves=["Tackle", "Seismic Toss", "Soft-Boiled", "Thunder Wave"])]

    result = run_battle(pivot_team, slow_attacker, PivotOutThenFirstLegal(), FirstLegal(), seed=0, max_turns=1)

    assert len(result.logs) == 1  # resolved inside the one step() call; no post-turn replacement needed
    lines = result.logs[0].rendered()
    assert any("withdrew Staraptor and sent out Blissey" in line for line in lines)
    assert any("Blissey took" in line and "damage" in line for line in lines)
    assert not any("Staraptor took" in line for line in lines)


def test_run_battle_reaches_an_outcome():
    result = run_battle(TEAM_A, TEAM_B, FirstLegal(), FirstLegal(), seed=0)
    assert isinstance(result, BattleResult)
    assert result.outcome in (Outcome.P1_WIN, Outcome.P2_WIN, Outcome.DRAW)
    assert 0 < result.turns < 1000
    assert result.logs


def test_run_battle_is_deterministic_per_seed():
    first = run_battle(TEAM_A, TEAM_B, FirstLegal(), FirstLegal(), seed=7)
    second = run_battle(TEAM_A, TEAM_B, FirstLegal(), FirstLegal(), seed=7)
    assert first == second


def test_chosen_order_decides_the_lead():
    result = run_battle(TEAM_A, TEAM_B, ReversedOrder(), FirstLegal(), seed=0, max_turns=1)
    lead_line = result.logs[0].rendered()
    assert any("Tyranitar" in line for line in lead_line)  # TEAM_A reversed leads with its last spec


def test_invalid_order_raises():
    with pytest.raises(ValueError, match="permutation"):
        run_battle(TEAM_A, TEAM_B, BadOrder(), FirstLegal(), seed=0)


def _solo_state(pokemon: Pokemon, opponent: Pokemon) -> BattleState:
    return BattleState(sides=(SideState(team=[pokemon]), SideState(team=[opponent])), rng=RNG(seed=0))


def test_legal_actions_moves_and_switches():
    team = [build_pokemon(spec) for spec in TEAM_A]
    state = BattleState(sides=(SideState(team=team), SideState(team=[build_pokemon(TEAM_B[0])])), rng=RNG(seed=0))
    actions = legal_actions(state, 0)
    moves = [a for a in actions if a.action is ActionType.USE_MOVE]
    switches = [a for a in actions if a.action is ActionType.SWITCH_OUT]
    assert len(moves) == 4
    assert len(switches) == 2


def test_legal_actions_forced_switch_only_offers_bench():
    team = [build_pokemon(spec) for spec in TEAM_A]
    state = BattleState(sides=(SideState(team=team), SideState(team=[build_pokemon(TEAM_B[0])])), rng=RNG(seed=0))
    team[0].live_stats.HP = 0
    actions = legal_actions(state, 0)
    assert all(a.action is ActionType.SWITCH_OUT for a in actions)
    assert len(actions) == 2


def test_legal_actions_choice_lock_restricts_to_locked_move():
    a = build_pokemon(PokemonSpec(species="Garchomp", item=Item.CHOICE_BAND, moves=["Earthquake", "Dragon Claw"]))
    state = _solo_state(a, build_pokemon(TEAM_B[0]))
    a.choice_locked_move = MoveSlot.FIRST
    moves = [x for x in legal_actions(state, 0) if x.action is ActionType.USE_MOVE]
    assert [m.move for m in moves] == [MoveSlot.FIRST]


def test_legal_actions_rampage_forbids_switching():
    team = [build_pokemon(spec) for spec in TEAM_A]
    state = BattleState(sides=(SideState(team=team), SideState(team=[build_pokemon(TEAM_B[0])])), rng=RNG(seed=0))
    team[0].volatiles[ExtraStatus.LOCKED_MOVE] = 2
    team[0].locked_slot = MoveSlot.FIRST
    actions = legal_actions(state, 0)
    assert len(actions) == 1
    assert actions[0].move is MoveSlot.FIRST


def test_legal_actions_all_pp_empty_offers_struggle_slot():
    a = build_pokemon(PokemonSpec(species="Garchomp", moves=["Earthquake"]))
    state = _solo_state(a, build_pokemon(TEAM_B[0]))
    a.pp = dict.fromkeys(a.pp, 0)
    moves = [x for x in legal_actions(state, 0) if x.action is ActionType.USE_MOVE]
    assert len(moves) == 1  # the Struggle stand-in
