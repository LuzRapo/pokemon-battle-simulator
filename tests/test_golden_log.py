"""Golden-log regression test.

Runs a fixed, seeded battle that exercises a wide slice of mechanics (weather
abilities, Intimidate, hazards, items, multi-hit, status, residuals, switching)
and compares the rendered log byte-for-byte against a checked-in golden file.

Any refactor that changes battle behaviour, effect ordering, or RNG draw order
shows up here as a diff. Regenerate deliberately with:

    uv run python -m tests.test_golden_log > tests/golden/battle_seed0.txt
"""

from pathlib import Path

from battle_sim.engine import apply_forced_switch, step
from battle_sim.maths.rng import RNG
from battle_sim.mechanics.battle import BattleState, SideState
from battle_sim.models.actions import Action, ActionType
from battle_sim.models.moves import MoveSlot
from battle_sim.models.spec import PokemonSpec
from battle_sim.teams import build_pokemon
from battle_sim.utils import Ability, Item, Target

GOLDEN_PATH = Path(__file__).parent / "golden" / "battle_seed0.txt"


def _move(slot: MoveSlot) -> Action:
    return Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=slot)


def _switch_to(state: BattleState, side: int, team_index: int) -> Action:
    return Action(action=ActionType.SWITCH_OUT, switch_in=state.sides[side].team[team_index])


def run_golden_battle() -> list[str]:
    p1_team = [
        build_pokemon(
            PokemonSpec(
                species="Tyranitar",
                ability=Ability.SAND_STREAM,
                item=Item.LEFTOVERS,
                moves=["Stealth Rock", "Rock Slide", "Crunch", "Earthquake"],
            )
        ),
        build_pokemon(
            PokemonSpec(
                species="Garchomp",
                item=Item.LIFE_ORB,
                moves=["Earthquake", "Dragon Claw", "Swords Dance", "Fire Fang"],
            )
        ),
    ]
    p2_team = [
        build_pokemon(
            PokemonSpec(
                species="Gengar",
                ability=Ability.LEVITATE,
                item=Item.BLACK_SLUDGE,
                moves=["Shadow Ball", "Will-O-Wisp", "Taunt", "Icy Wind"],
            )
        ),
        build_pokemon(
            PokemonSpec(
                species="Staraptor",
                ability=Ability.INTIMIDATE,
                item=Item.CHOICE_BAND,
                moves=["Brave Bird", "Double-Edge", "Quick Attack", "U-turn"],
            )
        ),
    ]
    state = BattleState(sides=(SideState(team=p1_team), SideState(team=p2_team)), rng=RNG(seed=0))

    lines: list[str] = []

    def turn(p1: Action, p2: Action) -> None:
        if state.outcome is not None:
            return
        lines.append(f"-- turn {state.turn} --")
        lines.extend(step(state, {0: p1, 1: p2}).rendered())
        for side in (0, 1):
            if state.outcome is None and (
                state.sides[side].active_pokemon.is_fainted() or state.sides[side].needs_switch
            ):
                bench = next(
                    (
                        i
                        for i, p in enumerate(state.sides[side].team)
                        if i != state.sides[side].active[0] and not p.is_fainted()
                    ),
                    None,
                )
                if bench is not None:
                    lines.extend(apply_forced_switch(state, side, _switch_to(state, side, bench)).rendered())

    turn(_move(MoveSlot.FIRST), _move(MoveSlot.SECOND))  # Stealth Rock vs Will-O-Wisp
    turn(_move(MoveSlot.SECOND), _move(MoveSlot.THIRD))  # Rock Slide vs Taunt
    turn(_move(MoveSlot.THIRD), _move(MoveSlot.FIRST))  # Crunch vs Shadow Ball
    turn(_switch_to(state, 0, 1), _move(MoveSlot.FOURTH))  # switch Garchomp in vs Icy Wind
    turn(_move(MoveSlot.THIRD), _switch_to(state, 1, 1))  # Swords Dance vs switch Staraptor (rocks + Intimidate)
    turn(_move(MoveSlot.SECOND), _move(MoveSlot.FIRST))  # Dragon Claw vs Brave Bird (choice lock)
    turn(_move(MoveSlot.FIRST), _move(MoveSlot.FIRST))  # Earthquake vs Brave Bird
    turn(_move(MoveSlot.FOURTH), _move(MoveSlot.FIRST))  # Fire Fang vs Brave Bird
    turn(_move(MoveSlot.FIRST), _move(MoveSlot.FIRST))
    turn(_move(MoveSlot.FIRST), _move(MoveSlot.FIRST))
    return lines


def test_golden_battle_log_is_stable():
    assert GOLDEN_PATH.exists(), f"golden file missing: {GOLDEN_PATH} (see module docstring to regenerate)"
    expected = GOLDEN_PATH.read_text().splitlines()
    assert run_golden_battle() == expected


if __name__ == "__main__":
    print("\n".join(run_golden_battle()))
