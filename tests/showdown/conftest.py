from typing import Any

from battle_sim.database.loader import normalize_id
from battle_sim.engine import step
from battle_sim.maths.rng import RNG
from battle_sim.mechanics.battle import BattleState, SideState
from battle_sim.models.actions import Action, ActionType
from battle_sim.models.moves import MoveSlot
from battle_sim.models.pokemon import Pokemon
from battle_sim.models.spec import PokemonSpec
from battle_sim.teams import build_pokemon
from battle_sim.utils import Status

_STATUS_FROM_SHOWDOWN: dict[str, Status] = {
    "": Status.NONE,
    "brn": Status.BURN,
    "psn": Status.POISON,
    "tox": Status.TOXIC,
    "par": Status.PARALYSIS,
    "slp": Status.SLEEP,
    "frz": Status.FREEZE,
}
_STAT_FROM_SHOWDOWN: dict[str, str] = {
    "atk": "ATTACK",
    "def": "DEFENCE",
    "spa": "SP_ATTACK",
    "spd": "SP_DEFENCE",
    "spe": "SPEED",
    "accuracy": "ACCURACY",
    "evasion": "EVASION",
}


type RawSpec = PokemonSpec | dict[str, Any]


def _coerce(spec: RawSpec) -> PokemonSpec:
    return spec if isinstance(spec, PokemonSpec) else PokemonSpec(**spec)


def create_battle(p1: list[RawSpec], p2: list[RawSpec], seed: int = 0) -> BattleState:
    p1_team = [build_pokemon(_coerce(s)) for s in p1]
    p2_team = [build_pokemon(_coerce(s)) for s in p2]
    return BattleState(sides=(SideState(team=p1_team), SideState(team=p2_team)), rng=RNG(seed=seed))


def make_choices(state: BattleState, p1: str, p2: str) -> list[str]:
    """Showdown parity tests assert on rendered text — that is their job — so this stays string-based."""
    return step(state, {0: _parse_choice(state, 0, p1), 1: _parse_choice(state, 1, p2)}).rendered()


def _parse_choice(state: BattleState, side_index: int, choice: str) -> Action:
    kind, _, arg = choice.partition(" ")
    side = state.sides[side_index]
    if kind == "move":
        target_id = normalize_id(arg)
        for slot in MoveSlot:
            move = side.active_pokemon.moves[slot]
            if move is not None and normalize_id(move.name) == target_id:
                return Action(action=ActionType.USE_MOVE, target=move.target, move=slot)
        raise ValueError(f"Move {arg!r} not in P{side_index + 1}'s active moveset")
    if kind == "switch":
        target_id = normalize_id(arg)
        for pokemon in side.team:
            if normalize_id(pokemon.name) == target_id:
                return Action(action=ActionType.SWITCH_OUT, switch_in=pokemon)
        raise ValueError(f"Pokemon {arg!r} not in P{side_index + 1}'s team")
    raise ValueError(f"Unknown choice kind: {kind!r}")


def assert_status(pokemon: Pokemon, showdown_id: str) -> None:
    expected = _STATUS_FROM_SHOWDOWN[showdown_id]
    assert pokemon.status is expected, f"Expected status {showdown_id!r}, got {pokemon.status.name}"


def assert_stat_stage(pokemon: Pokemon, stat: str, expected: int) -> None:
    actual = pokemon.stat_stages.model_dump()[_STAT_FROM_SHOWDOWN[stat]]
    assert actual == expected, f"Expected {stat} stage {expected}, got {actual}"
