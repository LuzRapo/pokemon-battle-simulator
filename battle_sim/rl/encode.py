"""`rust/src/obs.rs`, over a Python `BattleState`: one side's view of a battle as fixed-shape arrays.

The training loop encodes in Rust; this copy exists so a trained network can play inside the Python
engine — against `SearchPlayer` for evaluation, and in the bot. The two must produce identical
arrays for the same position, and `tests/test_rl_encode.py` checks that they do at every decision
of recorded battles. Change one, change the other.
"""

from enum import Enum

import numpy as np
from numpy.typing import NDArray

from battle_sim.mechanics.battle import BattleState
from battle_sim.models.moves import MoveSlot
from battle_sim.models.pokemon import Pokemon
from battle_sim.rl.vocab import ability_id, item_id, move_id, species_id
from battle_sim.utils import ExtraStatus, Hazards, PseudoWeather, Status, Terrain, Type, Weather


class Decision(Enum):
    """What is being asked for: the lead before turn 0, a turn's action, or a replacement."""

    LEAD = 0
    TURN = 1
    SWITCH = 2


SLOTS = 12
POKEMON_IDS = 7
STAGES = ("ATTACK", "DEFENCE", "SP_ATTACK", "SP_DEFENCE", "SPEED", "ACCURACY", "EVASION")
STATUSES = (Status.BURN, Status.POISON, Status.TOXIC, Status.PARALYSIS, Status.SLEEP, Status.FREEZE)
TYPES = tuple(Type)
VOLATILES = tuple(
    ExtraStatus[name]
    for name in (
        "CONFUSION", "DISABLE", "ENCORE", "FLINCH", "FOCUS_ENERGY", "IDENTIFIED", "MIRACLE_EYE", "LEECH_SEED",
        "LOCKED_MOVE", "NIGHTMARE", "PROTECT", "YAWN", "SUBSTITUTE", "TAUNT", "SALT_CURE", "CURSE", "PERISH",
        "DESTINY_BOND", "ENDURE", "CHARGING", "MUST_RECHARGE", "SLOW_START", "LOAFING", "ROOSTED",
        "PARTIALLY_TRAPPED",
    )
)  # fmt: skip
WEATHERS = tuple(w for w in Weather if w is not Weather.NONE)
TERRAINS = tuple(t for t in Terrain if t is not Terrain.NONE)
PSEUDO_WEATHERS = tuple(PseudoWeather)
HAZARDS = ((Hazards.STEALTH_ROCK, 1.0), (Hazards.SPIKES, 3.0), (Hazards.TOXIC_SPIKES, 2.0), (Hazards.STICKY_WEB, 1.0))
SCREENS = (Hazards.REFLECT, Hazards.LIGHT_SCREEN, Hazards.AURORA_VEIL)

POKEMON_FLOATS = 1 + 1 + 1 + 5 + len(STAGES) + len(STATUSES) + 1 + 1 + 1 + 1 + 1 + len(TYPES) + len(VOLATILES) + 4 + 12
SIDE_FLOATS = len(HAZARDS) + len(SCREENS) + 1 + 4 + 3
FIELD_FLOATS = len(WEATHERS) + 1 + len(TERRAINS) + 1 + len(PSEUDO_WEATHERS) + 2 * SIDE_FLOATS + 1 + len(Decision)

Observation = tuple[NDArray[np.int64], NDArray[np.float32], NDArray[np.float32]]


def encode(state: BattleState, viewer: int, decision: Decision) -> Observation:
    """`(ids[12, 7], pokemon[12, POKEMON_FLOATS], field[FIELD_FLOATS])`, the viewer's team first."""
    ids = np.zeros((SLOTS, POKEMON_IDS), dtype=np.int64)
    pokemon = np.zeros((SLOTS, POKEMON_FLOATS), dtype=np.float32)
    transformed = set(state.transforms)
    for block, side_index in enumerate((viewer, 1 - viewer)):
        side = state.sides[side_index]
        for index, member in enumerate(side.team):
            slot = block * 6 + index
            active = decision is not Decision.LEAD and index == side.active[0]
            ids[slot] = _ids(member)
            pokemon[slot] = _floats(member, active, id(member) in transformed)
    return ids, pokemon, _field(state, viewer, decision)


def _ids(p: Pokemon) -> list[int]:
    moves = [move_id(move.name) if move is not None else 0 for move in p.moves]
    return [species_id(p.name), ability_id(p.ability), item_id(p.item), *moves]


def _floats(p: Pokemon, active: bool, transformed: bool) -> list[float]:
    totals = p.stat_totals
    out: list[float] = [1.0, p.live_stats.HP / totals.HP if totals.HP > 0 else 0.0, totals.HP / 500.0]
    out += [stat / 500.0 for stat in (totals.ATTACK, totals.DEFENCE, totals.SP_ATTACK, totals.SP_DEFENCE, totals.SPEED)]
    out += [getattr(p.stat_stages, stage) / 6.0 for stage in STAGES]
    out += [float(p.status is status) for status in STATUSES]
    out.append(min(p.status_turns / 10.0, 1.0))
    out += [float(p.is_fainted()), float(active), float(transformed), float(p.item_consumed)]
    types = {t for t in p.battle_types if t is not None}
    out += [float(kind in types) for kind in TYPES]
    out += [float(p.volatiles.get(volatile, 0) > 0) for volatile in VOLATILES]
    out += [min(p.pp.get(slot, 0) / 40.0, 1.0) for slot in MoveSlot]
    for lock in (p.choice_locked_move, p.encored_slot, p.disabled_slot):
        out += [float(lock is slot) for slot in MoveSlot]
    return out


def _field(state: BattleState, viewer: int, decision: Decision) -> NDArray[np.float32]:
    field = state.field
    out: list[float] = [float(field.weather is weather) for weather in WEATHERS]
    out.append(field.weather_turns_left / 8.0)
    out += [float(field.terrain is terrain) for terrain in TERRAINS]
    out.append(field.terrain_turns_left / 8.0)
    out += [float(pseudo in field.pseudo_weather) for pseudo in PSEUDO_WEATHERS]
    for side_index in (viewer, 1 - viewer):
        side = state.sides[side_index]
        out += [side.hazards.get(hazard, 0) / most for hazard, most in HAZARDS]
        out += [side.screens.get(screen, 0) / 8.0 for screen in SCREENS]
        out.append(side.tailwind_turns / 4.0)
        used = (side.has_mega_evolved, side.has_primal_reverted, side.has_ultra_bursted, side.has_used_z_move)
        out += [float(flag) for flag in used]
        out += [side.wish_turns / 2.0, side.future_sight_turns / 3.0, float(side.healing_wish_pending)]
    out.append(min(state.turn / 100.0, 1.0))
    out += [float(decision is kind) for kind in Decision]
    return np.asarray(out, dtype=np.float32)
