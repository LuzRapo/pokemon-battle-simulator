"""The contract a second engine has to satisfy: play the same battle, event for event."""

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from battle_sim.engine import apply_forced_switch, legal_actions, step
from battle_sim.maths.rng import RNG
from battle_sim.mechanics.battle import BattleState, SideState
from battle_sim.mechanics.log import BattleLog
from battle_sim.models.actions import Action, ActionType
from battle_sim.models.log_events import LogEntry
from battle_sim.models.pokemon import Pokemon
from battle_sim.models.spec import PokemonSpec
from battle_sim.models.stats import STAGED_STATS, EVs, IVs
from battle_sim.teams import build_pokemon
from battle_sim.utils import Ability, Item, Nature, field_values

FORMAT_VERSION = 1

type Chooser = Callable[[BattleState, int], Action]


class TapeExhausted(RuntimeError):
    """A replay asked for more randomness than the recording holds."""


@dataclass
class TapeRNG(RNG):
    """An RNG that writes down what it rolled, or replays what was rolled before."""

    tape: list[float | int] = field(default_factory=list)
    _replaying: bool = False
    _position: int = 0

    @property
    def drawn(self) -> int:
        """How many draws have been taken so far, recording or replaying."""
        return self._position if self._replaying else len(self.tape)

    @classmethod
    def replaying(cls, tape: Sequence[float | int]) -> "TapeRNG":
        recorder = cls(seed=0)
        recorder.tape = list(tape)
        recorder._replaying = True
        return recorder

    def _next(self, live: float | int) -> float | int:
        if not self._replaying:
            self.tape.append(live)
            return live
        if self._position >= len(self.tape):
            raise TapeExhausted(
                f"the recording holds {len(self.tape)} draws and a {self._position + 1}th was asked for"
            )
        value = self.tape[self._position]
        self._position += 1
        return value

    def random_probability(self) -> float:
        return float(self._next(super().random_probability()))

    def random_integer(self, minimum: int, maximum: int) -> int:
        return int(self._next(super().random_integer(minimum, maximum)))


def name_action(action: Action, state: BattleState, side_index: int) -> str:
    """One action as a stable string. See the module docstring on why not an index."""
    if action.action is ActionType.SWITCH_OUT:
        assert action.switch_in is not None
        return f"switch:{action.switch_in.nickname}"
    assert action.move is not None
    move = state.sides[side_index].active_pokemon.moves[action.move]
    prefix = "zmove" if action.z_move else "move"
    # The slot as well as the name, because `build_pokemon` repeats moves to pad a short moveset.
    return f"{prefix}:{action.move.name}:{move.name if move is not None else '?'}"


def find_action(named: str, state: BattleState, side_index: int) -> Action:
    """The legal action that string refers to, or a failure naming what was on offer instead."""
    for candidate in legal_actions(state, side_index):
        if name_action(candidate, state, side_index) == named:
            return candidate
    offered = sorted(name_action(a, state, side_index) for a in legal_actions(state, side_index))
    raise LookupError(f"{named!r} is not legal for side {side_index}; the engine offered {offered}")


def _plain(value: object) -> object:
    """Anything the log carries, as something `json.dumps` will accept and compare."""
    if isinstance(value, Enum):
        return value.name
    if is_dataclass(value) and not isinstance(value, type):
        return _plain(field_values(value))
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(_plain(k)): _plain(v) for k, v in value.items()}
    return value


def entry_json(entry: LogEntry) -> dict[str, object]:
    return {"type": type(entry).__name__} | {name: _plain(value) for name, value in field_values(entry).items()}


def pokemon_digest(pokemon: Pokemon) -> dict[str, Any]:
    """Everything about one Pokémon that a turn can change."""
    return {
        "nickname": pokemon.nickname,
        "species": pokemon.name,
        "hp": pokemon.live_stats.HP,
        "max_hp": pokemon.stat_totals.HP,
        "status": pokemon.status.name,
        "status_turns": pokemon.status_turns,
        "fainted": pokemon.is_fainted(),
        "item": pokemon.item.name,
        "item_consumed": pokemon.item_consumed,
        "ability": pokemon.ability.name,
        "types": [t.name for t in pokemon.battle_types if t is not None],
        "stages": {stat.name: pokemon.stat_stages[stat] for stat in STAGED_STATS},
        "volatiles": {v.name: turns for v, turns in sorted(pokemon.volatiles.items(), key=lambda kv: kv[0].name)},
        "pp": {slot.name: left for slot, left in sorted(pokemon.pp.items(), key=lambda kv: kv[0].name)},
        "lives_used": pokemon.lives_used,
        "made_last_stand": pokemon.made_last_stand,
    }


def side_digest(side: SideState) -> dict[str, Any]:
    return {
        "active": side.active_index if hasattr(side, "active_index") else None,
        "team": [pokemon_digest(p) for p in side.team],
        "hazards": {h.name: n for h, n in sorted(side.hazards.items(), key=lambda kv: kv[0].name)},
        "screens": {s.name: n for s, n in sorted(side.screens.items(), key=lambda kv: kv[0].name)},
    }


def state_digest(state: BattleState) -> dict[str, Any]:
    return {
        "turn": state.turn,
        "outcome": state.outcome.name if state.outcome is not None else None,
        "weather": state.field.weather.name,
        "weather_turns_left": state.field.weather_turns_left,
        "terrain": state.field.terrain.name,
        "terrain_turns_left": state.field.terrain_turns_left,
        "sides": [side_digest(side) for side in state.sides],
    }


@dataclass
class Scenario:
    """One battle, reduced to inputs: who fought, what they did, and every roll it took."""

    teams: tuple[list[dict[str, Any]], list[dict[str, Any]]]
    actions: list[tuple[str, str]]  # per turn, the named action for each side
    tape: list[float | int]
    seed: int = 0

    def to_json(self) -> str:
        return json.dumps(
            {
                "version": FORMAT_VERSION,
                "seed": self.seed,
                "teams": list(self.teams),
                "actions": self.actions,
                "tape": self.tape,
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, text: str) -> "Scenario":
        payload = json.loads(text)
        if payload["version"] != FORMAT_VERSION:
            raise ValueError(f"scenario format v{payload['version']}, this build reads v{FORMAT_VERSION}")
        return cls(
            teams=(payload["teams"][0], payload["teams"][1]),
            actions=[(a, b) for a, b in payload["actions"]],
            tape=payload["tape"],
            seed=payload["seed"],
        )


def encode_spec(spec: PokemonSpec) -> dict[str, Any]:
    """One set as JSON, by *name* rather than by enum ordinal."""
    return {
        "species": spec.species,
        "nickname": spec.nickname,
        "level": spec.level,
        "ability": spec.ability.name,
        "item": spec.item.name,
        "nature": spec.nature.name,
        "effort_values": spec.effort_values.model_dump(),
        "individual_values": spec.individual_values.model_dump(),
        "moves": list(spec.moves),
        "pp_ups": spec.pp_ups,
    }


def decode_spec(entry: dict[str, Any]) -> PokemonSpec:
    return PokemonSpec(
        species=entry["species"],
        nickname=entry["nickname"],
        level=entry["level"],
        ability=Ability[entry["ability"]],
        item=Item[entry["item"]],
        nature=Nature[entry["nature"]],
        effort_values=EVs.model_validate(entry["effort_values"]),
        individual_values=IVs.model_validate(entry["individual_values"]),
        moves=list(entry["moves"]),
        pp_ups=entry["pp_ups"],
    )


def _specs(team: Sequence[dict[str, Any]]) -> list[PokemonSpec]:
    return [decode_spec(entry) for entry in team]


def build_state(scenario: Scenario) -> tuple[BattleState, TapeRNG]:
    """The opening position, with an RNG that will replay the scenario's rolls."""
    rng = TapeRNG.replaying(scenario.tape)
    sides = (
        SideState(team=[build_pokemon(spec) for spec in _specs(scenario.teams[0])]),
        SideState(team=[build_pokemon(spec) for spec in _specs(scenario.teams[1])]),
    )
    state = BattleState(sides=sides, rng=rng)
    return state, rng


def trace(scenario: Scenario) -> list[dict[str, Any]]:
    """Replay a scenario and produce the canonical per-turn trace to diff against."""
    state, rng = build_state(scenario)
    turns: list[dict[str, Any]] = []
    for first, second in scenario.actions:
        if state.outcome is not None:
            break
        chosen = {0: find_action(first, state, 0), 1: find_action(second, state, 1)}
        log = step(state, chosen, replacement_chooser)
        turns.append(
            {
                "actions": [first, second],
                "events": [entry_json(entry) for entry in log],
                "state": state_digest(state),
                "drawn": rng.drawn,
            }
        )
    return turns


@dataclass(frozen=True)
class Divergence:
    """Where two engines stopped agreeing, and about what."""

    turn: int
    where: str
    left: object
    right: object

    def __str__(self) -> str:
        return f"turn {self.turn}: {self.where}\n  python: {self.left!r}\n  other:  {self.right!r}"


def compare(left: Sequence[dict[str, Any]], right: Sequence[dict[str, Any]]) -> Divergence | None:
    """The first disagreement between two traces, or None if they played the same battle."""
    for index, (mine, theirs) in enumerate(zip(left, right, strict=False)):
        if mine["actions"] != theirs["actions"]:
            return Divergence(index + 1, "actions", mine["actions"], theirs["actions"])
        if (found := _first_event_difference(index + 1, mine["events"], theirs["events"])) is not None:
            return found
        if mine.get("drawn") != theirs.get("drawn"):
            return Divergence(index + 1, "draws taken", mine.get("drawn"), theirs.get("drawn"))
        if (found := _first_state_difference(index + 1, "state", mine["state"], theirs["state"])) is not None:
            return found
    if len(left) != len(right):
        return Divergence(min(len(left), len(right)) + 1, "battle length", len(left), len(right))
    return None


def _first_event_difference(turn: int, mine: Sequence[object], theirs: Sequence[object]) -> Divergence | None:
    for position, (a, b) in enumerate(zip(mine, theirs, strict=False)):
        if a != b:
            return Divergence(turn, f"event {position}", a, b)
    if len(mine) != len(theirs):
        extra = mine[len(theirs) :] or theirs[len(mine) :]
        return Divergence(turn, f"event count ({len(mine)} vs {len(theirs)})", extra[0] if extra else None, None)
    return None


def _first_state_difference(turn: int, path: str, mine: object, theirs: object) -> Divergence | None:
    """Walk two digests together and name the exact field that parted, not the whole tree."""
    if isinstance(mine, dict) and isinstance(theirs, dict):
        for key in sorted(set(mine) | set(theirs)):
            found = _first_state_difference(turn, f"{path}.{key}", mine.get(key), theirs.get(key))
            if found is not None:
                return found
        return None
    if isinstance(mine, list) and isinstance(theirs, list) and len(mine) == len(theirs):
        for index, (a, b) in enumerate(zip(mine, theirs, strict=True)):
            found = _first_state_difference(turn, f"{path}[{index}]", a, b)
            if found is not None:
                return found
        return None
    return None if mine == theirs else Divergence(turn, path, mine, theirs)


def write(scenario: Scenario, expected: Sequence[dict[str, Any]], directory: Path, name: str) -> None:
    """A scenario and the trace this engine produces for it, as two files a second engine can read."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.scenario.json").write_text(scenario.to_json())
    (directory / f"{name}.trace.json").write_text(json.dumps(list(expected), indent=2))


def replacement_chooser(state: BattleState, side_index: int) -> Action | None:
    """The lowest-index healthy benched Pokemon, always."""
    side = state.sides[side_index]
    for index, member in enumerate(side.team):
        if index != side.active[0] and not member.is_fainted():
            return Action(action=ActionType.SWITCH_OUT, switch_in=member)
    return None


def record(
    teams: tuple[Sequence[PokemonSpec], Sequence[PokemonSpec]],
    choose: "Chooser",
    seed: int = 0,
    max_turns: int = 200,
) -> tuple[Scenario, list[dict[str, Any]]]:
    """Play one battle through this engine, writing down everything needed to replay it elsewhere."""
    rng = TapeRNG(seed=seed)
    sides = (
        SideState(team=[build_pokemon(spec) for spec in teams[0]]),
        SideState(team=[build_pokemon(spec) for spec in teams[1]]),
    )
    state = BattleState(sides=sides, rng=rng)
    named: list[tuple[str, str]] = []
    turns: list[dict[str, Any]] = []
    while state.outcome is None and len(turns) < max_turns:
        actions = {side: choose(state, side) for side in (0, 1)}
        labels = (name_action(actions[0], state, 0), name_action(actions[1], state, 1))
        log = step(state, actions, replacement_chooser)
        named.append(labels)
        turns.append(
            {
                "actions": list(labels),
                "events": [entry_json(e) for e in log],
                "state": state_digest(state),
                "drawn": rng.drawn,
            }
        )
    scenario = Scenario(
        teams=(
            [encode_spec(spec) for spec in teams[0]],
            [encode_spec(spec) for spec in teams[1]],
        ),
        actions=named,
        tape=list(rng.tape),
        seed=seed,
    )
    return scenario, turns


def self_check(scenario: Scenario, expected: Sequence[dict[str, Any]]) -> Divergence | None:
    """Replay a scenario through *this* engine and confirm it still agrees with itself."""
    return compare(expected, trace(scenario))


type ActionPicker = Callable[[BattleState, int, list[Action]], Action]


@dataclass
class PlayedScenario:
    """A battle played as the bot plays one, reduced to its inputs and every decision with its legal actions."""

    teams: tuple[list[dict[str, Any]], list[dict[str, Any]]]
    orders: tuple[list[int], list[int]]
    decisions: list[dict[str, Any]]
    tape: list[float | int]
    seed: int = 0

    def to_json(self) -> str:
        return json.dumps(
            {
                "version": FORMAT_VERSION,
                "seed": self.seed,
                "teams": list(self.teams),
                "orders": list(self.orders),
                "decisions": self.decisions,
                "tape": self.tape,
            }
        )

    @classmethod
    def from_json(cls, text: str) -> "PlayedScenario":
        payload = json.loads(text)
        return cls(
            teams=(payload["teams"][0], payload["teams"][1]),
            orders=(payload["orders"][0], payload["orders"][1]),
            decisions=payload["decisions"],
            tape=payload["tape"],
            seed=payload["seed"],
        )


def _named_legal(state: BattleState, side: int) -> list[str]:
    return [name_action(action, state, side) for action in legal_actions(state, side)]


def lead_first(lead: int, size: int) -> list[int]:
    """A team order with `lead` first and the rest in their listed order."""
    return [lead, *(index for index in range(size) if index != lead)]


def record_played(
    teams: tuple[Sequence[PokemonSpec], Sequence[PokemonSpec]],
    orders: tuple[Sequence[int], Sequence[int]],
    pick: ActionPicker,
    seed: int = 0,
    max_turns: int = 200,
    observe: Callable[[BattleState, int, str], None] | None = None,
) -> tuple[PlayedScenario, list[dict[str, Any]]]:
    """Play one battle exactly as `runner.run_battle` does, recording decisions, legal actions, tape and trace."""
    rng = TapeRNG(seed=seed)
    sides = (
        SideState(team=[build_pokemon(teams[0][i]) for i in orders[0]]),
        SideState(team=[build_pokemon(teams[1][i]) for i in orders[1]]),
    )
    state = BattleState(sides=sides, rng=rng)
    decisions: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []

    def segment(labels: list[str], log: BattleLog) -> None:
        segments.append(
            {
                "actions": labels,
                "events": [entry_json(e) for e in log],
                "state": state_digest(state),
                "drawn": rng.drawn,
            }
        )

    def seen(live: BattleState, side: int, kind: str) -> None:
        if observe is not None:
            observe(live, side, kind)

    seen(state, 0, "lead")
    seen(state, 1, "lead")
    pivots: list[str] = []

    def switch_chooser(live: BattleState, side: int) -> Action:
        seen(live, side, "switch")
        offered = legal_actions(live, side)
        chosen = pick(live, side, offered)
        name = name_action(chosen, live, side)
        decisions.append({"kind": "pivot", "side": side, "action": name, "legal": _named_legal(live, side)})
        pivots.append(name)
        return chosen

    turns = 0
    while state.outcome is None and turns < max_turns:
        seen(state, 0, "turn")
        seen(state, 1, "turn")
        legal = [_named_legal(state, 0), _named_legal(state, 1)]
        actions = {side: pick(state, side, legal_actions(state, side)) for side in (0, 1)}
        labels = [name_action(actions[0], state, 0), name_action(actions[1], state, 1)]
        decisions.append({"kind": "turn", "actions": labels, "legal": legal})
        pivots.clear()
        log = step(state, actions, switch_chooser)
        segment(labels + pivots, log)
        turns += 1
        for side in (0, 1):
            own = state.sides[side]
            while state.outcome is None and (own.active_pokemon.is_fainted() or own.needs_switch):
                seen(state, side, "switch")
                offered = legal_actions(state, side)
                chosen = pick(state, side, offered)
                name = name_action(chosen, state, side)
                decisions.append({"kind": "replace", "side": side, "action": name, "legal": _named_legal(state, side)})
                segment([f"replace:{side}", name], apply_forced_switch(state, side, chosen))

    scenario = PlayedScenario(
        teams=([encode_spec(s) for s in teams[0]], [encode_spec(s) for s in teams[1]]),
        orders=(list(orders[0]), list(orders[1])),
        decisions=decisions,
        tape=list(rng.tape),
        seed=seed,
    )
    return scenario, segments
