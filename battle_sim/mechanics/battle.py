from functools import cached_property

from pydantic import BaseModel, Field, model_validator

from battle_sim.maths.rng import RNG
from battle_sim.mechanics.effects import EffectRegistry, register_active
from battle_sim.mechanics.events import EventBus
from battle_sim.models.actions import Action
from battle_sim.models.pokemon import FormSnapshot, Pokemon
from battle_sim.utils import Ability, BattleFormat, Hazards, Outcome, PseudoWeather, Terrain, Weather


class FieldState(BaseModel):
    weather: Weather = Weather.NONE
    weather_turns_left: int = Field(default=0, ge=0)
    terrain: Terrain = Terrain.NONE
    terrain_turns_left: int = Field(default=0, ge=0)
    pseudo_weather: dict[PseudoWeather, int] = Field(default_factory=dict)


class SideState(BaseModel):
    team: list[Pokemon] = Field(min_length=1, max_length=6)
    active: list[int] = Field(default_factory=lambda: [0])
    hazards: dict[Hazards, int] = Field(default_factory=dict)
    screens: dict[Hazards, int] = Field(default_factory=dict)
    tailwind_turns: int = Field(default=0, ge=0)
    needs_switch: bool = False
    chosen_action: Action | None = None  # this turn's submitted action (Sucker Punch reads it)
    acted_this_turn: bool = False
    wish_pending: int = Field(default=0, ge=0)  # HP a Wish restores when its turn comes
    wish_turns: int = Field(default=0, ge=0)
    healing_wish_pending: bool = False  # the next pokemon sent in is fully restored
    pending_substitute: int = Field(default=0, ge=0)  # Shed Tail: the substitute passed to the replacement

    @model_validator(mode="after")
    def _check_active_indices(self) -> "SideState":
        for i in self.active:
            if not 0 <= i < len(self.team):
                raise ValueError(f"Active index {i} out of range for team of size {len(self.team)}.")
        if len(set(self.active)) != len(self.active):
            raise ValueError(f"Active indices must be unique (got {self.active}).")
        return self

    @property
    def active_pokemon(self) -> Pokemon:
        return self.team[self.active[0]]

    @property
    def bench(self) -> list[Pokemon]:
        return [pokemon for i, pokemon in enumerate(self.team) if i not in self.active]


_ACTIVE_COUNT_BY_FORMAT: dict[BattleFormat, int] = {BattleFormat.SINGLES: 1}


class BattleState(BaseModel):
    format: BattleFormat = BattleFormat.SINGLES
    turn: int = Field(default=0, ge=0)
    sides: tuple[SideState, SideState]
    field: FieldState = Field(default_factory=FieldState)
    rng: RNG = Field(default_factory=RNG)
    outcome: Outcome | None = None
    transforms: dict[int, FormSnapshot] = Field(default_factory=dict)  # id(pokemon) -> pre-Transform form

    model_config = {"arbitrary_types_allowed": True}

    @cached_property
    def _wiring(self) -> tuple[EventBus, EffectRegistry]:
        """The battle-lifetime bus with both leads' handlers registered, built on first use."""
        bus = EventBus()
        registry = EffectRegistry()
        for side in self.sides:
            register_active(bus, registry, side.active_pokemon)
        return bus, registry

    @property
    def bus(self) -> EventBus:
        return self._wiring[0]

    @property
    def effects(self) -> EffectRegistry:
        return self._wiring[1]

    @model_validator(mode="after")
    def _check_active_count_matches_format(self) -> "BattleState":
        expected = _ACTIVE_COUNT_BY_FORMAT[self.format]
        for index, side in enumerate(self.sides):
            if len(side.active) != expected:
                raise ValueError(
                    f"Side {index} has {len(side.active)} active pokemon; {self.format.name} expects {expected}."
                )
        return self


def effective_weather(state: BattleState) -> Weather:
    """Air Lock suppresses weather effects while its holder is active (the weather itself persists)."""
    for side in state.sides:
        active = side.active_pokemon
        if not active.is_fainted() and active.ability is Ability.AIR_LOCK:
            return Weather.NONE
    return state.field.weather
