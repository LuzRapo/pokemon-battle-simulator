"""Validated views over the vendored Showdown JSON — only consumed fields are modelled, the rest ignored."""

from pydantic import BaseModel, ConfigDict, Field


class RawSelf(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    boosts: dict[str, int] | None = None
    volatile_status: str | None = Field(default=None, alias="volatileStatus")


class RawSecondary(BaseModel):
    """A secondary effect entry (`secondary` or one of `secondaries`)."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    chance: int = 0
    status: str | None = None
    volatile_status: str | None = Field(default=None, alias="volatileStatus")
    boosts: dict[str, int] | None = None
    self_effects: RawSelf | None = Field(default=None, alias="self")


class RawMoveData(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    name: str
    type: str
    category: str
    target: str
    pp: int
    priority: int
    accuracy: bool | int
    base_power: int = Field(default=0, alias="basePower")
    crit_ratio: int | None = Field(default=None, alias="critRatio")
    flags: dict[str, int] = Field(default_factory=dict)
    multihit: int | tuple[int, int] | None = None
    recoil: tuple[int, int] | None = None
    drain: tuple[int, int] | None = None
    heal: tuple[int, int] | None = None
    damage: int | str | None = None
    weather: str | None = None
    terrain: str | None = None
    pseudo_weather: str | None = Field(default=None, alias="pseudoWeather")
    side_condition: str | None = Field(default=None, alias="sideCondition")
    volatile_status: str | None = Field(default=None, alias="volatileStatus")
    status: str | None = None
    boosts: dict[str, int] | None = None
    self_effects: RawSelf | None = Field(default=None, alias="self")
    secondary: RawSecondary | None = None
    secondaries: list[RawSecondary] | None = None
    self_switch: bool | str = Field(default=False, alias="selfSwitch")
    self_destruct: str | None = Field(default=None, alias="selfdestruct")
    force_switch: bool = Field(default=False, alias="forceSwitch")
    stalling_move: bool = Field(default=False, alias="stallingMove")
    struggle_recoil: bool = Field(default=False, alias="struggleRecoil")
    has_crash_damage: bool = Field(default=False, alias="hasCrashDamage")
    is_nonstandard: str | None = Field(default=None, alias="isNonstandard")
    is_z: bool | str | None = Field(default=None, alias="isZ")
    is_max: bool | str | None = Field(default=None, alias="isMax")

    def all_secondaries(self) -> list[RawSecondary]:
        """Both the singular `secondary` and the plural `secondaries` (e.g. Fire Fang)."""
        singular = [self.secondary] if self.secondary is not None else []
        return [*singular, *(self.secondaries or [])]


class RawSpeciesData(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    name: str
    num: int | None = None
    types: list[str] = Field(default_factory=list)
    base_stats: dict[str, int] | None = Field(default=None, alias="baseStats")
    abilities: dict[str, str] = Field(default_factory=dict)
    height_m: float | None = Field(default=None, alias="heightm")
    weight_kg: float | None = Field(default=None, alias="weightkg")
    evos: list[str] = Field(default_factory=list)
    forme: str | None = None
    base_species: str | None = Field(default=None, alias="baseSpecies")
    required_item: str | None = Field(default=None, alias="requiredItem")
    battle_only: str | list[str] | None = Field(default=None, alias="battleOnly")
    is_nonstandard: str | None = Field(default=None, alias="isNonstandard")
    is_cosmetic_forme: bool = Field(default=False, alias="isCosmeticForme")
    tags: list[str] = Field(default_factory=list)

    def battle_only_parents(self) -> tuple[str, ...]:
        if self.battle_only is None:
            return ()
        return (self.battle_only,) if isinstance(self.battle_only, str) else tuple(self.battle_only)


class RawLearnsetData(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    learnset: dict[str, list[str]] = Field(default_factory=dict)


class RawRandbatsRole(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    abilities: list[str] = Field(default_factory=list)
    items: list[str] = Field(default_factory=list)
    moves: list[str] = Field(default_factory=list)


class RawRandbatsSpecies(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    items: list[str] = Field(default_factory=list)  # species-level fallback for the rare itemless role
    roles: dict[str, RawRandbatsRole] = Field(default_factory=dict)
