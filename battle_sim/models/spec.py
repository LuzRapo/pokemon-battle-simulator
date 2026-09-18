from dataclasses import dataclass
from enum import Enum, auto

from pydantic import BaseModel, Field, model_validator

from battle_sim.models.stats import EVs, IVs
from battle_sim.utils import Ability, Item, Nature


class PokemonSpec(BaseModel):
    species: str
    nickname: str | None = None
    level: int = Field(default=100, ge=1, le=100)
    ability: Ability = Ability.NONE
    item: Item = Item.NONE
    nature: Nature = Nature.HARDY
    effort_values: EVs = Field(default_factory=lambda: EVs())
    individual_values: IVs = Field(default_factory=lambda: IVs())
    moves: list[str] = Field(default_factory=list, max_length=4)
    # PP Ups, the real mechanic: three of them take a move to 8/5 of its listed PP. Written on the
    # spec rather than baked into a move, because it belongs to one Pokemon's copy of that move.
    pp_ups: int = Field(default=0, ge=0, le=3)

    @model_validator(mode="after")
    def _require_species(self) -> "PokemonSpec":
        if not self.species.strip():
            raise ValueError("species must be non-empty")
        return self


class ParseWarningKind(Enum):
    UNKNOWN_ABILITY = auto()
    UNKNOWN_ITEM = auto()
    UNKNOWN_NATURE = auto()
    MALFORMED_LEVEL = auto()
    UNKNOWN_STAT_KEY = auto()
    MALFORMED_STAT_VALUE = auto()
    EXTRA_MOVES_TRUNCATED = auto()


@dataclass(frozen=True)
class ParseWarning:
    kind: ParseWarningKind
    block_index: int
    detail: str


@dataclass(frozen=True)
class ParseResult:
    specs: tuple[PokemonSpec, ...]
    warnings: tuple[ParseWarning, ...] = ()
