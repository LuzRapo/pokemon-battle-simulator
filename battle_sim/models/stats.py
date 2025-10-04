from dataclasses import dataclass

from pydantic import BaseModel, Field, model_validator


@dataclass(frozen=True)
class BaseStats:
    HP: int  # Fixed stats belonging to the species
    ATTACK: int
    DEFENCE: int
    SP_ATTACK: int
    SP_DEFENCE: int
    SPEED: int


@dataclass(frozen=True)
class StatTotals:
    HP: int  # Fixed stats calculated from base stats, nature, and IVs/EVs
    ATTACK: int
    DEFENCE: int
    SP_ATTACK: int
    SP_DEFENCE: int
    SPEED: int


# From https://bulbapedia.bulbagarden.net/wiki/Stat:
# "Aside from Shedinja's HP (which is always 1), the lowest a stat can ever possibly be is 4."
class LiveStats(BaseModel):
    HP: int = Field(default=1, ge=0)
    ATTACK: int = Field(default=4, ge=4)
    DEFENCE: int = Field(default=4, ge=4)
    SP_ATTACK: int = Field(default=4, ge=4)
    SP_DEFENCE: int = Field(default=4, ge=4)
    SPEED: int = Field(default=4, ge=4)


class StatStages(BaseModel):
    ATTACK: int = Field(default=0, ge=-6, le=6)
    DEFENCE: int = Field(default=0, ge=-6, le=6)
    SP_ATTACK: int = Field(default=0, ge=-6, le=6)
    SP_DEFENCE: int = Field(default=0, ge=-6, le=6)
    SPEED: int = Field(default=0, ge=-6, le=6)
    ACCURACY: int = Field(default=0, ge=-6, le=6)
    EVASION: int = Field(default=0, ge=-6, le=6)


class EVs(BaseModel):
    HP: int = Field(default=0, ge=0, le=252)
    ATTACK: int = Field(default=0, ge=0, le=252)
    DEFENCE: int = Field(default=0, ge=0, le=252)
    SP_ATTACK: int = Field(default=0, ge=0, le=252)
    SP_DEFENCE: int = Field(default=0, ge=0, le=252)
    SPEED: int = Field(default=0, ge=0, le=252)

    @property
    def total(self) -> int:
        return self.HP + self.ATTACK + self.DEFENCE + self.SP_ATTACK + self.SP_DEFENCE + self.SPEED

    @property
    def remaining(self) -> int:
        return 510 - self.total

    @model_validator(mode="after")
    def _check_total(self) -> "EVs":
        if self.total > 510:
            raise ValueError(f"Total EVs exceed 510 (got {self.total}).")
        return self


class IVs(BaseModel):
    HP: int = Field(default=31, ge=0, le=31)
    ATTACK: int = Field(default=31, ge=0, le=31)
    DEFENCE: int = Field(default=31, ge=0, le=31)
    SP_ATTACK: int = Field(default=31, ge=0, le=31)
    SP_DEFENCE: int = Field(default=31, ge=0, le=31)
    SPEED: int = Field(default=31, ge=0, le=31)

    def total(self) -> int:
        return self.HP + self.ATTACK + self.DEFENCE + self.SP_ATTACK + self.SP_DEFENCE + self.SPEED
