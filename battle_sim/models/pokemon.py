from math import floor

from pydantic import BaseModel, Field

from battle_sim.models.stats import BaseStats, EVs, IVs, StatTotals
from battle_sim.utils import Nature, NatureEffect, Stats


class Pokemon(BaseModel):
    name: str
    nickname: str
    level: int = Field(default=50, ge=1, le=100)
    nature: Nature
    effort_values: EVs
    individual_values: IVs
    base_stats: BaseStats
    # items, abilities, types, moves, etc. can be added later

    @property
    def stat_totals(self) -> StatTotals:
        """Calculate final stats from base stats, IVs, EVs, level, and nature."""
        lvl = self.level
        nat = self.nature
        ivs = self.individual_values
        evs = self.effort_values
        bs = self.base_stats

        def _nature_multiplier(nature: Nature, stat: Stats) -> float:
            eff: NatureEffect = nature.value
            if eff.UP == eff.DOWN:
                return 1.0
            elif stat == eff.UP:
                return 1.1
            elif stat == eff.DOWN:
                return 0.9
            else:
                return 1.0

        def _calculate_total_hp(base: int, iv: int, ev: int, lvl: int) -> int:
            return floor(((2 * base + iv + floor(ev / 4)) * lvl) / 100) + lvl + 10

        def _calculate_total_stat(base: int, iv: int, ev: int, lvl: int, nature: Nature, stat: Stats) -> int:
            raw = floor(((2 * base + iv + floor(ev / 4)) * lvl) / 100) + 5
            mult = _nature_multiplier(nature, stat)
            return floor(raw * mult)

        return StatTotals(
            HP=_calculate_total_hp(bs.HP, ivs.HP, evs.HP, lvl),
            ATTACK=_calculate_total_stat(bs.ATTACK, ivs.ATTACK, evs.ATTACK, lvl, nat, "ATTACK"),
            DEFENCE=_calculate_total_stat(bs.DEFENCE, ivs.DEFENCE, evs.DEFENCE, lvl, nat, "DEFENCE"),
            SP_ATTACK=_calculate_total_stat(bs.SP_ATTACK, ivs.SP_ATTACK, evs.SP_ATTACK, lvl, nat, "SP_ATTACK"),
            SP_DEFENCE=_calculate_total_stat(bs.SP_DEFENCE, ivs.SP_DEFENCE, evs.SP_DEFENCE, lvl, nat, "SP_DEFENCE"),
            SPEED=_calculate_total_stat(bs.SPEED, ivs.SPEED, evs.SPEED, lvl, nat, "SPEED"),
        )
