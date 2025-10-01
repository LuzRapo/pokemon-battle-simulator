from pydantic import BaseModel, Field

from battle_sim.maths.stats import calculate_total_hp, calculate_total_stat
from battle_sim.models.stats import BaseStats, EVs, IVs, StatTotals
from battle_sim.models.type_matchups import TypePair
from battle_sim.utils import Nature


class Pokemon(BaseModel):
    name: str
    nickname: str
    level: int = Field(default=50, ge=1, le=100)
    nature: Nature
    effort_values: EVs
    individual_values: IVs
    base_stats: BaseStats
    types: TypePair
    # items, abilities, moves, etc. can be added later

    @property
    def stat_totals(self) -> StatTotals:
        """Calculate final stats from base stats, IVs, EVs, level, and nature."""
        lvl = self.level
        nat = self.nature
        ivs = self.individual_values
        evs = self.effort_values
        bs = self.base_stats

        return StatTotals(
            HP=calculate_total_hp(bs.HP, ivs.HP, evs.HP, lvl),
            ATTACK=calculate_total_stat(
                bs.ATTACK, ivs.ATTACK, evs.ATTACK, lvl, nat, "ATTACK"
            ),
            DEFENCE=calculate_total_stat(
                bs.DEFENCE, ivs.DEFENCE, evs.DEFENCE, lvl, nat, "DEFENCE"
            ),
            SP_ATTACK=calculate_total_stat(
                bs.SP_ATTACK, ivs.SP_ATTACK, evs.SP_ATTACK, lvl, nat, "SP_ATTACK"
            ),
            SP_DEFENCE=calculate_total_stat(
                bs.SP_DEFENCE, ivs.SP_DEFENCE, evs.SP_DEFENCE, lvl, nat, "SP_DEFENCE"
            ),
            SPEED=calculate_total_stat(
                bs.SPEED, ivs.SPEED, evs.SPEED, lvl, nat, "SPEED"
            ),
        )
