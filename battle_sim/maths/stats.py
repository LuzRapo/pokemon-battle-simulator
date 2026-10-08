from math import floor

from battle_sim.models.stats import StatStages, StatTotals
from battle_sim.utils import Nature, NatureEffect, StageBases, Stats


def calculate_total_hp(base: int, iv: int, ev: int, lvl: int) -> int:
    if base == 1:  # Shedinja: HP is always exactly 1, regardless of level/IVs/EVs
        return 1
    return floor(((2 * base + iv + floor(ev / 4)) * lvl) / 100) + lvl + 10


def calculate_total_stat(base: int, iv: int, ev: int, lvl: int, nature: Nature, stat: Stats) -> int:
    def _nature_multiplier(nature: Nature, stat: Stats) -> float:
        effect: NatureEffect = nature.value
        if effect.UP != effect.DOWN:
            if stat == effect.UP:
                return 1.1
            if stat == effect.DOWN:
                return 0.9
        return 1.0

    raw = floor(((2 * base + iv + floor(ev / 4)) * lvl) / 100) + 5
    mult = _nature_multiplier(nature, stat)
    return floor(raw * mult)


def apply_stage_multiplier(unmodified_value: int, stage_level: int) -> int:
    stage_base = int(StageBases.STATS)
    if stage_level >= 0:
        numerator, denominator = stage_base + stage_level, stage_base
    else:
        numerator, denominator = stage_base, stage_base - stage_level
    return max(1, unmodified_value * numerator // denominator)


def calculate_effective_stat(stat_totals: StatTotals, stat_stages: StatStages, stat: Stats) -> int:
    assert stat in (Stats.ATTACK, Stats.DEFENCE, Stats.SP_ATTACK, Stats.SP_DEFENCE, Stats.SPEED)
    return apply_stage_multiplier(stat_totals[stat], stat_stages[stat])
