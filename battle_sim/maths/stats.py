from math import floor

from battle_sim.models.stats import StatStages, StatTotals
from battle_sim.utils import Nature, NatureEffect, Stats


def calculate_total_hp(base: int, iv: int, ev: int, lvl: int) -> int:
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


def calculate_effective_stat(stat_totals: StatTotals, stat_stages: StatStages, stat: Stats) -> int:
    unmodified_value = getattr(stat_totals, stat.name)
    stage_level = getattr(stat_stages, stat.name)

    stage_base = 2

    # TODO: add accuracy/evasion handling
    # stage_base = 3 if stat in (Stats.ACCURACY, Stats.EVASION) else 2

    if stage_level >= 0:
        numerator, denominator = stage_base + stage_level, stage_base
    else:
        numerator, denominator = stage_base, stage_base - stage_level

    modified_value = unmodified_value * numerator // denominator

    # TODO: status/items/abilities/field (nerd modifiers)
    # if stat is Stats.SPEED and self.paralyzed:
    #     val = val * 1 // 2
    # if stat is Stats.SPEED and self.tailwind:
    #     val = val * 2 // 1
    # if stat is Stats.ATTACK and self.item == Item.CHOICE_BAND:
    #     val = val * 3 // 2

    return max(1, modified_value)
