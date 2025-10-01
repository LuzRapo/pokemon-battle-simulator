from math import floor

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
