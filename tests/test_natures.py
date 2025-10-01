from typing import Any

import pytest

from battle_sim.models.pokemon import Pokemon
from battle_sim.models.stats import BaseStats, EVs, IVs
from battle_sim.utils import Nature, Type


@pytest.fixture
def garchomp_setup():
    return dict(
        name="Garchomp",
        nickname="Chompy",
        level=100,
        base_stats=BaseStats(HP=108, ATTACK=130, DEFENCE=95, SP_ATTACK=80, SP_DEFENCE=85, SPEED=102),
        effort_values=EVs(HP=0, ATTACK=252, DEFENCE=0, SP_ATTACK=0, SP_DEFENCE=4, SPEED=252),
        individual_values=IVs(),  # defaults to 31s
        types=(Type.DRAGON, Type.GROUND),
    )


@pytest.mark.parametrize(
    "nature, expected",
    [
        (Nature.ADAMANT, dict(HP=357, ATTACK=394, DEFENCE=226, SP_ATTACK=176, SP_DEFENCE=207, SPEED=303)),
        (Nature.JOLLY, dict(HP=357, ATTACK=359, DEFENCE=226, SP_ATTACK=176, SP_DEFENCE=207, SPEED=333)),
        (Nature.BASHFUL, dict(HP=357, ATTACK=359, DEFENCE=226, SP_ATTACK=196, SP_DEFENCE=207, SPEED=303)),
    ],
)
def test_garchomp_nature_effects(garchomp_setup: dict[str, Any], nature: Nature, expected: dict[str, int]):
    chompy = Pokemon(nature=nature, **garchomp_setup)
    totals = chompy.stat_totals

    for stat, val in expected.items():
        assert getattr(totals, stat) == val, f"{nature.name} {stat} mismatch"


def test_all_natures_are_unique_pairs():
    seen = set()
    for nature in Nature:
        eff = nature.value
        pair = (eff.UP, eff.DOWN)
        assert pair not in seen, f"Duplicate UP/DOWN pair found in {nature}"
        seen.add(pair)
    assert len(seen) == 25
