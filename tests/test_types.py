from battle_sim.models.type_matchups import type_effectiveness
from battle_sim.utils import Type


def test_dual_typing_effectiveness():
    # Fighting vs Ghost/Dark (Spiritomb): 0x * 2x = 0x
    multiplier = type_effectiveness(Type.FIGHTING, (Type.GHOST, Type.DARK))
    assert multiplier == 0.0

    # Normal vs Rock/Steel (Aggron): 0.5x * 0.5x = 0.25x
    multiplier = type_effectiveness(Type.NORMAL, (Type.ROCK, Type.STEEL))
    assert multiplier == 0.25

    # Flying vs Rock/Ground (Golem): 0.5x * 1x = 0.5x
    multiplier = type_effectiveness(Type.FLYING, (Type.ROCK, Type.GROUND))
    assert multiplier == 0.5

    # Flying vs Grass/Rock (Cradily): 2x * 0.5x = 1x
    multiplier = type_effectiveness(Type.FLYING, (Type.GRASS, Type.ROCK))
    assert multiplier == 1.0

    # Water vs Fire/Flying (Charizard): 2x * 1x = 2x
    multiplier = type_effectiveness(Type.WATER, (Type.FIRE, Type.FLYING))
    assert multiplier == 2.0

    # Ice vs Dragon/Ground (Garchomp): 2x * 2x = 4x
    multiplier = type_effectiveness(Type.ICE, (Type.DRAGON, Type.GROUND))
    assert multiplier == 4.0
