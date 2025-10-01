from battle_sim.models.pokemon import Pokemon
from battle_sim.models.stats import BaseStats, EVs, IVs
from battle_sim.utils import Nature


def test_adamant_garchomp_stats():
    """Garchomp at Lv. 100 with 252 Atk / 252 Spe / 4 SpD and Adamant nature."""

    chomp = Pokemon(
        name="Garchomp",
        nickname="Chompy",
        nature=Nature.ADAMANT,
        level=100,
        base_stats=BaseStats(HP=108, ATTACK=130, DEFENCE=95, SP_ATTACK=80, SP_DEFENCE=85, SPEED=102),
        effort_values=EVs(HP=0, ATTACK=252, DEFENCE=0, SP_ATTACK=0, SP_DEFENCE=4, SPEED=252),
        individual_values=IVs(),  # all 31 IVs
    )
    totals = chomp.stat_totals

    assert totals.HP == 357
    assert totals.ATTACK == 394
    assert totals.SP_ATTACK == 176
    assert totals.SPEED == 303
    assert totals.ATTACK > totals.SP_ATTACK
    assert totals.HP > 0


def test_all_natures_are_unique_pairs():
    seen = set()
    for nature in Nature:
        eff = nature.value
        pair = (eff.UP, eff.DOWN)
        assert pair not in seen, f"Duplicate UP/DOWN pair found in {nature}"
        seen.add(pair)
    assert len(seen) == 25
