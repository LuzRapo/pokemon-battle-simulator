import pytest

from battle_sim.database.sample_moves import DRACO_METEOR, EARTHQUAKE, ROCK_SLIDE, SWORDS_DANCE
from battle_sim.models.moves import MoveSet
from battle_sim.models.pokemon import Pokemon
from battle_sim.models.stats import BaseStats, EVs, IVs
from battle_sim.utils import Nature, Stats, Type


@pytest.fixture
def garchomp_factory():
    """I Guess We Doin' Garchomps Now"""

    def _make(nickname: str) -> Pokemon:
        pokemon = Pokemon(
            name="Garchomp",
            nickname=nickname,
            level=78,
            base_stats=BaseStats(HP=108, ATTACK=130, DEFENCE=95, SP_ATTACK=80, SP_DEFENCE=85, SPEED=102),
            effort_values=EVs(HP=74, ATTACK=190, DEFENCE=91, SP_ATTACK=48, SP_DEFENCE=84, SPEED=23),
            individual_values=IVs(HP=24, ATTACK=12, DEFENCE=30, SP_ATTACK=16, SP_DEFENCE=23, SPEED=5),
            types=(Type.DRAGON, Type.GROUND),
            moves=MoveSet(EARTHQUAKE, SWORDS_DANCE, DRACO_METEOR, ROCK_SLIDE),
            nature=Nature.ADAMANT,
        )
        pokemon.reset_live_stats()
        return pokemon

    return _make


def assert_stats(pokemon: Pokemon, expected: dict):
    for stat_name, value in expected.items():
        assert getattr(pokemon.stat_totals, stat_name) == value, f"{stat_name} mismatch"


@pytest.fixture
def expected_garchomp_stats():
    # From: https://bulbapedia.bulbagarden.net/wiki/Stat#Formula
    return dict(HP=289, ATTACK=278, DEFENCE=193, SP_ATTACK=135, SP_DEFENCE=171, SPEED=171)


def test_hp_damage_heal_and_faint(garchomp_factory, expected_garchomp_stats):
    chompert = garchomp_factory("Chompert")
    assert_stats(chompert, expected_garchomp_stats)

    assert chompert.apply_damage(100) == 100
    assert chompert.live_stats.HP == 189

    assert chompert.apply_healing(11) == 11
    assert chompert.live_stats.HP == 200

    # Healing caps at max HP
    chompert.apply_healing(1000)
    assert chompert.live_stats.HP == chompert.stat_totals.HP

    # HP cannot drop below 0
    chompert.apply_damage(9999)
    assert chompert.live_stats.HP == 0
    assert chompert.is_fainted()


def test_stat_stage_raises_and_drops(garchomp_factory, expected_garchomp_stats):
    chompina = garchomp_factory("Chompina")
    assert_stats(chompina, expected_garchomp_stats)

    # +2 Attack (Swords Dance)
    chompina.stat_stages.ATTACK += 2
    assert chompina.effective_stat(Stats.ATTACK) == 556

    # -1 Attack (Intimidate)
    chompina.stat_stages.ATTACK -= 1
    assert chompina.effective_stat(Stats.ATTACK) == 417

    # +3 Sp. Atk (Tail Glow style)
    chompina.stat_stages.SP_ATTACK += 3
    assert chompina.effective_stat(Stats.SP_ATTACK) == 337


def test_reset_restores_hp_and_stages(garchomp_factory, expected_garchomp_stats):
    chomperinho = garchomp_factory("Chomperinho")
    assert_stats(chomperinho, expected_garchomp_stats)

    chomperinho.apply_damage(150)
    chomperinho.stat_stages.ATTACK = 4
    chomperinho.stat_stages.SP_ATTACK = -3
    chomperinho.reset_live_stats()
    chomperinho.reset_stat_stages()
    assert chomperinho.live_stats.HP == chomperinho.stat_totals.HP
    assert chomperinho.stat_stages.ATTACK == 0
    assert chomperinho.stat_stages.SP_ATTACK == 0
