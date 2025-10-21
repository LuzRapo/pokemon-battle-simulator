import pytest

from battle_sim.database.sample_moves import DRACO_METEOR, EARTHQUAKE, ROCK_SLIDE, SWORDS_DANCE
from battle_sim.models.moves import MoveSet
from battle_sim.models.pokemon import Pokemon
from battle_sim.models.stats import BaseStats, EVs, IVs
from battle_sim.utils import Nature, Type


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
