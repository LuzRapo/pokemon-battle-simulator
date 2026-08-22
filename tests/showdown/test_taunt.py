import pytest

from tests.showdown.conftest import assert_stat_stage, create_battle, make_choices


def test_taunt_should_prevent_status_moves():
    battle = create_battle(
        [{"species": "Sableye", "moves": ["taunt"]}],
        [{"species": "Chansey", "moves": ["calmmind"]}],
    )
    battle.sides[0].active_pokemon.stat_stages.SPEED = 6
    log = make_choices(battle, "move taunt", "move calmmind")
    assert_stat_stage(battle.sides[1].active_pokemon, "spa", 0)
    assert_stat_stage(battle.sides[1].active_pokemon, "spd", 0)
    assert any("Calm Mind" in line and "Taunt" in line for line in log)


@pytest.mark.skip(reason="Z-moves not implemented")
def test_taunt_does_not_prevent_z_powered_status_moves(): ...


@pytest.mark.skip(reason="Z-moves not implemented")
def test_taunt_hackmons_prevents_z_status_without_crystal(): ...
