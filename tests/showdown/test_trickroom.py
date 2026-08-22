import pytest

from tests.showdown.conftest import assert_status, create_battle, make_choices


def test_trickroom_slower_acts_first_in_priority_bracket():
    battle = create_battle(
        [{"species": "Bronzong", "moves": ["spore", "trickroom"]}],
        [{"species": "Ninjask", "moves": ["poisonjab", "spore"]}],
        seed=0,
    )
    make_choices(battle, "move trickroom", "move poisonjab")
    make_choices(battle, "move spore", "move spore")
    assert_status(battle.sides[0].active_pokemon, "")
    assert_status(battle.sides[1].active_pokemon, "slp")


def test_trickroom_does_not_change_priority_brackets():
    # Protect (+4) still acts before Poison Jab (+0) under Trick Room: TR reorders
    # within a priority bracket, it does not invert brackets.
    battle = create_battle(
        [{"species": "Bronzong", "moves": ["trickroom", "protect"]}],
        [{"species": "Ninjask", "moves": ["poisonjab"]}],
        seed=0,
    )
    bronzong = battle.sides[0].active_pokemon
    make_choices(battle, "move trickroom", "move poisonjab")
    hp_after_first_jab = bronzong.live_stats.HP
    make_choices(battle, "move protect", "move poisonjab")
    assert hp_after_first_jab == bronzong.live_stats.HP  # protected despite being the slower-and-TR-favoured side


@pytest.mark.skip(reason="simultaneous replacements are not modelled (forced switches resolve one side at a time)")
def test_trickroom_affects_ability_activation_order(): ...


@pytest.mark.skip(reason="Choice Scarf item and speed-glitch boundary not implemented")
def test_trickroom_1809_speed_glitch_rollover(): ...


@pytest.mark.skip(reason="Gyro Ball variable-power callback not implemented")
def test_trickroom_does_not_affect_gyro_ball_basepower(): ...
