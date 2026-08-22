from battle_sim.utils import ExtraStatus
from tests.showdown.conftest import create_battle, make_choices


def test_chatter_pierces_substitutes():
    battle = create_battle(
        [{"species": "Chatot", "moves": ["chatter"]}],
        [{"species": "Snorlax", "moves": ["substitute"]}],
        seed=0,
    )
    snorlax = battle.sides[1].active_pokemon
    battle.sides[1].active_pokemon.stat_stages.SPEED = 6  # substitute goes up before Chatter
    make_choices(battle, "move chatter", "move substitute")
    sub_cost = snorlax.stat_totals.HP // 4
    assert ExtraStatus.SUBSTITUTE in snorlax.volatiles  # sound bypasses the sub entirely: it never took the hit
    assert sub_cost < snorlax.stat_totals.HP - snorlax.live_stats.HP  # the mon itself was damaged
    assert ExtraStatus.CONFUSION in snorlax.volatiles  # and the secondary applied through the sub


def test_chatter_inflicts_confusion_on_hit():
    battle = create_battle(
        [{"species": "Chatot", "moves": ["chatter"]}],
        [{"species": "Snorlax", "moves": ["splash"]}],
        seed=0,
    )
    battle.sides[0].active_pokemon.stat_stages.SPEED = 6
    make_choices(battle, "move chatter", "move splash")
    assert ExtraStatus.CONFUSION in battle.sides[1].active_pokemon.volatiles
