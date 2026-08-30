from dataclasses import replace

from battle_sim.database.loader import get_move
from battle_sim.engine.power import _ate_boost_applies, move_type_override
from battle_sim.maths.rng import RNG
from battle_sim.mechanics.battle import BattleState, SideState
from battle_sim.models.moves import MoveSet
from battle_sim.models.pokemon import Pokemon
from battle_sim.models.stats import BaseStats, EVs, IVs
from battle_sim.utils import Ability, Item, Nature, Type

JUDGMENT = get_move("Judgment")
MULTI_ATTACK = get_move("Multi-Attack")
TECHNO_BLAST = get_move("Techno Blast")
DOUBLE_EDGE = get_move("Double-Edge")
HURRICANE = get_move("Hurricane")


def _mk(item: Item = Item.NONE, ability: Ability = Ability.NONE) -> Pokemon:
    return Pokemon(
        name="X",
        nickname="X",
        level=50,
        base_stats=BaseStats(HP=100, ATTACK=100, DEFENCE=100, SP_ATTACK=100, SP_DEFENCE=100, SPEED=100),
        effort_values=EVs(),
        individual_values=IVs(),
        types=(Type.NORMAL, None),
        moves=MoveSet(JUDGMENT, MULTI_ATTACK, TECHNO_BLAST, JUDGMENT),
        nature=Nature.HARDY,
        item=item,
        ability=ability,
    )


def _state(mine: Pokemon) -> BattleState:
    """None of these overrides read anything from the state but Weather Ball's, which has its own
    dedicated test elsewhere -- just enough of a real state to satisfy the type, not a scenario."""
    return BattleState(sides=(SideState(team=[mine]), SideState(team=[_mk()])), rng=RNG(seed=0))


def test_judgment_matches_the_held_plate():
    plate_user = _mk(Item.SPLASH_PLATE)
    assert move_type_override(JUDGMENT, plate_user, _state(plate_user)) is Type.WATER
    mind_plate_user = _mk(Item.MIND_PLATE)
    assert move_type_override(JUDGMENT, mind_plate_user, _state(mind_plate_user)) is Type.PSYCHIC


def test_judgment_without_a_plate_is_unchanged():
    itemless = _mk(Item.NONE)
    assert move_type_override(JUDGMENT, itemless, _state(itemless)) is None


def test_multi_attack_matches_the_held_memory():
    steel_memory = _mk(Item.STEEL_MEMORY)
    assert move_type_override(MULTI_ATTACK, steel_memory, _state(steel_memory)) is Type.STEEL
    water_memory = _mk(Item.WATER_MEMORY)
    assert move_type_override(MULTI_ATTACK, water_memory, _state(water_memory)) is Type.WATER


def test_techno_blast_matches_the_held_drive():
    burn_drive = _mk(Item.BURN_DRIVE)
    assert move_type_override(TECHNO_BLAST, burn_drive, _state(burn_drive)) is Type.FIRE
    chill_drive = _mk(Item.CHILL_DRIVE)
    assert move_type_override(TECHNO_BLAST, chill_drive, _state(chill_drive)) is Type.ICE


def test_techno_blast_without_a_drive_is_unchanged():
    itemless = _mk(Item.NONE)
    assert move_type_override(TECHNO_BLAST, itemless, _state(itemless)) is None


def test_aerilate_pixilate_refrigerate_convert_a_normal_move():
    aerilate = _mk(ability=Ability.AERILATE)
    assert move_type_override(DOUBLE_EDGE, aerilate, _state(aerilate)) is Type.FLYING
    pixilate = _mk(ability=Ability.PIXILATE)
    assert move_type_override(DOUBLE_EDGE, pixilate, _state(pixilate)) is Type.FAIRY
    refrigerate = _mk(ability=Ability.REFRIGERATE)
    assert move_type_override(DOUBLE_EDGE, refrigerate, _state(refrigerate)) is Type.ICE


def test_an_ate_ability_leaves_a_non_normal_move_alone():
    aerilate = _mk(ability=Ability.AERILATE)
    assert move_type_override(HURRICANE, aerilate, _state(aerilate)) is None


def test_ate_boost_applies_only_to_a_genuine_conversion():
    """The power boost must not fire for a naturally Flying/Fairy/Ice move on an -ate holder."""
    converted = replace(DOUBLE_EDGE, type=Type.FLYING)
    assert _ate_boost_applies(converted, _mk(ability=Ability.AERILATE))
    assert not _ate_boost_applies(HURRICANE, _mk(ability=Ability.AERILATE))
    assert not _ate_boost_applies(DOUBLE_EDGE, _mk(ability=Ability.NONE))
