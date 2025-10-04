from battle_sim.database.sample_moves import DRACO_METEOR, EARTHQUAKE, ROCK_SLIDE, SWORDS_DANCE, WILL_O_WISP
from battle_sim.models.moves import MoveSet, MoveSlot


def test_forget_move_shifts_left():
    move_set = MoveSet(EARTHQUAKE, DRACO_METEOR, ROCK_SLIDE, SWORDS_DANCE)
    move_set.forget_move(MoveSlot.SECOND)
    assert move_set.to_list() == [EARTHQUAKE, ROCK_SLIDE, SWORDS_DANCE]
    assert move_set[MoveSlot.FOURTH] is None


def test_learn_and_contains():
    move_set = MoveSet(EARTHQUAKE, DRACO_METEOR, None, None)
    move_set.learn_move(SWORDS_DANCE, MoveSlot.SECOND)
    assert move_set.contains(SWORDS_DANCE)
    assert move_set.to_list() == [EARTHQUAKE, DRACO_METEOR, SWORDS_DANCE]


def test_learn_move_in_first_empty_slot():
    move_set = MoveSet(EARTHQUAKE, None, None, None)
    move_set.learn_move(SWORDS_DANCE, MoveSlot.FIRST)
    assert move_set.to_list() == [EARTHQUAKE, SWORDS_DANCE]
    assert move_set[MoveSlot.SECOND] == SWORDS_DANCE


def test_learn_move_requires_slot_when_full():
    move_set = MoveSet(EARTHQUAKE, DRACO_METEOR, ROCK_SLIDE, SWORDS_DANCE)
    move_set.learn_move(WILL_O_WISP, MoveSlot.FIRST)
    assert move_set.to_list() == [WILL_O_WISP, DRACO_METEOR, ROCK_SLIDE, SWORDS_DANCE]
    assert move_set[MoveSlot.FIRST] == WILL_O_WISP
