from battle_sim.models.moves import DamageEffect, InflictStatusEffect, Move, StatStageChangeEffect
from battle_sim.utils import Category, ExtraStatus, PriorityLevel, Stats, Status, Target, Type

EARTHQUAKE = Move(
    name="Earthquake",
    type=Type.GROUND,
    category=Category.PHYSICAL,
    accuracy_probability=1.0,
    priority=PriorityLevel.NORMAL,
    pp=10,
    target=Target.ALL_ADJACENT,
    effects=[DamageEffect(power=100, category=Category.PHYSICAL, contact=True, crit_stage=0)],
)


SWORDS_DANCE = Move(
    name="Swords Dance",
    type=Type.NORMAL,
    category=Category.STATUS,
    accuracy_probability=None,
    priority=PriorityLevel.NORMAL,
    pp=20,
    target=Target.SELF,
    effects=[StatStageChangeEffect(target="SELF", stages={Stats.ATTACK: +2}, probability=1)],
)


WILL_O_WISP = Move(
    name="Will-O-Wisp",
    type=Type.FIRE,
    category=Category.STATUS,
    accuracy_probability=0.85,
    priority=PriorityLevel.NORMAL,
    pp=15,
    target=Target.SINGLE_OPPONENT,
    effects=[InflictStatusEffect(status=Status.BURN, probability=1)],
)


ROCK_SLIDE = Move(
    name="Rock Slide",
    type=Type.ROCK,
    category=Category.PHYSICAL,
    accuracy_probability=0.9,
    priority=PriorityLevel.NORMAL,
    pp=10,
    target=Target.ALL_ADJACENT_ENEMIES,
    effects=[
        DamageEffect(power=75, category=Category.PHYSICAL, contact=True, crit_stage=0),
        InflictStatusEffect(status=ExtraStatus.FLINCH, probability=0.3, only_if_contact=True),
    ],
)

DRACO_METEOR = Move(
    name="Draco Meteor",
    type=Type.DRAGON,
    category=Category.SPECIAL,
    accuracy_probability=0.9,
    priority=PriorityLevel.NORMAL,
    pp=5,
    target=Target.SINGLE_OPPONENT,
    effects=[
        DamageEffect(power=130, category=Category.SPECIAL, crit_stage=0, contact=False),
        StatStageChangeEffect(target="SELF", stages={Stats.SP_ATTACK: -2}, probability=1),
    ],
)
