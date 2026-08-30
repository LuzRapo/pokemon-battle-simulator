import pytest

from battle_sim.analysis import best_expected_damage, expected_damage, posterior_threat
from battle_sim.database.loader import get_move
from battle_sim.maths.rng import RNG
from battle_sim.mechanics.battle import BattleState, FieldState, SideState
from battle_sim.models.moves import MoveSet
from battle_sim.models.pokemon import BelievedSet, Pokemon
from battle_sim.models.stats import BaseStats, EVs, IVs
from battle_sim.utils import Ability, Item, Nature, Type

TACKLE = get_move("Tackle")
EMBER = get_move("Ember")
SPLASH = get_move("Splash")
FLAMETHROWER = get_move("Flamethrower")


def _mk(types: tuple[Type, Type | None] = (Type.NORMAL, None), item: Item = Item.NONE) -> Pokemon:
    return Pokemon(
        name="TestMon",
        nickname="X",
        level=50,
        base_stats=BaseStats(HP=100, ATTACK=100, DEFENCE=100, SP_ATTACK=100, SP_DEFENCE=100, SPEED=100),
        effort_values=EVs(),
        individual_values=IVs(),
        types=types,
        moves=MoveSet(TACKLE, EMBER, SPLASH, None),
        nature=Nature.HARDY,
        item=item,
    )


def _battle(attacker: Pokemon, defender: Pokemon) -> BattleState:
    return BattleState(
        sides=(SideState(team=[defender]), SideState(team=[attacker])), rng=RNG(seed=0), field=FieldState()
    )


def _set(weight: float, *moves, item: Item = Item.NONE, ability: Ability = Ability.NONE) -> BelievedSet:
    return BelievedSet(weight=weight, moves=tuple(moves), item=item, ability=ability)


def test_a_known_set_is_read_as_its_own_best_move():
    attacker, defender = _mk(), _mk()
    state = _battle(attacker, defender)
    assert attacker.believed_sets is None
    assert posterior_threat(attacker, defender, state) == best_expected_damage(attacker, defender, state)


def test_threat_is_the_weighted_mean_of_each_candidates_own_best_move():
    attacker, defender = _mk(), _mk(types=(Type.GRASS, None))
    state = _battle(attacker, defender)
    tackle = expected_damage(TACKLE, attacker, defender, state)
    ember = expected_damage(EMBER, attacker, defender, state)
    attacker.believed_sets = (_set(0.75, TACKLE, SPLASH), _set(0.25, EMBER, SPLASH))
    assert posterior_threat(attacker, defender, state) == pytest.approx(0.75 * tackle + 0.25 * ember)


def test_a_candidate_is_credited_only_with_the_coverage_it_owns():
    """The point of the estimator: one set holding both moves reads hotter than a mix holding one each."""
    attacker, defender = _mk(), _mk(types=(Type.GRASS, None))
    state = _battle(attacker, defender)
    attacker.believed_sets = (_set(1.0, TACKLE, EMBER),)
    both = posterior_threat(attacker, defender, state)
    attacker.believed_sets = (_set(0.5, TACKLE, SPLASH), _set(0.5, EMBER, SPLASH))
    split = posterior_threat(attacker, defender, state)
    assert both > split


def test_a_candidates_ability_is_priced_into_its_own_damage():
    """Mold Breaker ignores the Unaware wall's stat-stage immunity; a candidate without it does not.

    Item and most ability multipliers are bound to the damage event bus, which `damage_range`
    never runs — so the abilities it reads directly are what this can assert.
    """
    attacker, defender = _mk(), _mk()
    attacker.stat_stages.ATTACK = 2
    defender.ability = Ability.UNAWARE
    state = _battle(attacker, defender)
    attacker.believed_sets = (_set(1.0, TACKLE, ability=Ability.NONE),)
    ignored = posterior_threat(attacker, defender, state)
    attacker.believed_sets = (_set(1.0, TACKLE, ability=Ability.MOLD_BREAKER),)
    counted = posterior_threat(attacker, defender, state)
    assert counted > ignored
    assert attacker.ability is Ability.NONE  # pricing a candidate must not mutate the pokemon


def test_an_empty_candidate_pool_threatens_nothing():
    attacker, defender = _mk(), _mk()
    state = _battle(attacker, defender)
    attacker.believed_sets = (_set(1.0, SPLASH),)
    assert posterior_threat(attacker, defender, state) == 0.0
