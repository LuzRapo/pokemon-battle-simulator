from battle_sim.engine.turn import _resolve_mega_evolution, step
from battle_sim.maths.rng import RNG
from battle_sim.mechanics.battle import BattleState, SideState
from battle_sim.mechanics.log import BattleLog
from battle_sim.models.actions import Action, ActionType
from battle_sim.models.log_events import FormeChanged
from battle_sim.models.moves import MoveSlot
from battle_sim.models.spec import PokemonSpec
from battle_sim.teams import build_pokemon
from battle_sim.utils import Ability, Item, Nature, Target

_ATTACK = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.FIRST)


def _spec(species: str, ability: Ability, item: Item, moves: list[str], level: int = 80) -> PokemonSpec:
    return PokemonSpec(species=species, level=level, moves=moves, ability=ability, item=item, nature=Nature.ADAMANT)


def _battle(own: list[PokemonSpec], theirs: list[PokemonSpec]) -> BattleState:
    return BattleState(
        sides=(
            SideState(team=[build_pokemon(s) for s in own]),
            SideState(team=[build_pokemon(s) for s in theirs]),
        ),
        rng=RNG(seed=0),
    )


def _tyranitar(item: Item) -> PokemonSpec:
    return _spec("Tyranitar", Ability.SAND_STREAM, item, ["Crunch", "Stone Edge", "Earthquake", "Fire Punch"])


def _blissey() -> PokemonSpec:
    return _spec("Blissey", Ability.NATURAL_CURE, Item.NONE, ["Seismic Toss", "Soft-Boiled", "Toxic", "Protect"])


def test_holding_the_stone_mega_evolves_on_the_first_move_turn():
    state = _battle([_tyranitar(Item.TYRANITARITE)], [_blissey()])
    log = step(state, {0: _ATTACK, 1: _ATTACK})

    assert state.sides[0].active_pokemon.name == "Tyranitar-Mega"
    assert state.sides[0].has_mega_evolved
    assert [e.forme for e in log if isinstance(e, FormeChanged)] == ["Tyranitar-Mega"]


def test_without_the_stone_nothing_happens():
    state = _battle([_tyranitar(Item.LEFTOVERS)], [_blissey()])
    step(state, {0: _ATTACK, 1: _ATTACK})

    assert state.sides[0].active_pokemon.name == "Tyranitar"
    assert not state.sides[0].has_mega_evolved


def test_mega_evolves_only_once_per_side():
    """The stone is spent on the side, not the Pokemon: a second holder cannot also Mega."""
    state = _battle(
        [_tyranitar(Item.TYRANITARITE), _spec("Scizor", Ability.TECHNICIAN, Item.SCIZORITE, ["Bullet Punch"])],
        [_blissey()],
    )
    step(state, {0: _ATTACK, 1: _ATTACK})
    assert state.sides[0].active_pokemon.name == "Tyranitar-Mega"
    assert state.sides[0].has_mega_evolved

    state.sides[0].active[0] = 1  # bring the side's other stone-holder in
    state.sides[0].chosen_action = _ATTACK
    _resolve_mega_evolution(state, BattleLog())

    assert state.sides[0].active_pokemon.name == "Scizor"


def test_switching_does_not_mega_evolve():
    state = _battle([_blissey(), _tyranitar(Item.TYRANITARITE)], [_blissey()])
    switch = Action(action=ActionType.SWITCH_OUT, switch_in=state.sides[0].team[1])
    step(state, {0: switch, 1: _ATTACK})

    assert state.sides[0].active_pokemon.name == "Tyranitar"  # arrived this turn, cannot also Mega
    assert not state.sides[0].has_mega_evolved


def test_the_new_speed_decides_this_turn_s_order():
    """Aerodactyl is 130 base Speed and its Mega is 150; Accelgor sits between at 145."""
    aerodactyl = _spec("Aerodactyl", Ability.PRESSURE, Item.AERODACTYLITE, ["Stone Edge"])
    foe = _spec("Accelgor", Ability.HYDRATION, Item.NONE, ["Bug Buzz"])
    state = _battle([aerodactyl], [foe])
    mine, theirs = state.sides[0].active_pokemon, state.sides[1].active_pokemon
    assert mine.stat_totals.SPEED < theirs.stat_totals.SPEED  # slower before evolving

    step(state, {0: _ATTACK, 1: _ATTACK})

    assert state.sides[0].active_pokemon.name == "Aerodactyl-Mega"
    assert theirs.stat_totals.SPEED < mine.stat_totals.SPEED  # and faster after, within the same turn


def test_mega_rewires_the_new_forme_s_ability():
    """Mega Aerodactyl trades Pressure for Tough Claws, so the ability field must follow the forme."""
    state = _battle(
        [_spec("Aerodactyl", Ability.PRESSURE, Item.AERODACTYLITE, ["Stone Edge"])],
        [_blissey()],
    )
    step(state, {0: _ATTACK, 1: _ATTACK})

    assert state.sides[0].active_pokemon.ability is Ability.TOUGH_CLAWS
