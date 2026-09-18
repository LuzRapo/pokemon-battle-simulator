"""Damage effects of the Mega-forme abilities.

Measured from the `DamageDealt` log event rather than an HP delta, so healing and residual chip
can't confound the reading, and through `step` rather than `analysis.damage_range` — the latter
builds its own payload and deliberately bypasses the event bus, so no ability handler fires there.

Defenders are chosen with enough bulk that the boosted hit does not KO: a KO clamps the reported
damage to the target's remaining HP and would understate the multiplier.
"""

from battle_sim.engine.turn import step
from battle_sim.maths.rng import RNG
from battle_sim.mechanics.battle import BattleState, SideState
from battle_sim.models.actions import Action, ActionType
from battle_sim.models.log_events import DamageDealt
from battle_sim.models.moves import MoveSlot
from battle_sim.models.spec import PokemonSpec
from battle_sim.teams import build_pokemon
from battle_sim.utils import Ability, Item, Nature, Target, Weather


def _act(slot: MoveSlot) -> Action:
    return Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=slot)


def _damage(
    species: str,
    ability: Ability,
    moves: list[str],
    slot: MoveSlot,
    weather: Weather = Weather.NONE,
) -> int:
    attacker = PokemonSpec(
        species=species, level=50, moves=moves, ability=ability, item=Item.NONE, nature=Nature.ADAMANT
    )
    wall = PokemonSpec(
        species="Chansey", level=100, moves=["Splash"], ability=Ability.NATURAL_CURE, item=Item.NONE, nature=Nature.BOLD
    )
    state = BattleState(
        sides=(SideState(team=[build_pokemon(attacker)]), SideState(team=[build_pokemon(wall)])),
        rng=RNG(seed=3),
    )
    state.field.weather = weather
    log = step(state, {0: _act(slot), 1: _act(MoveSlot.FIRST)})
    return sum(e.amount for e in log if isinstance(e, DamageDealt) and e.side == 1)


def _ratio(
    species: str,
    base: Ability,
    new: Ability,
    moves: list[str],
    slot: MoveSlot,
    weather: Weather = Weather.NONE,
) -> float:
    """How much the new ability multiplies this move's damage by, all else equal."""
    before = _damage(species, base, moves, slot, weather=weather)
    assert before > 0
    return _damage(species, new, moves, slot, weather=weather) / before


def test_tough_claws_boosts_contact_moves_only():
    contact = _ratio("Aerodactyl", Ability.PRESSURE, Ability.TOUGH_CLAWS, ["Crunch", "Rock Slide"], MoveSlot.FIRST)
    other = _ratio("Aerodactyl", Ability.PRESSURE, Ability.TOUGH_CLAWS, ["Crunch", "Rock Slide"], MoveSlot.SECOND)
    assert 1.25 < contact < 1.35
    assert other == 1.0


def test_strong_jaw_boosts_biting_moves_only():
    bite = _ratio("Sharpedo", Ability.PRESSURE, Ability.STRONG_JAW, ["Crunch", "Waterfall"], MoveSlot.FIRST)
    other = _ratio("Sharpedo", Ability.PRESSURE, Ability.STRONG_JAW, ["Crunch", "Waterfall"], MoveSlot.SECOND)
    assert 1.45 < bite < 1.55
    assert other == 1.0


def test_mega_launcher_boosts_pulse_moves_only():
    pulse = _ratio("Blastoise", Ability.TORRENT, Ability.MEGA_LAUNCHER, ["Dark Pulse", "Surf"], MoveSlot.FIRST)
    other = _ratio("Blastoise", Ability.TORRENT, Ability.MEGA_LAUNCHER, ["Dark Pulse", "Surf"], MoveSlot.SECOND)
    assert 1.45 < pulse < 1.55
    assert other == 1.0


def test_parental_bond_totals_one_and_a_quarter():
    ratio = _ratio("Kangaskhan", Ability.SCRAPPY, Ability.PARENTAL_BOND, ["Body Slam"], MoveSlot.FIRST)
    assert 1.2 < ratio < 1.3


def test_sand_force_needs_a_sandstorm_and_the_right_type():
    moves = ["Earthquake", "Dragon Claw"]  # Ground is boosted, Dragon is not
    boosted = _ratio(
        "Garchomp", Ability.ROUGH_SKIN, Ability.SAND_FORCE, moves, MoveSlot.FIRST, weather=Weather.SANDSTORM
    )
    wrong_type = _ratio(
        "Garchomp", Ability.ROUGH_SKIN, Ability.SAND_FORCE, moves, MoveSlot.SECOND, weather=Weather.SANDSTORM
    )
    no_weather = _ratio("Garchomp", Ability.ROUGH_SKIN, Ability.SAND_FORCE, moves, MoveSlot.FIRST)
    assert 1.25 < boosted < 1.35
    assert wrong_type == 1.0
    assert no_weather == 1.0


def _chomp_vs(foe_ability: Ability) -> tuple[int, Ability]:
    """One turn: a Garchomp mega-evolves, then eats an Earthquake. Returns damage taken and ability."""
    chomp = build_pokemon(
        PokemonSpec(
            species="Garchomp",
            level=50,
            moves=["Dragon Claw"],
            ability=Ability.ROUGH_SKIN,
            item=Item.GARCHOMPITE_Z,
            nature=Nature.JOLLY,
        )
    )
    excadrill = build_pokemon(
        PokemonSpec(
            species="Excadrill",
            level=50,
            moves=["Earthquake"],
            ability=foe_ability,
            item=Item.NONE,
            nature=Nature.JOLLY,
        )
    )
    chomp.nickname, excadrill.nickname = "Garchomp", "Excadrill"
    state = BattleState(sides=(SideState(team=[chomp]), SideState(team=[excadrill])), rng=RNG(seed=3))
    log = step(state, {0: _act(MoveSlot.FIRST), 1: _act(MoveSlot.FIRST)})
    taken = sum(entry.amount for entry in log if isinstance(entry, DamageDealt) and entry.side == 0)
    return taken, chomp.ability


def test_mega_garchomp_z_floats_above_ground_moves() -> None:
    """A house rule, and one the forme argues for: this mega sheds the Ground type to become pure
    Dragon, so the Sand Force the vendor file gives it — a Ground/Rock/Steel booster — would boost
    nothing it has STAB on. It reads as a Garchomp that has stopped touching the ground.

    Worth testing through a real mid-battle mega evolution rather than by building the forme
    directly: the ability arrives *during* the turn, and would do nothing at all if the forme change
    failed to rewire the event bus behind it.
    """
    taken, ability = _chomp_vs(Ability.SAND_RUSH)
    assert ability is Ability.LEVITATE
    assert taken == 0


def test_mold_breaker_still_reaches_a_floating_mega_garchomp_z() -> None:
    """Which is why turn 6 of match 5a8744b1 was right either way — Ingo's Excadrill held Mold
    Breaker, and Mold Breaker unwires the defender's ability for the move's duration."""
    taken, ability = _chomp_vs(Ability.MOLD_BREAKER)
    assert ability is Ability.LEVITATE
    assert taken > 0
