import pytest

from battle_sim.database.loader import get_all_z_moves, get_move, normalize_id
from battle_sim.engine import legal_actions
from battle_sim.engine.turn import step
from battle_sim.maths.rng import RNG
from battle_sim.mechanics.battle import BattleState, SideState
from battle_sim.mechanics.log import BattleLog, render_text
from battle_sim.models.actions import Action, ActionType
from battle_sim.models.log_events import DamageDealt, MoveUsed
from battle_sim.models.moves import DamageEffect, Move, MoveSlot
from battle_sim.models.spec import PokemonSpec
from battle_sim.teams import build_pokemon, item_showdown_name
from battle_sim.utils import Ability, Item, Nature, Target
from battle_sim.zmoves import crystal_type, z_move_for, z_power

_GENERIC_CRYSTALS = 18  # one per type
_SIGNATURE_CRYSTALS = 17  # upgrade one specific move; the pairing is hand-written in `zmoves`


@pytest.mark.parametrize(
    ("base_power", "expected"),
    [(40, 100), (55, 100), (60, 120), (75, 140), (85, 160), (90, 175), (100, 180), (110, 185), (120, 190), (130, 195)],
)
def test_power_table_follows_the_gen7_bands(base_power: int, expected: int):
    assert z_power(base_power) == expected


@pytest.mark.parametrize("base_power", [140, 150, 250])
def test_power_is_capped(base_power: int):
    assert z_power(base_power) == 200


def test_every_type_has_exactly_one_generic_crystal():
    types = [crystal_type(item) for item in Item]
    found = [t for t in types if t is not None]
    assert len(found) == _GENERIC_CRYSTALS
    assert len(set(found)) == _GENERIC_CRYSTALS  # one crystal per type, no duplicates


def test_signature_crystals_are_not_treated_as_generic():
    """They share types with the generic crystals, so keying on type would mis-resolve them."""
    # Identified from the Z-move data, never from the item's name: three of the Legends Z-A *mega
    # stones* are called "Absolite Z", "Garchompite Z" and "Lucarionite Z", and a name-suffix test
    # sweeps them in as crystals. They are not; `crystal_type` is right about them for the right
    # reason, which is why this asks the data instead.
    z_moves = get_all_z_moves()
    signature = [
        item for item in Item if normalize_id(item_showdown_name(item)) in z_moves and crystal_type(item) is None
    ]
    assert len(signature) == _SIGNATURE_CRYSTALS
    assert Item.MIMIKIUM_Z in signature  # Ghost, same as the generic Ghostium Z
    assert Item.KOMMONIUM_Z in signature  # Dragon, same as the generic Dragonium Z


def test_a_generic_crystal_upgrades_a_move_of_its_type():
    z_move = z_move_for(Item.FLYINIUM_Z, get_move("Brave Bird"))
    assert z_move is not None
    assert z_move.name == "Supersonic Skystrike"
    assert z_move.accuracy_probability is None  # Z-moves never miss


def test_power_comes_from_the_base_move_not_the_placeholder():
    strong = z_move_for(Item.FLYINIUM_Z, get_move("Brave Bird"))  # 120 base
    weak = z_move_for(Item.FLYINIUM_Z, get_move("Acrobatics"))  # 55 base
    assert strong is not None and weak is not None
    assert _power(strong) == 190
    assert _power(weak) == 100


def test_category_is_inherited_from_the_base_move():
    """The vendored generic entries all claim Physical regardless of what triggered them."""
    special = z_move_for(Item.FIRIUM_Z, get_move("Flamethrower"))
    physical = z_move_for(Item.FIRIUM_Z, get_move("Flare Blitz"))
    assert special is not None and physical is not None
    assert special.category.name == "SPECIAL"
    assert physical.category.name == "PHYSICAL"


@pytest.mark.parametrize(
    ("item", "move"),
    [
        (Item.FLYINIUM_Z, "Earthquake"),  # wrong type
        (Item.NORMALIUM_Z, "Splash"),  # status move: gets a bonus effect instead, not modelled
        (Item.KOMMONIUM_Z, "Dragon Claw"),  # signature crystal, but not the move it upgrades
        (Item.LEFTOVERS, "Brave Bird"),  # not a crystal at all
    ],
)
def test_pairings_that_do_nothing(item: Item, move: str):
    assert z_move_for(item, get_move(move)) is None


def test_a_signature_crystal_upgrades_the_one_move_it_belongs_to():
    """These were inert until the crystal -> base-move pairing was written down, which made every
    holder give up a real item for nothing — Necrozma-Dusk-Mane carried Ultranecrozium Z through a
    whole battle and could never fire it."""
    fired = z_move_for(Item.ULTRANECROZIUM_Z, get_move("Photon Geyser"))
    assert fired is not None
    assert fired.name == "Light That Burns the Sky"
    assert fired.accuracy_probability is None  # Z-moves never miss
    assert z_move_for(Item.ULTRANECROZIUM_Z, get_move("Sunsteel Strike")) is None


def test_every_signature_crystal_names_a_move_the_database_has():
    """A typo in the hand-written pairing table would silently make that crystal inert again."""
    from battle_sim.zmoves import _SIGNATURE_BASES

    for crystal, base in _SIGNATURE_BASES.items():
        assert get_move(base) is not None, f"{crystal} names an unknown move {base!r}"


def test_every_vendored_z_move_is_addressable_by_a_real_crystal():
    """Guards the crystal keying: an entry we load but no Item can reach is dead data."""
    z_moves = get_all_z_moves()
    assert len(z_moves) == 35
    reachable = {normalize_id(item_showdown_name(item)) for item in Item}
    assert set(z_moves) <= reachable


def _power(move: Move) -> int | None:
    return next((effect.power for effect in move.effects if isinstance(effect, DamageEffect)), None)


# -- Engine integration ---------------------------------------------------------


def _z_battle(item: Item) -> BattleState:
    attacker = PokemonSpec(
        species="Salamence",
        level=80,
        moves=["Dragon Claw", "Fire Blast", "Roost", "Dragon Dance"],
        ability=Ability.MOXIE,
        item=item,
        nature=Nature.ADAMANT,
    )
    wall = PokemonSpec(
        species="Chansey", level=100, moves=["Splash"], ability=Ability.NATURAL_CURE, item=Item.NONE, nature=Nature.BOLD
    )
    return BattleState(
        sides=(SideState(team=[build_pokemon(attacker)]), SideState(team=[build_pokemon(wall)])), rng=RNG(seed=5)
    )


_PLAIN = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.FIRST)
_Z = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.FIRST, z_move=True)


def test_only_matching_type_moves_get_a_z_variant():
    """One extra action, not a doubled move set: only the Dragon move qualifies for Dragonium Z."""
    actions = legal_actions(_z_battle(Item.DRAGONIUM_Z), 0)
    assert sum(1 for a in actions if a.z_move) == 1
    assert sum(1 for a in actions if not a.z_move) == 4


def test_no_crystal_means_no_z_variant():
    assert not any(a.z_move for a in legal_actions(_z_battle(Item.LEFTOVERS), 0))


def test_the_z_move_hits_harder_and_spends_the_crystal():
    plain = _z_battle(Item.DRAGONIUM_Z)
    boosted = _z_battle(Item.DRAGONIUM_Z)
    plain_damage = _dealt(step(plain, {0: _PLAIN, 1: _PLAIN}))
    z_damage = _dealt(step(boosted, {0: _Z, 1: _PLAIN}))

    assert plain_damage < z_damage
    assert boosted.sides[0].has_used_z_move
    assert not plain.sides[0].has_used_z_move
    assert not any(a.z_move for a in legal_actions(boosted, 0))  # the crystal is spent


def test_move_used_names_the_slot_so_beliefs_stay_buildable():
    """`BattleObserver` treats every MoveUsed name as slot-fillable; a Z-move name fills no slot."""
    state = _z_battle(Item.DRAGONIUM_Z)
    log = step(state, {0: _Z, 1: _PLAIN})
    entries = [e for e in log if isinstance(e, MoveUsed) and e.side == 0]

    assert [e.move for e in entries] == ["Dragon Claw"]
    assert [e.unleashed_as for e in entries] == ["Devastating Drake"]
    get_move(entries[0].move)  # resolvable, which is what keeps SetPrior.believe able to rebuild the set


def test_a_z_move_reads_as_one_move_becoming_its_z_move():
    """One line, not two separately-named moves: it was never really two uses."""
    entry = MoveUsed(side=0, pokemon="Flygon", move="Outrage", unleashed_as="Devastating Drake")
    assert render_text(entry) == "P1's Flygon's Outrage became Devastating Drake!"


def test_a_plain_move_still_just_says_it_was_used():
    entry = MoveUsed(side=0, pokemon="Flygon", move="Outrage")
    assert render_text(entry) == "P1's Flygon used Outrage!"


def _dealt(log: object) -> int:
    assert isinstance(log, BattleLog)
    return sum(e.amount for e in log if isinstance(e, DamageDealt) and e.side == 1)
