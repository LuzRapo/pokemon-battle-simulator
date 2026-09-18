import pytest

from battle_sim.database.loader import get_all_species, get_move, get_species
from battle_sim.engine import legal_actions, step
from battle_sim.formes import (
    _forme_by_base_and_item,
    _is_transformed_forme,
    apply_forme,
    mega_forme,
    mega_stones,
    ultra_bursts,
    unplayable_formes,
)
from battle_sim.maths.damage import move_effectiveness
from battle_sim.maths.rng import RNG
from battle_sim.mechanics.battle import BattleState, FieldState, SideState
from battle_sim.models.actions import ActionType
from battle_sim.models.pokemon import Pokemon
from battle_sim.models.spec import PokemonSpec
from battle_sim.teams import ability_from_showdown, build_pokemon, item_from_showdown, parse_showdown_team
from battle_sim.utils import Ability, Item, Nature, Type, Weather


def _tyranitar(item: Item) -> PokemonSpec:
    return PokemonSpec(
        species="Tyranitar",
        level=80,
        moves=["Stone Edge", "Crunch", "Earthquake", "Fire Punch"],
        ability=Ability.SAND_STREAM,
        item=item,
        nature=Nature.ADAMANT,
    )


@pytest.mark.parametrize(
    ("species", "item", "expected"),
    [
        ("Tyranitar", Item.TYRANITARITE, "Tyranitar-Mega"),
        ("Charizard", Item.CHARIZARDITE_X, "Charizard-Mega-X"),
        ("Charizard", Item.CHARIZARDITE_Y, "Charizard-Mega-Y"),
        ("Groudon", Item.RED_ORB, "Groudon-Primal"),
        ("Kyogre", Item.BLUE_ORB, "Kyogre-Primal"),
    ],
)
def test_stone_reaches_its_forme(species: str, item: Item, expected: str):
    assert mega_forme(species, item) == expected


@pytest.mark.parametrize(
    ("species", "item"),
    [
        ("Charizard", Item.LEFTOVERS),  # not a stone
        ("Charizard", Item.TYRANITARITE),  # someone else's stone
        ("Tyranitar", Item.NONE),
    ],
)
def test_wrong_pairing_reaches_nothing(species: str, item: Item):
    assert mega_forme(species, item) is None


def test_mega_swaps_species_data_and_refreshes_cached_stats():
    pokemon = build_pokemon(_tyranitar(Item.TYRANITARITE))
    before = pokemon.stat_totals.ATTACK  # read first, so the cached_property is populated pre-swap

    apply_forme(pokemon, "Tyranitar-Mega")

    mega = get_species("Tyranitar-Mega")
    assert pokemon.name == "Tyranitar-Mega"
    assert pokemon.base_stats == mega.base_stats
    assert pokemon.weight_kg == mega.weight_kg
    assert before < pokemon.stat_totals.ATTACK  # a stale cache would still report the base forme's total


def test_mega_keeps_what_the_trainer_chose():
    pokemon = build_pokemon(_tyranitar(Item.TYRANITARITE))
    moves = [move.name for move in pokemon.moves if move is not None]
    pokemon.live_stats.HP = 150

    apply_forme(pokemon, "Tyranitar-Mega")

    assert [move.name for move in pokemon.moves if move is not None] == moves
    assert pokemon.item is Item.TYRANITARITE
    assert pokemon.live_stats.HP == 150
    assert pokemon.stat_totals.HP == build_pokemon(_tyranitar(Item.TYRANITARITE)).stat_totals.HP


def test_mega_can_change_typing_and_ability():
    spec = PokemonSpec(
        species="Charizard",
        level=80,
        moves=["Flamethrower", "Air Slash", "Roost", "Dragon Pulse"],
        ability=Ability.BLAZE,
        item=Item.CHARIZARDITE_X,
        nature=Nature.ADAMANT,
    )
    pokemon = build_pokemon(spec)
    assert Type.FLYING in pokemon.types

    apply_forme(pokemon, "Charizard-Mega-X")

    assert pokemon.types == (Type.FIRE, Type.DRAGON)
    assert pokemon.ability is Ability.TOUGH_CLAWS


def test_every_stone_forme_is_constructible():
    """Guards the whole table: a forme we can name but not build would crash mid-battle."""
    table = _forme_by_base_and_item()
    # Counted from the data rather than pinned to a number: the table grew from 49 to 98 when the
    # Legends Z-A stones were added, and a hardcoded total only ever tells you the data moved.
    # Item and move tables together, because Rayquaza-Mega is reached by knowing Dragon Ascent.
    from battle_sim.formes import _forme_by_base_and_move

    reachable = set(table.values()) | set(_forme_by_base_and_move().values())
    every = {s.name for s in get_all_species().values() if _is_transformed_forme(s.name)}
    # Everything not reachable is held back for a stated reason, never merely missed.
    assert every - reachable == set(unplayable_formes())
    for (_base, item), forme in table.items():
        species = get_species(forme)
        assert species.regular_abilities, f"{forme} has no ability to assign"
        assert ability_from_showdown(species.regular_abilities[0]) is not Ability.NONE, f"{forme} ability unmapped"
        assert item is not Item.NONE


def test_mega_stones_are_exactly_the_items_the_forme_table_uses():
    assert mega_stones() == {item for _, item in _forme_by_base_and_item()}
    assert Item.CHARIZARDITE_X in mega_stones()
    assert Item.LEFTOVERS not in mega_stones()


def test_forme_that_changes_max_hp_keeps_the_hp_fraction():
    """Zygarde's Power Construct doubles base HP — the only corpus forme that does."""
    spec = PokemonSpec(
        species="Zygarde",
        level=80,
        moves=["Thousand Arrows", "Dragon Dance", "Extreme Speed", "Substitute"],
        ability=Ability.AURA_BREAK,
        item=Item.NONE,
        nature=Nature.ADAMANT,
    )
    pokemon = build_pokemon(spec)
    pokemon.live_stats.HP = pokemon.stat_totals.HP // 2

    apply_forme(pokemon, "Zygarde-Complete")

    assert pokemon.name == "Zygarde-Complete"
    assert abs(pokemon.live_stats.HP - pokemon.stat_totals.HP // 2) <= 1


def test_fainted_stays_fainted_across_a_forme_change():
    pokemon = build_pokemon(_tyranitar(Item.TYRANITARITE))
    pokemon.live_stats.HP = 0

    apply_forme(pokemon, "Tyranitar-Mega")

    assert pokemon.is_fainted()


# -- Mega Rayquaza and the weather it brings ----------------------------------------
# Gen 7 Anything Goes runs on these three. Mega Rayquaza was unreachable (the stone table is built
# from `requiredItem`, and it has none — it is gated on knowing Dragon Ascent), and the two primals
# arrived without their weather, because the ability binders fire on switch-in and a forme swap
# happens well after that.


def _brought(text: str) -> Pokemon:
    return build_pokemon(parse_showdown_team(text).specs[0])


FOE = "Snorlax @ Leftovers\nAbility: Immunity\nLevel: 50\n- Body Slam\n- Rest\n- Curse\n- Crunch\n"
RAYQUAZA = "Rayquaza @ {item}\nAbility: Air Lock\nLevel: 50\n- {first}\n- Extreme Speed\n- Earthquake\n- Swords Dance\n"


def _after_one_turn(text: str) -> tuple[Pokemon, BattleState]:
    mine, foe = _brought(text), _brought(FOE)
    state = BattleState(sides=(SideState(team=[mine]), SideState(team=[foe])), rng=RNG(seed=1), field=FieldState())
    mine_action = next(a for a in legal_actions(state, 0) if a.action is ActionType.USE_MOVE)
    foe_action = next(a for a in legal_actions(state, 1) if a.action is ActionType.USE_MOVE)
    step(state, {0: mine_action, 1: foe_action})
    return mine, state


def test_rayquaza_megas_off_dragon_ascent_rather_than_a_stone() -> None:
    mine, state = _after_one_turn(RAYQUAZA.format(item="Life Orb", first="Dragon Ascent"))
    assert mine.name == "Rayquaza-Mega"
    assert mine.ability is Ability.DELTA_STREAM
    assert state.field.weather is Weather.STRONG_WINDS


def test_rayquaza_without_dragon_ascent_stays_as_it_is() -> None:
    mine, _ = _after_one_turn(RAYQUAZA.format(item="Life Orb", first="Outrage"))
    assert mine.name == "Rayquaza"


def test_a_z_crystal_locks_rayquaza_out_of_mega_evolving() -> None:
    """The real rule, and the whole reason a Z-move Rayquaza is a different Pokemon rather than a
    strictly worse one."""
    mine, _ = _after_one_turn(RAYQUAZA.format(item="Flyinium Z", first="Dragon Ascent"))
    assert mine.name == "Rayquaza"


@pytest.mark.parametrize(
    ("text", "forme", "ability", "weather"),
    [
        (
            "Groudon @ Red Orb\nAbility: Drought\nLevel: 50\n"
            "- Precipice Blades\n- Fire Punch\n- Stealth Rock\n- Rest\n",
            "Groudon-Primal",
            Ability.DESOLATE_LAND,
            Weather.HARSH_SUN,
        ),
        (
            "Kyogre @ Blue Orb\nAbility: Drizzle\nLevel: 50\n- Origin Pulse\n- Ice Beam\n- Thunder\n- Rest\n",
            "Kyogre-Primal",
            Ability.PRIMORDIAL_SEA,
            Weather.HEAVY_RAIN,
        ),
    ],
)
def test_a_primal_brings_its_own_weather_not_the_ordinary_kind(
    text: str, forme: str, ability: Ability, weather: Weather
) -> None:
    """It used to set plain sun/rain: the pre-reversion ability set the weather on switch-in and the
    primal ability that replaced it never got a turn to."""
    mine, state = _after_one_turn(text)
    assert (mine.name, mine.ability) == (forme, ability)
    assert state.field.weather is weather


def test_strong_winds_take_the_flying_half_out_of_a_weakness() -> None:
    """Delta Stream neutralises what is super effective against Flying, and nothing else: Ice on a
    Dragon/Flying goes 4x -> 2x, because only the Flying half of it is cancelled."""
    ray = _brought(RAYQUAZA.format(item="Life Orb", first="Dragon Ascent"))
    weavile = _brought(
        "Weavile @ Life Orb\nAbility: Pressure\nLevel: 50\n- Ice Shard\n- Knock Off\n- Swords Dance\n- Pursuit\n"
    )
    shard = get_move("Ice Shard")
    assert move_effectiveness(shard, weavile, ray, Weather.NONE) == 4.0
    assert move_effectiveness(shard, weavile, ray, Weather.STRONG_WINDS) == 2.0


def test_ultra_burst_works_from_both_fused_formes_but_not_from_plain_necrozma():
    """The vendored entry says `baseSpecies: Necrozma`, and plain Necrozma is exactly what cannot
    Ultra Burst — only the two fused formes can. Reading the pairing out of the data would have let
    the wrong Pokemon transform and stopped the right ones."""
    assert mega_forme("Necrozma-Dusk-Mane", Item.ULTRANECROZIUM_Z) == "Necrozma-Ultra"
    assert mega_forme("Necrozma-Dawn-Wings", Item.ULTRANECROZIUM_Z) == "Necrozma-Ultra"
    assert mega_forme("Necrozma", Item.ULTRANECROZIUM_Z) is None


def test_ultra_burst_does_not_spend_the_sides_mega_evolution():
    """Separate mechanics: a team may Mega Evolve a Rayquaza *and* Ultra Burst a Necrozma."""
    assert ultra_bursts("Necrozma-Dusk-Mane", Item.ULTRANECROZIUM_Z)
    assert not ultra_bursts("Rayquaza", Item.LIFE_ORB)
    assert SideState.model_fields["has_ultra_bursted"].default is False
    assert SideState.model_fields["has_mega_evolved"].default is False


def test_every_mega_forme_in_the_data_is_reachable_by_its_stone():
    """Forty-nine of them were not: the formes and their stats were vendored all along, but the
    stones were missing from the Item enum, so `item_from_showdown` returned None and the pairing
    was skipped. A mega nobody can reach is dead data."""
    from battle_sim.database.loader import get_all_species
    from battle_sim.formes import _forme_by_base_and_item, _forme_by_base_and_move

    reachable = set(_forme_by_base_and_item().values()) | set(_forme_by_base_and_move().values())
    megas = {s.name for s in get_all_species().values() if "Mega" in s.name and s.base_species}
    assert megas - reachable == {name for name in unplayable_formes() if "Mega" in name}


def test_a_forme_of_a_forme_reaches_its_own_mega():
    """Meowstic-M and Meowstic-F share one stone and one `baseSpecies`, so keying on the base alone
    collided and only one of them survived."""
    male = mega_forme("Meowstic", Item.MEOWSTICITE)
    female = mega_forme("Meowstic-F", Item.MEOWSTICITE)
    assert male == "Meowstic-M-Mega"
    assert female == "Meowstic-F-Mega"
    assert male != female


def test_a_forme_whose_ability_we_cannot_model_is_held_back():
    """An unmapped ability resolves to `Ability.NONE`, so the forme would transform and then play
    without the thing worth transforming for. Better absent than quietly wrong."""
    held = unplayable_formes()
    assert "Victreebel-Mega" in held  # Innards Out
    assert held["Victreebel-Mega"] == "Innards Out"
    assert mega_forme("Victreebel", item_from_showdown("Victreebelite")) is None
    # And nothing that already worked is caught by it.
    assert not {name for name in held if name.startswith(("Charizard", "Mewtwo", "Rayquaza", "Groudon", "Kyogre"))}


def test_a_mega_that_changes_base_hp_keeps_its_share_of_the_bar():
    """No Gen 6 or 7 mega changed base HP, so the old table could assume it never happened. Some of
    the Legends Z-A ones do — Floette-Mega goes from 54 to 74 — and a Pokemon halfway through a
    battle must not be healed or hurt by transforming."""
    spec = PokemonSpec(species="Floette", level=50, item=Item.FLOETTITE, moves=["Moonblast"])
    pokemon = build_pokemon(spec)
    pokemon.live_stats.HP = pokemon.stat_totals.HP // 2
    before = pokemon.live_stats.HP / pokemon.stat_totals.HP
    apply_forme(pokemon, "Floette-Mega")
    assert pytest.approx(before, abs=0.02) == pokemon.live_stats.HP / pokemon.stat_totals.HP
    assert pokemon.stat_totals.HP > 0


def test_a_forme_we_cannot_model_is_held_back_with_a_reason():
    """An unmapped ability resolves to `Ability.NONE`, so the forme would transform and then play
    without the thing worth transforming for. Better absent, and absent for a stated reason."""
    held = unplayable_formes()
    assert held["Victreebel-Mega"] == "Innards Out"
    assert held["Floette-Mega"] == "changes base HP"
    assert mega_forme("Victreebel", item_from_showdown("Victreebelite")) is None
    # Nothing that already worked is caught by it.
    assert not {n for n in held if n.startswith(("Charizard", "Mewtwo", "Rayquaza", "Groudon", "Kyogre"))}
