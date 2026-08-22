import pytest

from battle_sim.models.spec import ParseWarningKind, PokemonSpec
from battle_sim.models.stats import EVs, IVs
from battle_sim.teams import build_pokemon, export_to_showdown, parse_showdown_team
from battle_sim.utils import Ability, Item, Nature, Type

USER_SAMPLE_TEAM = """Birdy (Staraptor) @ Choice Band
Ability: Intimidate
EVs: 252 Atk / 4 Def / 252 Spe
Adamant Nature
- Brave Bird
- Return
- Close Combat
- U-turn

Rocky (Golem) @ Chople Berry
Ability: Rock Head
EVs: 4 HP / 252 Atk / 252 Spe
Adamant Nature
- Earthquake
- Rock Blast
- Sucker Punch
- Stealth Rock

Spooky (Gengar) @ Black Sludge
Ability: Levitate
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
- Shadow Ball
- Focus Blast
- Will-O-Wisp
- Taunt

Punchy (Machamp) @ Focus Sash
Ability: No Guard
EVs: 4 HP / 252 Atk / 252 Spe
Adamant Nature
- Dynamic Punch
- Payback
- Bullet Punch
- Stone Edge

Zappy (Electivire) @ Expert Belt
Ability: Motor Drive
EVs: 252 Atk / 252 SpA / 4 Spe
Lonely Nature
- Thunderbolt
- Ice Punch
- Cross Chop
- Flamethrower

Speedy (Deoxys-Attack) @ Life Orb
Ability: Pressure
EVs: 4 Atk / 252 SpA / 252 Spe
Rash Nature
- Ice Beam
- Thunderbolt
- Superpower
- Extreme Speed
"""


def test_parse_full_six_mon_team():
    specs = parse_showdown_team(USER_SAMPLE_TEAM).specs
    assert len(specs) == 6
    assert [s.nickname for s in specs] == ["Birdy", "Rocky", "Spooky", "Punchy", "Zappy", "Speedy"]
    assert [s.species for s in specs] == [
        "Staraptor",
        "Golem",
        "Gengar",
        "Machamp",
        "Electivire",
        "Deoxys-Attack",
    ]
    assert [s.nature for s in specs] == [
        Nature.ADAMANT,
        Nature.ADAMANT,
        Nature.TIMID,
        Nature.ADAMANT,
        Nature.LONELY,
        Nature.RASH,
    ]


def test_parse_extracts_evs_with_correct_stat_keys():
    specs = parse_showdown_team(USER_SAMPLE_TEAM).specs
    assert specs[0].effort_values == EVs(ATTACK=252, DEFENCE=4, SPEED=252)
    assert specs[4].effort_values == EVs(ATTACK=252, SP_ATTACK=252, SPEED=4)


def test_parse_each_pokemon_has_four_moves():
    for spec in parse_showdown_team(USER_SAMPLE_TEAM).specs:
        assert len(spec.moves) == 4


def test_parse_ability_is_mapped_when_known():
    specs = parse_showdown_team(USER_SAMPLE_TEAM).specs
    assert specs[0].ability is Ability.INTIMIDATE


def test_parse_unknown_ability_warns_and_defaults_to_none():
    result = parse_showdown_team("Garchomp\nAbility: Made Up Ability\n- Earthquake\n")
    assert result.specs[0].ability is Ability.NONE
    (warning,) = result.warnings
    assert warning.kind is ParseWarningKind.UNKNOWN_ABILITY
    assert warning.block_index == 0
    assert warning.detail == "Made Up Ability"


def test_parse_item_is_mapped_when_known():
    specs = parse_showdown_team(USER_SAMPLE_TEAM).specs
    assert specs[0].item is Item.CHOICE_BAND
    assert specs[2].item is Item.BLACK_SLUDGE
    assert specs[3].item is Item.FOCUS_SASH
    assert specs[5].item is Item.LIFE_ORB


def test_parse_unknown_item_warns_and_defaults_to_none():
    result = parse_showdown_team("Garchomp @ Made Up Item\n- Earthquake\n")
    assert result.specs[0].item is Item.NONE
    (warning,) = result.warnings
    assert warning.kind is ParseWarningKind.UNKNOWN_ITEM
    assert warning.detail == "Made Up Item"


def test_parse_unknown_nature_warns_and_defaults_to_hardy():
    result = parse_showdown_team("Garchomp\nSpicy Nature\n- Earthquake\n")
    assert result.specs[0].nature is Nature.HARDY
    (warning,) = result.warnings
    assert warning.kind is ParseWarningKind.UNKNOWN_NATURE


def test_parse_malformed_level_warns_and_defaults_to_100():
    result = parse_showdown_team("Garchomp\nLevel: lots\n- Earthquake\n")
    assert result.specs[0].level == 100
    (warning,) = result.warnings
    assert warning.kind is ParseWarningKind.MALFORMED_LEVEL


def test_parse_out_of_range_level_warns_and_defaults_to_100():
    result = parse_showdown_team("Garchomp\nLevel: 250\n- Earthquake\n")
    assert result.specs[0].level == 100
    (warning,) = result.warnings
    assert warning.kind is ParseWarningKind.MALFORMED_LEVEL


def test_parse_unknown_stat_key_warns():
    result = parse_showdown_team("Garchomp\nEVs: 252 Atk / 4 Vigor\n- Earthquake\n")
    assert result.specs[0].effort_values == EVs(ATTACK=252)
    (warning,) = result.warnings
    assert warning.kind is ParseWarningKind.UNKNOWN_STAT_KEY
    assert warning.detail == "Vigor"


def test_parse_extra_moves_truncated_warns():
    text = "Garchomp\n- Earthquake\n- Dragon Claw\n- Stone Edge\n- Fire Fang\n- Swords Dance\n"
    result = parse_showdown_team(text)
    assert result.specs[0].moves == ["Earthquake", "Dragon Claw", "Stone Edge", "Fire Fang"]
    (warning,) = result.warnings
    assert warning.kind is ParseWarningKind.EXTRA_MOVES_TRUNCATED


def test_parse_sample_team_is_fully_supported():
    result = parse_showdown_team(USER_SAMPLE_TEAM)
    assert result.warnings == ()


def test_parse_species_only_no_nickname_no_item():
    text = """Garchomp
- Earthquake
- Dragon Claw
- Stone Edge
- Fire Fang
"""
    result = parse_showdown_team(text)
    assert result.warnings == ()
    (spec,) = result.specs
    assert spec.species == "Garchomp"
    assert spec.nickname is None
    assert spec.item is Item.NONE
    assert spec.moves == ["Earthquake", "Dragon Claw", "Stone Edge", "Fire Fang"]


def test_parse_gender_marker_is_not_treated_as_nickname():
    text = """Garchomp (M) @ Choice Band
Ability: Sand Veil
Adamant Nature
- Earthquake
- Outrage
- Stone Edge
- Iron Head
"""
    result = parse_showdown_team(text)
    assert result.specs[0].species == "Garchomp"
    assert result.specs[0].nickname is None


def test_parse_nickname_species_and_gender_marker():
    text = """Leviathan (Dragapult) (M) @ Life Orb
Ability: Clear Body
Adamant Nature
- Dragon Darts
- Phantom Force
- U-turn
- Sucker Punch
"""
    result = parse_showdown_team(text)
    assert result.specs[0].species == "Dragapult"
    assert result.specs[0].nickname == "Leviathan"
    assert result.specs[0].item is Item.LIFE_ORB


def test_parse_level_line():
    text = """Pichu
Level: 5
- Tackle
"""
    assert parse_showdown_team(text).specs[0].level == 5


def test_parse_hidden_power_strips_type_bracket():
    text = """Manectric
- Thunderbolt
- Hidden Power [Fire]
- Flamethrower
- Snarl
"""
    assert parse_showdown_team(text).specs[0].moves == ["Thunderbolt", "Hidden Power", "Flamethrower", "Snarl"]


def test_parse_ivs_when_specified():
    text = """Magikarp
IVs: 0 Atk / 0 SpA
- Splash
"""
    assert parse_showdown_team(text).specs[0].individual_values == IVs(ATTACK=0, SP_ATTACK=0)


def test_parse_empty_text_returns_empty_result():
    assert parse_showdown_team("").specs == ()
    assert parse_showdown_team("\n\n").specs == ()


def test_parse_extra_blank_lines_between_blocks_are_tolerated():
    text = """Garchomp\n- Earthquake\n\n\n\nGengar\n- Shadow Ball"""
    assert [s.species for s in parse_showdown_team(text).specs] == ["Garchomp", "Gengar"]


def test_build_pokemon_from_full_spec():
    specs = parse_showdown_team(USER_SAMPLE_TEAM).specs
    team = [build_pokemon(s) for s in specs]
    assert team[0].name == "Staraptor"
    assert team[0].nickname == "Birdy"
    assert team[0].level == 100
    assert team[0].nature is Nature.ADAMANT
    assert team[0].effort_values.ATTACK == 252
    assert team[0].effort_values.DEFENCE == 4
    assert team[0].effort_values.SPEED == 252
    assert team[0].types[0] is Type.NORMAL
    assert team[0].types[1] is Type.FLYING
    assert [m.name for m in team[0].known_moves()] == ["Brave Bird", "Return", "Close Combat", "U-turn"]


def test_build_defaults_to_level_100_when_unspecified():
    spec = parse_showdown_team("Garchomp\n- Earthquake").specs[0]
    assert build_pokemon(spec).level == 100


def test_build_defaults_to_hardy_nature_when_unspecified():
    spec = parse_showdown_team("Garchomp\n- Earthquake").specs[0]
    assert build_pokemon(spec).nature is Nature.HARDY


def test_build_pokemon_unknown_species_raises():
    with pytest.raises(KeyError):
        build_pokemon(PokemonSpec(species="Notamon", moves=["Earthquake"]))


def test_build_pokemon_unknown_move_raises():
    with pytest.raises(KeyError):
        build_pokemon(PokemonSpec(species="Garchomp", moves=["FakeMove"]))


def test_spec_rejects_blank_species():
    with pytest.raises(ValueError):
        PokemonSpec(species="   ")


def test_spec_rejects_out_of_range_level():
    with pytest.raises(ValueError):
        PokemonSpec(species="Garchomp", level=0)


def test_export_includes_nickname_when_different_from_species():
    team = [build_pokemon(s) for s in parse_showdown_team(USER_SAMPLE_TEAM).specs]
    text = export_to_showdown(team)
    assert "Birdy (Staraptor)" in text
    assert "Spooky (Gengar)" in text


def test_export_omits_nickname_when_same_as_species():
    pokemon = build_pokemon(PokemonSpec(species="Garchomp", moves=["Earthquake"]))
    assert export_to_showdown([pokemon]).splitlines()[0] == "Garchomp"


def test_export_omits_zero_evs_and_full_ivs():
    pokemon = build_pokemon(PokemonSpec(species="Garchomp", moves=["Earthquake"]))
    text = export_to_showdown([pokemon])
    assert "EVs:" not in text
    assert "IVs:" not in text


def test_export_omits_hardy_nature():
    pokemon = build_pokemon(PokemonSpec(species="Garchomp", moves=["Earthquake"]))
    assert "Nature" not in export_to_showdown([pokemon])


def test_export_includes_level_when_not_100():
    pokemon = build_pokemon(PokemonSpec(species="Garchomp", level=50, moves=["Earthquake"]))
    assert "Level: 50" in export_to_showdown([pokemon])


def test_roundtrip_preserves_supported_fields():
    specs = parse_showdown_team(USER_SAMPLE_TEAM).specs
    team = [build_pokemon(s) for s in specs]
    re_text = export_to_showdown(team)
    re_specs = parse_showdown_team(re_text).specs
    assert len(re_specs) == 6
    for original, recovered in zip(specs, re_specs, strict=True):
        assert recovered.species == original.species
        assert recovered.nickname == original.nickname
        assert recovered.nature == original.nature
        assert recovered.moves == original.moves
        assert recovered.effort_values == original.effort_values
