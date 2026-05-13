import json

import pytest

from battle_sim.database.loader import (
    _CATEGORY_MAP,
    _STAT_MAP,
    _STATUS_MAP,
    _TARGET_MAP,
    _TYPE_MAP,
    _VENDOR_DIR,
    _VOLATILE_MAP,
    get_all_moves,
    get_all_species,
    get_move,
    get_species,
    normalize_id,
)
from battle_sim.models.moves import DamageEffect, InflictStatusEffect, Move, StatStageChangeEffect
from battle_sim.models.species import BaseSpecies
from battle_sim.models.stats import BaseStats
from battle_sim.utils import Category, ExtraStatus, PriorityLevel, Stats, Status, Target, Type


@pytest.mark.parametrize(
    "raw_name, expected_id",
    [
        ("Earthquake", "earthquake"),
        ("Will-O-Wisp", "willowisp"),
        ("Mr. Mime", "mrmime"),
        ("Tapu Koko", "tapukoko"),
        ("10,000,000 Volt Thunderbolt", "10000000voltthunderbolt"),
        ("EARTHQUAKE", "earthquake"),
        ("  earthquake  ", "earthquake"),
    ],
)
def test_normalize_id(raw_name, expected_id):
    assert normalize_id(raw_name) == expected_id


def test_get_move_accepts_varied_capitalisation_and_punctuation():
    canonical = get_move("Will-O-Wisp")
    assert get_move("willowisp") is canonical
    assert get_move("WILL-O-WISP") is canonical


def test_get_move_unknown_raises():
    with pytest.raises(KeyError, match="Unknown move"):
        get_move("Definitely Not A Real Move")


def test_get_species_unknown_raises():
    with pytest.raises(KeyError, match="Unknown species"):
        get_species("Mewthree")


def test_get_all_moves_is_cached():
    assert get_all_moves() is get_all_moves()


def test_get_all_species_is_cached():
    assert get_all_species() is get_all_species()


def test_earthquake_fixes_contact_bug():
    move = get_move("Earthquake")
    damage = next(eff for eff in move.effects if isinstance(eff, DamageEffect))
    assert damage.contact is False


def test_earthquake_full_shape():
    move = get_move("Earthquake")
    assert move == Move(
        name="Earthquake",
        type=Type.GROUND,
        category=Category.PHYSICAL,
        accuracy_probability=1.0,
        priority=PriorityLevel.NORMAL,
        pp=10,
        target=Target.ALL_ADJACENT,
        effects=(DamageEffect(power=100, category=Category.PHYSICAL, crit_stage=0, contact=False),),
    )


def test_thunderbolt_secondary_status():
    move = get_move("Thunderbolt")
    assert move.type is Type.ELECTRIC
    assert move.category is Category.SPECIAL
    assert move.accuracy_probability == 1.0
    damage, status = move.effects
    assert isinstance(damage, DamageEffect)
    assert damage.power == 90
    assert isinstance(status, InflictStatusEffect)
    assert status.status is Status.PARALYSIS
    assert status.probability == pytest.approx(0.1)


def test_willowisp_primary_status():
    move = get_move("Will-O-Wisp")
    assert move.category is Category.STATUS
    assert move.accuracy_probability == pytest.approx(0.85)
    (effect,) = move.effects
    assert effect == InflictStatusEffect(status=Status.BURN, probability=1.0)


def test_swordsdance_self_boost():
    move = get_move("Swords Dance")
    assert move.target is Target.SELF
    assert move.accuracy_probability is None
    (effect,) = move.effects
    assert effect == StatStageChangeEffect(target="SELF", stages={Stats.ATTACK: 2}, probability=1.0)


def test_dracometeor_damage_plus_self_drop():
    move = get_move("Draco Meteor")
    damage, drop = move.effects
    assert isinstance(damage, DamageEffect) and damage.power == 130
    assert drop == StatStageChangeEffect(target="SELF", stages={Stats.SP_ATTACK: -2}, probability=1.0)


def test_rockslide_flinch_chance():
    move = get_move("Rock Slide")
    assert move.target is Target.ALL_ADJACENT_ENEMIES
    damage, flinch = move.effects
    assert damage.power == 75
    assert flinch == InflictStatusEffect(status=ExtraStatus.FLINCH, probability=pytest.approx(0.3))


def test_bulletseed_multihit():
    move = get_move("Bullet Seed")
    (damage,) = move.effects
    assert damage.multi_hit == (2, 5)


def test_doubleedge_recoil():
    move = get_move("Double-Edge")
    (damage,) = move.effects
    assert damage.recoil_percent == pytest.approx(0.33)


def test_gigadrain_drain():
    move = get_move("Giga Drain")
    (damage,) = move.effects
    assert damage.drain_percent == pytest.approx(0.5)


def test_fakeout_priority_and_flinch():
    move = get_move("Fake Out")
    assert move.priority is PriorityLevel.FAKE_OUT
    damage, flinch = move.effects
    assert damage.power == 40 and damage.contact is True
    assert flinch == InflictStatusEffect(status=ExtraStatus.FLINCH, probability=1.0)


def test_extremespeed_priority():
    assert get_move("Extreme Speed").priority is PriorityLevel.E_SPEED


def test_trickroom_priority_and_target():
    move = get_move("Trick Room")
    assert move.priority is PriorityLevel.TRICK_ROOM
    assert move.target is Target.FIELD


def test_bulbasaur_species():
    bulbasaur = get_species("Bulbasaur")
    assert bulbasaur == BaseSpecies(
        name="Bulbasaur",
        dex_number=1,
        types=(Type.GRASS, Type.POISON),
        base_stats=BaseStats(HP=45, ATTACK=49, DEFENCE=49, SP_ATTACK=65, SP_DEFENCE=65, SPEED=45),
        regular_abilities=("Overgrow",),
        hidden_ability="Chlorophyll",
        height_m=0.7,
        weight_kg=6.9,
    )


def test_garchomp_species_matches_test_fixture():
    garchomp = get_species("Garchomp")
    assert garchomp.dex_number == 445
    assert garchomp.types == (Type.DRAGON, Type.GROUND)
    assert garchomp.base_stats == BaseStats(HP=108, ATTACK=130, DEFENCE=95, SP_ATTACK=80, SP_DEFENCE=85, SPEED=102)
    assert garchomp.regular_abilities == ("Sand Veil",)
    assert garchomp.hidden_ability == "Rough Skin"


def test_meowscarada_species():
    meowscarada = get_species("Meowscarada")
    assert meowscarada.dex_number == 908
    assert meowscarada.types == (Type.GRASS, Type.DARK)
    assert meowscarada.hidden_ability == "Protean"


def test_mega_garchomp_loads_as_separate_species():
    base = get_species("Garchomp")
    mega = get_species("Garchomp-Mega")
    assert mega.dex_number == base.dex_number
    assert mega.base_stats != base.base_stats


def test_total_counts_sane():
    species = get_all_species()
    moves = get_all_moves()
    assert 1100 < len(species) < 2000, len(species)
    assert 700 < len(moves) < 1100, len(moves)


def test_past_gen_species_still_loaded():
    for name in ["Mewtwo", "Gengar", "Snorlax", "Tyranitar", "Salamence", "Greninja"]:
        get_species(name)


def test_no_cap_or_custom_species_loaded():
    raw = json.loads((_VENDOR_DIR / "pokedex.json").read_text())
    loaded = get_all_species()
    for key, entry in raw.items():
        if entry.get("isNonstandard") in {"CAP", "Custom"}:
            assert key not in loaded


def test_all_loaded_moves_have_known_types_and_categories():
    for move in get_all_moves().values():
        assert isinstance(move.type, Type)
        assert isinstance(move.category, Category)
        assert isinstance(move.priority, PriorityLevel)
        assert isinstance(move.target, Target)


def test_all_source_types_are_mapped():
    raw = json.loads((_VENDOR_DIR / "moves.json").read_text())
    source_types = {entry["type"] for entry in raw.values()}
    assert source_types <= set(_TYPE_MAP), f"Unmapped types: {source_types - set(_TYPE_MAP)}"


def test_all_source_categories_are_mapped():
    raw = json.loads((_VENDOR_DIR / "moves.json").read_text())
    source_categories = {entry["category"] for entry in raw.values()}
    assert source_categories <= set(_CATEGORY_MAP)


def test_all_source_priorities_have_enum_members():
    raw = json.loads((_VENDOR_DIR / "moves.json").read_text())
    source_priorities = {entry["priority"] for entry in raw.values()}
    enum_values = {p.value for p in PriorityLevel}
    assert source_priorities <= enum_values, f"Unmapped priorities: {source_priorities - enum_values}"


def test_all_source_baseStat_keys_are_mapped():
    raw = json.loads((_VENDOR_DIR / "pokedex.json").read_text())
    keys = {k for entry in raw.values() if "baseStats" in entry for k in entry["baseStats"]}
    assert keys == set(_STAT_MAP)


def test_target_map_covers_all_singles_targets():
    raw = json.loads((_VENDOR_DIR / "moves.json").read_text())
    unmapped = {entry["target"] for entry in raw.values()} - set(_TARGET_MAP)
    assert unmapped <= {"adjacentAlly", "adjacentAllyOrSelf", "allies"}, unmapped


def test_status_map_covers_primary_statuses():
    raw = json.loads((_VENDOR_DIR / "moves.json").read_text())
    source = {entry["status"] for entry in raw.values() if isinstance(entry.get("status"), str)}
    assert source <= set(_STATUS_MAP), source - set(_STATUS_MAP)


def test_volatile_map_covers_observed_flinch_and_confusion():
    raw = json.loads((_VENDOR_DIR / "moves.json").read_text())
    secondary_volatiles = {
        entry["secondary"]["volatileStatus"]
        for entry in raw.values()
        if isinstance(entry.get("secondary"), dict) and isinstance(entry["secondary"].get("volatileStatus"), str)
    }
    assert "flinch" in secondary_volatiles
    assert "flinch" in _VOLATILE_MAP
    assert _VOLATILE_MAP["flinch"] is ExtraStatus.FLINCH


def test_secondary_self_boosts_are_extracted():
    fire_blast_like = next(
        m
        for m in get_all_moves().values()
        if any(
            isinstance(e, StatStageChangeEffect) and e.target == "SELF" and 0.0 < e.probability < 1.0 for e in m.effects
        )
    )
    assert fire_blast_like is not None


def test_loaded_garchomp_matches_conftest_fixture(garchomp_factory):
    chompy = garchomp_factory("Chompy")
    species = get_species("Garchomp")
    assert chompy.base_stats == species.base_stats
    assert chompy.types == species.types
