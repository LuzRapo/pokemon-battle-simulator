import pytest

from battle_sim.database.scope import gen7_high_tier_species, gen7_movepool, in_scope_species


def test_gen1_species_is_in_scope():
    assert "bulbasaur" in in_scope_species()


def test_mega_forme_is_excluded():
    assert "charizardmegax" not in in_scope_species()


def test_post_gen7_species_is_excluded():
    assert "zacian" not in in_scope_species()


def test_regional_forme_sharing_an_old_dex_number_is_excluded():
    assert "meowthgalar" not in in_scope_species()


def test_battle_only_forme_of_a_post_gen7_forme_is_excluded():
    """Darmanitan-Galar-Zen's base_species is plain Darmanitan (Gen 5); its real parent,

    reachable only via battleOnly, is Darmanitan-Galar (Gen 8) — the Gen-8 parent must win.
    """
    assert "darmanitangalarzen" not in in_scope_species()


def test_event_only_forme_with_no_learnset_entry_inherits_its_base_movepool():
    """Arceus' type formes have a learnsets.json entry with no `learnset` key at all

    ({"eventOnly": true}) — that must count as "no data of its own", not "zero Gen-7 moves".
    """
    assert "arceusbug" in in_scope_species()
    assert gen7_movepool("arceusbug") == gen7_movepool("arceus")


def test_pre_gen8_alternate_forme_inherits_its_base_movepool():
    assert "rotomfan" in in_scope_species()
    assert "airslash" in gen7_movepool("rotomfan")
    assert "thunderbolt" in gen7_movepool("rotomfan")  # inherited from base Rotom


def test_cosmetic_forme_is_excluded():
    assert "vivillonpolar" not in in_scope_species()


def test_unknown_species_raises():
    with pytest.raises(KeyError):
        gen7_movepool("not-a-real-species")


def test_movepool_excludes_moves_only_taught_after_gen7():
    assert "acidspray" not in gen7_movepool("bulbasaur")  # Gen 9 TM move only
    assert "tackle" in gen7_movepool("bulbasaur")


def test_high_tier_species_is_gen7s_own_ou_uu_uber_band():
    pool = gen7_high_tier_species()
    assert "moltres" in pool  # Gen 7's own UU, unlike the current-gen `pokedex.json` tier field
    assert "mewtwo" in pool  # Uber
    assert "bulbasaur" not in pool  # LC
    assert "zubat" not in pool  # ZU
    assert pool <= in_scope_species()  # never offers something unbattleable in this engine


def test_high_tier_species_excludes_mega_only_placements():
    """`charizardmegax` is tiered OU in Gen 7, but it is not an `in_scope_species()` entry — Mega
    formes are reached mid-battle via the base species' held item, never selected directly."""
    assert "charizardmegax" not in gen7_high_tier_species()
