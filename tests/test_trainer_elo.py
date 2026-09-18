import pytest

from battle_sim.trainer_elo import GameResult, _margin_multiplier, compute_elo


def test_a_dominant_player_ends_up_rated_higher():
    games = [GameResult(a="Strong", b="Weak", winner="Strong") for _ in range(20)]
    ratings, _ = compute_elo(["Strong", "Weak"], games)
    assert ratings["Strong"] > ratings["Weak"]


def test_fixed_ratings_never_move():
    games = [GameResult(a="Newcomer", b="Champion", winner="Newcomer") for _ in range(20)]
    ratings, _ = compute_elo(["Newcomer"], games, fixed={"Champion": 2200.0})
    assert ratings["Champion"] == 2200.0


def test_floating_rating_rises_after_beating_a_fixed_strong_opponent():
    games = [GameResult(a="Newcomer", b="Champion", winner="Newcomer") for _ in range(20)]
    ratings, _ = compute_elo(["Newcomer"], games, fixed={"Champion": 2200.0})
    assert ratings["Newcomer"] > 1500.0


def test_floating_rating_falls_after_losing_to_a_fixed_weak_opponent():
    games = [GameResult(a="Underperformer", b="Scrub", winner="Scrub") for _ in range(20)]
    ratings, _ = compute_elo(["Underperformer"], games, fixed={"Scrub": 800.0})
    assert ratings["Underperformer"] < 1500.0


def test_a_name_cannot_be_both_floating_and_fixed():
    with pytest.raises(ValueError, match="Both"):
        compute_elo(["Both"], [], fixed={"Both": 1500.0})


def test_a_sweep_moves_ratings_more_than_a_last_pokemon_win():
    """Plain Elo throws the scoreline away, so 6-0 and 6-5 were worth the same. For a difficulty
    ladder they are not: the trainer who sweeps you is the harder trainer."""
    assert _margin_multiplier(1, 0) < _margin_multiplier(3, 0) < _margin_multiplier(6, 0)
    assert _margin_multiplier(3, 0) == pytest.approx(1.0)  # the average game is left exactly as it was


def test_a_favourites_blowout_counts_for_less_than_an_underdogs():
    """Stronger players win by more *by definition*, so an uncorrected margin weight feeds a
    favourite's rating back into its own margin and inflates it."""
    assert _margin_multiplier(6, 400) < _margin_multiplier(6, 0)


def test_a_draw_is_unweighted():
    assert _margin_multiplier(0, 0) == 1.0
