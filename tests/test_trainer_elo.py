import pytest

from battle_sim.trainer_elo import GameResult, compute_elo


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
