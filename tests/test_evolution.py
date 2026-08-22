import random
from pathlib import Path

import pytest

from battle_sim.evolution import (
    EvolutionConfig,
    crossover,
    evolve,
    fitness,
    from_genes,
    load_weights,
    mutate,
    sample_matchups,
    save_weights,
    to_genes,
)
from battle_sim.matchup import MatchupPlayer, MatchupWeights
from battle_sim.observation import SetPrior
from battle_sim.players import BasicPlayer
from battle_sim.runner import Player
from battle_sim.teams import parse_showdown_team
from tests.test_teams import USER_SAMPLE_TEAM

TEAM = parse_showdown_team(USER_SAMPLE_TEAM).specs


def test_genes_roundtrip():
    weights = MatchupWeights(ko_now=7.0, hazard_value=0.9)
    assert from_genes(to_genes(weights)) == weights


def test_mutate_is_seeded_and_non_negative():
    rng_a, rng_b = random.Random(5), random.Random(5)
    mutant_a = mutate(MatchupWeights(), rng_a, rate=1.0, sigma=3.0)
    mutant_b = mutate(MatchupWeights(), rng_b, rate=1.0, sigma=3.0)
    assert mutant_a == mutant_b
    assert all(gene >= 0.0 for gene in to_genes(mutant_a))


def test_mutate_with_zero_rate_is_identity():
    assert mutate(MatchupWeights(), random.Random(0), rate=0.0, sigma=1.0) == MatchupWeights()


def test_crossover_takes_every_gene_from_a_parent():
    a, b = MatchupWeights(), mutate(MatchupWeights(), random.Random(1), rate=1.0, sigma=0.5)
    child = crossover(a, b, random.Random(2))
    for gene, gene_a, gene_b in zip(to_genes(child), to_genes(a), to_genes(b), strict=True):
        assert gene in (gene_a, gene_b)


def test_sample_matchups_builds_same_seed_side_pairs():
    matchups = sample_matchups([TEAM], count=4, rng=random.Random(0))
    assert [m.candidate_is_p1 for m in matchups] == [True, False, True, False]
    assert matchups[0].seed == matchups[1].seed
    assert matchups[0].candidate_team == matchups[1].candidate_team
    assert matchups[0].seed != matchups[2].seed  # fresh seed per pair


def test_sample_matchups_pins_each_pair_to_one_pooled_opponent():
    matchups = sample_matchups([TEAM], count=20, rng=random.Random(0), opponent_count=3)
    for first, second in zip(matchups[::2], matchups[1::2], strict=True):
        assert first.opponent_index == second.opponent_index
    indices = {m.opponent_index for m in matchups}
    assert indices <= {0, 1, 2}
    assert len(indices) > 1  # 10 pairs over 3 opponents: the pool is actually mixed in


def test_sample_matchups_rejects_odd_counts():
    with pytest.raises(ValueError, match="even"):
        sample_matchups([TEAM], count=3, rng=random.Random(0))


def test_fitness_is_a_margin_within_bounds():
    matchups = sample_matchups([TEAM], count=2, rng=random.Random(0))
    score = fitness(MatchupWeights(), matchups, [BasicPlayer()], SetPrior.from_teams([TEAM]))
    assert 0.0 <= score <= 1.0


def test_fitness_of_stock_weights_against_stock_matchup_player_is_exactly_half():
    """Same-seed side pairs against an identically-behaved opponent must cancel to 0.5."""
    matchups = sample_matchups([TEAM], count=4, rng=random.Random(0))
    assert fitness(MatchupWeights(), matchups, [MatchupPlayer()], SetPrior.from_teams([TEAM])) == 0.5


def test_weights_roundtrip_through_json(tmp_path: Path):
    weights = MatchupWeights(ko_now=0.0, matchup_gain=2.56, entry_cost=1.42)
    path = tmp_path / "champion.json"
    save_weights(weights, path)
    assert load_weights(path) == weights


def test_load_weights_rejects_gene_mismatches(tmp_path: Path):
    weights_path = tmp_path / "champion.json"
    save_weights(MatchupWeights(), weights_path)
    mangled = weights_path.read_text().replace("ko_now", "ko_later")
    weights_path.write_text(mangled)
    with pytest.raises(ValueError, match="ko_later"):
        load_weights(weights_path)


def test_evolve_is_deterministic_and_tracks_the_champion():
    config = EvolutionConfig(population=4, generations=2, matchups_per_eval=2, elites=1, tournament=2, seed=7)
    opponents: list[Player] = [BasicPlayer(), MatchupPlayer()]
    ticks = 0

    def tick() -> None:
        nonlocal ticks
        ticks += 1

    history_a = list(evolve([TEAM], config, opponents, tick=tick))
    history_b = list(evolve([TEAM], config, opponents))
    assert history_a == history_b
    assert len(history_a) == 2
    assert ticks == config.generations * config.population
    assert all(0.0 <= stats.best_fitness <= 1.0 for stats in history_a)
    assert all(stats.mean_fitness <= stats.best_fitness for stats in history_a)
