import argparse
import json
import os
import random
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, astuple, dataclass, field, fields
from pathlib import Path

from loguru import logger

from battle_sim.matchup import MatchupPlayer, MatchupWeights
from battle_sim.models.spec import PokemonSpec
from battle_sim.observation import SetPrior
from battle_sim.players import BasicPlayer
from battle_sim.runner import BattleResult, Player, run_battle
from battle_sim.search import SearchPlayer, SearchProfile
from battle_sim.teams import parse_showdown_team

type Team = tuple[PokemonSpec, ...]

_GENE_SCALES = MatchupWeights()  # mutation steps are sized relative to the stock priors


@dataclass(frozen=True)
class EvolutionConfig:
    population: int = 24
    generations: int = 30
    matchups_per_eval: int = 60
    elites: int = 2
    tournament: int = 3
    mutation_rate: float = 0.5
    mutation_sigma: float = 0.25
    seed: int = 0
    workers: int = 1
    search: SearchProfile | None = None  # None: the genome plays myopically instead of driving a lookahead


@dataclass(frozen=True)
class Matchup:
    candidate_team: Team
    opponent_team: Team
    seed: int
    candidate_is_p1: bool
    opponent_index: int


@dataclass(frozen=True)
class GenerationStats:
    generation: int
    best_fitness: float
    mean_fitness: float
    best: MatchupWeights
    population: tuple[MatchupWeights, ...]  # the generation as evaluated: champion selection reads the last one


def to_genes(weights: MatchupWeights) -> tuple[float, ...]:
    return astuple(weights)


def from_genes(genes: Sequence[float]) -> MatchupWeights:
    return MatchupWeights(*genes)


def mutate(weights: MatchupWeights, rng: random.Random, rate: float, sigma: float) -> MatchupWeights:
    genes = [
        max(0.0, gene + rng.gauss(0.0, sigma * scale)) if rng.random() < rate else gene
        for gene, scale in zip(to_genes(weights), to_genes(_GENE_SCALES), strict=True)
    ]
    return from_genes(genes)


def crossover(a: MatchupWeights, b: MatchupWeights, rng: random.Random) -> MatchupWeights:
    genes = [x if rng.random() < 0.5 else y for x, y in zip(to_genes(a), to_genes(b), strict=True)]
    return from_genes(genes)


def sample_matchups(teams: Sequence[Team], count: int, rng: random.Random, opponent_count: int = 1) -> list[Matchup]:
    """Same-seed side pairs: every battle is scored from both sides, cancelling side bias exactly.

    Each pair is also pinned to one pooled opponent, so the side flip cancels against
    the same player it was scored against.
    """
    if count % 2:
        raise ValueError(f"Matchup count must be even to form side pairs, got {count}.")
    matchups = []
    for _ in range(count // 2):
        candidate_team, opponent_team = rng.choice(teams), rng.choice(teams)
        seed = rng.randrange(2**32)
        opponent_index = rng.randrange(opponent_count)
        for candidate_is_p1 in (True, False):
            matchups.append(Matchup(candidate_team, opponent_team, seed, candidate_is_p1, opponent_index))
    return matchups


def fitness(
    weights: MatchupWeights,
    matchups: Sequence[Matchup],
    opponents: Sequence[Player],
    prior: SetPrior,
    search: SearchProfile | None = None,
) -> float:
    """Mean battle margin vs each matchup's pooled opponent: 0.5 + (own survivors - theirs) / 12.

    Margin makes partial progress count — "up two mons" beats "up one" — so selection
    has a gradient even between weight sets that both lose. Battles run under the shared
    metagame prior: both players believe sets from the pool, not the true ones. A search
    profile evolves the genome as a leaf evaluator inside SearchPlayer's lookahead.
    """
    candidate: Player = MatchupPlayer(weights) if search is None else SearchPlayer(weights, profile=search)
    total = 0.0
    for matchup in matchups:
        opponent = opponents[matchup.opponent_index]
        if matchup.candidate_is_p1:
            result = run_battle(
                matchup.candidate_team, matchup.opponent_team, candidate, opponent, seed=matchup.seed, prior=prior
            )
            total += battle_margin(result, candidate_index=0)
        else:
            result = run_battle(
                matchup.opponent_team, matchup.candidate_team, opponent, candidate, seed=matchup.seed, prior=prior
            )
            total += battle_margin(result, candidate_index=1)
    return total / len(matchups)


def battle_margin(result: BattleResult, candidate_index: int) -> float:
    own = result.survivors[candidate_index]
    theirs = result.survivors[1 - candidate_index]
    return 0.5 + (own - theirs) / 12


def evolve(
    teams: Sequence[Team],
    config: EvolutionConfig,
    opponents: Sequence[Player],
    tick: Callable[[], None] | None = None,
) -> Iterator[GenerationStats]:
    """Yield one GenerationStats per generation; `tick` fires after every genome evaluation."""
    rng = random.Random(config.seed)
    prior = SetPrior.from_teams(teams)
    population = [MatchupWeights()] + [
        mutate(MatchupWeights(), rng, config.mutation_rate, config.mutation_sigma) for _ in range(config.population - 1)
    ]
    for generation in range(config.generations):
        matchups = sample_matchups(teams, config.matchups_per_eval, rng, len(opponents))
        scores = _evaluate(population, matchups, opponents, prior, config.search, config.workers, tick)
        ranked = sorted(zip(scores, population, strict=True), key=lambda pair: pair[0], reverse=True)
        yield GenerationStats(
            generation=generation,
            best_fitness=ranked[0][0],
            mean_fitness=sum(scores) / len(scores),
            best=ranked[0][1],
            population=tuple(population),
        )
        population = [weights for _, weights in ranked[: config.elites]] + [
            _breed(ranked, rng, config) for _ in range(config.population - config.elites)
        ]


def _breed(
    ranked: Sequence[tuple[float, MatchupWeights]], rng: random.Random, config: EvolutionConfig
) -> MatchupWeights:
    child = crossover(
        tournament_select(ranked, rng, config.tournament), tournament_select(ranked, rng, config.tournament), rng
    )
    return mutate(child, rng, config.mutation_rate, config.mutation_sigma)


def tournament_select[T](ranked: Sequence[tuple[float, T]], rng: random.Random, tournament: int) -> T:
    return max(rng.sample(list(ranked), tournament), key=lambda pair: pair[0])[1]


def _evaluate(
    population: Sequence[MatchupWeights],
    matchups: Sequence[Matchup],
    opponents: Sequence[Player],
    prior: SetPrior,
    search: SearchProfile | None,
    workers: int,
    tick: Callable[[], None] | None,
) -> list[float]:
    if workers == 1:
        scores = []
        for weights in population:
            scores.append(fitness(weights, matchups, opponents, prior, search))
            if tick is not None:
                tick()
        return scores
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fitness, weights, matchups, opponents, prior, search) for weights in population]
        if tick is not None:
            for _ in as_completed(futures):
                tick()
        return [future.result() for future in futures]


def load_teams(directory: Path) -> list[Team]:
    """Every full, warning-free 6-mon team in the directory's showdown pastes."""
    teams: list[Team] = []
    for path in sorted(directory.glob("*.txt")):
        result = parse_showdown_team(path.read_text())
        if not result.warnings and len(result.specs) == 6:
            teams.append(result.specs)
    return teams


def load_team(path: Path) -> Team:
    result = parse_showdown_team(path.read_text())
    if result.warnings:
        raise ValueError(f"{path} has parse warnings: {result.warnings}")
    return result.specs


def save_weights(weights: MatchupWeights, path: Path) -> None:
    path.write_text(json.dumps(asdict(weights), indent=2) + "\n")


def load_weights(path: Path) -> MatchupWeights:
    """Genes absent from an older champion file fall back to their default.

    Genes get added as the feature set grows, and a champion written before a gene existed is still a
    valid baseline to measure against — refusing to load it would throw away exactly the comparison
    we want. An *unknown* gene is still an error: that means the file and the code disagree.
    """
    data = json.loads(path.read_text())
    expected = {gene.name for gene in fields(MatchupWeights)}
    unknown = set(data) - expected
    if unknown:
        raise ValueError(f"{path} has genes MatchupWeights does not: {sorted(unknown)}")
    missing = expected - set(data)
    if missing:
        logger.warning(f"{path} predates {len(missing)} gene(s), defaulting them: {sorted(missing)}")
    return MatchupWeights(**{name: float(value) for name, value in data.items()})


@dataclass
class Progress:
    """Single-line terminal progress bar; one tick per completed genome evaluation."""

    total: int
    started: float = field(default_factory=time.monotonic)
    completed: int = 0
    status: str = ""

    def tick(self) -> None:
        self.completed += 1
        self.render()

    def note(self, status: str) -> None:
        self.status = status
        self.render()

    def render(self) -> None:
        elapsed = time.monotonic() - self.started
        fraction = self.completed / self.total
        eta = f"{elapsed * (1 - fraction) / fraction / 60:5.1f}m" if self.completed else "    ?"
        filled = int(28 * fraction)
        bar = "█" * filled + "░" * (28 - filled)
        print(f"\r{bar} {fraction:6.1%}  elapsed {elapsed / 60:5.1f}m  eta {eta}  {self.status}", end="", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evolve MatchupPlayer's scoring weights against an opponent pool.")
    parser.add_argument("--teams-dir", type=Path, default=Path("sample_teams"))
    parser.add_argument("--team", type=Path, default=None, help="pin one paste: every battle is a mirror of it")
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--matchups", type=int, default=60)
    parser.add_argument("--elites", type=int, default=EvolutionConfig.elites)
    parser.add_argument("--tournament", type=int, default=EvolutionConfig.tournament)
    parser.add_argument("--mutation-rate", type=float, default=EvolutionConfig.mutation_rate)
    parser.add_argument("--mutation-sigma", type=float, default=EvolutionConfig.mutation_sigma)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--holdout", type=int, default=400, help="held-out matchups for the final champion eval")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument(
        "--opponent",
        type=Path,
        action="append",
        default=[],
        dest="opponents",
        help="champion weights JSON to add to the opponent pool (repeatable)",
    )
    parser.add_argument(
        "--search-opponent",
        type=Path,
        action="append",
        default=[],
        dest="search_opponents",
        help="champion weights JSON added to the pool as a SearchPlayer, not myopic (repeatable)",
    )
    parser.add_argument(
        "--opponent-budget",
        type=int,
        default=30,
        help="search budget for --search-opponent pool members",
    )
    parser.add_argument(
        "--drop-reference-pool",
        action="store_true",
        help="omit BasicPlayer and stock MatchupPlayer, leaving only the champions supplied",
    )
    parser.add_argument("--save", type=Path, default=None, help="write the champion's weights JSON here")
    parser.add_argument(
        "--search-budget",
        type=int,
        default=0,
        help="engine simulations per candidate decision (0: myopic; the genome becomes a search leaf evaluator)",
    )
    parser.add_argument("--root-k", type=int, default=5, help="actions per side kept at the search root")
    parser.add_argument("--chance-samples", type=int, default=2, help="roll realizations averaged per root cell")
    parser.add_argument("--determinizations", type=int, default=1, help="sampled belief worlds hedged per decision")
    parser.add_argument(
        "--validation",
        type=int,
        default=200,
        help="fixed matchups for picking the champion from the final population (escapes last-gen sample noise)",
    )
    args = parser.parse_args()
    if args.save is not None and not args.save.parent.is_dir():
        parser.error(f"--save directory {args.save.parent} does not exist")  # fail before the run, not after

    teams = [load_team(args.team)] if args.team is not None else load_teams(args.teams_dir)
    # A league, not two reference bots. Evolving against BasicPlayer and stock MatchupPlayer selects
    # for exploiting them: a round-robin showed every champion so evolved drifts to roughly twice the
    # switch rate of expert play, which beats weak opposition and little else. Searching pool members
    # matter most — they are the strongest agents available, and a myopic copy of a search champion is
    # a much softer target than the champion itself.
    pool: list[tuple[str, Player]] = []
    if not args.drop_reference_pool:
        pool += [("BasicPlayer", BasicPlayer()), ("MatchupPlayer(stock)", MatchupPlayer())]
    pool += [(path.stem, MatchupPlayer(load_weights(path))) for path in args.opponents]
    opponent_profile = SearchProfile(
        budget=args.opponent_budget,
        root_k=args.root_k,
        chance_samples=args.chance_samples,
        determinizations=args.determinizations,
    )
    pool += [
        (f"{path.stem}+S", SearchPlayer(load_weights(path), profile=opponent_profile)) for path in args.search_opponents
    ]
    if not pool:
        parser.error("empty opponent pool: --drop-reference-pool needs at least one --opponent/--search-opponent")
    opponents = [player for _, player in pool]
    logger.info(f"loaded {len(teams)} team(s) from {args.team or args.teams_dir}; pool of {len(pool)} opponents")
    logger.info(f"pool: {', '.join(label for label, _ in pool)}")
    search = (
        SearchProfile(
            budget=args.search_budget,
            root_k=args.root_k,
            chance_samples=args.chance_samples,
            determinizations=args.determinizations,
        )
        if args.search_budget
        else None
    )
    config = EvolutionConfig(
        population=args.population,
        generations=args.generations,
        matchups_per_eval=args.matchups,
        elites=args.elites,
        tournament=args.tournament,
        mutation_rate=args.mutation_rate,
        mutation_sigma=args.mutation_sigma,
        seed=args.seed,
        workers=args.workers,
        search=search,
    )
    progress = Progress(total=(config.generations + 1) * config.population)  # +1: the validation pass
    final: GenerationStats | None = None
    for stats in evolve(teams, config, opponents, tick=progress.tick):
        final = stats
        if args.save is not None:
            save_weights(stats.best, args.save)  # checkpoint: an interrupted run still leaves its champion
        progress.note(
            f"gen {stats.generation + 1}/{config.generations}"
            f"  best {stats.best_fitness:.3f}  mean {stats.mean_fitness:.3f}"
        )
    assert final is not None  # generations >= 1
    prior = SetPrior.from_teams(teams)
    progress.note("validating the final population...")
    validation = sample_matchups(teams, args.validation, random.Random(config.seed + 2), len(opponents))
    scores = _evaluate(final.population, validation, opponents, prior, config.search, config.workers, progress.tick)
    champion = max(zip(scores, final.population, strict=True), key=lambda pair: pair[0])[1]
    progress.note("held-out champion eval...")
    holdout = sample_matchups(teams, args.holdout, random.Random(config.seed + 1), len(opponents))
    subsets = [(label, [m for m in holdout if m.opponent_index == index]) for index, (label, _) in enumerate(pool)]
    margins = [
        (label, fitness(champion, subset, opponents, prior, config.search), len(subset))
        for label, subset in subsets
        if subset
    ]
    overall = sum(margin * count for _, margin, count in margins) / len(holdout)
    print(f"\nchampion after {final.generation + 1} generations (validated on {len(validation)} fixed matchups):")
    print(f"  validation fitness {max(scores):.3f}, held-out fitness {overall:.3f} ({len(holdout)} matchups)")
    for label, margin, count in margins:
        print(f"    vs {label}: {margin:.3f} ({count} matchups)")
    print(f"  {champion}")
    if args.save is not None:
        save_weights(champion, args.save)
        print(f"  champion saved to {args.save}")


if __name__ == "__main__":
    main()
