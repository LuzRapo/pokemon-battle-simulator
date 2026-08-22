import argparse
import json
import os
import random
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from battle_sim.database.loader import get_move, get_species
from battle_sim.evolution import Matchup, Progress, Team, fitness, load_teams, load_weights, tournament_select
from battle_sim.matchup import MatchupPlayer, MatchupWeights
from battle_sim.models.spec import PokemonSpec
from battle_sim.observation import SetPrior
from battle_sim.players import BasicPlayer
from battle_sim.runner import Player
from battle_sim.search import SearchProfile
from battle_sim.utils import Ability, Item, Nature

type Slate = list[tuple[Team, int, int]]  # shared per generation: (opponent roster, seed, pool index)


@dataclass(frozen=True)
class TeamSearchConfig:
    population: int = 32
    generations: int = 30
    matchups_per_eval: int = 60
    elites: int = 2
    tournament: int = 3
    seed: int = 0
    workers: int = 1
    search: SearchProfile | None = None


@dataclass(frozen=True)
class TeamGenerationStats:
    generation: int
    best_fitness: float
    mean_fitness: float
    best: Team
    population: tuple[Team, ...]


def set_universe(teams: Sequence[Team]) -> list[PokemonSpec]:
    """Every distinct set in the pool, nickname-stripped, in first-seen order."""
    seen: dict[str, PokemonSpec] = {}
    for team in teams:
        for spec in team:
            canonical = spec.model_copy(update={"nickname": None, "moves": [get_move(m).name for m in spec.moves]})
            seen.setdefault(canonical.model_dump_json(), canonical)
    return list(seen.values())


def _species(spec: PokemonSpec) -> str:
    return get_species(spec.species).name


def draft_team(pool: Sequence[PokemonSpec], rng: random.Random) -> Team:
    """Six sets from a shuffle of the pool, greedily honoring species clause."""
    team: list[PokemonSpec] = []
    taken: set[str] = set()
    for spec in rng.sample(list(pool), len(pool)):
        if _species(spec) in taken:
            continue
        team.append(spec)
        taken.add(_species(spec))
        if len(team) == 6:
            return tuple(team)
    raise ValueError(f"Need 6 distinct species to draft a team, the pool has {len(taken)}.")


def crossover_teams(a: Team, b: Team, rng: random.Random) -> Team:
    return draft_team([*a, *b], rng)


def mutate_team(team: Team, universe: Sequence[PokemonSpec], rng: random.Random) -> Team:
    """Replace one slot with any set whose species doesn't collide with the other five."""
    slot = rng.randrange(len(team))
    others = {_species(spec) for i, spec in enumerate(team) if i != slot}
    replacement = rng.choice([spec for spec in universe if _species(spec) not in others])
    return tuple(replacement if i == slot else spec for i, spec in enumerate(team))


def sample_slate(teams: Sequence[Team], count: int, rng: random.Random, opponent_count: int) -> Slate:
    """The generation's shared opposition: every candidate roster faces exactly these draws."""
    if count % 2:
        raise ValueError(f"Matchup count must be even to form side pairs, got {count}.")
    return [(rng.choice(teams), rng.randrange(2**32), rng.randrange(opponent_count)) for _ in range(count // 2)]


def instantiate(team: Team, slate: Slate) -> list[Matchup]:
    return [
        Matchup(team, opponent_team, seed, candidate_is_p1, opponent_index)
        for opponent_team, seed, opponent_index in slate
        for candidate_is_p1 in (True, False)
    ]


def evolve_teams(
    weights: MatchupWeights,
    teams: Sequence[Team],
    config: TeamSearchConfig,
    opponents: Sequence[Player],
    tick: Callable[[], None] | None = None,
) -> Iterator[TeamGenerationStats]:
    """Yield one TeamGenerationStats per generation; `tick` fires after every roster evaluation."""
    rng = random.Random(config.seed)
    prior = SetPrior.from_teams(teams)
    universe = set_universe(teams)
    population = [draft_team(universe, rng) for _ in range(config.population)]
    for generation in range(config.generations):
        slate = sample_slate(teams, config.matchups_per_eval, rng, len(opponents))
        scores = _evaluate(weights, population, slate, opponents, prior, config.search, config.workers, tick)
        ranked = sorted(zip(scores, population, strict=True), key=lambda pair: pair[0], reverse=True)
        yield TeamGenerationStats(
            generation=generation,
            best_fitness=ranked[0][0],
            mean_fitness=sum(scores) / len(scores),
            best=ranked[0][1],
            population=tuple(population),
        )
        population = [team for _, team in ranked[: config.elites]] + [
            _breed(ranked, universe, rng, config) for _ in range(config.population - config.elites)
        ]


def _breed(
    ranked: Sequence[tuple[float, Team]], universe: Sequence[PokemonSpec], rng: random.Random, config: TeamSearchConfig
) -> Team:
    child = crossover_teams(
        tournament_select(ranked, rng, config.tournament), tournament_select(ranked, rng, config.tournament), rng
    )
    return mutate_team(child, universe, rng)


def _evaluate(
    weights: MatchupWeights,
    population: Sequence[Team],
    slate: Slate,
    opponents: Sequence[Player],
    prior: SetPrior,
    search: SearchProfile | None,
    workers: int,
    tick: Callable[[], None] | None,
) -> list[float]:
    if workers == 1:
        scores = []
        for team in population:
            scores.append(fitness(weights, instantiate(team, slate), opponents, prior, search))
            if tick is not None:
                tick()
        return scores
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(fitness, weights, instantiate(team, slate), opponents, prior, search) for team in population
        ]
        if tick is not None:
            for _ in as_completed(futures):
                tick()
        return [future.result() for future in futures]


def save_team(team: Team, path: Path) -> None:
    """Spec JSON with enums by name: Nature's values are NatureEffect objects, so names are the stable form."""
    path.write_text(json.dumps([_dump_spec(spec) for spec in team], indent=2) + "\n")


def _dump_spec(spec: PokemonSpec) -> dict[str, object]:
    dumped: dict[str, object] = spec.model_dump(mode="json")
    return dumped | {"ability": spec.ability.name, "item": spec.item.name, "nature": spec.nature.name}


def load_saved_team(path: Path) -> Team:
    loaded = json.loads(path.read_text())
    return tuple(
        PokemonSpec.model_validate(
            item | {"ability": Ability[item["ability"]], "item": Item[item["item"]], "nature": Nature[item["nature"]]}
        )
        for item in loaded
    )


def _summary(team: Team) -> str:
    return ", ".join(f"{spec.species} @ {spec.item.name}" for spec in team)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evolve a roster for a fixed champion genome, blind to opponents.")
    parser.add_argument("--teams-dir", type=Path, default=Path("sample_teams"))
    parser.add_argument("--weights", type=Path, default=None, help="champion weights JSON (default: stock genome)")
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--matchups", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--holdout", type=int, default=400, help="held-out matchups for the final champion eval")
    parser.add_argument(
        "--validation", type=int, default=200, help="fixed matchups for picking the champion roster at the end"
    )
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument(
        "--opponent",
        type=Path,
        action="append",
        default=[],
        dest="opponents",
        help="champion weights JSON to add to the opponent pool (repeatable)",
    )
    parser.add_argument("--save", type=Path, default=None, help="write the champion roster JSON here")
    parser.add_argument("--search-budget", type=int, default=0, help="0: the champion genome plays myopically")
    parser.add_argument("--root-k", type=int, default=5, help="actions per side kept at the search root")
    parser.add_argument("--chance-samples", type=int, default=2, help="roll realizations averaged per root cell")
    parser.add_argument("--determinizations", type=int, default=1, help="sampled belief worlds hedged per decision")
    args = parser.parse_args()
    if args.save is not None and not args.save.parent.is_dir():
        parser.error(f"--save directory {args.save.parent} does not exist")

    teams = load_teams(args.teams_dir)
    weights = load_weights(args.weights) if args.weights is not None else MatchupWeights()
    pool: list[tuple[str, Player]] = [("BasicPlayer", BasicPlayer()), ("MatchupPlayer(stock)", MatchupPlayer())]
    pool += [(path.stem, MatchupPlayer(load_weights(path))) for path in args.opponents]
    opponents = [player for _, player in pool]
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
    config = TeamSearchConfig(
        population=args.population,
        generations=args.generations,
        matchups_per_eval=args.matchups,
        seed=args.seed,
        workers=args.workers,
        search=search,
    )
    universe = set_universe(teams)
    logger.info(f"loaded {len(teams)} teams ({len(universe)} distinct sets); pool of {len(pool)} opponents")
    progress = Progress(total=(config.generations + 1) * config.population)
    final: TeamGenerationStats | None = None
    for stats in evolve_teams(weights, teams, config, opponents, tick=progress.tick):
        final = stats
        if args.save is not None:
            save_team(stats.best, args.save)  # checkpoint: an interrupted run still leaves its champion
        progress.note(
            f"gen {stats.generation + 1}/{config.generations}"
            f"  best {stats.best_fitness:.3f}  mean {stats.mean_fitness:.3f}"
        )
    assert final is not None  # generations >= 1
    prior = SetPrior.from_teams(teams)
    progress.note("validating the final population...")
    validation = sample_slate(teams, args.validation, random.Random(config.seed + 2), len(opponents))
    scores = _evaluate(
        weights, final.population, validation, opponents, prior, config.search, args.workers, progress.tick
    )
    champion = max(zip(scores, final.population, strict=True), key=lambda pair: pair[0])[1]
    progress.note("held-out champion eval...")
    holdout = sample_slate(teams, args.holdout, random.Random(config.seed + 1), len(opponents))
    overall = fitness(weights, instantiate(champion, holdout), opponents, prior, config.search)
    print(f"\nchampion roster after {final.generation + 1} generations (validated on {args.validation} matchups):")
    print(f"  validation fitness {max(scores):.3f}, held-out fitness {overall:.3f} ({args.holdout} matchups)")
    print(f"  {_summary(champion)}")
    if args.save is not None:
        save_team(champion, args.save)
        print(f"  champion roster saved to {args.save}")


if __name__ == "__main__":
    main()
