"""Generates a bank of randomly-drafted, legal 6v6 teams spanning the full Gen-7-in-scope dex.

Written as real Showdown-paste `.txt` files so the existing training pipeline (`evolution.py`,
`team_search.py`, `observation.SetPrior`) consumes them completely unchanged via `--teams-dir` —
no code there needs to know these teams were generated rather than scraped.

`uv run python -m battle_sim.team_bank --count 3000 --out random_teams`
"""

import argparse
import random
from pathlib import Path

from loguru import logger

from battle_sim.database.scope import in_scope_species
from battle_sim.setgen import random_set
from battle_sim.teams import build_pokemon, export_to_showdown

type Team = tuple[str, ...]


def draft_random_team(rng: random.Random) -> Team:
    """Six distinct species (species clause is automatic: sampling unique names can't collide)."""
    return tuple(rng.sample(sorted(in_scope_species()), 6))


def write_team_bank(directory: Path, count: int, rng: random.Random) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    digits = len(str(count - 1))
    for index in range(count):
        species = draft_random_team(rng)
        pokemons = [build_pokemon(random_set(name, rng)) for name in species]
        (directory / f"team_{index:0{digits}d}.txt").write_text(export_to_showdown(pokemons) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a bank of random legal teams for training.")
    parser.add_argument("--count", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("random_teams"))
    args = parser.parse_args()

    write_team_bank(args.out, args.count, random.Random(args.seed))
    logger.info(f"wrote {args.count} teams to {args.out}")


if __name__ == "__main__":
    main()
