"""Round-robin tournament across every trainer in `trainers_db.txt`: the same AI genome (the
newest tuned champion, wrapped in search) pilots both sides of every matchup, each played as a
best-of-3 series with sides swapped between games so neither trainer is always "P1". Ranks
trainers by series record, then by game differential.

Does not touch any training script or champion weights file — this only reads them.

`uv run python -m battle_sim.trainer_db_tournament`
"""

import argparse
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

from battle_sim.evolution import Progress, Team, load_weights
from battle_sim.matchup import MatchupWeights
from battle_sim.observation import SetPrior
from battle_sim.runner import run_battle
from battle_sim.search import SearchPlayer, SearchProfile
from battle_sim.teams import parse_showdown_team
from battle_sim.utils import Outcome

_HEADER_RE = re.compile(r"^=== (?P<name>.+) \((?P<version>.+)\) ===$", re.MULTILINE)
_GEN_RE = re.compile(r"^##### (?P<gen>.+?) #####$", re.MULTILINE)  # any section title, not only "Generation N"
_GAMES_PER_SERIES = 3
_WINS_TO_CLINCH = 2


@dataclass(frozen=True)
class Trainer:
    name: str
    generation: str
    team: Team


@dataclass(frozen=True)
class SeriesResult:
    a: str
    b: str
    a_wins: int
    b_wins: int
    draws: int

    @property
    def winner(self) -> str | None:
        if self.a_wins == self.b_wins:
            return None
        return self.a if self.a_wins > self.b_wins else self.b


@dataclass
class Standing:
    name: str
    generation: str
    series_w: int = 0
    series_l: int = 0
    series_d: int = 0
    game_w: int = 0
    game_l: int = 0
    game_d: int = 0

    @property
    def points(self) -> int:
        return 3 * self.series_w + self.series_d

    @property
    def game_diff(self) -> int:
        return self.game_w - self.game_l


def load_trainers(path: Path) -> list[Trainer]:
    """Every `=== Name (Version) ===` team in the file, tagged with its `##### Generation N #####` section."""
    text = path.read_text()
    gen_markers = [(m.start(), m.group("gen")) for m in _GEN_RE.finditer(text)]
    header_matches = list(_HEADER_RE.finditer(text))
    trainers: list[Trainer] = []
    for i, m in enumerate(header_matches):
        start = m.end()
        end = header_matches[i + 1].start() if i + 1 < len(header_matches) else len(text)
        body = "\n".join(line for line in text[start:end].splitlines() if not line.strip().startswith("#"))
        result = parse_showdown_team(body)
        if result.warnings:
            raise ValueError(f"{m.group('name')}: unexpected parse warnings {result.warnings}")
        generation = next((gen for pos, gen in reversed(gen_markers) if pos < m.start()), "Unknown")
        trainers.append(Trainer(name=m.group("name"), generation=generation, team=result.specs))
    names = [t.name for t in trainers]
    if len(names) != len(set(names)):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise ValueError(f"Duplicate trainer names collide in standings: {duplicates}")
    return trainers


def play_series(
    a: Trainer, b: Trainer, weights: MatchupWeights, profile: SearchProfile, prior: SetPrior, series_seed: int
) -> SeriesResult:
    a_wins = b_wins = draws = 0
    for game in range(1, _GAMES_PER_SERIES + 1):
        if a_wins == _WINS_TO_CLINCH or b_wins == _WINS_TO_CLINCH:
            break
        seed = series_seed * 10 + game
        player_a, player_b = SearchPlayer(weights, profile=profile), SearchPlayer(weights, profile=profile)
        a_is_p1 = game % 2 == 1  # alternate who's "P1" each game so neither side always moves first
        if a_is_p1:
            result = run_battle(a.team, b.team, player_a, player_b, seed=seed, prior=prior)
        else:
            result = run_battle(b.team, a.team, player_b, player_a, seed=seed, prior=prior)
        if result.outcome is None or result.outcome is Outcome.DRAW:
            draws += 1
        elif (result.outcome is Outcome.P1_WIN) == a_is_p1:
            a_wins += 1
        else:
            b_wins += 1
    return SeriesResult(a=a.name, b=b.name, a_wins=a_wins, b_wins=b_wins, draws=draws)


def tally(trainers: list[Trainer], results: list[SeriesResult]) -> list[Standing]:
    standings = {t.name: Standing(name=t.name, generation=t.generation) for t in trainers}
    for r in results:
        sa, sb = standings[r.a], standings[r.b]
        sa.game_w += r.a_wins
        sa.game_l += r.b_wins
        sa.game_d += r.draws
        sb.game_w += r.b_wins
        sb.game_l += r.a_wins
        sb.game_d += r.draws
        if r.winner == r.a:
            sa.series_w += 1
            sb.series_l += 1
        elif r.winner == r.b:
            sb.series_w += 1
            sa.series_l += 1
        else:
            sa.series_d += 1
            sb.series_d += 1
    return sorted(standings.values(), key=lambda s: (s.points, s.game_diff), reverse=True)


def _short_generation(generation: str) -> str:
    return generation.split(" — ")[0].replace("Generation ", "Gen ")


def render_standings(standings: list[Standing]) -> str:
    header = f"{'#':>3} {'Trainer':<45} {'Gen':<6} {'Series':>10} {'Games':>12} {'Pts':>4}"
    lines = [header, "-" * len(header)]
    for rank, s in enumerate(standings, start=1):
        series = f"{s.series_w}-{s.series_l}-{s.series_d}"
        games = f"{s.game_w}-{s.game_l}-{s.game_d}"
        gen = _short_generation(s.generation)
        lines.append(f"{rank:>3} {s.name:<45} {gen:<6} {series:>10} {games:>12} {s.points:>4}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Round-robin best-of-3 tournament across every trainer_db.txt team.")
    parser.add_argument("--teams-file", type=Path, default=Path("trainers_db.txt"))
    parser.add_argument("--champion", type=Path, default=Path("champions/random-a-myopic-tuned.json"))
    parser.add_argument("--search-budget", type=int, default=30)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("tournament_results.txt"))
    args = parser.parse_args()

    trainers = load_trainers(args.teams_file)
    prior = SetPrior.from_teams([t.team for t in trainers])
    weights = load_weights(args.champion)
    profile = SearchProfile(budget=args.search_budget)

    pairs = list(combinations(trainers, 2))
    progress = Progress(total=len(pairs))
    results: list[SeriesResult] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(play_series, a, b, weights, profile, prior, idx + args.seed * len(pairs)): (a, b)
            for idx, (a, b) in enumerate(pairs)
        }
        for future in as_completed(futures):
            a, b = futures[future]
            progress.note(f"{a.name} vs {b.name}")
            results.append(future.result())
            progress.tick()
    print()

    standings = tally(trainers, results)
    report = render_standings(standings)
    print(report)
    args.out.write_text(
        f"Trainer DB round-robin — {len(trainers)} trainers, {len(pairs)} best-of-3 series, "
        f"champion={args.champion.stem}, search-budget={args.search_budget}\n\n{report}\n"
    )


if __name__ == "__main__":
    main()
