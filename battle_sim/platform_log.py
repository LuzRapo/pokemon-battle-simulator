"""Append-only record of the teams and battles the platform actually sees."""

import json
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from battle_sim.evolution import Team
from battle_sim.teams import build_pokemon, export_to_showdown, parse_showdown_team
from battle_sim.utils import Outcome


class Origin(StrEnum):
    """Where a team came from, because the two need different priors."""

    GENERATED = "generated"
    HUMAN = "human"


def log_team(path: Path, trainer: str, origin: Origin, team: Team) -> None:
    """One team, stored as the Showdown paste the rest of the project already reads and writes."""
    _append(
        path,
        {
            "kind": "team",
            "at": _now(),
            "trainer": trainer,
            "origin": str(origin),
            "team": export_to_showdown([build_pokemon(spec) for spec in team]),
        },
    )


def log_matchup(path: Path, trainer: str, opponent: str, teams: Sequence[Team], origins: Sequence[Origin]) -> None:
    """Both sides of one battle's teams, written before it starts so a crash cannot lose them."""
    for name, team, origin in zip((trainer, opponent), teams, origins, strict=True):
        log_team(path, name, origin, team)


def log_battle(path: Path, trainer: str, opponent: str, seed: int, outcome: Outcome | None, turns: int) -> None:
    _append(
        path,
        {
            "kind": "battle",
            "at": _now(),
            "trainer": trainer,
            "opponent": opponent,
            "seed": seed,
            "outcome": None if outcome is None else outcome.name,
            "turns": turns,
        },
    )


def read_teams(path: Path, origin: Origin | None = None) -> list[Team]:
    """Every logged team, optionally only those from one origin, oldest first."""
    return [team for entry_origin, team in _teams(path) if origin is None or entry_origin is origin]


def human_teams(path: Path) -> list[Team]:
    """The teams people actually built: the prior for believing what a human opponent is running."""
    return read_teams(path, Origin.HUMAN)


def _teams(path: Path) -> Iterator[tuple[Origin, Team]]:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        entry = json.loads(line)
        if entry["kind"] == "team":
            yield Origin(entry["origin"]), tuple(parse_showdown_team(entry["team"]).specs)


def _append(path: Path, entry: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(entry) + "\n")


def _now() -> str:
    return datetime.now(UTC).isoformat()
