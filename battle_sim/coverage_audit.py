"""Reports the ability-coverage gap between in-scope species and what the engine mechanically models.

`uv run python -m battle_sim.coverage_audit`
"""

import argparse
import pathlib
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from battle_sim.database.loader import get_species, normalize_id
from battle_sim.database.scope import in_scope_species
from battle_sim.teams import _ABILITY_BY_SHOWDOWN_NAME
from battle_sim.utils import Ability

_ENGINE_SOURCE = "".join(p.read_text() for p in pathlib.Path(__file__).parent.rglob("*.py") if p.name != "teams.py")


def _wired_members[E: Enum](enum_cls: type[E], name_pattern: Callable[[E], str]) -> frozenset[E]:
    """Enum members whose engine name is referenced anywhere outside teams.py's display-name table."""
    return frozenset(member for member in enum_cls if re.search(name_pattern(member), _ENGINE_SOURCE))


@dataclass(frozen=True)
class AbilityGap:
    showdown_name: str
    species_affected: tuple[str, ...]
    has_enum_member: bool  # True: mapped but not mechanically wired; False: no Ability member at all


def ability_gaps() -> list[AbilityGap]:
    """One entry per real ability name used by an in-scope species that isn't mechanically wired, by impact."""
    wired = _wired_members(Ability, lambda a: rf"Ability\.{a.name}\b")
    species_by_name: dict[str, list[str]] = {}
    for key in in_scope_species():
        base_species = get_species(key)
        for name in (*base_species.regular_abilities, base_species.hidden_ability):
            if name is not None:
                species_by_name.setdefault(name, []).append(key)

    gaps = []
    for name, affected in species_by_name.items():
        ability = _ABILITY_BY_SHOWDOWN_NAME.get(normalize_id(name))
        if ability is None:
            gaps.append(AbilityGap(name, tuple(affected), has_enum_member=False))
        elif ability not in wired:
            gaps.append(AbilityGap(name, tuple(affected), has_enum_member=True))
    return sorted(gaps, key=lambda gap: len(gap.species_affected), reverse=True)


def species_blocked_summary(gaps: list[AbilityGap]) -> tuple[int, int, int]:
    """(fully blocked, partially blocked, fully fine) counts among in-scope species."""
    blocked_names = {gap.showdown_name for gap in gaps}
    fully_blocked = partially_blocked = fully_fine = 0
    for key in in_scope_species():
        species = get_species(key)
        names = [n for n in (*species.regular_abilities, species.hidden_ability) if n is not None]
        blocked = sum(1 for n in names if n in blocked_names)
        if blocked == len(names):
            fully_blocked += 1
        elif blocked > 0:
            partially_blocked += 1
        else:
            fully_fine += 1
    return fully_blocked, partially_blocked, fully_fine


def main() -> None:
    parser = argparse.ArgumentParser(description="Ability-coverage gap among Gen-7-in-scope species.")
    parser.add_argument("--top", type=int, default=40, help="how many ability gaps to list")
    args = parser.parse_args()

    scope = in_scope_species()
    gaps = ability_gaps()
    fully_blocked, partially_blocked, fully_fine = species_blocked_summary(gaps)

    print(f"in-scope species: {len(scope)}")
    print(f"  fully fine (every ability slot already wired): {fully_fine}")
    print(f"  partially blocked (>=1 usable ability among their slots): {partially_blocked}")
    print(f"  fully blocked (NO usable ability at all): {fully_blocked}")
    print()
    print(f"ability names with no engine mechanics, ranked by species affected (top {args.top}):")
    for gap in gaps[: args.top]:
        kind = "mapped, unwired" if gap.has_enum_member else "no enum member"
        print(f"  {len(gap.species_affected):4d}  {gap.showdown_name:<20s} ({kind})")


if __name__ == "__main__":
    main()
