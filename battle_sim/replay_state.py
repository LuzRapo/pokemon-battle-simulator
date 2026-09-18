"""Coarse battle state, reconstructed from a replay log and computable from a live battle.

The position evaluator's coefficients are guesses. `timer_value` prices a status as a flat constant,
`hazard_value` priced hazards as a headcount until today, and `residual_pressure = 6.0` because that
scale makes it commensurate with material -- not because anything measured what a landed Toxic is
actually worth. Five thousand replays contain the ground truth for exactly that question: what a
position was worth is what the side holding it went on to score.

The constraint that makes this usable rather than merely interesting is that **every feature here
must be computable from a log line and from a `BattleState` alike**. Fit on logs, evaluate in the
engine, with no state reconstruction, no approximated EV spreads and no divergence over forty turns.
Anything a log cannot see (exact damage rolls, held items before they are revealed) is therefore not
a feature, however much the engine knows about it.

Percentages, not absolute HP: the ladder runs with HP Percentage Mod, so a log never states a real
HP total. `_material` in the engine is already a sum of HP *fractions*, so the two agree.
"""

from dataclasses import dataclass, field

_HAZARD_NAMES = {"Stealth Rock", "Spikes", "Toxic Spikes", "Sticky Web"}
_LINGERING = {"tox", "psn", "brn"}  # the statuses that take HP every turn


@dataclass
class SideSnapshot:
    """One side of the board, as coarsely as a replay log can describe it."""

    hp: dict[str, float] = field(default_factory=dict)  # nickname -> share of its own maximum
    status: dict[str, str] = field(default_factory=dict)
    toxic_turns: dict[str, int] = field(default_factory=dict)
    hazards: dict[str, int] = field(default_factory=dict)
    boosts: dict[str, int] = field(default_factory=dict)  # nickname -> net positive stages
    active: str | None = None

    def material(self) -> float:
        """Sum of surviving HP fractions, 0..6 — the same quantity `search._material` computes."""
        return sum(share for share in self.hp.values() if share > 0)

    def alive(self) -> int:
        return sum(1 for share in self.hp.values() if share > 0)

    def lingering(self) -> int:
        """Members losing HP every turn to a status, which is not the same as merely statused."""
        return sum(1 for name, s in self.status.items() if s in _LINGERING and self.hp.get(name, 0) > 0)

    def toxic_pressure(self) -> float:
        """Badly-poisoned members weighted by how far the counter has climbed, capped at one each.

        The quantity `timer_value` flattens to a bare count and the whole stall investigation turned
        on: a Toxic on its sixth tick is a dead Pokemon, and one landed this turn is a promise.
        """
        total = 0.0
        for name, s in self.status.items():
            if s != "tox" or self.hp.get(name, 0) <= 0:
                continue
            turns = self.toxic_turns.get(name, 1)
            total += min(1.0, turns * (turns + 1) / 32)  # cumulative share of a bar, 1/16 + 2/16 + ...
        return total

    def hazard_layers(self) -> int:
        return sum(self.hazards.values())

    def active_boosts(self) -> int:
        return self.boosts.get(self.active, 0) if self.active else 0


def _slot(token: str) -> tuple[str, str]:
    """`p1a: Arceus` -> ("p1", "Arceus"); `p2: TENCHIN097` -> ("p2", "TENCHIN097")."""
    side, _, name = token.partition(": ")
    return side[:2], name.strip()


def _hp_share(field_text: str) -> float:
    """`84/100` -> 0.84; `0 fnt` -> 0.0."""
    head = field_text.split()[0]
    if head in {"0", "0 fnt"} or head.startswith("0/"):
        return 0.0
    numerator, _, denominator = head.partition("/")
    try:
        return max(0.0, min(1.0, float(numerator) / float(denominator or 100)))
    except ValueError:
        return 0.0


def apply(line: str, sides: dict[str, SideSnapshot]) -> None:
    """Fold one log line into the running snapshots. Unknown lines are ignored by design."""
    parts = line.split("|")
    if len(parts) < 3:
        return
    _apply_health(parts, sides)
    _apply_conditions(parts, sides)


def _apply_health(parts: list[str], sides: dict[str, SideSnapshot]) -> None:
    """Who is out and how much of them is left."""
    tag = parts[1]
    if tag in {"switch", "drag", "replace"} and len(parts) > 4:
        key, name = _slot(parts[2])
        side = sides[key]
        side.active = name
        side.hp[name] = _hp_share(parts[4])
        side.boosts[name] = 0  # stat stages do not survive a switch
    elif tag in {"-damage", "-heal", "-sethp"} and len(parts) > 3:
        key, name = _slot(parts[2])
        sides[key].hp[name] = _hp_share(parts[3])
    elif tag == "faint":
        key, name = _slot(parts[2])
        sides[key].hp[name] = 0.0


def _apply_conditions(parts: list[str], sides: dict[str, SideSnapshot]) -> None:
    """Status, hazards and stat stages."""
    tag = parts[1]
    if tag == "-status" and len(parts) > 3:
        key, name = _slot(parts[2])
        sides[key].status[name] = parts[3]
        if parts[3] == "tox":
            sides[key].toxic_turns[name] = 1
    elif tag == "-curestatus" and len(parts) > 2:
        key, name = _slot(parts[2])
        sides[key].status.pop(name, None)
        sides[key].toxic_turns.pop(name, None)
    elif tag == "-sidestart" and len(parts) > 3:
        key, _ = _slot(parts[2])
        hazard = parts[3].removeprefix("move: ").strip()
        if hazard in _HAZARD_NAMES:
            sides[key].hazards[hazard] = sides[key].hazards.get(hazard, 0) + 1
    elif tag == "-sideend" and len(parts) > 3:
        key, _ = _slot(parts[2])
        sides[key].hazards.pop(parts[3].removeprefix("move: ").strip(), None)
    elif tag in {"-boost", "-unboost"} and len(parts) > 4:
        key, name = _slot(parts[2])
        delta = int(parts[4]) * (1 if tag == "-boost" else -1)
        sides[key].boosts[name] = max(0, sides[key].boosts.get(name, 0) + delta)


def tick_toxic(sides: dict[str, SideSnapshot]) -> None:
    """Advance every badly-poisoned counter by a turn. Called once per `|turn|`."""
    for side in sides.values():
        for name in list(side.toxic_turns):
            if side.hp.get(name, 0) > 0:
                side.toxic_turns[name] += 1
