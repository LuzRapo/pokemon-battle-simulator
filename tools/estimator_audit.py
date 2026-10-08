"""Where does the AI's damage estimate disagree with what the engine actually deals?"""

import random
import sys
from collections import defaultdict
from dataclasses import dataclass

from loguru import logger

from battle_sim.analysis import damage_range
from battle_sim.engine import legal_actions, step
from battle_sim.maths.rng import RNG
from battle_sim.mechanics.battle import BattleState, FieldState, SideState
from battle_sim.models.actions import ActionType
from battle_sim.models.log_events import DamageDealt, MoveMissed, MoveUsed
from battle_sim.models.pokemon import Pokemon
from battle_sim.setgen import random_set
from battle_sim.teams import build_pokemon
from battle_sim.utils import Category

SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 3000


@dataclass(frozen=True)
class Sample:
    attacker: Pokemon
    defender: Pokemon
    low: int
    high: int
    dealt: int


def sample(rng: random.Random, pool: list[str]) -> Sample | None:
    """One move played once: what was predicted, and what actually landed."""
    attacker = build_pokemon(random_set(rng.choice(pool), rng))
    defender = build_pokemon(random_set(rng.choice(pool), rng))
    attacker.nickname, defender.nickname = "A", "D"
    state = BattleState(
        sides=(SideState(team=[attacker]), SideState(team=[defender])),
        rng=RNG(seed=rng.randrange(2**30)),
        field=FieldState(),
    )
    attacking = [
        action
        for action in legal_actions(state, 0)
        if action.action is ActionType.USE_MOVE
        and action.move is not None
        and (move := attacker.moves[action.move]) is not None
        and move.category is not Category.STATUS
        and not action.z_move
    ]
    if not attacking:
        return None
    choice = rng.choice(attacking)
    assert choice.move is not None
    move = attacker.moves[choice.move]
    assert move is not None
    low, high = damage_range(move, attacker, defender, state)

    theirs = [a for a in legal_actions(state, 1) if a.action is ActionType.USE_MOVE]
    if not theirs:
        return None
    before = defender.live_stats.HP
    entries = step(state, {0: choice, 1: theirs[0]}).entries

    # Only the damage this move did.
    dealt, seen_ours, missed = None, False, False
    for entry in entries:
        if isinstance(entry, MoveUsed):
            if entry.pokemon == "A" and not seen_ours:
                seen_ours = True
            elif seen_ours:
                break
        elif seen_ours and isinstance(entry, MoveMissed):
            missed = True
            break
        elif seen_ours and isinstance(entry, DamageDealt) and entry.side == 1:
            dealt = entry.amount
            break
    if missed or dealt is None:
        return None
    return Sample(attacker, defender, low, high, min(dealt, before))


def main() -> None:
    rng = random.Random(7)
    pool = sorted({random_set(name, random.Random(0)).species for name in _pool_names()})
    by_holder: dict[str, list[float]] = defaultdict(list)
    counted = over = under = 0
    for _ in range(SAMPLES):
        result = sample(rng, pool)
        if result is None:
            continue
        if result.high <= 0:
            continue
        counted += 1
        ratio = result.dealt / max(1, (result.low + result.high) / 2)
        if result.dealt > result.high:
            over += 1
        elif result.dealt < result.low:
            under += 1
        for key in _keys(result.attacker, result.defender):
            by_holder[key].append(ratio)
    logger.info(f"{counted} hits compared\n  landed above the predicted band: {over / counted:.1%}")
    logger.info(f"  landed below it:                 {under / counted:.1%}\n")
    logger.info("worst systematic offenders (mean actual/predicted, 12+ samples):")
    ranked = sorted(
        ((sum(v) / len(v), key, len(v)) for key, v in by_holder.items() if len(v) >= 12),
        key=lambda row: abs(row[0] - 1),
        reverse=True,
    )
    for mean, key, n in ranked[:18]:
        logger.info(f"  {key:<44} {mean:5.2f}x  (n={n})")


def _keys(attacker: Pokemon, defender: Pokemon) -> list[str]:
    """Items and abilities are where the unmodelled multipliers live, on both sides of the hit."""
    return [
        f"attacker item {attacker.item.name}",
        f"attacker ability {attacker.ability.name}",
        f"defender item {defender.item.name}",
        f"defender ability {defender.ability.name}",
    ]


def _pool_names() -> list[str]:
    from battle_sim.database.scope import in_scope_species

    return list(in_scope_species())


if __name__ == "__main__":
    main()
