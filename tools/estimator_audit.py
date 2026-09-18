"""Where does the AI's damage estimate disagree with what the engine actually deals?

`analysis.damage_range` predicts a [low, high] band. The engine then rolls inside that band — unless
something the estimator cannot see is in play, because every ability, item and side condition that
multiplies damage does it from an event-bus handler and the estimator never runs the bus.

So: play one move, compare the damage to the band it was predicted to land in, and group the misses
by the attacker's and defender's item and ability. Anything unmodelled shows up as a near-total
over- or under-shoot for one particular holder, standing out from the ~6% of samples a critical hit
pushes over the top.
"""

import random
import sys
from collections import defaultdict

from battle_sim.analysis import damage_range
from battle_sim.engine import legal_actions, step
from battle_sim.maths.rng import RNG
from battle_sim.mechanics.battle import BattleState, FieldState, SideState
from battle_sim.models.actions import ActionType
from battle_sim.models.log_events import DamageDealt, MoveMissed, MoveUsed
from battle_sim.setgen import random_set
from battle_sim.teams import build_pokemon
from battle_sim.utils import Category

SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 3000


def sample(rng: random.Random, pool: list[str]):
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
    logs = step(state, {0: choice, 1: theirs[0]})
    entries = list(logs.entries) if hasattr(logs, "entries") else [e for log in logs for e in log.entries]

    # Only the damage this move did: the first hit on the defender after our own move announcement,
    # and nothing after the opponent has started theirs.
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
    return attacker, defender, move.name, low, high, min(dealt, before)


def main() -> None:
    rng = random.Random(7)
    pool = sorted({random_set(name, random.Random(0)).species for name in _pool_names()})
    by_holder: dict[str, list[float]] = defaultdict(list)
    counted = over = under = 0
    for _ in range(SAMPLES):
        try:
            result = sample(rng, pool)
        except Exception:
            continue
        if result is None:
            continue
        attacker, defender, move_name, low, high, dealt = result
        if high <= 0:
            continue
        counted += 1
        ratio = dealt / max(1, (low + high) / 2)
        if dealt > high:
            over += 1
        elif dealt < low:
            under += 1
        for key in _keys(attacker, defender):
            by_holder[key].append(ratio)
    print(f"{counted} hits compared\n  landed above the predicted band: {over / counted:.1%}")
    print(f"  landed below it:                 {under / counted:.1%}\n")
    print("worst systematic offenders (mean actual/predicted, 12+ samples):")
    ranked = sorted(
        ((sum(v) / len(v), key, len(v)) for key, v in by_holder.items() if len(v) >= 12),
        key=lambda row: abs(row[0] - 1),
        reverse=True,
    )
    for mean, key, n in ranked[:18]:
        print(f"  {key:<44} {mean:5.2f}x  (n={n})")


def _keys(attacker, defender) -> list[str]:
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
