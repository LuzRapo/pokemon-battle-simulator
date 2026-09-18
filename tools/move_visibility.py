"""Which moves can the action scorer actually see?

`MatchupPlayer` scores an action from a feature vector. A move whose effect has no feature produces
the same vector as Splash — so the search cannot tell it apart from doing nothing, and it only ever
gets picked by accident. That is invisible in the ordinary sense too: nothing errors, the AI just
never plays a whole class of move.

This builds a position where every listed move could plausibly be the right play (the user hurt and
behind hazards, the opponent boosted and holding an item), asks for each move's features, and prints
the ones that come back identical to Splash's.

    uv run python -m tools.move_visibility
"""

from dataclasses import fields

from battle_sim.database.loader import get_move
from battle_sim.matchup import MatchupPlayer
from battle_sim.maths.rng import RNG
from battle_sim.mechanics.battle import BattleState, FieldState, SideState
from battle_sim.models.actions import Action, ActionType
from battle_sim.models.moves import MoveSet, MoveSlot
from battle_sim.models.pokemon import Pokemon
from battle_sim.models.stats import BaseStats, EVs, IVs
from battle_sim.utils import Hazards, Item, Nature, Type

# Terms every action in a position shares, so they say nothing about the move itself.
GENERIC = frozenset({"exchange_edge", "setup_concession", "wincon_preservation", "fodder_exploit", "wincon_support"})
# Psychic so that Ghost moves are neither immune nor resisted: a type chart artifact would otherwise
# read as "the scorer cannot see this move".
FOE_TYPES = (Type.PSYCHIC, None)
SURVEYED = [
    "Splash",
    "Haze",
    "Leech Seed",
    "Taunt",
    "Substitute",
    "Protect",
    "Encore",
    "Disable",
    "Light Screen",
    "Reflect",
    "Tailwind",
    "Trick Room",
    "Heal Bell",
    "Baton Pass",
    "Perish Song",
    "Destiny Bond",
    "Defog",
    "Rapid Spin",
    "Spikes",
    "Stealth Rock",
    "Toxic Spikes",
    "Sticky Web",
    "Whirlwind",
    "Roar",
    "Recover",
    "Roost",
    "Wish",
    "Toxic",
    "Thunder Wave",
    "Will-O-Wisp",
    "Swords Dance",
    "Calm Mind",
    "Geomancy",
    "Knock Off",
    "Trick",
    "Spectral Thief",
    "Thousand Arrows",
]


def _mk(nickname: str, moves: list[object], types: tuple[Type, Type | None], item: Item = Item.NONE) -> Pokemon:
    return Pokemon(
        name="T",
        nickname=nickname,
        level=50,
        base_stats=BaseStats(HP=100, ATTACK=100, DEFENCE=100, SP_ATTACK=100, SP_DEFENCE=100, SPEED=100),
        effort_values=EVs(),
        individual_values=IVs(),
        types=types,
        moves=MoveSet(*moves, *[None] * (4 - len(moves))),
        nature=Nature.HARDY,
        item=item,
    )


def features_for(name: str) -> dict[str, float] | None:
    """The non-generic features one move moves, or None if the scorer treats it as a dead action."""
    move = get_move(name)
    mine = _mk("Mine", [move, get_move("Tackle")], (Type.NORMAL, None))
    mine.live_stats.HP = mine.stat_totals.HP // 2  # something for healing to do
    theirs = _mk("Theirs", [get_move("Tackle")], FOE_TYPES, item=Item.LEFTOVERS)
    theirs.stat_stages.ATTACK = 2  # something for Haze/Taunt/Encore/Spectral Thief to answer
    state = BattleState(
        sides=(
            SideState(team=[mine, _mk("A2", [get_move("Tackle")], (Type.NORMAL, None))], hazards={Hazards.SPIKES: 2}),
            SideState(team=[theirs, _mk("B2", [get_move("Tackle")], FOE_TYPES)]),
        ),
        rng=RNG(seed=0),
        field=FieldState(),
    )
    action = Action(action=ActionType.USE_MOVE, target=move.target, move=MoveSlot.FIRST)
    featured, _ = MatchupPlayer().feature_actions(state, 0, [action])[0]
    if featured is None:
        return None
    return {
        field.name: round(getattr(featured, field.name), 2)
        for field in fields(featured)
        if abs(getattr(featured, field.name)) > 1e-9 and field.name not in GENERIC
    }


def main() -> None:
    baseline = features_for("Splash")
    blind: list[str] = []
    print("what the action scorer can see about each move (Splash is the do-nothing baseline)\n")
    for name in SURVEYED:
        if name == "Splash":
            continue
        live = features_for(name)
        if live is None:
            print(f"  {name:<16} dead action — priced below every real move")
        elif live == baseline:
            blind.append(name)
            print(f"  {name:<16} INVISIBLE — scores exactly as Splash does")
        else:
            print(f"  {name:<16} {', '.join(f'{k}={v}' for k, v in live.items())}")
    print(f"\n{len(blind)} invisible: {', '.join(blind)}")


if __name__ == "__main__":
    main()
