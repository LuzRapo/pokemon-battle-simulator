"""The forme-changing abilities: Stance Change, Disguise, Schooling, Shields Down, Zen Mode.

Each of these swaps its holder onto a different species entry mid-battle. Until 2026-09-08 none of
them was wired at all, so Aegislash battled as a wall that could not hurt anything and Wishiwashi as
the 175-BST solo fish — which is what they were rated as. The engine applies them at controlled
points rather than from event handlers, because a swap rebinds ability handlers and doing that from
inside the bus's own walk of those handlers loses them.
"""

from battle_sim.engine import step
from battle_sim.formes import hp_forme, stance_forme
from battle_sim.maths.rng import RNG
from battle_sim.mechanics.battle import BattleState, SideState
from battle_sim.mechanics.log import BattleLog
from battle_sim.models.actions import Action, ActionType
from battle_sim.models.log_events import FormeChanged, VolatileInflicted
from battle_sim.models.moves import MoveSlot
from battle_sim.models.pokemon import Pokemon
from battle_sim.models.spec import PokemonSpec
from battle_sim.teams import build_pokemon
from battle_sim.utils import Ability, ExtraStatus, Status, Target

ATTACK = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.FIRST)
SECOND = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.SECOND)
GUARD = Action(action=ActionType.USE_MOVE, target=Target.SELF, move=MoveSlot.SECOND)
SETUP = Action(action=ActionType.USE_MOVE, target=Target.SELF, move=MoveSlot.THIRD)


def _mk(species: str, moves: list[str], ability: Ability, level: int = 50) -> Pokemon:
    return build_pokemon(PokemonSpec(species=species, level=level, moves=moves, ability=ability))


def _battle(mine: Pokemon, theirs: Pokemon) -> BattleState:
    return BattleState(sides=(SideState(team=[mine]), SideState(team=[theirs])), rng=RNG(seed=1))


def _aegislash() -> Pokemon:
    return _mk("Aegislash", ["Shadow Ball", "King's Shield", "Swords Dance", "Sacred Sword"], Ability.STANCE_CHANGE)


def _foe(move: str = "Tackle") -> Pokemon:
    return _mk("Snorlax", [move], Ability.NONE)


# -- Stance Change ------------------------------------------------------------------


def test_attacking_draws_the_blade() -> None:
    aegislash = _aegislash()
    state = _battle(aegislash, _foe())
    shielded_attack = aegislash.stat_totals.SP_ATTACK

    step(state, {0: ATTACK, 1: ATTACK})

    assert aegislash.name == "Aegislash-Blade"
    assert shielded_attack < aegislash.stat_totals.SP_ATTACK


def test_kings_shield_puts_the_blade_away() -> None:
    aegislash = _aegislash()
    state = _battle(aegislash, _foe())
    shielded_defence = aegislash.stat_totals.DEFENCE
    step(state, {0: ATTACK, 1: ATTACK})

    step(state, {0: GUARD, 1: ATTACK})

    assert aegislash.name == "Aegislash"
    assert shielded_defence == aegislash.stat_totals.DEFENCE


def test_the_blade_is_drawn_before_the_hit_it_is_drawn_for() -> None:
    """The point of the ability: the attack lands off Blade forme's Attack, not Shield forme's."""
    blade = _aegislash()
    plain = _mk("Aegislash", ["Shadow Ball", "King's Shield", "Swords Dance", "Sacred Sword"], Ability.NONE)

    dealt = []
    for attacker in (blade, plain):
        defender = _mk("Gengar", ["Tackle"], Ability.NONE)  # Shadow Ball has to be able to land
        state = _battle(attacker, defender)
        before = defender.live_stats.HP
        step(state, {0: ATTACK, 1: ATTACK})
        dealt.append(before - defender.live_stats.HP)

    assert dealt[0] > dealt[1]


def test_an_ordinary_status_move_leaves_the_stance_alone() -> None:
    aegislash = _aegislash()
    state = _battle(aegislash, _foe())

    step(state, {0: SETUP, 1: ATTACK})

    assert aegislash.name == "Aegislash"


def test_stance_change_only_answers_for_aegislash() -> None:
    ordinary = _mk("Snorlax", ["Tackle"], Ability.NONE)
    tackle = ordinary.moves[MoveSlot.FIRST]
    assert tackle is not None
    assert stance_forme(ordinary, tackle) is None


def test_the_swap_is_logged_like_a_mega_evolution() -> None:
    aegislash = _aegislash()
    state = _battle(aegislash, _foe())

    log = step(state, {0: ATTACK, 1: ATTACK})

    assert any(isinstance(e, FormeChanged) and e.forme == "Aegislash-Blade" for e in log)


# -- Disguise -----------------------------------------------------------------------


def _mimikyu() -> Pokemon:
    return _mk("Mimikyu", ["Play Rough", "Shadow Sneak", "Swords Dance", "Drain Punch"], Ability.DISGUISE)


def test_the_disguise_eats_the_first_attack_whole() -> None:
    mimikyu = _mimikyu()
    state = _battle(mimikyu, _foe("Earthquake"))  # Ground, since Normal cannot touch a Ghost at all

    step(state, {0: ATTACK, 1: ATTACK})

    assert mimikyu.name == "Mimikyu-Busted"
    assert mimikyu.live_stats.HP == mimikyu.stat_totals.HP


def test_the_second_attack_gets_through() -> None:
    mimikyu = _mimikyu()
    state = _battle(mimikyu, _foe("Earthquake"))
    step(state, {0: ATTACK, 1: ATTACK})

    step(state, {0: ATTACK, 1: ATTACK})

    assert mimikyu.live_stats.HP < mimikyu.stat_totals.HP


def test_a_busted_disguise_stays_busted() -> None:
    """The forme is the record of it, so nothing has to be reset — or forgotten to be reset."""
    mimikyu = _mimikyu()
    state = _battle(mimikyu, _foe("Earthquake"))
    step(state, {0: ATTACK, 1: ATTACK})

    step(state, {0: ATTACK, 1: ATTACK})

    assert mimikyu.name == "Mimikyu-Busted"


# -- Schooling ----------------------------------------------------------------------


def _wishiwashi(level: int = 50) -> Pokemon:
    return _mk("Wishiwashi", ["Waterfall", "Protect", "Rest", "Sleep Talk"], Ability.SCHOOLING, level=level)


def test_a_healthy_wishiwashi_schools_on_the_first_turn() -> None:
    wishiwashi = _wishiwashi()
    state = _battle(wishiwashi, _foe())
    solo_attack = wishiwashi.stat_totals.ATTACK

    step(state, {0: ATTACK, 1: ATTACK})

    assert wishiwashi.name == "Wishiwashi-School"
    assert solo_attack < wishiwashi.stat_totals.ATTACK


def test_the_school_breaks_up_below_a_quarter_health() -> None:
    wishiwashi = _wishiwashi()
    state = _battle(wishiwashi, _foe())
    step(state, {0: ATTACK, 1: ATTACK})
    assert wishiwashi.name == "Wishiwashi-School"

    wishiwashi.live_stats.HP = wishiwashi.stat_totals.HP // 6
    step(state, {0: ATTACK, 1: ATTACK})

    assert wishiwashi.name == "Wishiwashi"


def test_a_young_wishiwashi_cannot_school_however_healthy() -> None:
    assert hp_forme(_wishiwashi(level=10)) is None


# -- Shields Down -------------------------------------------------------------------


def _minior() -> Pokemon:
    return _mk("Minior", ["Shell Smash", "Acrobatics", "Earthquake", "Stone Edge"], Ability.SHIELDS_DOWN)


def test_minior_keeps_its_shell_up_while_healthy() -> None:
    minior = _minior()
    state = _battle(minior, _foe())

    step(state, {0: SECOND, 1: ATTACK})

    assert minior.name == "Minior-Meteor"


def test_the_shell_cracks_at_half_health_and_the_core_hits_harder() -> None:
    minior = _minior()
    state = _battle(minior, _foe())
    step(state, {0: SECOND, 1: ATTACK})
    shelled_attack = minior.stat_totals.ATTACK

    minior.live_stats.HP = minior.stat_totals.HP // 4
    step(state, {0: SECOND, 1: ATTACK})

    assert minior.name == "Minior"
    assert shelled_attack < minior.stat_totals.ATTACK


# -- Zen Mode -----------------------------------------------------------------------


def _darmanitan() -> Pokemon:
    """Rock Slide leads rather than Flare Blitz: recoil would put it under the line on turn one, and
    then the test would pass without the threshold ever being the reason."""
    return _mk("Darmanitan", ["Rock Slide", "Earthquake", "Flare Blitz", "U-turn"], Ability.ZEN_MODE)


def test_darmanitan_enters_zen_mode_at_half_health() -> None:
    darmanitan = _darmanitan()
    state = _battle(darmanitan, _foe())
    step(state, {0: ATTACK, 1: ATTACK})
    assert darmanitan.name == "Darmanitan"
    lean_defence = darmanitan.stat_totals.DEFENCE

    darmanitan.live_stats.HP = darmanitan.stat_totals.HP // 3
    step(state, {0: ATTACK, 1: ATTACK})

    assert darmanitan.name == "Darmanitan-Zen"
    assert lean_defence < darmanitan.stat_totals.DEFENCE  # Zen trades the glass cannon for a wall


def test_healing_back_over_the_line_leaves_zen_mode() -> None:
    darmanitan = _darmanitan()
    state = _battle(darmanitan, _foe())
    darmanitan.live_stats.HP = darmanitan.stat_totals.HP // 3
    step(state, {0: ATTACK, 1: ATTACK})
    assert darmanitan.name == "Darmanitan-Zen"

    darmanitan.live_stats.HP = darmanitan.stat_totals.HP
    step(state, {0: ATTACK, 1: ATTACK})

    assert darmanitan.name == "Darmanitan"


def test_a_fainted_pokemon_does_not_change_forme_on_the_way_out() -> None:
    darmanitan = _darmanitan()
    darmanitan.live_stats.HP = 0
    assert hp_forme(darmanitan) is None


def test_zen_mode_survives_reverting_and_can_trigger_again() -> None:
    """Zen Mode is Darmanitan's *hidden* ability, so a naive swap back hands it Sheer Force — the
    forme reverts once and can never enter Zen again."""
    darmanitan = _darmanitan()
    state = _battle(darmanitan, _foe())
    darmanitan.live_stats.HP = darmanitan.stat_totals.HP // 3
    step(state, {0: ATTACK, 1: ATTACK})
    darmanitan.live_stats.HP = darmanitan.stat_totals.HP
    step(state, {0: ATTACK, 1: ATTACK})
    assert darmanitan.name == "Darmanitan"
    assert darmanitan.ability is Ability.ZEN_MODE

    darmanitan.live_stats.HP = darmanitan.stat_totals.HP // 3
    step(state, {0: ATTACK, 1: ATTACK})

    assert darmanitan.name == "Darmanitan-Zen"


def test_a_swapped_forme_keeps_the_ability_that_swapped_it() -> None:
    aegislash = _aegislash()
    state = _battle(aegislash, _foe())
    step(state, {0: ATTACK, 1: ATTACK})
    assert aegislash.ability is Ability.STANCE_CHANGE


# -- Power Construct ----------------------------------------------------------------
# The one forme change here that is permanent, and the only one that changes its holder's maximum
# HP. Zygarde is Uber entirely because of it; unwired, it rated as an ordinary dragon.


def _zygarde() -> Pokemon:
    return _mk("Zygarde", ["Thousand Arrows", "Earthquake", "Toxic", "Rest"], Ability.POWER_CONSTRUCT)


def test_the_cells_swarm_in_at_half_health() -> None:
    zygarde = _zygarde()
    state = _battle(zygarde, _foe())
    step(state, {0: ATTACK, 1: ATTACK})
    assert zygarde.name == "Zygarde"

    zygarde.live_stats.HP = zygarde.stat_totals.HP // 2
    step(state, {0: ATTACK, 1: ATTACK})

    assert zygarde.name == "Zygarde-Complete"
    assert zygarde.ability is Ability.POWER_CONSTRUCT


def test_completing_is_a_second_wind_rather_than_a_rescaling() -> None:
    """The arriving cells are extra HP, so the fraction of the bar goes *up*, not across.

    Rescaling would land Complete on half of a doubled maximum, which reads on the scoreboard as
    "nothing happened" — the opposite of the ability that makes Zygarde Uber.
    """
    zygarde = _zygarde()
    state = _battle(zygarde, _foe())
    zygarde.live_stats.HP = zygarde.stat_totals.HP // 2
    was_live, was_max = zygarde.live_stats.HP, zygarde.stat_totals.HP
    step(state, {0: ATTACK, 1: ATTACK})

    assert was_max < zygarde.stat_totals.HP  # Complete's bar is far longer
    assert was_live < zygarde.live_stats.HP  # and it arrives holding more than it had
    assert zygarde.live_stats.HP / zygarde.stat_totals.HP > 0.5


def test_the_cells_never_leave_again() -> None:
    """Every other forme ability here reverts across its threshold. This one cannot: there is no
    Zygarde to go back to once the swarm has assembled."""
    zygarde = _zygarde()
    state = _battle(zygarde, _foe())
    zygarde.live_stats.HP = zygarde.stat_totals.HP // 2
    step(state, {0: ATTACK, 1: ATTACK})
    assert zygarde.name == "Zygarde-Complete"

    zygarde.live_stats.HP = zygarde.stat_totals.HP  # healed to full
    step(state, {0: ATTACK, 1: ATTACK})

    assert zygarde.name == "Zygarde-Complete"


def test_a_healthy_zygarde_stays_as_it_is() -> None:
    zygarde = _zygarde()
    state = _battle(zygarde, _foe())
    zygarde.live_stats.HP = int(zygarde.stat_totals.HP * 0.9)
    step(state, {0: ATTACK, 1: ATTACK})
    assert zygarde.name == "Zygarde"


def test_the_randbats_set_actually_carries_the_ability() -> None:
    """It used to fall through to Aura Break: the name mapped to nothing, and `setgen` then picked
    one of the species' other real abilities instead."""
    import random

    from battle_sim.setgen import random_set
    from battle_sim.teams import build_pokemon as build

    abilities = {build(random_set("Zygarde", random.Random(seed))).ability for seed in range(5)}
    assert abilities == {Ability.POWER_CONSTRUCT}


def test_a_disguise_takes_the_flinch_along_with_the_hit() -> None:
    """From a real match: Iron Head broke the disguise and the Mimikyu behind it flinched anyway.

    Gen 7 Disguise absorbs the whole move rather than only its damage — Showdown does it by
    returning 0 from `onTryHit`, which skips secondaries too — so a Mimikyu that was never actually
    hit cannot be made to flinch. Twenty attempts against a 30% flinch is not luck.
    """
    for seed in range(20):
        mimikyu = _mimikyu()
        state = BattleState(sides=(SideState(team=[mimikyu]), SideState(team=[_foe("Iron Head")])), rng=RNG(seed=seed))

        played = step(state, {0: ATTACK, 1: ATTACK})

        assert mimikyu.name == "Mimikyu-Busted", "the disguise did not take the hit at all"
        assert not _flinched(played), "flinched through an intact disguise"


def test_a_busted_disguise_can_be_flinched_like_anything_else() -> None:
    """The disguise is one free move, not a standing immunity to being flinched."""
    flinched = False
    for seed in range(30):
        state = BattleState(
            sides=(SideState(team=[_mimikyu()]), SideState(team=[_foe("Iron Head")])), rng=RNG(seed=seed)
        )
        step(state, {0: ATTACK, 1: ATTACK})  # this one goes on the disguise
        flinched = flinched or _flinched(step(state, {0: ATTACK, 1: ATTACK}))  # and this one lands
    assert flinched, "thirty Iron Heads at a busted disguise and never a single flinch"


def _flinched(played: BattleLog) -> bool:
    """Read off the log rather than the volatile: a flinch is cleared at the end of the turn it
    happens in, so by the time `step` returns there is nothing left on the Pokemon to find."""
    return any(
        isinstance(entry, VolatileInflicted) and entry.volatile is ExtraStatus.FLINCH for entry in played.entries
    )


def test_a_status_move_goes_straight_through_a_disguise() -> None:
    """It absorbs attacks. Will-O-Wisp is not an attack, and Gen 7 lets it through."""
    mimikyu = _mimikyu()
    state = _battle(mimikyu, _foe("Will-O-Wisp"))

    step(state, {0: ATTACK, 1: ATTACK})

    assert mimikyu.name == "Mimikyu", "a status move broke the disguise"
    assert mimikyu.status is Status.BURN
