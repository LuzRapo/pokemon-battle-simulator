import random

import pytest

from battle_sim.database.loader import get_move
from battle_sim.maths.rng import RNG
from battle_sim.mechanics.battle import BattleState, SideState
from battle_sim.mechanics.log import BattleLog
from battle_sim.models.log_events import ItemChipDamage, ItemRemoved, LogEntry, MoveUsed, Switched
from battle_sim.models.moves import MoveSlot
from battle_sim.models.spec import PokemonSpec
from battle_sim.models.stats import EVs
from battle_sim.observation import BattleObserver, Knowledge, SetPrior
from battle_sim.teams import build_pokemon
from battle_sim.utils import Ability, Hazards, Item, Status

GARCHOMP_LEAD = PokemonSpec(
    species="Garchomp",
    ability=Ability.ROUGH_SKIN,
    item=Item.LEFTOVERS,
    moves=["Earthquake", "Dragon Claw", "Stealth Rock", "Protect"],
)
GARCHOMP_SCARF = PokemonSpec(
    species="Garchomp",
    ability=Ability.ROUGH_SKIN,
    item=Item.CHOICE_SCARF,
    moves=["Earthquake", "Outrage", "Fire Blast", "Stone Edge"],
)
GARCHOMP_BULKY = PokemonSpec(
    species="Garchomp",
    ability=Ability.ROUGH_SKIN,
    item=Item.ROCKY_HELMET,
    effort_values=EVs(HP=252, DEFENCE=252),
    moves=["Earthquake", "Dragon Claw", "Stealth Rock", "Protect"],
)
HEATRAN = PokemonSpec(
    species="Heatran",
    ability=Ability.FLASH_FIRE,
    item=Item.HEAVY_DUTY_BOOTS,
    moves=["Lava Plume", "Stealth Rock", "Protect", "Earth Power"],
)

# The lead set appears twice, the scarf set once: the lead set is the modal belief.
PRIOR = SetPrior.from_teams([[GARCHOMP_LEAD, HEATRAN], [GARCHOMP_LEAD, HEATRAN], [GARCHOMP_SCARF, HEATRAN]])


def test_preview_is_the_most_common_set():
    believed = PRIOR.preview("Garchomp", level=100)
    assert believed.item is Item.LEFTOVERS
    assert believed.level == 100


def test_revealed_move_flips_belief_to_the_consistent_set():
    believed = PRIOR.believe("Garchomp", Knowledge(moves={"Outrage"}))
    assert believed.item is Item.CHOICE_SCARF


def test_unknown_revealed_move_is_grafted_into_the_belief():
    believed = PRIOR.believe("Garchomp", Knowledge(moves={"Swords Dance"}))
    assert "Swords Dance" in believed.moves
    assert len(believed.moves) == 4


def test_revealed_ability_and_item_override_the_candidate():
    believed = PRIOR.believe("Garchomp", Knowledge(ability=Ability.SAND_VEIL, item=Item.LUM_BERRY))
    assert believed.ability is Ability.SAND_VEIL
    assert believed.item is Item.LUM_BERRY


def test_item_gone_means_holding_nothing():
    believed = PRIOR.believe("Garchomp", Knowledge(item=Item.CHOICE_SCARF, item_gone=True))
    assert believed.item is Item.NONE


def test_species_outside_the_prior_crashes():
    with pytest.raises(KeyError):
        PRIOR.believe("Pikachu", Knowledge())


def _battle() -> tuple[BattleState, BattleObserver]:
    state = BattleState(
        sides=(
            SideState(team=[build_pokemon(GARCHOMP_BULKY), build_pokemon(HEATRAN)]),
            SideState(team=[build_pokemon(GARCHOMP_SCARF), build_pokemon(HEATRAN)]),
        ),
        rng=RNG(seed=0),
    )
    return state, BattleObserver(state, PRIOR)


def _log(*entries: LogEntry) -> BattleLog:
    log = BattleLog()
    for entry in entries:
        log.add(entry)
    return log


def test_own_side_is_the_true_object_and_the_opponent_is_not():
    state, observer = _battle()
    view = observer.view(0)
    assert view.sides[0] is state.sides[0]
    assert view.sides[1] is not state.sides[1]
    assert view.sides[1].team[0] is not state.sides[1].team[0]


def test_hidden_details_come_from_the_prior_not_the_truth():
    state, observer = _battle()
    believed = observer.view(1).sides[0].team[0]  # side 1's belief about the bulky Garchomp
    true = state.sides[0].team[0]
    assert true.item is Item.ROCKY_HELMET
    assert believed.item is Item.LEFTOVERS  # the modal set, not the true one
    assert believed.stat_totals.HP < true.stat_totals.HP  # no HP investment in the modal set


def test_opponent_hp_is_a_percent_of_the_believed_max():
    state, observer = _battle()
    true = state.sides[0].team[0]
    true.apply_damage(true.stat_totals.HP // 2)
    believed = observer.view(1).sides[0].team[0]
    percent = round(100 * true.live_stats.HP / true.stat_totals.HP)
    assert round(percent / 100 * believed.stat_totals.HP) == believed.live_stats.HP


def test_move_use_updates_the_believed_set():
    _, observer = _battle()
    observer.ingest(_log(MoveUsed(side=1, pokemon="Garchomp", move="Outrage")))
    believed = observer.view(0).sides[1].team[0]
    assert believed.has_move(get_move("Outrage"))
    assert believed.item is Item.CHOICE_SCARF  # Outrage is unique to the scarf set


def test_rocky_helmet_chip_reveals_the_defenders_item():
    _, observer = _battle()
    observer.ingest(_log(ItemChipDamage(side=1, pokemon="Garchomp", item=Item.ROCKY_HELMET, amount=10)))
    believed = observer.view(1).sides[0].team[0]  # the chipped attacker was side 1: the helmet is side 0's
    assert believed.item is Item.ROCKY_HELMET


def test_knocked_off_item_is_believed_gone():
    _, observer = _battle()
    observer.ingest(_log(ItemRemoved(side=0, pokemon="Garchomp", item=Item.ROCKY_HELMET)))
    believed = observer.view(1).sides[0].team[0]
    assert believed.item is Item.NONE


def test_public_state_is_mirrored():
    state, observer = _battle()
    true = state.sides[1].team[0]
    true.status = Status.PARALYSIS
    true.stat_stages.ATTACK = 2
    state.sides[1].hazards[Hazards.STEALTH_ROCK] = 1
    view = observer.view(0)
    believed = view.sides[1].team[0]
    assert believed.status is Status.PARALYSIS
    assert believed.stat_stages.ATTACK == 2
    assert view.sides[1].hazards[Hazards.STEALTH_ROCK] == 1


def test_forme_change_is_public_but_the_set_stays_believed():
    state, observer = _battle()
    state.sides[0].team[1].name = "Garchomp"  # stand-in for an engine forme change (Palafin-Hero style)
    believed = observer.view(1).sides[0].team[1]
    assert believed.name == "Garchomp"  # the new forme's public identity (and stats) are seen...
    assert believed.has_move(get_move("Lava Plume"))  # ...but the set still comes from the Heatran prior


def _slot_of(believed, name: str) -> MoveSlot:
    return next(s for s in MoveSlot if (move := believed.moves[s]) is not None and move.name == name)


def test_observed_uses_floor_the_believed_pp():
    _, observer = _battle()
    observer.ingest(_log(MoveUsed(side=1, pokemon="Garchomp", move="Outrage")))
    observer.ingest(_log(MoveUsed(side=1, pokemon="Garchomp", move="Outrage")))
    believed = observer.view(0).sides[1].team[0]
    assert believed.pp[_slot_of(believed, "Outrage")] == get_move("Outrage").pp - 2


def test_choice_item_belief_implies_a_lock_until_the_switch():
    _, observer = _battle()
    observer.ingest(_log(MoveUsed(side=1, pokemon="Garchomp", move="Outrage")))  # Outrage implies the scarf set
    believed = observer.view(0).sides[1].team[0]
    assert believed.choice_locked_move is _slot_of(believed, "Outrage")
    observer.ingest(_log(Switched(side=1, withdrew="Garchomp", sent_out="Heatran")))
    assert observer.view(0).sides[1].team[0].choice_locked_move is None


def test_no_lock_without_a_believed_choice_item():
    _, observer = _battle()
    observer.ingest(_log(MoveUsed(side=0, pokemon="Garchomp", move="Earthquake")))  # consistent with Leftovers sets
    assert observer.view(1).sides[0].team[0].choice_locked_move is None


def test_sampled_views_draw_only_reveal_consistent_sets():
    _, observer = _battle()
    # No reveals: both prior Garchomp sets are equally consistent, so sampling mixes them 2:1 by frequency.
    items = {observer.sample_view(1, random.Random(seed)).sides[0].team[0].item for seed in range(16)}
    assert items == {Item.LEFTOVERS, Item.CHOICE_SCARF}
    observer.ingest(_log(MoveUsed(side=0, pokemon="Garchomp", move="Stealth Rock")))  # rules out the scarf set
    locked = {observer.sample_view(1, random.Random(seed)).sides[0].team[0].item for seed in range(16)}
    assert locked == {Item.LEFTOVERS}


def test_sampled_views_are_seeded_and_fresh_objects():
    _, observer = _battle()
    first = observer.sample_view(1, random.Random(3))
    again = observer.sample_view(1, random.Random(3))
    assert first.sides[0].team[0].item is again.sides[0].team[0].item
    assert first.sides[0].team[0] is not again.sides[0].team[0]
    assert first.sides[1] is observer.view(1).sides[1]  # own side stays the true object


def test_view_is_stable_until_a_reveal_changes_belief():
    _, observer = _battle()
    first = observer.view(0)
    again = observer.view(0)
    assert first is again  # players cache per-battle work keyed on state identity
    observer.ingest(_log(MoveUsed(side=1, pokemon="Garchomp", move="Outrage")))
    assert observer.view(0) is not first
    observer.ingest(_log(MoveUsed(side=1, pokemon="Garchomp", move="Outrage")))  # nothing new
    assert observer.view(0) is observer.view(0)
