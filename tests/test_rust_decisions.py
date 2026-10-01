"""Every decision a player makes, made by the player — in both engines, agreeing.

`test_rust_battles.py` plays the differential harness's battles, where a pivot's replacement is
always the lowest-index healthy Pokemon and every lead is team slot 0. Self-play needs the agent to
choose those, the way `runner.run_battle` (and the bot's own AI) lets a player choose them: a lead
before turn 0, a pivot's replacement the instant it is forced mid-turn, a faint replacement after
the turn. These battles are played exactly that way (`differential.record_played`) and replayed
through the Rust engine (`rust_bridge.replay_played`), checking the trace and — at every decision —
that the Rust engine offered exactly the legal actions the Python did.
"""

import random
from collections.abc import Callable, Sequence
from typing import Any

import pytest

from battle_sim.differential import PlayedScenario, compare, name_action, record_played
from battle_sim.mechanics.battle import BattleState
from battle_sim.models.actions import Action, ActionType
from battle_sim.models.spec import PokemonSpec
from battle_sim.rust_bridge import LIBRARY, replay_played
from battle_sim.utils import Ability, Item

needs_bridge = pytest.mark.skipif(not LIBRARY.exists(), reason="run `cargo build --release` in rust/")


def _preferring(*names: str) -> Callable[[BattleState, int, list[Action]], Action]:
    """Pick the first offered action whose name ends with one of `names`, in preference order."""

    def pick(state: BattleState, side: int, offered: list[Action]) -> Action:
        named = {name_action(a, state, side): a for a in offered}
        for wanted in names:
            for label, action in named.items():
                if label.endswith(wanted):
                    return action
        return offered[0]

    return pick


def _played(
    team_a: Sequence[PokemonSpec],
    team_b: Sequence[PokemonSpec],
    pick: Callable[[BattleState, int, list[Action]], Action],
    orders: tuple[list[int], list[int]] | None = None,
    max_turns: int = 1,
) -> tuple[PlayedScenario, list[dict[str, Any]]]:
    orders = orders or (list(range(len(team_a))), list(range(len(team_b))))
    scenario, expected = record_played((team_a, team_b), orders, pick, seed=0, max_turns=max_turns)
    theirs, legality = replay_played(scenario)
    assert compare(expected, theirs) is None, compare(expected, theirs)
    assert all(check is None for check in legality), [c for c in legality if c]
    return scenario, expected


def _spec(species: str, nickname: str, moves: list[str], **extra: Any) -> PokemonSpec:
    return PokemonSpec(species=species, nickname=nickname, level=50, moves=moves, **extra)


@needs_bridge
def test_a_pivot_sends_in_the_pokemon_its_side_chose_mid_turn() -> None:
    """U-turn's user picks its replacement the instant it leaves — here the *last* benched Pokemon,
    not the harness's lowest-index one — and the slower foe's Tackle hits whoever arrived."""
    team_a = [
        _spec("Tauros", "A0", ["U-turn"]),
        _spec("Machamp", "A1", ["Tackle"]),
        _spec("Kangaskhan", "A2", ["Tackle"]),
    ]
    team_b = [_spec("Rhydon", "B0", ["Tackle"])]
    scenario, expected = _played(team_a, team_b, _preferring("U-turn", "Tackle", "switch:A2"))
    assert [d["kind"] for d in scenario.decisions] == ["turn", "pivot"]
    events = expected[0]["events"]
    assert {"type": "Switched", "side": 0, "withdrew": "A0", "sent_out": "A2"} in events, events
    hits = [e for e in events if e["type"] == "DamageDealt" and e["side"] == 0]
    assert hits and hits[-1]["pokemon"] == "A2", events


@needs_bridge
def test_an_eject_button_replacement_is_chosen_and_the_ejected_side_forfeits_its_move() -> None:
    team_a = [_spec("Tauros", "A0", ["Tackle"])]
    team_b = [
        _spec("Rhydon", "B0", ["Tackle"], item=Item.EJECT_BUTTON),
        _spec("Machamp", "B1", ["Tackle"]),
        _spec("Golem", "B2", ["Tackle"]),
    ]
    scenario, expected = _played(team_a, team_b, _preferring("Tackle", "switch:B2"))
    assert [d["kind"] for d in scenario.decisions] == ["turn", "pivot"]
    events = expected[0]["events"]
    assert {"type": "Switched", "side": 1, "withdrew": "B0", "sent_out": "B2"} in events, events
    assert not any(e["type"] == "MoveUsed" and e["side"] == 1 for e in events), events


@needs_bridge
def test_both_sides_replace_after_a_double_knockout() -> None:
    """Explosion takes both leads down; each side names its replacement after the turn, side 0 first."""
    team_a = [_spec("Golem", "A0", ["Explosion"]), _spec("Machamp", "A1", ["Tackle"])]
    team_b = [_spec("Tauros", "B0", ["Tackle"]), _spec("Rhydon", "B1", ["Tackle"])]
    team_b[0] = team_b[0].model_copy(update={"level": 5})
    scenario, _ = _played(team_a, team_b, _preferring("Explosion", "Tackle"))
    kinds = [(d["kind"], d.get("side")) for d in scenario.decisions]
    assert kinds == [("turn", None), ("replace", 0), ("replace", 1)], kinds


@needs_bridge
@pytest.mark.parametrize("lead", [0, 2])
def test_each_side_leads_with_the_pokemon_it_chose(lead: int) -> None:
    team = [_spec("Tauros", "P0", ["Tackle"]), _spec("Machamp", "P1", ["Tackle"]), _spec("Rhydon", "P2", ["Tackle"])]
    order = [lead, *[i for i in range(3) if i != lead]]
    _, expected = _played(team, team, _preferring("Tackle"), orders=(order, order))
    users = {e["pokemon"] for e in expected[0]["events"] if e["type"] == "MoveUsed"}
    assert users == {f"P{lead}"}, users


@needs_bridge
@pytest.mark.parametrize(
    ("lead", "free"),
    [
        (_spec("Rhydon", "B0", ["Tackle"]), False),
        (_spec("Blissey", "B0", ["Tackle"], item=Item.SHED_SHELL), True),
        (_spec("Gengar", "B0", ["Shadow Ball"]), True),
    ],
    ids=["trapped", "shed-shell", "ghost"],
)
def test_shadow_tag_traps_all_but_a_shed_shell_or_a_ghost(lead: PokemonSpec, free: bool) -> None:
    team_a = [_spec("Gothitelle", "A0", ["Psychic"], ability=Ability.SHADOW_TAG)]
    team_b = [lead, _spec("Machamp", "B1", ["Tackle"])]
    scenario, _ = _played(team_a, team_b, _preferring("Psychic", "Tackle", "Shadow Ball"))
    offered = scenario.decisions[0]["legal"][1]
    assert any(label.startswith("switch:") for label in offered) is free, offered


@needs_bridge
def test_a_choice_item_locks_the_next_turns_moves() -> None:
    team_a = [
        _spec("Tauros", "A0", ["Tackle", "Rock Slide"], item=Item.CHOICE_SCARF),
        _spec("Machamp", "A1", ["Tackle"]),
    ]
    team_b = [_spec("Blissey", "B0", ["Soft-Boiled"])]
    scenario, _ = _played(team_a, team_b, _preferring("Rock Slide", "Soft-Boiled"), max_turns=2)
    second = scenario.decisions[1]["legal"][0]
    moves = [label for label in second if not label.startswith("switch:")]
    assert moves == ["move:SECOND:Rock Slide"], second


@needs_bridge
@pytest.mark.parametrize("seed", range(10))
def test_random_mirror_battles_agree_decision_for_decision(seed: int) -> None:
    from battle_sim.ag_sets import mirror_team

    rng = random.Random(f"decisions:{seed}")
    team = mirror_team(rng)
    orders = tuple([lead, *[i for i in range(6) if i != lead]] for lead in (rng.randrange(6), rng.randrange(6)))

    def pick(state: BattleState, side: int, offered: list[Action]) -> Action:
        moves = [a for a in offered if a.action is ActionType.USE_MOVE]
        return rng.choice(moves) if moves and rng.random() < 0.8 else rng.choice(offered)

    scenario, expected = record_played((team, team), orders, pick, seed=seed, max_turns=150)  # type: ignore[arg-type]
    theirs, legality = replay_played(scenario)
    assert compare(expected, theirs) is None, compare(expected, theirs)
    assert all(check is None for check in legality), [c for c in legality if c]


@needs_bridge
@pytest.mark.parametrize(
    ("victim", "move"),
    [
        (_spec("Tauros", "B0", ["Memento"]), "Pain Split"),
        (_spec("Vaporeon", "B0", ["Memento"], ability=Ability.WATER_ABSORB), "Scald"),
    ],
    ids=["pain-split", "water-absorb"],
)
def test_a_move_aimed_at_a_fainted_pokemon_fails_and_does_not_revive_it(victim: PokemonSpec, move: str) -> None:
    """The faster side Mementos itself out; the slower one's move has no target and fails before
    anything is rolled. It used to go ahead — Pain Split averaged the fainted Pokemon's zero HP back
    up, and Water Absorb healed a fainted Vaporeon — standing a knocked-out Pokemon back up."""
    team_a = [_spec("Rhydon", "A0", [move]), _spec("Machamp", "A1", ["Tackle"])]
    team_b = [victim.model_copy(update={"level": 100}), _spec("Golem", "B1", ["Tackle"])]
    _, expected = _played(team_a, team_b, _preferring("Memento", move, "Tackle"))
    events = expected[0]["events"]
    assert events[-1] == {"type": "MoveFailed"} or {"type": "MoveFailed"} in events, events
    assert expected[0]["state"]["sides"][1]["team"][0]["fainted"], expected[0]["state"]["sides"][1]["team"][0]
    assert not any(e["type"] in ("Healed", "AbsorbHealed") and e["side"] == 1 for e in events), events


@needs_bridge
def test_future_sight_is_queued_against_a_target_immune_to_it() -> None:
    """Its effectiveness is asked when it lands, not when it is used: aimed at a Dark type it still
    goes up. The Rust engine used to stop it at the immunity gate, which the MatchupPlayer port's
    parity battles found (a Chimecho foreseeing into Liepard)."""
    team_a = [_spec("Chimecho", "A0", ["Future Sight"]), _spec("Machamp", "A1", ["Tackle"])]
    team_b = [_spec("Umbreon", "B0", ["Tackle"]), _spec("Rhydon", "B1", ["Tackle"])]
    _, expected = _played(team_a, team_b, _preferring("Future Sight", "Tackle"))
    events = expected[0]["events"]
    assert {"type": "FutureAttackQueued", "side": 0, "pokemon": "A0", "move": "Future Sight"} in events, events


@needs_bridge
@pytest.mark.parametrize(
    ("attacker", "target", "expected"),
    [
        (_spec("Breloom", "A0", ["Spore"]), _spec("Venusaur", "B0", ["Tackle"]), "NoEffect"),
        (
            _spec("Vileplume", "A0", ["Sleep Powder"]),
            _spec("Forretress", "B0", ["Tackle"], ability=Ability.OVERCOAT),
            "DoesNotAffect",
        ),
        (
            _spec("Excadrill", "A0", ["Spore"], ability=Ability.MOLD_BREAKER),
            _spec("Forretress", "B0", ["Tackle"], ability=Ability.OVERCOAT),
            "StatusInflicted",
        ),
        (_spec("Breloom", "A0", ["Spore"]), _spec("Tauros", "B0", ["Tackle"]), "StatusInflicted"),
    ],
    ids=["grass-type", "overcoat", "mold-breaker-through-overcoat", "ordinary-target"],
)
def test_powder_moves_do_not_affect_grass_types_or_overcoat(
    attacker: PokemonSpec, target: PokemonSpec, expected: str
) -> None:
    """Spore and the other powder moves have no effect on a Grass type, or on Overcoat unless a Mold
    Breaker uses them. Neither engine knew this: Spore put Grass types to sleep, which a trainer
    noticed and asked Meowfred about."""
    _, played = _played([attacker], [target], _preferring("Spore", "Sleep Powder", "Tackle"))
    events = played[0]["events"]
    assert any(event["type"] == expected for event in events), events
    if expected != "StatusInflicted":
        assert not any(event["type"] == "StatusInflicted" for event in events), events
