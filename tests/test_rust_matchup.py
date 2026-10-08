"""The Rust `MatchupPlayer` (`rust/src/matchup.rs`) scores every action exactly as the Python's does."""

import json
import random
from collections.abc import Sequence

import numpy as np
import pytest

from battle_sim.ag_sets import mirror_team
from battle_sim.differential import compare, encode_spec, name_action, record_played
from battle_sim.matchup import MatchupPlayer
from battle_sim.mechanics.battle import BattleState
from battle_sim.models.actions import Action
from battle_sim.models.spec import PokemonSpec
from battle_sim.rust_bridge import LIBRARY, RustBattle, database, load, replay_played

pytestmark = pytest.mark.skipif(not LIBRARY.exists(), reason="run `cargo build --release` in rust/")

Teams = tuple[list[PokemonSpec], list[PokemonSpec]]
Scored = list[tuple[str, list[float] | None]]


def _tournament_teams(seed: int) -> Teams:
    """Dealt as the tournament deals them, Megas and all, with Magnitude re-rolled away."""
    from battle_sim.rust_ratings import deal, playable_set, species_pool

    rng = random.Random(f"matchup:{seed}")
    pairing = next(deal(species_pool()[0], 1, rng))

    def dealt(name: str) -> PokemonSpec:
        spec = playable_set(name, rng)
        while "Magnitude" in spec.moves:
            spec = playable_set(name, rng)
        return spec

    return [dealt(n) for n in pairing.team_a], [dealt(n) for n in pairing.team_b]


def _mirror_teams(seed: int) -> Teams:
    team = list(mirror_team(random.Random(f"matchup-mirror:{seed}")))
    return team, team


def _team_json(team: Sequence[PokemonSpec]) -> str:
    return json.dumps([encode_spec(spec) for spec in team])


def _played(teams: Teams, seed: int) -> tuple[list[Scored], list[Scored]]:
    a, b = teams
    orders = (list(MatchupPlayer().choose_order(a, b)), list(MatchupPlayer().choose_order(b, a)))
    rust_orders = [load().matchup_order(database(), _team_json(own), _team_json(foe)) for own, foe in ((a, b), (b, a))]
    assert rust_orders == list(orders)

    players = (MatchupPlayer(), MatchupPlayer())
    ours: list[Scored] = []

    def pick(state: BattleState, side: int, offered: list[Action]) -> Action:
        featured = players[side].feature_actions(state, side, offered)
        ours.append([(name_action(a, state, side), None if f is None else list(f.as_vector())) for f, a in featured])
        return players[side].choose_action(state, side, offered)

    scenario, expected = record_played(teams, orders, pick, seed=seed, max_turns=200)
    theirs: list[Scored] = []

    def observe(battle: RustBattle, side: int, kind: str) -> None:
        if kind != "lead":
            theirs.append(battle.matchup_features(side))

    trace, _ = replay_played(scenario, observe=observe)
    assert compare(expected, trace) is None, compare(expected, trace)
    return ours, theirs


def _assert_same(ours: list[Scored], theirs: list[Scored]) -> None:
    assert len(ours) == len(theirs)
    for at, (python, rust) in enumerate(zip(ours, theirs, strict=True)):
        assert [name for name, _ in python] == [name for name, _ in rust], f"decision {at}: different actions"
        for (name, p), (_, r) in zip(python, rust, strict=True):
            assert (p is None) == (r is None), f"decision {at}, {name}: python {p} rust {r}"
            if p is not None and r is not None:
                np.testing.assert_allclose(r, p, rtol=1e-9, atol=1e-12, err_msg=f"decision {at}, {name}")


@pytest.mark.parametrize("seed", range(12))
def test_tournament_battles_score_the_same(seed: int) -> None:
    _assert_same(*_played(_tournament_teams(seed), seed))


@pytest.mark.parametrize("seed", range(8))
def test_mirror_battles_with_items_score_the_same(seed: int) -> None:
    _assert_same(*_played(_mirror_teams(seed), seed))


def test_the_rust_tournament_plays_whole_battles() -> None:
    from battle_sim.rust_ratings import deal, play_batch, species_pool

    pairings = list(deal(species_pool()[0], 8, random.Random(0)))
    records = play_batch(pairings, None, threads=1)
    assert len(records) == 16
    played = [r for r in records if "margin" in r]
    assert len(played) >= 14, [r.get("error") for r in records]
    assert all(0.0 <= r["margin"] <= 1.0 and r["turns"] > 0 for r in played)


def test_damage_estimates_match_on_the_quirks_random_battles_rarely_reach() -> None:
    """`damage_range` for every move against every foe, on hand-picked quirk cases."""
    from battle_sim.analysis import damage_range
    from battle_sim.maths.rng import RNG
    from battle_sim.mechanics.battle import SideState
    from battle_sim.teams import build_pokemon
    from battle_sim.utils import Ability, Item

    def spec(species: str, moves: list[str], ability: Ability = Ability.NONE, item: Item = Item.NONE) -> PokemonSpec:
        return PokemonSpec(species=species, level=50, moves=moves, ability=ability, item=item)

    ours = [
        spec("Tauros", ["Body Slam"]),
        spec("Butterfree", ["Bug Buzz", "Air Slash"], ability=Ability.TINTED_LENS, item=Item.LIFE_ORB),
        spec("Weavile", ["Beat Up", "Ice Shard"]),
        spec("Salamence", ["Dragon Claw"], item=Item.LIFE_ORB),
    ]
    theirs = [
        spec("Charizard", ["Flamethrower"]),
        spec("Dragonite", ["Dragon Claw"], ability=Ability.MULTISCALE),
        spec("Snorlax", ["Body Slam"]),
        spec("Machamp", ["Cross Chop"]),
    ]
    teams = ([build_pokemon(s) for s in ours], [build_pokemon(s) for s in theirs])
    state = BattleState(sides=(SideState(team=teams[0]), SideState(team=teams[1])), rng=RNG(seed=0))
    rust = load().matchup_damage(database(), _team_json(ours), _team_json(theirs))
    assert rust, "nothing was estimated"
    for side, index, name, foe, low, high in rust:
        attacker = teams[side][index]
        move = next(m for m in attacker.moves if m is not None and m.name == name)
        expected = damage_range(move, attacker, teams[1 - side][foe], state)
        assert (low, high) == expected, f"{attacker.name} {name} -> {teams[1 - side][foe].name}"
