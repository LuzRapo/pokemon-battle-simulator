from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Protocol, runtime_checkable

from battle_sim.engine import apply_forced_switch, legal_actions, step
from battle_sim.maths.rng import RNG
from battle_sim.mechanics.battle import BattleState, SideState
from battle_sim.mechanics.log import BattleLog
from battle_sim.models.actions import Action
from battle_sim.models.spec import PokemonSpec
from battle_sim.observation import BattleObserver, BeliefSampler, SetPrior
from battle_sim.teams import build_pokemon
from battle_sim.utils import Outcome


class Player(Protocol):
    def choose_order(self, own: Sequence[PokemonSpec], opponent: Sequence[PokemonSpec]) -> Sequence[int]:
        """Team preview: return a permutation of own-team indices; index 0 leads.

        `opponent` carries believed sets — the prior's best guess per previewed species —
        not the true ones.
        """
        ...

    def choose_action(self, state: BattleState, side_index: int, actions: Sequence[Action]) -> Action:
        """Pick one of the offered legal actions, reading the battle as this side sees it."""
        ...


@runtime_checkable
class Determinizing(Protocol):
    """A player that hedges over sampled belief-consistent worlds when given a sampler."""

    def bind_belief_sampler(self, sampler: BeliefSampler) -> None: ...


@dataclass(frozen=True)
class BattleResult:
    outcome: Outcome | None  # None: the turn cap was reached
    turns: int
    survivors: tuple[int, int]  # unfainted pokemon per side at the end
    logs: tuple[BattleLog, ...]  # one per step/forced switch, in order


def run_battle(
    team_a: Sequence[PokemonSpec],
    team_b: Sequence[PokemonSpec],
    player_a: Player,
    player_b: Player,
    seed: int = 0,
    max_turns: int = 1000,
    prior: SetPrior | None = None,
    belief_candidates: tuple[int, int] | None = None,
) -> BattleResult:
    metagame = prior if prior is not None else SetPrior.from_teams([team_a, team_b])
    preview_of_b = [metagame.preview(spec.species, spec.level) for spec in team_b]
    preview_of_a = [metagame.preview(spec.species, spec.level) for spec in team_a]
    sides = (
        build_side(team_a, player_a.choose_order(team_a, preview_of_b)),
        build_side(team_b, player_b.choose_order(team_b, preview_of_a)),
    )
    state = BattleState(sides=sides, rng=RNG(seed=seed))
    players = (player_a, player_b)
    observer = BattleObserver(state, metagame, belief_candidates)
    for i, player in enumerate(players):
        if isinstance(player, Determinizing):
            player.bind_belief_sampler(partial(observer.sample_view, i))

    def switch_chooser(live: BattleState, side_index: int) -> Action:
        # Both sides are bots and always have an answer ready — a pivot switch resolves the instant
        # it's forced, exactly as the games do, rather than waiting for the rest of the turn. Reads
        # through the belief view, same as every other decision this side makes.
        view = observer.view(side_index)
        return players[side_index].choose_action(view, side_index, legal_actions(live, side_index))

    logs: list[BattleLog] = []
    while state.outcome is None and state.turn < max_turns:
        actions = {i: players[i].choose_action(observer.view(i), i, legal_actions(state, i)) for i in (0, 1)}
        logs.append(step(state, actions, switch_chooser=switch_chooser))
        observer.ingest(logs[-1])
        for i in (0, 1):
            side = state.sides[i]
            # Replacements repeat until one survives its entry (hazards can faint them).
            while state.outcome is None and (side.active_pokemon.is_fainted() or side.needs_switch):
                replacement = players[i].choose_action(observer.view(i), i, legal_actions(state, i))
                logs.append(apply_forced_switch(state, i, replacement))
                observer.ingest(logs[-1])
    survivors = tuple(sum(1 for p in side.team if not p.is_fainted()) for side in state.sides)
    return BattleResult(
        outcome=state.outcome, turns=state.turn, survivors=(survivors[0], survivors[1]), logs=tuple(logs)
    )


def build_side(team: Sequence[PokemonSpec], order: Sequence[int]) -> SideState:
    if sorted(order) != list(range(len(team))):
        raise ValueError(f"Order must be a permutation of 0..{len(team) - 1}, got {list(order)}.")
    return SideState(team=[build_pokemon(team[i]) for i in order])
