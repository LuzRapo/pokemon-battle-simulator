"""Transform / Imposter: copy the target's battle form; everything restores on switch-out.

The copy covers base stats (HP excepted), nature/EVs/IVs (so computed totals match the
target's exactly), types, ability (re-wired), moves (5 PP each), and stat stages.
"""

from battle_sim.mechanics.battle import BattleState
from battle_sim.mechanics.effects import register_active, unregister_active
from battle_sim.mechanics.log import BattleLog
from battle_sim.models.log_events import MoveFailed, Transformed
from battle_sim.models.moves import MoveSet, MoveSlot
from battle_sim.models.pokemon import FormSnapshot, Pokemon
from battle_sim.models.stats import BaseStats

_TRANSFORM_PP = 5


def transform_into(
    attacker: Pokemon, attacker_side_index: int, defender: Pokemon, state: BattleState, log: BattleLog
) -> None:
    if id(attacker) in state.transforms or id(defender) in state.transforms or defender.is_fainted():
        log.add(MoveFailed())
        return
    state.transforms[id(attacker)] = FormSnapshot(
        base_stats=attacker.base_stats,
        nature=attacker.nature,
        effort_values=attacker.effort_values,
        individual_values=attacker.individual_values,
        types=attacker.types,
        ability=attacker.ability,
        moves=attacker.moves,
        pp=dict(attacker.pp),
    )
    unregister_active(state.bus, state.effects, attacker)
    attacker.base_stats = BaseStats(
        HP=attacker.base_stats.HP,  # HP stays the user's own
        ATTACK=defender.base_stats.ATTACK,
        DEFENCE=defender.base_stats.DEFENCE,
        SP_ATTACK=defender.base_stats.SP_ATTACK,
        SP_DEFENCE=defender.base_stats.SP_DEFENCE,
        SPEED=defender.base_stats.SPEED,
    )
    attacker.nature = defender.nature
    attacker.effort_values = defender.effort_values
    attacker.individual_values = defender.individual_values
    attacker.types = defender.types
    attacker.ability = defender.ability
    moves = defender.moves
    attacker.moves = MoveSet(moves.move_one, moves.move_two, moves.move_three, moves.move_four)
    attacker.pp = {slot: _TRANSFORM_PP for slot in MoveSlot if attacker.moves[slot] is not None}
    attacker.stat_stages = defender.stat_stages.model_copy()
    attacker.refresh_stats()
    register_active(state.bus, state.effects, attacker)
    log.add(Transformed(side=attacker_side_index, pokemon=attacker.nickname, into=defender.nickname))


def restore_form(pokemon: Pokemon, state: BattleState) -> None:
    """Called on every switch-out; a no-op unless the pokemon is transformed."""
    snapshot = state.transforms.pop(id(pokemon), None)
    if snapshot is None:
        return
    pokemon.base_stats = snapshot.base_stats
    pokemon.nature = snapshot.nature
    pokemon.effort_values = snapshot.effort_values
    pokemon.individual_values = snapshot.individual_values
    pokemon.types = snapshot.types
    pokemon.ability = snapshot.ability
    pokemon.moves = snapshot.moves
    pokemon.pp = dict(snapshot.pp)
    pokemon.refresh_stats()
