from loguru import logger
from pydantic import BaseModel, Field

from battle_sim.maths.stats import calculate_total_hp, calculate_total_stat
from battle_sim.models.moves import Move, MoveSet, MoveSlot
from battle_sim.models.stats import BaseStats, EVs, IVs, StatTotals
from battle_sim.models.type_matchups import TypePair
from battle_sim.utils import Nature


class Pokemon(BaseModel):
    name: str
    nickname: str
    level: int = Field(default=50, ge=1, le=100)
    nature: Nature
    effort_values: EVs
    individual_values: IVs
    base_stats: BaseStats
    types: TypePair
    moves: MoveSet
    # items, abilities, etc. can be added later

    @property
    def stat_totals(self) -> StatTotals:
        """Calculate final stats from base stats, IVs, EVs, level, and nature."""
        lvl = self.level
        nat = self.nature
        ivs = self.individual_values
        evs = self.effort_values
        bs = self.base_stats

        return StatTotals(
            HP=calculate_total_hp(bs.HP, ivs.HP, evs.HP, lvl),
            ATTACK=calculate_total_stat(bs.ATTACK, ivs.ATTACK, evs.ATTACK, lvl, nat, "ATTACK"),
            DEFENCE=calculate_total_stat(bs.DEFENCE, ivs.DEFENCE, evs.DEFENCE, lvl, nat, "DEFENCE"),
            SP_ATTACK=calculate_total_stat(bs.SP_ATTACK, ivs.SP_ATTACK, evs.SP_ATTACK, lvl, nat, "SP_ATTACK"),
            SP_DEFENCE=calculate_total_stat(bs.SP_DEFENCE, ivs.SP_DEFENCE, evs.SP_DEFENCE, lvl, nat, "SP_DEFENCE"),
            SPEED=calculate_total_stat(bs.SPEED, ivs.SPEED, evs.SPEED, lvl, nat, "SPEED"),
        )

    def known_moves(self) -> list[Move]:
        return self.moves.to_list()

    def has_move(self, move: Move) -> bool:
        return self.moves.contains(move)

    def learn_move(self, new_move: Move, move_slot: MoveSlot) -> None:
        """Learn a move. If not full, fills first empty slot; otherwise overwrites move_slot."""
        if self.has_move(new_move):
            return
        self.moves.learn_move(new_move, move_slot)
        logger.info(f"{self.nickname} has learned the move {new_move.name}.")

    def forget_move(self, move_slot: MoveSlot) -> None:
        """Forget a move and shift later moves left, leaving the last slot empty."""
        old_move = self.moves[move_slot]
        self.moves.forget_move(move_slot)
        if old_move is not None:
            logger.info(f"{self.nickname} has forgotten the move {old_move.name}.")
