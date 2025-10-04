from loguru import logger
from pydantic import BaseModel, Field, model_validator

from battle_sim.maths.stats import calculate_effective_stat, calculate_total_hp, calculate_total_stat
from battle_sim.models.moves import Move, MoveSet, MoveSlot
from battle_sim.models.stats import BaseStats, EVs, IVs, LiveStats, StatStages, StatTotals
from battle_sim.models.type_matchups import TypePair
from battle_sim.utils import Nature, Stats


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

    # TODO: item
    # TODO: ability
    # TODO: status conditions

    live_stats: LiveStats = Field(default_factory=lambda: LiveStats())
    stat_stages: StatStages = Field(default_factory=lambda: StatStages())

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
            ATTACK=calculate_total_stat(bs.ATTACK, ivs.ATTACK, evs.ATTACK, lvl, nat, Stats.ATTACK),
            DEFENCE=calculate_total_stat(bs.DEFENCE, ivs.DEFENCE, evs.DEFENCE, lvl, nat, Stats.DEFENCE),
            SP_ATTACK=calculate_total_stat(bs.SP_ATTACK, ivs.SP_ATTACK, evs.SP_ATTACK, lvl, nat, Stats.SP_ATTACK),
            SP_DEFENCE=calculate_total_stat(bs.SP_DEFENCE, ivs.SP_DEFENCE, evs.SP_DEFENCE, lvl, nat, Stats.SP_DEFENCE),
            SPEED=calculate_total_stat(bs.SPEED, ivs.SPEED, evs.SPEED, lvl, nat, Stats.SPEED),
        )

    @model_validator(mode="after")
    def _init_live_stats(self):
        """Populate live_stats using stat_totals when initialised."""
        totals = self.stat_totals
        self.live_stats = LiveStats(
            HP=totals.HP,
            ATTACK=totals.ATTACK,
            DEFENCE=totals.DEFENCE,
            SP_ATTACK=totals.SP_ATTACK,
            SP_DEFENCE=totals.SP_DEFENCE,
            SPEED=totals.SPEED,
        )
        return self

    def reset_live_stats(self) -> None:
        totals = self.stat_totals
        self.live_stats.HP = totals.HP
        self.live_stats.ATTACK = totals.ATTACK
        self.live_stats.DEFENCE = totals.DEFENCE
        self.live_stats.SP_ATTACK = totals.SP_ATTACK
        self.live_stats.SP_DEFENCE = totals.SP_DEFENCE
        self.live_stats.SPEED = totals.SPEED

    def _adjust_hp(self, amount: int) -> int:
        old_hp = self.live_stats.HP
        max_hp = self.stat_totals.HP
        new_hp = max(0, min(max_hp, old_hp + amount))
        self.live_stats.HP = new_hp
        return new_hp - old_hp

    def apply_damage(self, amount: int) -> int:
        return -self._adjust_hp(-abs(amount))

    def apply_healing(self, amount: int) -> int:
        return self._adjust_hp(abs(amount))

    def change_stat_stage(self, stat: Stats, stages: int) -> int:
        old_stage = getattr(self.stat_stages, stat.name)
        new_stage = max(-6, min(6, old_stage + stages))
        setattr(self.stat_stages, stat.name, new_stage)
        return new_stage - old_stage

    def reset_stat_stages(self) -> None:
        self.stat_stages = StatStages()

    def effective_stat(self, stat: Stats) -> int:
        assert stat in (Stats.ATTACK, Stats.DEFENCE, Stats.SP_ATTACK, Stats.SP_DEFENCE, Stats.SPEED)
        return calculate_effective_stat(self.stat_totals, self.stat_stages, stat)

    def is_fainted(self) -> bool:
        return self.live_stats.HP <= 0

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
