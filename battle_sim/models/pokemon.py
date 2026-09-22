from dataclasses import dataclass
from functools import cached_property

from loguru import logger
from pydantic import BaseModel, Field, model_validator

from battle_sim.maths.stats import calculate_effective_stat, calculate_total_hp, calculate_total_stat
from battle_sim.models.moves import Move, MoveSet, MoveSlot
from battle_sim.models.stats import BaseStats, EVs, IVs, LiveStats, StatStages, StatTotals
from battle_sim.models.type_matchups import TypePair
from battle_sim.utils import Ability, Category, ExtraStatus, Item, Nature, Stats, Status, Type

NINE_LIVES = 9  # how many times the butler gets up again
# Every move the butler has, however short its list says it is. Nine lives is a long fight by design,
# and his real set runs 24/16/56/16 — against Pressure, which doubles every cost, that is eight uses
# of Warm Dinner. Stalling him into Struggle and letting the recoil take the lives one by one is not
# beating him so much as waiting him out, and it was the cheapest line against him on the board.
BUTLERS_PP = 99


@dataclass(frozen=True)
class FormSnapshot:
    """A pokemon's pre-Transform form, restored on switch-out (see engine/transform.py)."""

    base_stats: BaseStats
    nature: Nature
    effort_values: EVs
    individual_values: IVs
    types: TypePair
    ability: Ability
    moves: MoveSet
    pp: dict[MoveSlot, int]


@dataclass(frozen=True)
class BelievedSet:
    """One candidate set an opponent's pokemon might be running, with its posterior weight.

    Threat estimation over a *believed* attacker has to average the damage each candidate set
    could deal, not take the best move of a set nobody holds: the union of a species' role
    movepool credits coverage no single four-move set owns (see `analysis.posterior_threat`).
    """

    weight: float
    moves: tuple[Move, ...]
    item: Item
    ability: Ability


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
    pp_ups: int = 0  # 0-3 PP Ups; each is +1/5 of the move's listed PP, as in the games
    # Nine Lives. A plain field rather than a volatile: `switching.volatiles.clear()` would hand a
    # free reset, and the search's cloned futures would each decrement a shared counter -- which is
    # exactly what happened, and produced a life count that repeated and ran backwards (8 8 7 7 6 5
    # 4 3 6) and let one battle spend thirteen of nine.
    lives_used: int = 0
    just_revived: bool = False  # set for the log to pick up, cleared once it has
    # The ninth life spent and a tenth blow landed: he stands at 1 HP instead of falling, once, and
    # yields. Set here because this is where every kind of damage passes; the engine reads it to
    # sweep the field and end the battle (see `damage_apply._log_revival`).
    made_last_stand: bool = False
    # Set the instant the stand is made, cleared once the log has said so — `just_revived`'s twin,
    # and needed for the same reason. `made_last_stand` is a permanent fact ("he has had his one"),
    # so a log reading it directly announced a fresh concession on every hit that followed, and the
    # butler's desperate form — which starts with that fact already true — conceded the second
    # anybody scratched it.
    just_stood: bool = False
    # What an opponent took off him, held until a life is spent and it comes back with him. Nine
    # Lives restores everything else about him -- full HP, status burned away -- so an item knocked
    # off on the way down was the one lasting wound in a fight that is meant to have none, and a
    # single Knock Off early on disarmed every one of the nine lives that followed.
    stripped_item: Item = Item.NONE
    restored_item: Item = Item.NONE  # set alongside `just_revived`; the log and the rewire read it
    # An item Tricked onto him. Held separately from `stripped_item` because a trade has two
    # halves: what he lost, and what he was left holding in its place — and the second half is
    # what he eats, and what goes back across the field when he rises.
    tricked_item: Item = Item.NONE
    owed_item: Item = Item.NONE  # what to hand back on revival; read and cleared by `_log_revival`

    item: Item = Item.NONE
    ability: Ability = Ability.NONE
    weight_kg: float = Field(default=100.0, gt=0)
    flash_fire_active: bool = False
    item_consumed: bool = False  # distinguishes "used up" (Unburden) from "never held"
    last_consumed_item: Item = Item.NONE  # what Harvest can regrow
    fully_evolved: bool = True  # Eviolite's defence boost applies only to not-fully-evolved holders
    paradox_boost: Stats | None = None  # Protosynthesis/Quark Drive: the boosted stat while active
    paradox_from_booster: bool = False  # a Booster Energy activation outlasts the weather/terrain
    switch_in_boost_used: bool = False  # Dauntless Shield / Intrepid Sword fire once per battle
    times_hit: int = Field(default=0, ge=0)  # lifetime hits taken (Rage Fist)
    last_hit_taken: int = Field(default=0, ge=0)  # damage from the most recent hit this turn (Counter family)
    last_hit_category: Category | None = None
    # Whether this one's own strength can be turned back on it: Counter and Mirror Coat fail against
    # a Pokemon that sets this. Nothing in the games does, and neither does anything in the bot bar
    # the butler's desperate form, which is a single Pokemon with one HP bar and no lives left —
    # the most Counter-vulnerable thing that can exist, and a ceremony rather than a ladder match.
    counter_proof: bool = False
    eject_pending: bool = False  # an Eject Pack waits for the action to resolve before pulling the holder
    flee_pending: bool = False  # Emergency Exit / Wimp Out, waiting on the action the same way
    turns_active: int = Field(default=0, ge=0)  # whole turns since this stint's switch-in (Fake Out)
    # True through the end of the turn this stint's switch-in happened (lead or mid-battle), then
    # cleared for good — unlike turns_active, this doesn't wait for the Pokemon to also act on its
    # own. Speed Boost's rule is "every turn-end except the one it entered on", not Fake Out's
    # "my first turn actually acting", and the two diverge whenever a switch spends the whole turn.
    just_switched_in: bool = True
    believed_sets: tuple[BelievedSet, ...] | None = None  # None: this side's set is known, not believed

    live_stats: LiveStats = Field(default_factory=lambda: LiveStats())
    stat_stages: StatStages = Field(default_factory=lambda: StatStages())
    # The lowest a named stage may ever go, empty for everything in the game. Where it is set it
    # makes a boost part of what the Pokemon *is* rather than something it did: Haze, Clear Smog,
    # Intimidate, a switch-out and a Sticky Web all take that stage down to its floor and no
    # further. Only stats listed here are held up, so a form that is locked at +3 Attack can still
    # be Sand-Attacked. Only the butler's desperate form sets it (see `last_stand_ui._past_caring`
    # in the bot), and only because a ceremony that ends in his execution should not be winnable by
    # opening with Haze.
    stage_floor: dict[Stats, int] = Field(default_factory=dict)
    status: Status = Status.NONE
    status_turns: int = Field(default=0, ge=0)
    volatiles: dict[ExtraStatus, int] = Field(default_factory=dict)
    choice_locked_move: MoveSlot | None = None
    protect_streak: int = Field(default=0, ge=0)  # consecutive successful Protect-likes; failure odds scale 3^n
    pp: dict[MoveSlot, int] = Field(default_factory=dict)
    last_move_slot: MoveSlot | None = None
    encored_slot: MoveSlot | None = None
    disabled_slot: MoveSlot | None = None
    locked_slot: MoveSlot | None = None  # Outrage-style rampage
    charging_slot: MoveSlot | None = None  # two-turn move committed last turn
    rolling_hits: int = Field(default=0, ge=0)  # consecutive Rollout/Ice Ball connections; each doubles the power

    @cached_property
    def stat_totals(self) -> StatTotals:
        """Final stats from base stats, IVs, EVs, level, and nature.

        Cached: the inputs only change via Transform / forme changes, which call refresh_stats().
        """
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
    def _init_pp(self) -> "Pokemon":
        if not self.pp:
            bonus = 1 + self.pp_ups / 5
            self.pp = {slot: int(move.pp * bonus) for slot in MoveSlot if (move := self.moves[slot]) is not None}
            if self.ability is Ability.NINE_LIVES:
                self.pp = dict.fromkeys(self.pp, BUTLERS_PP)
        return self

    @model_validator(mode="after")
    def _init_live_stats(self) -> "Pokemon":
        """Populate live_stats from stat_totals iff still at construction-time defaults."""
        if self.live_stats != LiveStats():
            return self
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

    def refresh_stats(self) -> None:
        """Drop the cached totals after base stats / nature / EVs / IVs changed."""
        self.__dict__.pop("stat_totals", None)

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
        if new_hp == 0 and self._revive():
            return self.live_stats.HP - old_hp
        return new_hp - old_hp

    def _revive(self) -> bool:
        """Nine Lives: rather than faint, get up whole. True if a life was spent.

        Placed here, at the one point every kind of damage passes through, because the engine checks
        for a faint in half a dozen places and only one of them emits `ON_FAINT` -- an ability bound
        to that event would have let poison, hazards, recoil and confusion kill him outright while
        eight lives went unused.
        """
        from battle_sim.utils import Ability, Status

        if self.ability is not Ability.NINE_LIVES:
            return False
        if self.lives_used >= NINE_LIVES:
            return self._last_stand()
        self.lives_used += 1
        self.live_stats.HP = self.stat_totals.HP
        self.status = Status.NONE
        self.status_turns = 0
        self.just_revived = True
        # He comes back holding what was taken from him. Only what an opponent stripped: an item he
        # spent himself stays spent, or a Sash behind nine lives would be nine free survivals.
        #
        # A Trick is taken back too, and that one is a trade rather than a theft — so whatever he was
        # left holding goes back across the field with it. `owed_item` carries that half; if he has
        # already eaten it there is nothing to give, and whoever tricked him simply loses both.
        if self.stripped_item is not Item.NONE:
            self.owed_item = self.item
            self.item = self.stripped_item
            self.restored_item = self.stripped_item
            self.stripped_item = Item.NONE
            self.tricked_item = Item.NONE
            self.item_consumed = False
        self.clear_stat_drops()
        return True

    def _last_stand(self) -> bool:
        """Out of lives and still standing, once. True if this blow was the one he survived.

        Nine lives spent is not nine deaths and a tenth: the tenth blow finds him with nothing left
        to spend, and he takes it on his feet. He keeps exactly one hit point, sheds whatever was
        wearing him down, and stops fighting — the engine turns the rest of it (the weather, the
        hazards, the battle itself) into the concession it is.

        Once per Pokemon, because a butler who cannot be killed twice cannot be killed at all.
        """
        from battle_sim.utils import Status

        if self.made_last_stand:
            return False
        self.made_last_stand = True
        self.just_stood = True
        self.live_stats.HP = 1
        self.status = Status.NONE
        self.status_turns = 0
        self.volatiles.clear()
        self.clear_stat_drops()
        return True

    def clear_stat_drops(self) -> None:
        """Every stage an opponent pushed below zero, back to zero. Boosts of his own are left alone.

        The same principle as the status and the item: rising whole undoes what was done to him, and
        an Intimidate landing early otherwise followed him through all nine lives — 0.67x Attack on
        a Pokemon that is supposed to come back untouched, with nothing in the log to explain it.

        Only the negative half, though. A Swords Dance he earned is not a wound, and wiping it would
        make each life a punishment for having set up before losing one.
        """
        for name in StatStages.model_fields:  # the seven that have stages; `Stats` also carries HP
            if getattr(self.stat_stages, name) < 0:
                setattr(self.stat_stages, name, 0)

    def apply_damage(self, amount: int) -> int:
        return -self._adjust_hp(-abs(amount))

    def apply_healing(self, amount: int) -> int:
        return self._adjust_hp(abs(amount))

    def consume_item(self) -> None:
        self.last_consumed_item = self.item
        self.item = Item.NONE
        self.item_consumed = True

    def change_stat_stage(self, stat: Stats, stages: int) -> int:
        old_stage = self.stat_stages[stat]
        new_stage = max(self.stage_floor.get(stat, -6), min(6, old_stage + stages))
        self.stat_stages[stat] = new_stage
        return new_stage - old_stage

    def reset_stat_stages(self) -> None:
        """Back to nothing — or back to the floor, for the stages that have one."""
        self.stat_stages = StatStages()
        for stat, floor in self.stage_floor.items():
            self.stat_stages[stat] = floor

    def effective_stat(self, stat: Stats) -> int:
        assert stat in (Stats.ATTACK, Stats.DEFENCE, Stats.SP_ATTACK, Stats.SP_DEFENCE, Stats.SPEED)
        return calculate_effective_stat(self.stat_totals, self.stat_stages, stat)

    def is_fainted(self) -> bool:
        return self.live_stats.HP <= 0

    @property
    def battle_types(self) -> "TypePair":
        """What this Pokemon counts as right now, which is not always what it is.

        Roost puts a bird on the ground for the rest of the turn: its Flying type is ignored, so
        Earthquake hits it and Ice Beam stops being four times effective. Read rather than written,
        because a Pokemon whose `types` were edited in place and then switched out, fainted or
        changed forme mid-turn would carry the edit for the rest of the battle.

        A pure Flying-type has nothing left to be, and the engine has no way to say "typeless", so
        it keeps its typing. That is Tornadus alone, and the alternative is a crash.
        """
        if ExtraStatus.ROOSTED not in self.volatiles or Type.FLYING not in self.types:
            return self.types
        remaining = [kind for kind in self.types if kind is not None and kind is not Type.FLYING]
        return (remaining[0], None) if remaining else self.types

    def is_grounded(self) -> bool:
        """Intrinsic grounding only (types, item, ability); field effects like Gravity would need field state."""
        return (
            Type.FLYING not in self.battle_types
            and self.item is not Item.AIR_BALLOON
            and self.ability is not Ability.LEVITATE
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
