from collections.abc import Callable, Sequence
from enum import Enum, auto
from functools import partial
from pathlib import Path

from battle_sim.engine import apply_forced_switch, legal_actions, step
from battle_sim.evolution import Team
from battle_sim.maths.rng import RNG
from battle_sim.mechanics.battle import BattleState
from battle_sim.mechanics.log import BattleLog, render_text
from battle_sim.models.actions import Action
from battle_sim.observation import BattleObserver, SetPrior
from battle_sim.platform_log import Origin, log_battle, log_matchup
from battle_sim.runner import Determinizing, Player, build_side
from battle_sim.teams import build_pokemon, export_to_showdown

HUMAN = 0
AI = 1


class Phase(Enum):
    CHOOSE = auto()  # waiting for the human's turn action
    REPLACE = auto()  # waiting for the human's forced replacement
    OVER = auto()


class InteractiveBattle:
    def __init__(
        self,
        human_team: Team,
        ai_team: Team,
        ai: Player,
        prior: SetPrior,
        seed: int,
        human_order: Sequence[int],
        record: Path | None = None,
        trainer: str = "human",
    ):
        """`record` is the platform log both teams are written to before the first turn.

        Before, because a battle abandoned halfway still tells us what a person chose to bring,
        and that is the half of the record a set prior is built from.
        """
        if record is not None:
            log_matchup(record, trainer, "boss", (tuple(human_team), tuple(ai_team)), (Origin.HUMAN, Origin.GENERATED))
        self._record = record
        self._trainer = trainer
        self._seed = seed
        preview_of_human = [prior.preview(spec.species, spec.level) for spec in human_team]
        ai_order = ai.choose_order(ai_team, preview_of_human)
        self.state = BattleState(
            sides=(build_side(human_team, human_order), build_side(ai_team, ai_order)),
            rng=RNG(seed=seed),
        )
        self.observer = BattleObserver(self.state, prior)
        if isinstance(ai, Determinizing):
            ai.bind_belief_sampler(partial(self.observer.sample_view, AI))
        self._ai = ai
        self.logs: list[BattleLog] = []
        self._turns: list[int] = []
        # Team sheets rendered now, from fresh builds: live pokemon shed consumed/knocked items.
        self._sheets = tuple(
            export_to_showdown([build_pokemon(spec) for spec in team]) for team in (human_team, ai_team)
        )

    @property
    def phase(self) -> Phase:
        if self.state.outcome is not None:
            return Phase.OVER
        side = self.state.sides[HUMAN]
        if side.active_pokemon.is_fainted() or side.needs_switch:
            return Phase.REPLACE
        return Phase.CHOOSE

    def view(self) -> BattleState:
        """The battle as the human sees it: own side true, the AI's side believed."""
        return self.observer.view(HUMAN)

    def options(self) -> list[Action]:
        return legal_actions(self.state, HUMAN)

    def submit(self, action: Action, pivot_reply: Action | None = None) -> None:
        """Advance the battle with the human's action for the current phase.

        `pivot_reply` is who should come in if `action` turns out to force a same-turn switch (a
        self-switch move connecting) — decided up front, the same way the AI always answers, since
        nobody can be asked mid-resolution. A fast pivot now protects its user exactly like the AI's
        always has, provided the caller collected a reply; without one (or outside Phase.CHOOSE, or
        a switch forced by something other than this move, e.g. Eject Button) it is left for
        `Phase.REPLACE`, same as before this existed.
        """
        phase = self.phase
        if phase is Phase.OVER:
            raise ValueError("The battle is over; nothing left to submit.")
        if phase is Phase.CHOOSE:
            ai_action = self._ai.choose_action(self.observer.view(AI), AI, legal_actions(self.state, AI))
            chooser = self._switch_chooser(pivot_reply)
            self._log(step(self.state, {HUMAN: action, AI: ai_action}, switch_chooser=chooser))
        else:
            self._log(apply_forced_switch(self.state, HUMAN, action))
        self._resolve_ai_replacements()
        if self._record is not None and self.state.outcome is not None:
            log_battle(self._record, self._trainer, "boss", self._seed, self.state.outcome, self.state.turn)

    def _switch_chooser(self, pivot_reply: Action | None) -> Callable[[BattleState, int], Action | None]:
        """The AI answers a mid-turn pivot switch on the spot; the human's own reply is whatever
        `submit` was given for this action, decided before either side's move was known."""

        def chooser(state: BattleState, side_index: int) -> Action | None:
            if side_index == AI:
                return self._ai.choose_action(self.observer.view(AI), AI, legal_actions(state, AI))
            return pivot_reply

        return chooser

    def _resolve_ai_replacements(self) -> None:
        """The runner's replacement loop, but only for the AI; the human's wait for the interface.

        The human replaces first, exactly as the runner resolves side 0 before side 1.
        """
        while self.state.outcome is None and self.phase is not Phase.REPLACE:
            side = self.state.sides[AI]
            if not (side.active_pokemon.is_fainted() or side.needs_switch):
                return
            replacement = self._ai.choose_action(self.observer.view(AI), AI, legal_actions(self.state, AI))
            self._log(apply_forced_switch(self.state, AI, replacement))

    def _log(self, log: BattleLog) -> None:
        self.logs.append(log)
        self._turns.append(self.state.turn)
        self.observer.ingest(log)

    def transcript(self, header: str = "") -> str:
        """The whole battle as text: optional caller header, both original teams, every log line.

        True sets are fine to include — a transcript is read after the match, when reveals
        no longer matter. Turn markers match the interface's post-step turn counter;
        replacements stay under the turn that forced them.
        """
        sections = [header] if header else []
        for name, sheet in zip(("== Human team ==", "== Boss team =="), self._sheets, strict=True):
            sections.append(name + "\n" + sheet.strip())
        lines: list[str] = []
        last_turn = None
        for turn, log in zip(self._turns, self.logs, strict=True):
            if turn != last_turn:
                lines.append(f"-- turn {turn} --")
                last_turn = turn
            lines.extend(render_text(entry) for entry in log)
        sections.append("== Battle ==\n" + "\n".join(lines))
        if self.state.outcome is not None:
            sections.append(f"== Result: {self.state.outcome.name} ==")
        return "\n\n".join(sections) + "\n"
