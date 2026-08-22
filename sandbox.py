import streamlit as st

from battle_sim.engine import apply_forced_switch, legal_actions, step
from battle_sim.maths.rng import RNG
from battle_sim.mechanics.battle import BattleState, SideState
from battle_sim.models.actions import Action
from battle_sim.teams import build_pokemon, export_to_showdown, parse_showdown_team
from battle_sim.utils import Outcome
from ui_shared import action_label, render_battle_state, render_field, render_log

P1_SAMPLE_TEAM = """Birdy (Staraptor) @ Choice Band
Ability: Intimidate
EVs: 252 Atk / 4 Def / 252 Spe
Adamant Nature
- Brave Bird
- Return
- Close Combat
- U-turn

Rocky (Golem) @ Chople Berry
Ability: Rock Head
EVs: 4 HP / 252 Atk / 252 Spe
Adamant Nature
- Earthquake
- Rock Blast
- Sucker Punch
- Stealth Rock

Spooky (Gengar) @ Black Sludge
Ability: Levitate
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
- Shadow Ball
- Focus Blast
- Will-O-Wisp
- Taunt

Punchy (Machamp) @ Focus Sash
Ability: No Guard
EVs: 4 HP / 252 Atk / 252 Spe
Adamant Nature
- Dynamic Punch
- Payback
- Bullet Punch
- Stone Edge

Zappy (Electivire) @ Expert Belt
Ability: Motor Drive
EVs: 252 Atk / 252 SpA / 4 Spe
Lonely Nature
- Thunderbolt
- Ice Punch
- Cross Chop
- Flamethrower

Speedy (Deoxys-Attack) @ Life Orb
Ability: Pressure
EVs: 4 Atk / 252 SpA / 252 Spe
Rash Nature
- Ice Beam
- Thunderbolt
- Superpower
- Extreme Speed
"""

P2_SAMPLE_TEAM = """Sandstorm (Garchomp) @ Rocky Helmet
Ability: Rough Skin
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Earthquake
- Stone Edge
- Dragon Claw
- Fire Fang

Inferno (Charizard) @ Choice Specs
Ability: Solar Power
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
- Flamethrower
- Air Slash
- Focus Blast
- Dragon Pulse

Boulder (Tyranitar) @ Assault Vest
Ability: Sand Stream
EVs: 252 HP / 252 Atk / 4 SpD
Adamant Nature
- Stone Edge
- Earthquake
- Crunch
- Ice Punch

Aerial (Salamence) @ Life Orb
Ability: Moxie
EVs: 4 HP / 252 Atk / 252 Spe
Jolly Nature
- Dragon Claw
- Earthquake
- Stone Edge
- Flamethrower

Steel (Metagross) @ Iron Plate
Ability: Clear Body
EVs: 252 HP / 252 Atk / 4 SpD
Adamant Nature
- Meteor Mash
- Earthquake
- Bullet Punch
- Ice Punch

Wave (Gyarados) @ Choice Band
Ability: Intimidate
EVs: 4 HP / 252 Atk / 252 Spe
Jolly Nature
- Waterfall
- Earthquake
- Ice Fang
- Stone Edge
"""


def action_picker(side_index: int, state: BattleState, phase: str = "turn") -> Action | None:
    active = state.sides[side_index].active_pokemon
    options = [(action_label(active, action), action) for action in legal_actions(state, side_index)]
    if not options:
        st.warning(f"P{side_index + 1} has no legal actions.")
        return None
    labels = [label for label, _ in options]
    choice = st.selectbox(f"P{side_index + 1} action", labels, key=f"action_pick_{phase}_{side_index}_{state.turn}")
    return options[labels.index(choice)][1]


def _load_sample_teams() -> None:
    st.session_state.p1_team_text = P1_SAMPLE_TEAM
    st.session_state.p2_team_text = P2_SAMPLE_TEAM


def _start_battle() -> None:
    p1_result = parse_showdown_team(st.session_state.p1_team_text)
    p2_result = parse_showdown_team(st.session_state.p2_team_text)
    for player, parse_result in (("P1", p1_result), ("P2", p2_result)):
        for warning in parse_result.warnings:
            kind = warning.kind.name.replace("_", " ").title()
            st.warning(f"{player} mon #{warning.block_index + 1}: {kind} — {warning.detail}")
    if not p1_result.specs:
        st.error("Player 1 team is empty.")
    elif not p2_result.specs:
        st.error("Player 2 team is empty.")
    else:
        p1_team = [build_pokemon(s) for s in p1_result.specs]
        p2_team = [build_pokemon(s) for s in p2_result.specs]
        st.session_state.battle = BattleState(
            sides=(SideState(team=p1_team), SideState(team=p2_team)),
            rng=RNG(seed=int(st.session_state.sandbox_seed)),
        )
        st.session_state.log = []
        st.rerun()


def _forced_switch_panel(state: BattleState, forced_sides: list[int]) -> None:
    st.subheader("Forced switch")
    st.warning("Pick a replacement for each fainted active before the next turn begins.")
    pending: dict[int, Action] = {}
    cols = st.columns(2)
    for i, col in enumerate(cols):
        with col:
            if i in forced_sides:
                chosen = action_picker(i, state, phase="forced")
                if chosen is not None:
                    pending[i] = chosen
            else:
                st.info(f"P{i + 1} waiting for opponent to send out a replacement.")
    if st.button("Confirm switch", type="primary"):
        if len(pending) != len(forced_sides):
            st.error("All fainted sides need a switch action.")
        else:
            switch_lines: list[str] = []
            for i, action in pending.items():
                switch_lines.extend(render_log(apply_forced_switch(state, i, action)))
            st.session_state.log.append(("Replacement", switch_lines))
            st.rerun()


def _turn_panel(state: BattleState) -> None:
    st.subheader("Next turn")
    pick_left, pick_right = st.columns(2)
    with pick_left:
        a1 = action_picker(0, state)
    with pick_right:
        a2 = action_picker(1, state)
    if st.button("Step turn", type="primary"):
        if a1 is None or a2 is None:
            st.error("Both players need a legal action.")
        else:
            new_lines = render_log(step(state, {0: a1, 1: a2}))
            st.session_state.log.append((f"Turn {state.turn}", new_lines))
            st.rerun()


def _log_panel() -> None:
    st.divider()
    st.subheader("Battle log")
    for label, lines in reversed(st.session_state.log[-10:]):
        st.write(f"**{label}**")
        for line in lines:
            st.write(f"- {line}")
    with st.expander("Copy full log"):
        full_lines = []
        for label, lines in st.session_state.log:
            full_lines.append(f"=== {label} ===")
            full_lines.extend(lines)
            full_lines.append("")
        st.code("\n".join(full_lines), language=None)


st.title("Sandbox — both sides, full information")

if "battle" not in st.session_state:
    st.session_state.battle = None
if "p1_team_text" not in st.session_state:
    st.session_state.p1_team_text = P1_SAMPLE_TEAM
if "p2_team_text" not in st.session_state:
    st.session_state.p2_team_text = P2_SAMPLE_TEAM
if "log" not in st.session_state:
    st.session_state.log = []

with st.sidebar:
    st.header("Player 1 team")
    st.caption("Paste a Showdown export-format team; anything unrecognised is skipped with a warning.")
    st.text_area("p1 team", height=260, key="p1_team_text", label_visibility="collapsed")
    st.header("Player 2 team")
    st.text_area("p2 team", height=260, key="p2_team_text", label_visibility="collapsed")
    st.button("Load sample teams", on_click=_load_sample_teams)
    st.divider()
    if st.session_state.battle is None:
        st.number_input("RNG seed", value=0, step=1, key="sandbox_seed")
        if st.button("Start Battle", type="primary"):
            _start_battle()
    else:
        if st.button("Reset battle"):
            st.session_state.battle = None
            st.session_state.log = []
            st.rerun()
        with st.expander("Export current teams"):
            st.code(export_to_showdown(st.session_state.battle.sides[0].team), language=None)
            st.code(export_to_showdown(st.session_state.battle.sides[1].team), language=None)

state: BattleState | None = st.session_state.battle
if state is None:
    st.info("Build your teams in the sidebar and click **Start Battle**.")
else:
    st.subheader(f"Turn {state.turn}")
    render_field(state.field, state.sides, ("P1 side", "P2 side"))
    render_battle_state(state)
    if state.outcome is not None:
        result = {Outcome.P1_WIN: "Player 1 wins!", Outcome.P2_WIN: "Player 2 wins!", Outcome.DRAW: "Draw."}[
            state.outcome
        ]
        st.success(f"Battle over — {result}")
    else:
        forced_sides = [i for i in (0, 1) if state.sides[i].active_pokemon.is_fainted() or state.sides[i].needs_switch]
        st.divider()
        if forced_sides:
            _forced_switch_panel(state, forced_sides)
        else:
            _turn_panel(state)
    if st.session_state.log:
        _log_panel()
