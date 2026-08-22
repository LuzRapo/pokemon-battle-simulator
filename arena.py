import random
from datetime import datetime
from pathlib import Path

import streamlit as st

from battle_sim.coach import grade_choice
from battle_sim.database.loader import get_species
from battle_sim.evolution import Team, load_teams, load_weights
from battle_sim.interactive import AI, HUMAN, InteractiveBattle, Phase
from battle_sim.matchup import MatchupWeights
from battle_sim.models.actions import Action, ActionType
from battle_sim.models.spec import PokemonSpec
from battle_sim.observation import SetPrior
from battle_sim.search import SearchPlayer, SearchProfile
from battle_sim.team_search import draft_team, set_universe
from ui_shared import (
    action_label,
    bench_html,
    hp_bar_html,
    move_chip_html,
    pretty,
    render_field,
    render_log,
    sprite_html,
    stages_line,
    status_chip_html,
    type_line_html,
)

_GRADE_BADGES = {
    "Brilliant": "💎",
    "Best": "✅",
    "Great": "👍",
    "Good": "🙂",
    "Inaccuracy": "🤔",
    "Mistake": "⚠️",
    "Blunder": "💀",
}


@st.cache_resource
def _arsenal() -> tuple[SetPrior, list[PokemonSpec], dict[str, list[PokemonSpec]]]:
    teams = load_teams(Path("sample_teams"))
    universe = set_universe(teams)
    by_species: dict[str, list[PokemonSpec]] = {}
    for spec in universe:
        by_species.setdefault(get_species(spec.species).name, []).append(spec)
    return SetPrior.from_teams(teams), universe, by_species


def _set_label(spec: PokemonSpec) -> str:
    return f"{pretty(spec.item)} — {' / '.join(spec.moves)}"


def _boss() -> SearchPlayer:
    weights = load_weights(Path(st.session_state.arena_genome)) if st.session_state.arena_genome else MatchupWeights()
    profile = SearchProfile(
        budget=int(st.session_state.arena_budget),
        determinizations=int(st.session_state.arena_determinizations),
    )
    return SearchPlayer(weights, profile=profile)


def _coach() -> SearchPlayer:
    """The boss's brain pointed at your decisions: same genome and budget, modal beliefs only."""
    weights = load_weights(Path(st.session_state.arena_genome)) if st.session_state.arena_genome else MatchupWeights()
    return SearchPlayer(weights, profile=SearchProfile(budget=int(st.session_state.arena_budget)))


def _surprise_me(universe: list[PokemonSpec], by_species: dict[str, list[PokemonSpec]]) -> None:
    drafted = draft_team(universe, random.Random())
    for slot, spec in enumerate(drafted):
        species = get_species(spec.species).name
        st.session_state[f"arena_species_{slot}"] = species
        st.session_state[f"arena_set_{slot}"] = _set_label(spec)


def _builder(universe: list[PokemonSpec], by_species: dict[str, list[PokemonSpec]]) -> None:
    st.subheader("Build your team")
    st.caption(
        f"Pick six from the {len(by_species)} known species and {len(universe)} known sets — the same arsenal "
        "the boss drafts from, blind, before it ever sees your picks."
    )
    st.button("Surprise me", on_click=_surprise_me, args=(universe, by_species))
    species_names = sorted(by_species)
    chosen: list[PokemonSpec] = []
    columns = st.columns(6)
    for slot, column in enumerate(columns):
        with column:
            default = species_names[slot] if f"arena_species_{slot}" not in st.session_state else None
            species = st.selectbox(
                f"Slot {slot + 1}",
                species_names,
                index=species_names.index(default) if default else None,
                key=f"arena_species_{slot}",
            )
            assert species is not None
            st.markdown(
                f'<div style="text-align:center;">{sprite_html(species, height=64)}</div>',
                unsafe_allow_html=True,
            )
            labels = [_set_label(spec) for spec in by_species[species]]
            label = st.selectbox("Set", labels, key=f"arena_set_{slot}", label_visibility="collapsed")
            chosen.append(by_species[species][labels.index(label)])
    species_picked = [get_species(spec.species).name for spec in chosen]
    if len(set(species_picked)) != 6:
        st.warning("Species clause: six different species, please.")
        return
    if st.button("Lock it in", type="primary"):
        seed = int(st.session_state.arena_seed)
        st.session_state.arena_human_team = tuple(chosen)
        st.session_state.arena_ai_team = draft_team(universe, random.Random(seed + 0xB055))  # blind: seed-only
        st.session_state.arena_stage = "preview"
        st.rerun()


def _preview(prior: SetPrior) -> None:
    human_team: Team = st.session_state.arena_human_team
    ai_team: Team = st.session_state.arena_ai_team
    st.subheader("Team preview — choose your lead")
    st.caption("The boss's species are public; its sets are not. It is choosing its own lead the same way.")
    st.markdown(
        '<div style="text-align:center;">'
        + "".join(sprite_html(spec.species, height=72) for spec in ai_team)
        + "</div>",
        unsafe_allow_html=True,
    )
    st.divider()
    columns = st.columns(6)
    for slot, (column, spec) in enumerate(zip(columns, human_team, strict=True)):
        with column:
            st.markdown(
                f'<div style="text-align:center;">{sprite_html(spec.species, back=True, height=64)}</div>',
                unsafe_allow_html=True,
            )
            if st.button(spec.species, key=f"arena_lead_{slot}", use_container_width=True):
                order = [slot, *[i for i in range(6) if i != slot]]
                seed = int(st.session_state.arena_seed)
                with st.spinner("The boss is picking its lead..."):
                    st.session_state.arena_battle = InteractiveBattle(
                        human_team, ai_team, _boss(), prior=prior, seed=seed, human_order=order
                    )
                stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                st.session_state.arena_transcript = Path("battles") / f"arena-{stamp}-seed{seed}.log"
                st.session_state.arena_coach_player = _coach()
                st.session_state.arena_verdicts = []
                _save_transcript()
                st.session_state.arena_stage = "battle"
                st.rerun()
    if st.button("Back to the builder"):
        st.session_state.arena_stage = "build"
        st.rerun()


def _active_cards(battle: InteractiveBattle) -> None:
    view = battle.view()
    mine, theirs = view.sides[HUMAN], view.sides[AI]
    me_column, foe_column = st.columns(2)
    with foe_column:
        mon = theirs.active_pokemon
        st.markdown(f"#### {mon.name} Lv{mon.level}{status_chip_html(mon)}", unsafe_allow_html=True)
        st.markdown(type_line_html(mon), unsafe_allow_html=True)
        st.markdown(
            f'<div style="text-align:center;">{sprite_html(mon.name, height=140, fainted=mon.is_fainted())}</div>',
            unsafe_allow_html=True,
        )
        percent = 100 * mon.live_stats.HP / mon.stat_totals.HP
        st.markdown(hp_bar_html(percent / 100, f"{percent:.0f}%"), unsafe_allow_html=True)
        if stages_line(mon):
            st.caption(stages_line(mon))
        st.markdown(bench_html(theirs), unsafe_allow_html=True)
        with st.expander("Metagame read (educated guesses, not reveals)"):
            st.caption(f"Ability: {pretty(mon.ability)} • Item: {pretty(mon.item)}")
            st.caption("Moves: " + ", ".join(move.name for move in mon.moves if move is not None))
    with me_column:
        mon = mine.active_pokemon
        st.markdown(f"#### {mon.nickname} Lv{mon.level}{status_chip_html(mon)}", unsafe_allow_html=True)
        st.markdown(type_line_html(mon), unsafe_allow_html=True)
        sprite = sprite_html(mon.name, back=True, height=140, fainted=mon.is_fainted())
        st.markdown(f'<div style="text-align:center;">{sprite}</div>', unsafe_allow_html=True)
        fraction = mon.live_stats.HP / mon.stat_totals.HP
        st.markdown(hp_bar_html(fraction, f"HP {mon.live_stats.HP} / {mon.stat_totals.HP}"), unsafe_allow_html=True)
        if stages_line(mon):
            st.caption(stages_line(mon))
        st.markdown(bench_html(mine, back=True), unsafe_allow_html=True)


def _header() -> str:
    return (
        f"== Arena match ==\nwhen: {datetime.now().isoformat(timespec='seconds')}\n"
        f"genome: {st.session_state.arena_genome}\n"
        f"budget: {st.session_state.arena_budget}  determinizations: {st.session_state.arena_determinizations}\n"
        f"seed: {st.session_state.arena_seed}"
    )


def _coach_section() -> str:
    verdicts = st.session_state.get("arena_verdicts", [])
    if not verdicts:
        return ""
    lines = [
        f"turn {verdict['turn']}: {verdict['label']} — {verdict['chosen']}"
        + (f" (-{verdict['loss']:.2f}, preferred {verdict['best']})" if verdict["loss"] > 0.05 else "")
        for verdict in verdicts
    ]
    return "\n\n== Coach ==\n" + "\n".join(lines) + "\n"


def _save_transcript() -> None:
    """Rewrite the whole transcript after every action: a dead session still leaves the replay."""
    battle: InteractiveBattle = st.session_state.arena_battle
    path: Path = st.session_state.arena_transcript
    path.parent.mkdir(exist_ok=True)
    path.write_text(battle.transcript(header=_header()).rstrip("\n") + "\n" + _coach_section())


def _grade(battle: InteractiveBattle, options: list[Action], action: Action) -> None:
    active = battle.state.sides[HUMAN].active_pokemon
    verdict = grade_choice(st.session_state.arena_coach_player, battle.view(), HUMAN, options, action)
    st.session_state.arena_verdicts.append(
        {
            "turn": battle.state.turn,
            "chosen": action_label(active, action),
            "label": verdict.label,
            "loss": verdict.loss,
            "best": action_label(active, verdict.best),
        }
    )


def _submit(battle: InteractiveBattle, options: list[Action], action: Action) -> None:
    with st.spinner("The boss is thinking..."):
        if st.session_state.arena_coach:
            _grade(battle, options, action)
        battle.submit(action)
    _save_transcript()
    st.rerun()


def _last_verdict_banner() -> None:
    if not (st.session_state.arena_coach and st.session_state.arena_verdicts):
        return
    last = st.session_state.arena_verdicts[-1]
    text = f"{_GRADE_BADGES[last['label']]} Coach on **{last['chosen']}**: **{last['label']}**"
    if last["loss"] > 0.05:
        text += f" (−{last['loss']:.2f} mons) — preferred: {last['best']}"
    st.caption(text)


def _action_panel(battle: InteractiveBattle) -> None:
    active = battle.state.sides[HUMAN].active_pokemon
    options = battle.options()
    moves = [a for a in options if a.action is ActionType.USE_MOVE]
    switches = [a for a in options if a.action is ActionType.SWITCH_OUT]
    _last_verdict_banner()
    if battle.phase is Phase.REPLACE:
        if active.is_fainted():
            st.warning(f"{active.nickname} is out of the fight — pick a replacement.")
        else:
            st.warning(f"{active.nickname} is being forced out (Eject Button, Roar...) — pick who takes the field.")
    elif moves:
        columns = st.columns(max(len(moves), 1))
        for index, (column, action) in enumerate(zip(columns, moves, strict=False)):
            with column:
                assert action.move is not None  # the Action validator guarantees it
                move = active.moves[action.move]
                if move is not None:
                    st.markdown(move_chip_html(move), unsafe_allow_html=True)
                if st.button(action_label(active, action), key=f"arena_move_{index}", use_container_width=True):
                    _submit(battle, options, action)
    if switches:
        st.caption("Bench")
        columns = st.columns(max(len(switches), 1))
        for index, (column, action) in enumerate(zip(columns, switches, strict=False)):
            assert action.switch_in is not None  # the Action validator guarantees it
            with column:
                st.markdown(
                    f'<div style="text-align:center;">{sprite_html(action.switch_in.name, height=48)}</div>',
                    unsafe_allow_html=True,
                )
                if st.button(action.switch_in.nickname, key=f"arena_switch_{index}", use_container_width=True):
                    _submit(battle, options, action)


def _coach_report() -> None:
    verdicts = st.session_state.arena_verdicts
    if not (st.session_state.arena_coach and verdicts):
        return
    counts: dict[str, int] = {}
    for verdict in verdicts:
        counts[verdict["label"]] = counts.get(verdict["label"], 0) + 1
    total_loss = sum(verdict["loss"] for verdict in verdicts)
    summary = "  ".join(
        f"{_GRADE_BADGES[label]} {label} ×{counts[label]}" for label in _GRADE_BADGES if label in counts
    )
    st.markdown(f"**Coach report** — {summary} — total value given up: **{total_loss:.2f} mons**")
    with st.expander("Move by move"):
        for verdict in verdicts:
            line = (
                f"turn {verdict['turn']}: {_GRADE_BADGES[verdict['label']]} **{verdict['label']}**"
                f" — {verdict['chosen']}"
            )
            if verdict["loss"] > 0.05:
                line += f" (−{verdict['loss']:.2f}, preferred {verdict['best']})"
            st.write(line)


def _battle_screen() -> None:
    battle: InteractiveBattle = st.session_state.arena_battle
    view = battle.view()
    stage_column, log_column = st.columns([5, 2])
    with stage_column:
        st.subheader(f"Turn {view.turn}")
        render_field(view.field, (view.sides[HUMAN], view.sides[AI]), ("You", "Boss"))
        _active_cards(battle)
        st.divider()
        if battle.phase is Phase.OVER:
            survivors = sum(1 for p in battle.state.sides[HUMAN].team if not p.is_fainted())
            if survivors:
                st.success(f"You win — {survivors} pokemon still standing. The boss demands a rematch.")
                st.balloons()
            else:
                st.error("The boss wins. It says nothing, which somehow makes it worse.")
            _coach_report()
            if st.button("Run it back (new seed)", type="primary"):
                st.session_state.arena_seed = int(st.session_state.arena_seed) + 1
                st.session_state.arena_stage = "preview"
                st.session_state.arena_battle = None
                st.rerun()
            if st.button("New teams"):
                st.session_state.arena_stage = "build"
                st.session_state.arena_battle = None
                st.rerun()
        else:
            _action_panel(battle)
    with log_column:
        st.subheader("Log")
        st.caption(f"Replay: `{st.session_state.arena_transcript}`")
        for index, log in enumerate(reversed(battle.logs[-8:])):
            for line in render_log(log):
                st.write(line)
            if index < len(battle.logs[-8:]) - 1:
                st.divider()


prior, universe, by_species = _arsenal()
st.title("⚔️ Battle the boss")

# Streamlit purges local modules whenever any watched file changes, minting new classes while
# session_state and cache_resource still hold instances of the old ones — every is-check and
# dict lookup then silently mismatches. Detect the identity break and reset cleanly: the
# replay is persisted after every action, so only the in-flight battle is lost.
if universe and not isinstance(universe[0], PokemonSpec):
    _arsenal.clear()
    prior, universe, by_species = _arsenal()
if st.session_state.get("arena_battle") is not None and not isinstance(
    st.session_state.arena_battle, InteractiveBattle
):
    st.session_state.arena_battle = None
    st.session_state.arena_stage = "build"
    st.warning("The app's code changed under a live battle — battle reset. The replay so far is saved in battles/.")

with st.sidebar:
    st.header("The boss")
    champions = sorted(str(path) for path in Path("champions").glob("*.json"))
    default = champions.index("champions/search-b.json") if "champions/search-b.json" in champions else 0
    st.selectbox("Genome", champions, index=default, key="arena_genome")
    st.number_input("Search budget (30 ≈ depth 1, 400 ≈ depth 2, 3500 ≈ depth 3)", value=3500, key="arena_budget")
    st.slider("Determinizations (sampled belief worlds)", 1, 5, 2, key="arena_determinizations")
    st.number_input("Battle seed", value=0, step=1, key="arena_seed")
    st.toggle("Coach mode (grades your moves)", value=True, key="arena_coach")

if "arena_stage" not in st.session_state:
    st.session_state.arena_stage = "build"
    st.session_state.arena_battle = None

if st.session_state.arena_stage == "build":
    _builder(universe, by_species)
elif st.session_state.arena_stage == "preview":
    _preview(prior)
else:
    _battle_screen()
