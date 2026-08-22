import json
import urllib.error
import urllib.request
from enum import Enum
from functools import lru_cache
from pathlib import Path

import streamlit as st

from battle_sim.mechanics.battle import BattleState, FieldState, SideState
from battle_sim.mechanics.log import BattleLog, render_text
from battle_sim.models.actions import Action, ActionType
from battle_sim.models.log_events import BattleEnded, Fainted
from battle_sim.models.moves import DamageEffect, Move
from battle_sim.models.pokemon import Pokemon
from battle_sim.utils import Category, Status, Terrain, Type, Weather

_DEX = Path(__file__).parent / "battle_sim" / "database" / "_vendor" / "pokedex.json"
_SPRITES = "https://play.pokemonshowdown.com/sprites"
_STATUS_CHIPS = {
    Status.BURN: ("BRN", "#e4572e"),
    Status.PARALYSIS: ("PAR", "#d4a017"),
    Status.POISON: ("PSN", "#9c27b0"),
    Status.TOXIC: ("TOX", "#6a1b9a"),
    Status.SLEEP: ("SLP", "#78909c"),
    Status.FREEZE: ("FRZ", "#4fc3f7"),
}
_TYPE_COLORS = {
    Type.NORMAL: "#A8A77A",
    Type.FIRE: "#EE8130",
    Type.WATER: "#6390F0",
    Type.ELECTRIC: "#F7D02C",
    Type.GRASS: "#7AC74C",
    Type.ICE: "#96D9D6",
    Type.FIGHTING: "#C22E28",
    Type.POISON: "#A33EA1",
    Type.GROUND: "#E2BF65",
    Type.FLYING: "#A98FF3",
    Type.PSYCHIC: "#F95587",
    Type.BUG: "#A6B91A",
    Type.ROCK: "#B6A136",
    Type.GHOST: "#735797",
    Type.DRAGON: "#6F35FC",
    Type.DARK: "#705746",
    Type.STEEL: "#B7B7CE",
    Type.FAIRY: "#D685AD",
}
_CATEGORY_GLYPHS = {Category.PHYSICAL: "⚔", Category.SPECIAL: "✦", Category.STATUS: "◌"}


def pretty(kind: Enum) -> str:
    return kind.name.replace("_", " ").title()


def _to_id(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


@lru_cache(maxsize=1)
def _sprite_ids() -> dict[str, str]:
    """Display name -> Showdown sprite id, forme-aware: Chi-Yu -> chiyu, Palafin-Hero -> palafin-hero."""
    dex = json.loads(_DEX.read_text())
    ids = {}
    for entry in dex.values():
        base = entry.get("baseSpecies", entry["name"])
        forme = entry.get("forme")
        ids[entry["name"]] = _to_id(base) + (f"-{_to_id(forme)}" if forme else "")
    return ids


@lru_cache(maxsize=4096)
def _first_available(urls: tuple[str, ...]) -> str:
    """The first URL that answers a HEAD: Streamlit's sanitizer strips onerror handlers, so
    sprite-set fallbacks must be resolved server-side. Cached per species for the process.

    The sprite CDN 403s python's default User-Agent, so the probe must dress as a browser.
    """
    for url in urls:
        request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(request, timeout=3):
                return url
        except (urllib.error.URLError, TimeoutError):
            continue
    return urls[-1]


def sprite_html(species: str, back: bool = False, height: int = 96, fainted: bool = False) -> str:
    """An <img> for the Showdown sprite, falling back across sets for mons the retro one never got.

    Chain: gen5 animated (the charm) -> modern animated (full gen-9 coverage) -> static dex
    art, mirrored when a back sprite was wanted since dex art only faces forward.
    """
    sprite_id = _sprite_ids()[species]
    front_only = (f"{_SPRITES}/home/{sprite_id}.png", f"{_SPRITES}/dex/{sprite_id}.png")
    url = _first_available(
        (
            f"{_SPRITES}/{'gen5ani-back' if back else 'gen5ani'}/{sprite_id}.gif",
            f"{_SPRITES}/{'ani-back' if back else 'ani'}/{sprite_id}.gif",
            f"{_SPRITES}/{'gen5-back' if back else 'gen5'}/{sprite_id}.png",
            *front_only,
        )
    )
    style = f"height:{height}px;image-rendering:auto;"
    if back and url in front_only:
        style += "transform:scaleX(-1);"
    if fainted:
        style += "filter:grayscale(1) brightness(0.55);"
    return f'<img src="{url}" style="{style}">'


def hp_bar_html(fraction: float, label: str) -> str:
    fraction = max(0.0, min(1.0, fraction))
    color = "#4caf50" if fraction > 0.5 else "#ffc107" if fraction > 0.2 else "#f44336"
    return (
        '<div style="background:#3a3a3a;border-radius:7px;height:14px;width:100%;overflow:hidden;">'
        f'<div style="background:{color};width:{fraction * 100:.1f}%;height:100%;'
        'border-radius:7px;transition:width .4s;"></div></div>'
        f'<div style="font-size:0.85em;opacity:0.75;margin:2px 0 8px 0;">{label}</div>'
    )


def type_chip_html(kind: Type) -> str:
    return (
        f'<span style="background:{_TYPE_COLORS[kind]};color:white;border-radius:4px;padding:1px 8px;'
        f'font-size:0.72em;font-weight:bold;text-shadow:0 1px 1px rgba(0,0,0,.4);">{kind.name.title()}</span>'
    )


def type_line_html(pokemon: Pokemon) -> str:
    primary, secondary = pokemon.types
    chips = type_chip_html(primary) + (f" {type_chip_html(secondary)}" if secondary is not None else "")
    return f'<div style="margin:2px 0 6px 0;">{chips}</div>'


def move_chip_html(move: Move) -> str:
    """Type chip plus category glyph and base power for a move button's caption row."""
    damage = next((e for e in move.effects if isinstance(e, DamageEffect)), None)
    power = f" · {damage.power}" if damage is not None and damage.power else ""
    detail = f"{_CATEGORY_GLYPHS[move.category]}{power}"
    return (
        f'<div style="text-align:center;margin-bottom:2px;">{type_chip_html(move.type)}'
        f'<span style="font-size:0.75em;opacity:0.7;margin-left:6px;">{detail}</span></div>'
    )


def status_chip_html(pokemon: Pokemon) -> str:
    if pokemon.status is Status.NONE:
        return ""
    text, color = _STATUS_CHIPS[pokemon.status]
    if pokemon.status in (Status.TOXIC, Status.SLEEP) and pokemon.status_turns > 0:
        text += f" {pokemon.status_turns}"
    return (
        f'<span style="background:{color};color:white;border-radius:4px;padding:1px 7px;'
        f'font-size:0.75em;font-weight:bold;margin-left:6px;">{text}</span>'
    )


def bench_html(side: SideState, back: bool = False) -> str:
    """A strip of small sprites for the whole team, active first, fainted greyed out."""
    cells = []
    for index in [*side.active, *[i for i in range(len(side.team)) if i not in side.active]]:
        mon = side.team[index]
        bar = (
            ""
            if mon.is_fainted()
            else (
                f'<div style="background:#4caf50;height:3px;border-radius:2px;'
                f'width:{100 * mon.live_stats.HP / mon.stat_totals.HP:.0f}%;"></div>'
            )
        )
        cells.append(
            f'<div style="display:inline-block;text-align:center;margin-right:6px;" title="{mon.nickname}">'
            f"{sprite_html(mon.name, back=back, height=40, fainted=mon.is_fainted())}{bar}</div>"
        )
    return '<div style="white-space:nowrap;">' + "".join(cells) + "</div>"


def stages_line(pokemon: Pokemon) -> str:
    nonzero = {stat: stage for stat, stage in pokemon.stat_stages.model_dump().items() if stage != 0}
    parts = [f"{name.replace('_', ' ').title()} {stage:+d}" for name, stage in nonzero.items()]
    parts.extend(volatile.name.replace("_", " ").title() for volatile in pokemon.volatiles)
    if pokemon.flash_fire_active:
        parts.append("Flash Fire")
    return " • ".join(parts)


def render_log(log: BattleLog) -> list[str]:
    return [
        f"**{render_text(entry)}**" if isinstance(entry, (Fainted, BattleEnded)) else render_text(entry)
        for entry in log
    ]


def render_field(field: FieldState, sides: tuple[SideState, SideState], side_names: tuple[str, str]) -> None:
    parts = []
    if field.weather is not Weather.NONE:
        parts.append(f"{pretty(field.weather)} ({field.weather_turns_left})")
    if field.terrain is not Terrain.NONE:
        parts.append(f"{field.terrain.name.title()} Terrain ({field.terrain_turns_left})")
    parts.extend(f"{pretty(pw)} ({turns})" for pw, turns in field.pseudo_weather.items())
    if parts:
        st.info("Field: " + " • ".join(parts))
    for side, name in zip(sides, side_names, strict=True):
        side_parts = [f"{pretty(hazard)} ×{value}" for hazard, value in side.hazards.items()]
        side_parts.extend(f"{pretty(screen)} ({turns})" for screen, turns in side.screens.items())
        if side.tailwind_turns > 0:
            side_parts.append(f"Tailwind ({side.tailwind_turns})")
        if side_parts:
            st.caption(f"{name}: " + " • ".join(side_parts))


def action_label(active: Pokemon, action: Action) -> str:
    if action.action is ActionType.SWITCH_OUT:
        assert action.switch_in is not None  # the Action validator guarantees it
        return f"Switch to {action.switch_in.nickname}"
    assert action.move is not None  # the Action validator guarantees it
    if active.pp[action.move] == 0:
        return "Struggle"
    move = active.moves[action.move]
    assert move is not None  # legal actions only reference filled slots
    return f"{move.name}  ·  {active.pp[action.move]} PP"


def render_battle_state(state: BattleState) -> None:
    """Sandbox card pair: both sides at full information."""
    columns = st.columns(2)
    for column, side, label, back in zip(columns, state.sides, ("P1", "P2"), (True, False), strict=True):
        with column:
            mon = side.active_pokemon
            st.markdown(f"### {label}: {mon.nickname} Lv{mon.level}{status_chip_html(mon)}", unsafe_allow_html=True)
            st.markdown(sprite_html(mon.name, back=back, fainted=mon.is_fainted()), unsafe_allow_html=True)
            fraction = mon.live_stats.HP / mon.stat_totals.HP
            st.markdown(hp_bar_html(fraction, f"HP {mon.live_stats.HP} / {mon.stat_totals.HP}"), unsafe_allow_html=True)
            if stages_line(mon):
                st.caption(stages_line(mon))
            st.markdown(bench_html(side, back=back), unsafe_allow_html=True)
