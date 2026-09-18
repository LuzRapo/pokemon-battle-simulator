"""Which species and moves are in scope for battling.

Design constraint (Sam, 2026-08-22): battles model Generation 7 mechanics only — no Mega
Evolution, no Terastallization (Dynamax/Gigantamax and Z-Moves are already excluded by
`loader.get_all_moves`). A species is in scope if some usable forme of it existed by Gen 7;
its movepool is whatever moves that forme (or the forme it's cosmetically/functionally
based on) could legally know by Gen 7.
"""

import json
from functools import cache
from pathlib import Path
from typing import Final

from battle_sim.database.loader import gen7_singles_tier, normalize_id
from battle_sim.database.raw import RawLearnsetData, RawSpeciesData

MAX_GENERATION: Final = 7
HIGH_COMPETITIVE_TIERS: Final = frozenset({"OU", "UU", "Uber"})

_VENDOR_DIR: Final = Path(__file__).parent / "_vendor"
_EXCLUDE_NONSTANDARD: Final = frozenset({"CAP", "Custom"})
_MEGA_LIKE_FORME_PREFIXES: Final = ("Mega", "Primal", "Gmax")


@cache
def _raw_species() -> dict[str, RawSpeciesData]:
    raw_data = json.loads((_VENDOR_DIR / "pokedex.json").read_text())
    return {key: RawSpeciesData.model_validate(entry) for key, entry in raw_data.items()}


@cache
def _raw_learnsets() -> dict[str, RawLearnsetData]:
    raw_data = json.loads((_VENDOR_DIR / "learnsets.json").read_text())
    return {key: RawLearnsetData.model_validate(entry) for key, entry in raw_data.items()}


def _own_movepool(key: str) -> frozenset[str]:
    entry = _raw_learnsets().get(key)
    if entry is None:
        return frozenset()
    return frozenset(
        move
        for move, tags in entry.learnset.items()
        if any(tag[0].isdigit() and int(tag[0]) <= MAX_GENERATION for tag in tags)
    )


def _parent_key(key: str) -> str | None:
    """The forme this one inherits from: its immediate `battleOnly` trigger forme if it has

    one (e.g. Darmanitan-Galar-Zen's real parent is Darmanitan-Galar, not root `base_species`
    "Darmanitan"), else its `base_species`, else none.
    """
    raw = _raw_species().get(key)
    if raw is None:
        return None
    parents = raw.battle_only_parents()
    if parents:
        return normalize_id(parents[0])
    return normalize_id(raw.base_species) if raw.base_species is not None else None


def _predates_gen8(key: str) -> bool:
    """Whether this forme existed by Gen 7, per its own learnset entry or (recursively) its parent's.

    A forme with no *moves* of its own on record (e.g. a battle-only fusion forme, or an
    event-only entry like Arceus' type formes that carries no `learnset` key at all) is a pure
    alternate presentation of its parent, not a new introduction, and defers to it. A forme
    whose own entry lists moves where EVERY tag is Gen 8+ (e.g. Meowth-Galar) is a genuine
    post-Gen-7 introduction that merely reuses an old species' dex number — it must not inherit
    the parent's movepool.
    """
    entry = _raw_learnsets().get(key)
    if entry is not None and entry.learnset:
        return any(
            tag[0].isdigit() and int(tag[0]) <= MAX_GENERATION for tags in entry.learnset.values() for tag in tags
        )
    parent = _parent_key(key)
    return _predates_gen8(parent) if parent is not None else True


@cache
def gen7_movepool(species: str) -> frozenset[str]:
    """Every move `species` could legally know by Generation 7.

    Three sources, and the third was missing. A Pokemon keeps everything it learned before it
    evolved, and the learnsets file stores those moves on the *pre-evolution* rather than repeating
    them: Sucker Punch is listed under Pawniard, so Bisharp could not be taught its own signature
    priority move. That is not a Bisharp problem -- every evolved species was short its earlier
    stages' moves, including anything bred onto the base form as an egg move.
    """
    key = normalize_id(species)
    if key not in _raw_species():
        raise KeyError(f"Unknown species: {species!r}")
    if not _predates_gen8(key):
        return frozenset()
    moves = set(_own_movepool(key))
    for earlier in (_parent_key(key), _prevo_key(key)):
        if earlier is not None and earlier != key and _predates_gen8(earlier):
            moves |= gen7_movepool(earlier)
    return frozenset(moves)


def _prevo_key(key: str) -> str | None:
    """What this species evolved from, if anything."""
    raw = _raw_species().get(key)
    if raw is None or raw.prevo is None:
        return None
    return normalize_id(raw.prevo)


def _is_mega_like(raw: RawSpeciesData) -> bool:
    return raw.forme is not None and raw.forme.startswith(_MEGA_LIKE_FORME_PREFIXES)


@cache
def in_scope_species() -> frozenset[str]:
    """Loader-normalized species keys battleable under the Gen-7-only, no-Mega/no-Tera design."""
    return frozenset(
        key
        for key, raw in _raw_species().items()
        if raw.is_nonstandard not in _EXCLUDE_NONSTANDARD
        and not raw.is_cosmetic_forme
        and not _is_mega_like(raw)
        and gen7_movepool(key)
    )


@cache
def gen7_high_tier_species() -> frozenset[str]:
    """In-scope species whose Gen 7 singles placement was OU, UU, or Uber — the genuinely
    competitive band, for anything (mirror-mode team generation) that wants real Smogon-viable
    picks rather than the full Gen 7 dex, tournament rating included."""
    return frozenset(key for key in in_scope_species() if gen7_singles_tier(key) in HIGH_COMPETITIVE_TIERS)
