import json
import re
from datetime import date
from pathlib import Path
from urllib.request import Request, urlopen

from loguru import logger

SOURCES = {
    "moves.json": "https://play.pokemonshowdown.com/data/moves.json",
    "pokedex.json": "https://play.pokemonshowdown.com/data/pokedex.json",
    "learnsets.json": "https://play.pokemonshowdown.com/data/learnsets.json",
    "randbats_gen7.json": "https://pkmn.github.io/randbats/data/gen7randombattle.json",
}

# Not JSON at the source — Showdown ships this generation's own competitive tier list (OU/UU/Uber/...)
# as a TS module, since `pokedex.json`'s own `tier` field always reflects the *current* generation's
# metagame, not Gen 7's. `_write_gen7_tiers` below converts it to the plain `{species_id: tier}` this
# repo actually wants.
GEN7_TIERS_SOURCE = "https://raw.githubusercontent.com/smogon/pokemon-showdown/master/data/mods/gen7/formats-data.ts"
GEN7_TIERS_FILENAME = "gen7_tiers.json"

VENDOR_DIR = Path(__file__).parent / "_vendor"
_ENTRY_RE = re.compile(r"(\w+):\s*\{([^{}]*)\}")
_TIER_RE = re.compile(r'tier:\s*"([^"]+)"')


def refresh() -> None:
    VENDOR_DIR.mkdir(exist_ok=True)
    for filename, url in SOURCES.items():
        (VENDOR_DIR / filename).write_bytes(_fetch(url))
    _write_gen7_tiers()
    _write_source_manifest()


def _fetch(url: str) -> bytes:
    logger.info(f"Fetching {url}")
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})  # the CDN 403s python's default UA
    with urlopen(request) as response:
        payload: bytes = response.read()
    logger.info(f"Fetched {url} ({len(payload):,} bytes)")
    return payload


def _write_gen7_tiers() -> None:
    """`{species_id: "OU"|"UU"|"Uber"|...}`, parsed from Showdown's own per-species TS blocks —
    each already reads `key: { tier: "...", ... }`, so a small regex is all a real parser buys here."""
    source = _fetch(GEN7_TIERS_SOURCE).decode()
    tiers = {
        key: tier_match.group(1)
        for key, body in _ENTRY_RE.findall(source)
        if (tier_match := _TIER_RE.search(body)) is not None
    }
    payload = (json.dumps(tiers, indent=1, sort_keys=True) + "\n").encode()
    (VENDOR_DIR / GEN7_TIERS_FILENAME).write_bytes(payload)
    logger.info(f"Wrote {GEN7_TIERS_FILENAME} ({len(tiers):,} species)")


def _write_source_manifest() -> None:
    today = date.today().isoformat()
    lines = [
        "# Vendored data",
        "",
        "| File | URL |",
        "| --- | --- |",
    ]
    for filename, url in SOURCES.items():
        lines.append(f"| `{filename}` | {url} |")
    lines.append(f"| `{GEN7_TIERS_FILENAME}` | {GEN7_TIERS_SOURCE} (converted from TS to `{{id: tier}}` JSON) |")
    lines += [
        "",
        f"Fetched: {today}.",
        "",
        "Refresh with `uv run python -m battle_sim.database.vendor`.",
        "",
        "Type chart and natures are not vendored — they live in "
        "`battle_sim/models/type_matchups.py` and `battle_sim/utils.py` respectively.",
        "",
    ]
    (VENDOR_DIR / "SOURCE.md").write_text("\n".join(lines))


if __name__ == "__main__":
    refresh()
