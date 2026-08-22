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

VENDOR_DIR = Path(__file__).parent / "_vendor"


def refresh() -> None:
    VENDOR_DIR.mkdir(exist_ok=True)
    for filename, url in SOURCES.items():
        logger.info(f"Fetching {url}")
        request = Request(url, headers={"User-Agent": "Mozilla/5.0"})  # the CDN 403s python's default UA
        with urlopen(request) as response:
            payload = response.read()
        (VENDOR_DIR / filename).write_bytes(payload)
        logger.info(f"Wrote {filename} ({len(payload):,} bytes)")

    _write_source_manifest()


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
