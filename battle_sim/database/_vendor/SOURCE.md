# Vendored data

Source: https://play.pokemonshowdown.com/data/

| File | URL |
| --- | --- |
| `moves.json` | https://play.pokemonshowdown.com/data/moves.json |
| `pokedex.json` | https://play.pokemonshowdown.com/data/pokedex.json |

Fetched: 2026-05-13.

Refresh with `uv run python -m battle_sim.database.vendor`.

Type chart and natures are not vendored — they live in `battle_sim/models/type_matchups.py` and `battle_sim/utils.py` respectively.
