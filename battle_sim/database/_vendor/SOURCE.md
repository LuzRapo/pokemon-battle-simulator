# Vendored data

| File | URL |
| --- | --- |
| `moves.json` | https://play.pokemonshowdown.com/data/moves.json |
| `pokedex.json` | https://play.pokemonshowdown.com/data/pokedex.json |
| `learnsets.json` | https://play.pokemonshowdown.com/data/learnsets.json |
| `randbats_gen7.json` | https://pkmn.github.io/randbats/data/gen7randombattle.json |

Fetched: 2026-05-13 (`moves.json`, `pokedex.json`); 2026-08-22 (`learnsets.json`, `randbats_gen7.json`).

Refresh with `uv run python -m battle_sim.database.vendor`.

Type chart and natures are not vendored — they live in `battle_sim/models/type_matchups.py` and `battle_sim/utils.py` respectively.
