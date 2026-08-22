# Showdown-ported tests

Tests in this directory are adapted from [smogon/pokemon-showdown](https://github.com/smogon/pokemon-showdown), which is distributed under the MIT License (Copyright (c) 2011–2026 Guangcong Luo and contributors).

## Sources

| Local test | Showdown source |
| --- | --- |
| `test_taunt.py` | [`test/sim/moves/taunt.js`](https://github.com/smogon/pokemon-showdown/blob/master/test/sim/moves/taunt.js) |
| `test_trickroom.py` | [`test/sim/moves/trickroom.js`](https://github.com/smogon/pokemon-showdown/blob/master/test/sim/moves/trickroom.js) |
| `test_chatter.py` | [`test/sim/moves/chatter.js`](https://github.com/smogon/pokemon-showdown/blob/master/test/sim/moves/chatter.js) |

## Harness

`conftest.py` provides a thin Python shim around our engine that mirrors Showdown's test API:

| Showdown | Local equivalent |
| --- | --- |
| `common.createBattle([[p1...], [p2...]])` | `create_battle([p1...], [p2...], seed=0)` |
| `battle.makeChoices('move foo', 'move bar')` | `make_choices(state, 'move foo', 'move bar')` |
| `assert.equal(p.status, 'slp')` | `assert_status(p, 'slp')` |
| `assert.statStage(p, 'spa', 1)` | `assert_stat_stage(p, 'spa', 1)` |

`ability` and `item` fields in pokemon specs are silently dropped — we don't implement those yet. When an ability is *load-bearing* for a test (e.g. Prankster making Taunt go first), the port either adapts the setup (set `stat_stages.SPEED = 6` to ensure Taunt-user moves first) or skips with a reason.

## Adaptation policy

Tests fall into three buckets:

1. **Portable directly** — the mechanic under test (Taunt blocking status, Trick Room reversing speed, Chatter applying confusion) is implemented. Filler `ability`/`item` fields are dropped.
2. **Adapt** — the original relies on Prankster/Speed Boost/etc. as setup, not as the thing being tested. We adapt the setup (manual speed boost, different seed) so the assertion still exercises the same mechanic.
3. **Skip with `pytest.skip`** — the unsupported mechanic *is* the thing being tested (Z-moves bypassing Taunt, Substitute interactions, Gyro Ball's variable BP). Each skip has a one-line `reason=` so it's clear what unlocks it.

## What's currently skipped and why

- **Z-moves** (Taunt #2, #3): no Z-move infrastructure.
- **Protect/Detect** (Trick Room #2): no Protect volatile.
- **Switch-in abilities** (Trick Room #3): no ability hooks.
- **Choice items / 1809-speed boundary** (Trick Room #4): no items, no speed glitch boundary.
- **Variable-BP callbacks** (Trick Room #5 / Gyro Ball): no callback support — `basePowerCallback` moves load with `power=0`.
- **Substitute** (Chatter #1, #2): no Substitute volatile.
- **Sheer Force** (Chatter #3): no ability hooks.

As we add abilities, items, and more volatiles, these unlock with no harness changes.

## License

The original tests are © Smogon contributors and distributed under the [MIT License](https://github.com/smogon/pokemon-showdown/blob/master/LICENSE). The ported versions here are likewise MIT-licensed for the parts derived from upstream.
