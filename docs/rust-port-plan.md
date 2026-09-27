# Finishing the Rust port

Written to survive a context compaction. Everything a fresh session needs to carry this to 100% and
then replace the Python engine.

## Where it stands

Branch `feature/vectorised-engine`, last commit `95baec9`.

| | done | total |
|---|---|---|
| moves | 773 | 843 (92%) |
| abilities | 127 | 220 (58%) |
| items | 33 | 110 (30%) |
| volatiles | 10 | ~18 |

6,000 randomly generated battles agree turn for turn, event for event, draw for draw. ~240k
turns/s single-threaded against the Python's ~2.2k.

## The loop

This is the whole method. Do not deviate from it; every bug so far was caught by it, usually within
seconds, including several where the edit silently failed to apply.

```
# widen the scenario generator by one feature, then:
uv run python tools/differential_sweep.py --battles 3000 --slice status --switches --abilities \
    --quiet --team-size 4 --max-turns 100
# something diverged? ask it which turn and why:
uv run python tools/differential_sweep.py --battles 3000 --slice status --switches --abilities \
    --team-size 4 --max-turns 100 --explain <seed>
# and if the divergence is about *how many* draws a turn took rather than what happened:
TRACE_DRAWS=1 rust/target/release/replay <scenario.json> rust/data
```

Also, every time: `cargo build --release` (the binary the harness runs), `cargo test --release`
(the crate's own unit tests — `cargo build` does **not** compile `#[cfg(test)]`, and they rotted
unnoticed once), `cargo clippy --release --all-targets`, `uv run ruff check`, and
`uv run pytest -q tests/test_rust_battles.py`.

Coverage is measured, never tallied: `rust/target/release/replay --coverage rust/data`. It asks the
same `unsupported()` that refuses a real scenario. It has twice started flattering the port when a
new refusal path was added without teaching `unsupported` about it — check it after any change to
what is refused.

## Invariants that must not be broken

1. **A refusal and a divergence are different things.** Exit 2 is "not ported", exit 3 is "we have
   already taken different paths". The harness may skip the first and must fail on the second.
   Collapsing them once hid five real divergences inside a green run.
2. **Anything unported is refused loudly**, never quietly played wrong. Names come off the ported
   lists only after the differential has agreed about them across a sweep.
3. **Reproduce the Python, do not correct it.** See `docs/python-oddities.md`. Four probable bugs
   and several deliberate simplifications are matched on purpose.
4. **Check a claim before making it.** For every ability or item marked ported, grep where the
   Python actually mentions it. Four claims failed that check (Levitate, Air Balloon, Punching
   Glove, Choice Scarf) — each does more somewhere else than the clause that was ported.
5. **Beware a move in two tables.** Weather Ball (type override *and* power condition), Raging Bull
   (type override *and* screen breaker) and Thunder (reacher *and* weather accuracy) each cost an
   afternoon. There is no complete list of tables; check every table when freeing a name.
6. **A green run over dead battles is worse than a red one.** The pivot work turned up six turns of
   empty event lists that both engines agreed about perfectly. Assert that a feature actually
   *fires* in the recorded traces, not merely that the two engines match.

## Remaining work

### 1. Charge / two-turn moves — 17 moves *(attempted, reverted, see below)*

`Fly, Bounce, Dig, Dive, Sky Drop, Phantom Force, Shadow Force, Solar Beam, Solar Blade,
Meteor Beam, Electro Shot, Freeze Shock, Geomancy, Ice Burn, Razor Wind, Skull Bash, Sky Attack`

Design that got to 794/800 before being reverted:

- `Pokemon.charging_slot: Option<usize>` plus a `CHARGING` volatile.
- In `resolve_move`, **before the PP spend**: if `CHARGING` is set and `charging_slot` is `Some`,
  redirect `slot` to `charging_slot` whatever slot the action names — a short moveset is padded by
  repeating its first move, so the same move sits in several slots and the recorded action can name
  a different one. Skip the PP spend on that turn (`_spend_pp`'s `if rampaging or releasing_charge`).
- After `MoveUsed`, before the stall check: if `move.charge && !releasing`, apply
  `_CHARGE_TURN_BOOSTS` (Meteor Beam and Electro Shot, +1 SpA, **whether or not the charge is
  skipped**), then skip the turn unless `_SUN_SKIP_CHARGE` applies in sun (Power Herb also skips
  and is unported). Set `CHARGING`, `charging_slot`, log `ChargingUp`, return.
- `_out_of_reach`, after the Protect check and before the accuracy roll: a defender mid-Fly/Dig is
  untouchable except by `_REACHES_THROUGH`. Logs `MoveMissed`, breaks the rolling run, applies crash
  damage, takes **no** draw. An unlisted charge (Solar Beam) leaves its user visible.
- `switch_out` clears `charging_slot`.
- Take the 11 charge names off `PORTED_ORDINARY_DESPITE_BEING_NAMED` and put them in a
  `PORTED_CHARGES` list — the reachers were only safe *because* charges were refused, and that
  comment is in `power.rs`.

**Unresolved bug.** `--explain 485` (slice status, switches, abilities, team-size 4, max-turns 100):
turn 48 charges Bounce and matches; turn 49 releases it and Rust asks the tape for an integer at
draw 202 where Python has a probability. Python's turn 49 spends 8 draws
`[0.3348, 0.9839, 0.8150, 0.3309, 97, 0.3963, 0.2779, 0.5668]`; two tie-breaks, then Bounce's
accuracy/crit/damage/paralysis-secondary, then **two draws that are not accounted for** — Astral
Barrage logs `NoEffect` and the immunity gate returns before its accuracy roll. Find those two
draws first; the released Bounce is probably not where the discrepancy is. `TRACE_DRAWS=1` prints
the tape position at each move.

### 2. Volatiles — 12 moves

`Substitute` (the big one: gates secondaries, drain, Infiltrator), `Taunt`, `Encore`, `Disable`,
`Destiny Bond`, `LOCKED_MOVE` (Outrage/Petal Dance/Thrash/Raging Fury, plus Rollout and Ice Ball,
which need it for their lock), `Foresight`/`Odor Sleuth`/`Miracle Eye` (the `IDENTIFIED` bypass).

`Taunt`, `Encore` and `Disable` need move-choice restriction, which lives in `engine/choices.py` —
the harness records actions, so the engine side is only the blocking check (`TauntBlocked`,
`DisabledBlocked`) plus the volatile's countdown.

### 3. The remaining coded moves — ~40

Each needs its Python special case read individually. The mechanical ones are gone; what is left is
`Rest`, `Trick`/`Switcheroo`, `Knock Off`'s item removal, `Pain Split`, `Perish Song`, `Haze`,
`Belly Drum`, `Curse`, `Strength Sap`, `Court Change`, `Tidy Up`, `Wish`, `Healing Wish`,
`Revival Blessing`, `Transform`, `Role Play`/`Skill Swap`/`Entrainment`/`Simple Beam`/`Worry Seed`
(ability swapping), `Future Sight`/`Doom Desire` (delayed damage), `Shed Tail`, `Sleep Talk`,
`Roost`, `King's Shield`, the four `WEATHER_HEAL` moves, the party-cure pair, and the four
pseudo-weather rooms (`Trick Room` also inverts the speed sort).

### 4. Abilities — 93 left

The clusters, roughly in order of value:

- **Damage/ordering one-liners at sites that already exist**: Sturdy, Shield Dust, Serene Grace,
  Skill Link, Steadfast, Scrappy/Minds Eye, Soundproof, Bulletproof, Unburden, Surge Surfer,
  Prankster, Gale Wings, Quick Draw, Mycelium Might, Triage, Pressure, Mold Breaker/Teravolt/
  Turboblaze, Corrosion, Synchronize, Natural Cure, Regenerator.
- **Type absorption**: Volt Absorb, Water Absorb, Earth Eater, Sap Sipper, Well Baked Body, Flash
  Fire, Motor Drive, Lightning Rod, Storm Drain, Wind Rider, Dry Skin, Wonder Guard, Good as Gold.
  These cancel a move at `ON_BEFORE_MOVE` — a hook this engine does not have yet.
- **Trapping**: Arena Trap, Shadow Tag, Magnet Pull (affects `legal_actions`, not `step`).
- **Ability swapping / copying**: Trace, Imposter, Mummy, Power of Alchemy-likes, Neutralizing Gas.
- **Formes**: Stance Change, Zen Mode, Schooling, Shields Down, Power Construct, Zero to Hero,
  Tera Shift, Disguise, Multitype/RKS System, Battle Bond, Libero/Protean.
- **Paradox**: Protosynthesis, Quark Drive, Booster Energy, Hadron Engine, Orichalcum Pulse.
- **Forced switches**: Emergency Exit, Wimp Out.
- **The butler**: Nine Lives, and the last-stand machinery in `damage_apply`. Bespoke to this bot.

### 5. Items — 77 left

Focus Sash, Life Orb, Heavy-Duty Boots, the Choice trio (need `choice_locked_move`), the plates and
memories (type boosters *and* Judgment/Multi-Attack's type selector — both halves), the weather and
terrain rocks, Power Herb, Light Clay, Terrain Extender, Loaded Dice, White Herb, Mental Herb,
Mirror Herb, Clear Amulet, Covert Cloak, Eject Button/Pack, Red Card, Shed Shell, Quick Claw,
Custap Berry, the cure berries, Leppa, Air Balloon, Punching Glove, Adrenaline Orb, the seeds,
Griseous Orb/Core, Ultranecrozium Z.

### 6. Hooks still missing

`ON_BEFORE_MOVE` (absorption, Magic Bounce, Air Balloon's float), `ON_BEFORE_HIT` (survival clamps:
Sturdy, Focus Sash, Endure's item cousins), `ON_FAINT`, `ON_SWITCH_OUT` (Regenerator, Natural Cure),
`ON_ACTION_RESOLVE` (Life Orb), `ON_TURN_START`/`ON_TURN_END`.

### 7. Z-moves, megas, formes

`zmove:` actions are refused where actions are parsed. Mega evolution, Z-moves and forme changes are
a whole action/state dimension the port has not touched. Check what the bot actually uses before
deciding how much is needed.

### 8. Integration — after parity

1. Decide the interface. PyO3 in-process is the obvious one (the search calls `step` millions of
   times; a subprocess per call is hopeless). `rust/src/lib.rs` is already a library crate.
2. The search needs `legal_actions` too, which currently lives only in Python
   (`engine/choices.py`) — it reads locks, traps, charges and `needs_switch`.
3. Swap the search over behind a flag, then re-run the AI validation (mirror-match win rate) to
   confirm the engine change did not move play strength.
4. Keep the differential running in CI against the Python for as long as both exist.

## Rough sizing

Sections 1–3 are perhaps two sessions. Section 4 is the largest single block but most of it is
one-liners once the missing hooks exist — call it two sessions with section 6 folded in. Section 5
is one. Integration is one, plus whatever the AI re-validation turns up.

The tail is not uniform: absorption abilities and formes are each a small architecture change, and
the butler's revival mechanic has no reference outside this codebase.
