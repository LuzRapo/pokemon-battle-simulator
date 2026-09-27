# Finishing the Rust port

Written to survive a context compaction. Everything a fresh session needs to carry this to 100% and
then replace the Python engine.

## Where it stands

Branch `feature/vectorised-engine`. Sections 1 and 2 below (charges and volatiles) are both done.

| | done | total |
|---|---|---|
| moves | 804 | 843 (95%) |
| abilities | 127 | 220 (58%) |
| items | 33 | 110 (30%) |
| volatiles | 18 | 18 — every volatile in the database is ported |

The 39 moves still refused are 4 that do something to their user or the field (pseudo-weather
rooms) and 35 special-cased by name (the remaining coded moves, section 3).

Several thousand randomly generated battles agree turn for turn, event for event, draw for draw,
including a pool built specifically to force charging on both the semi-invulnerable and the
ordinary two-turn moves (`test_charge_moves_actually_charge_and_agree`), and a vacuity check that
disabling `power::out_of_reach` turns 251/300 of them red. ~240k turns/s single-threaded against
the Python's ~2.2k.

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

### 1. Charge / two-turn moves — 17 moves — **done**

`Fly, Bounce, Dig, Dive, Sky Drop, Phantom Force, Shadow Force, Solar Beam, Solar Blade,
Meteor Beam, Electro Shot, Freeze Shock, Geomancy, Ice Burn, Razor Wind, Skull Bash, Sky Attack`

Re-implemented from scratch after the first attempt (794/800) was lost to a rejected-but-already-run
`git checkout`. Shipped design, differs from the first attempt in one respect noted below:

- `Pokemon.charging_slot: Option<usize>` plus a `CHARGING` volatile.
- In `resolve_move`, **before the PP spend**: if `CHARGING` is set and `charging_slot` is `Some`,
  redirect `slot` to `charging_slot` whatever slot the action names — a short moveset is padded by
  repeating its first move, so the same move sits in several slots and the recorded action can name
  a different one. Skip the PP spend on that turn (`_spend_pp`'s `if rampaging or releasing_charge`).
- After `MoveUsed`, before the stall check: if `move.charge && !releasing`, apply
  `_CHARGE_TURN_BOOSTS` (Meteor Beam and Electro Shot, +1 SpA, **whether or not the charge is
  skipped**), then skip the turn unless `_SUN_SKIP_CHARGE` applies in sun (Power Herb also skips
  and is unported — a Pokemon holding it is refused upstream, so that branch is dead code for now).
  Set `CHARGING`, `charging_slot`, log `ChargingUp`, return.
- `_out_of_reach` (`power::out_of_reach`), after the Protect check and before the accuracy roll: a
  defender mid-Fly/Dig/Phantom-Force is untouchable except by `_REACHES_THROUGH`. Logs
  `MoveMissed`, breaks the rolling run, applies crash damage, takes **no** draw. An unlisted charge
  (Solar Beam) leaves its user visible.
- `switch_out`/`withdraw` clears `charging_slot`.
- **Not needed, unlike the first attempt's plan:** the reacher names (Gust, Earthquake, Surf, ...)
  stay in `PORTED_ORDINARY_DESPITE_BEING_NAMED` untouched. `out_of_reach` is a generic function
  keyed on the *defender's* charging move, not on the attacker's move being specially tagged, so
  there was nothing to move out of that list after all.
- One thing the first attempt's notes didn't mention and this one had to solve: most of the 17
  charge names are themselves in Python's `coded_moves` AST sweep (they're string literals in
  `_REACHES_THROUGH`/`_CHARGE_TURN_BOOSTS`/`_SUN_SKIP_CHARGE`), so they were being refused by the
  `Gap::CodedByName` gate *before* `unsupported`'s `the_move.charge` check ever ran. Fixed by adding
  `power::PORTED_CHARGES` into `turn::ported_coded_moves()`'s set, same pattern as
  `PORTED_ORDINARY_DESPITE_BEING_NAMED`.
- Found and fixed in passing: `SCREEN_BREAKERS` was applied right after the Protect check, before
  the accuracy roll and the effectiveness/immunity check — so a screen-breaking move that missed, or
  was shrugged off as `NoEffect`, broke the screen anyway. Python applies it after both (`moves.py`
  line 279, well after the accuracy check at 248). Moved down; this was a real latent bug, not
  something charges introduced, just adjacent code the charge work required reading closely.

Verified: `test_charge_moves_actually_charge_and_agree` forces a pool of all 17 charges plus the
four reachers used across the multiple `_REACHES_THROUGH` tables (Earthquake, Surf, Gust, Thunder)
and asserts both that a `ChargingUp` event actually appears and that the traces agree. Vacuity
check: commenting out the `out_of_reach` call turns 251/300 of a charge-heavy stress batch red.

### 2. Volatiles — **done**

Every volatile in the database now has a home, across four commits:

- **Rampage** (Outrage, Petal Dance, Raging Fury, Thrash): `locked_slot`/`last_move_slot` on
  `Pokemon`, the same redirect-and-skip-PP shape as a charge, `LOCKED_MOVE` dispatched through its
  own bespoke path (`start_rampage`) rather than the generic volatile one — no log line, no
  ability/type immunity, matching `_apply_status`'s own special case for it exactly. A `LOCKED_MOVE`
  residual counts the lock down and confuses the user on natural expiry, unless the lock was a roll
  rather than a rampage.
- **Rollout / Ice Ball**: reuse the same `LOCKED_MOVE` machinery via `_continue_rolling`'s shape,
  and turned up a real bug in the process — the "any other move ends the escalation" reset checked
  only for `"Fury Cutter"` by name instead of the whole `ESCALATING_MOVES` set, so a landed Rollout
  zeroed its own hit counter before its own power formula read it. Fixed; `break_rolling` now gates
  the lock-clearing half of a miss on `_is_rolling_slot`, since a rampage's lock must survive a miss
  that a roll's must not.
- **Taunt, Encore, Disable**: move-choice restriction. The harness already records legal actions via
  Python's own `legal_actions`, so the engine side is only the blocking check (`TauntBlocked` after
  PP is spent, `DisabledBlocked` *before* — a real ordering difference worth getting right) plus each
  volatile's countdown (`tick_countdown`, a small shared residual helper).
- **Destiny Bond**: cleared at the very top of `resolve_move`, before any redirect is even
  determined — that single line is the entire "lasts until the user's next action" rule. The payoff
  is one check where a hit's damage is applied: a fainting Destiny-Bond holder takes its
  still-standing attacker with it.
- **Foresight / Odor Sleuth / Miracle Eye**: `Pokemon::identify_bypass()` plus
  `Database::effectiveness_bypassing()`. Needed at **two** independent call sites that both ask the
  type chart the same question — the immunity gate in `resolve_move` and `damage::calculate_hit`'s
  own internal effectiveness check — and missing either one leaves a hit correctly announced as
  landing and then dealing zero damage anyway. (These three moves' evasion-ignoring half was never
  written in the Python at all; see `docs/python-oddities.md`.)
- **Substitute**, last and largest: `behind_substitute` computed once per move at the top of the
  effects loop (SUBSTITUTE present, move doesn't bypass it, attacker isn't Infiltrator — still
  unreachable, but written to match), then threaded through the damage loop (a fresh per-hit check,
  which is how a multi-hit move that breaks the sub partway through finishes on the real Pokemon —
  reproduced, see `docs/python-oddities.md`), fixed damage, and the status/stage dispatch (an
  opponent-targeted effect is skipped **before its probability draw**, not after). `start_substitute`
  is bespoke like the other three ExtraStatus branches. Also required: a `bypass_substitute` field on
  `Move` that had never been read on the Rust side before; a `behind_substitute` field on the
  ability/item `Calc` context, since the resist berries (already ported) ask it directly; and one
  side-effect fix once a Pokemon could actually have a substitute up — Intimidate did not check for
  one.

Nine vacuity checks across these commits (the rolling-hits reset, the Disable gate, the Taunt gate,
Destiny Bond's retaliation, the `calculate_hit` bypass site, the per-hit substitute soak, the
status/stage substitute block, and Intimidate-vs-Substitute) each turned somewhere between 105/300
and 296/300 of a targeted batch red when disabled.

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
