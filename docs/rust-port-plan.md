# Finishing the Rust port

Written to survive a context compaction. Everything a fresh session needs to carry this to 100% and
then replace the Python engine.

## Where it stands

Branch `feature/vectorised-engine`. Sections 1, 2 and 3 (charges, volatiles, and the remaining
coded moves) are all done — every `CodedMoveKind` is ported, and with it every move in the
database. Section 4 (abilities) is underway: the turn-order cluster (seven abilities), Sturdy, and
two one-liner batches (Skill Link/Serene Grace/Shield Dust/Scrappy/Mind's Eye/Synchronize/Pressure/
Steadfast/Corrosion, then Sheer Force/Solar Power/Thermal Exchange/Toxic Debris/Cursed Body/Wonder
Guard/Wind Rider/Liquid Voice/Poison Puppeteer) are in the tree and verified. Section 5 (items) has
now started: the first batch — Quick Claw, Custap Berry, Leppa Berry, Chesto Berry, Lum Berry, the
four weather rocks, and the four terrain seeds — is in the tree and verified; see the batch
write-up below. Its dedicated tests, individual vacuity checks (all thirteen items, each confirmed
to turn red on its own), and confirming differential sweeps have all been run since the last update
to this doc; per invariant 2 the batch now belongs on the "done" side.

| | done | total |
|---|---|---|
| moves | 843 | 843 (100%) |
| abilities | 180 | 220 (82%) |
| items | 105 | 193 |
| volatiles | 18 | 18 — every volatile in the database is ported |

The counts above are confirmed against `rust/target/release/replay --coverage rust/data`, run with
a release build for this update:
`{"abilities":{"live":220,"ported":180},"items":{"live":193,"ported":105},"moves":{"playable":843,
"refused_by_cause":{},"total":843}}` — an empty `refused_by_cause` for moves, matching the 100%
row above.

The items total jumped from 110 to 193 mid-section: 96 Mega Stones (and Primal orbs, Rusted
Sword/Shield) turned out to have real behaviour — `_resolve_mega_evolution` — that the coverage
export had never been able to see, for reasons worth reading in section 3's own notes below. They
are correctly refused now rather than silently wrong, which is why the *ported* item count did not
move even though total climbed by 83. (An earlier revision of this doc had the items row and
section 5's header disagree with each other — 33/193 in the table but "77 left" in the section
header, which would only be consistent with a different total. The table above is the one to trust;
section 5 below has been corrected to match it.)

Every refusal cause `--coverage` used to report — coded-by-name, no modelled effect, does something
to its user or the field, variable power, an unported volatile — is gone. `Transform`, the last
move in the database, closed out section 3: see its own write-up below. Wish, Healing Wish/Lunar
Dance, Revival Blessing, Shed Tail, and Future Sight/Doom Desire were done just before it — see the
batch write-up below for the two real bugs that batch turned up (an immediate-switch gap and a
log-ordering gap, both fixed).

Several thousand randomly generated battles agree turn for turn, event for event, draw for draw,
including a pool built specifically to force charging on both the semi-invulnerable and the
ordinary two-turn moves (`test_charge_moves_actually_charge_and_agree`), and a vacuity check that
disabling `power::out_of_reach` turns 251/300 of them red. ~240k turns/s single-threaded against
the Python's ~2.2k. The latest full sweep, run with Transform in the pool: 3000 status-slice
battles with switching enabled, 2971/3000 agreed, 29 refused (all the same documented Future Sight
scope limit below), 0 diverged; a 2000-battle plain-slice sweep came back 2000/2000 agreed, 0
diverged.

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

### 3. The remaining coded moves — most of it done

Each needed its Python special case read individually; there was no shortcut for any of them.
**Done, across three commits:**

- `Sleep Talk` (a real redirect: computed by the caller from the *chosen* slot before any other
  redirect, `power::sleep_talk_choice` drawing from the padded four-slot moveset — Python pads
  `known_moves()` identically, so no new state was needed), `Roost` (`Pokemon::battle_types()`,
  needed at every effectiveness call site a roosted defender could reach — the immunity gate,
  `calculate_hit`'s own independent check, `is_grounded`, and half a dozen ability/item
  super-effective reads), `King's Shield` (a pure false positive — Aegislash's forme swap, gated on
  an unported ability, so nothing behind the name needed porting).
- Twelve `CodedMoveKind`s dispatched by `apply_coded` on the exported kind name (several moves
  share one — the four `WEATHER_HEAL` moves, the two `CURE_PARTY` ones): `REST`, `WEATHER_HEAL`,
  `PAIN_SPLIT`, `STRENGTH_SAP`, `BELLY_DRUM`, `HAZE`, `COURT_CHANGE`, `CURSE`, `TIDY_UP`,
  `PERISH_SONG`, `CURE_SELF`, `CURE_PARTY`. Curse's target takes a permanent quarter-HP chip every
  turn (new `ResidualOrder.CURSE` slot, between `SALT_CURE` and `BAD_DREAMS`); Perish Song's
  four-turn silent countdown ends in a mutual faint rather than a cleared status.
- Seven more: `KNOCK_OFF_ITEM`, `TRICK`, `SKILL_SWAP`, `ROLE_PLAY`, `ENTRAINMENT`, `WORRY_SEED`,
  `SIMPLE_BEAM` — the item- and ability-swap family. Needed `Database::is_fused_to` (a Mega Stone,
  Primal orb, Rusted Sword/Shield or Z-Crystal cannot be knocked off or traded away) and turned up
  a real, previously-invisible gap in the process, worth its own paragraph:

  **Mega Evolution was never refused, and was never played either.** `_resolve_mega_evolution`
  auto-Mega-Evolves a Pokemon holding its matching stone before a single move is ordered each turn
  — it changes the active Pokemon's stats (hence turn order), ability, and is permanent for the
  rest of the battle. Nothing in this port has ever touched that mechanic, and nothing refused a
  Pokemon holding a Mega Stone either, because `live_behaviour()`'s coverage sweep can see a literal
  `Item.GARCHOMPITE` but not `item_from_showdown(species.required_item)` — the runtime lookup
  `_forme_by_base_and_item()` is actually built from. Ninety-six items were invisible to the sweep
  this way. Fixed at the export layer: `live_behaviour()` now unions in `formes.mega_stones()`
  directly (`battle_sim/export_data.py`), and a separate, tiny table
  (`_forme_by_base_and_move`, exported as `move_gated_formes` — one entry, Mega Rayquaza gated on
  knowing Dragon Ascent with no item at all) is checked in `unsupported_pokemon` on its own, since
  it is not an item fact. `Species.fused_item` is a new exported field too — the Showdown display
  name (`"Charizardite X"`) resolved to this engine's own item name once, in Python, rather than
  reimplemented as a second normalisation table in Rust.

**Also done:** the four pseudo-weather rooms (`PseudoWeatherEffect`, a field-level `kind -> turns
left` map since more than one can stand at once, unlike weather or terrain). Three of the four —
Gravity, Magic Room, Wonder Room — have no gameplay effect anywhere in the Python beyond standing
up and ticking down; that is a deliberate simplification already documented, not a gap this port
introduced. Only Trick Room does anything mechanical (`order_actions`'s speed sort itself inverts,
not the speed stat — the one file in the whole Python that reads `PseudoWeather` directly).
Vacuity-checked: disabling the inversion alone, with the field state otherwise untouched, turns
177/300 of a deliberately slow-vs-fast matchup red.

**Also done: Wish, Healing Wish/Lunar Dance, Revival Blessing, Shed Tail, Future Sight/Doom
Desire.** Each got its own `Side`-level pending-effect field this engine didn't have before
(`wish_turns`/`wish_pending`, `healing_wish_pending`, `pending_substitute`,
`future_sight_turns`/`_attacker`/`_move`), and a shared `grant_switch_in_bonuses` helper (healing
wish's heal, a pending Shed Tail substitute) now runs from all three switch-in sites — the direct
`Action::Switch` handler, `send_out_replacement`, and `force_random_switch` — rather than just the
one that was obvious at first. Future Sight/Doom Desire store `(side, team_index)` for the
attacker, a stable identity across switches; landing after that attacker has switched out is an
explicit, documented `Refusal::Unported` rather than an attempt to thread an attacker index through
a damage-calc pipeline that assumes attacker == the active Pokemon on its side.

Two real bugs turned up while testing this batch, both found by the differential sweep rather than
by inspection:

- **Shed Tail's forced switch was silently never happening.** The initial assumption — that
  `needs_switch` is only ever consumed by the AI's own switch-choice — was wrong:
  `_resolve_pending_switches` in the Python's `turn.py` runs after *every* action resolves, not at
  some later checkpoint, so Shed Tail's substitute-then-switch has to happen inline, in the same
  action, not on the next `step()` call. Fixed by calling `send_out_replacement` directly from the
  `SHED_TAIL` dispatch arm. Covered by `test_shed_tail_forces_an_immediate_switch`.
- **Two same-turn pseudo-weather expiries logged out of order.** Python's `dict` preserves cast
  (insertion) order; a Rust `BTreeMap` sorts by key, so `Field::pseudo_weather` alphabetised
  `PseudoWeatherEnded` lines instead of preserving cast order — agreeing whenever the two happened
  to coincide alphabetically and diverging the rest of the time. Found by a 6000-battle sweep (1
  divergence, seed 3368). Fixed with a new `OrderedCounts` newtype (a `Vec<(String,i32)>` with
  dict-like update-in-place-else-append semantics) applied to `Field::pseudo_weather` and, since
  they share the identical shape and the identical latent bug, `Side::hazards` and `Side::screens`
  too. Covered by `test_two_rooms_fading_together_log_in_cast_order_not_alphabetical`, which scripts
  both cast orders deterministically (Trick Room and Wonder Room, cast turn 0 by two Pokemon of
  equal speed, never recast afterward) rather than relying on random play to hold still for the
  five turns both durations need to align.

**Also done: `Transform`, the last move in the database.** `Pokemon` gained the fields it had never
needed to keep past construction before this — `base_stats`, `nature`, `ivs`, `evs` — alongside the
`totals: StatTotals` folded from them, plus a `recompute_totals` method standing in for
`refresh_stats`. `Side` gained `transforms: BTreeMap<usize, FormSnapshot>`, keyed by team index
(the same stable-identity substitution `future_sight_attacker` already made for the Python's
`id(pokemon)`): a snapshot of everything Transform overwrites — base stats, nature, EVs, IVs,
types, ability, moves, PP — taken before the copy and popped back by `withdraw` on that Pokemon's
next switch-out. Stat stages are deliberately *not* in the snapshot: `withdraw` already zeroes them
on every switch-out, transformed or not, matching the Python's own separate reset rather than
Transform's own restore. HP's base stat is the one field Transform never touches, copied through
from the attacker's own `base_stats` rather than the target's, exactly as `BaseStats(HP=attacker...,
ATTACK=defender..., ...)` does. Refuses (`MoveFailed`) against a fainted target or a Pokemon already
on either side of a transformation, matching `id(attacker) in state.transforms or id(defender) in
state.transforms` with the same identity substitution. Covered by
`test_transform_copies_the_target_and_reverts_on_switch_out`; vacuity-checked by skipping the copy
entirely (leaving the `Transformed` log line in place) — a still-Rhydon Pokemon holding a move name
from Machamp's copied moveset has nowhere to put it, and the very first of 40 seeds came back a
refusal rather than a quiet pass.

### 4. Abilities — 76 left

**Done: the turn-order cluster — Prankster, Gale Wings, Triage, Mycelium Might, Surge Surfer,
Unburden, Quick Draw.** All seven live in `priority.py`, a single self-contained file, and all
seven landed at the two sites that already existed for this purpose: `inline::priority_bonus`
(Prankster/Gale Wings/Triage — each adds a fixed amount to a move's priority, folded together
rather than short-circuited since a Pokemon can only hold one ability but the Python adds all
three anyway), `turn::order_actions`'s sort key (Mycelium Might, which overrides the bracket-jump
component to 1 for the Pokemon's own status moves — sorting them last within their bracket rather
than changing the priority number itself), and `turn::effective_speed` (Surge Surfer's Electric
Terrain doubler, Unburden's post-consumption doubler). Quick Draw is the one genuinely new piece:
`_bracket_jump`, a 30% chance to move first within the bracket that this port had never touched —
Quick Claw and Custap Berry are the item two-thirds of that same Python function, left unwired
since both are still-unported *items*, whose live-item gate already refuses any Pokemon holding
one before turn one, so the two never reach this code in a playable scenario. The draw itself is
unconditional in the Python (taken and then discarded whenever Mycelium Might overrides the
result), which is exactly the kind of tape-order subtlety this project's own invariants exist to
catch — and every one of the seven was vacuity-checked individually, each turning at least one of
20-60 seeds red (several as an outright tape divergence rather than a quiet digest mismatch) the
moment its own effect was neutralised. Covered by `test_priority_abilities_reorder_moves_and_agree`,
`test_surge_surfer_and_unburden_double_speed_and_agree`, and
`test_quick_draw_occasionally_wins_the_bracket_and_agrees`.

**Also done: Sturdy.** The same survival clamp Endure already had in `turn::apply_damage`
(`_land_hit`'s Endure check), keyed off full HP instead of a volatile — `ON_BEFORE_HIT` fires
ahead of `_land_hit` in the Python, so Sturdy's own clamp is checked first, though the two can
never actually collide (a hit Sturdy has already reduced can't also satisfy Endure's threshold).
Found along the way: fixed-damage moves — Fissure, Seismic Toss, Super Fang and the rest —
bypass *both* Sturdy and Endure in this codebase; `_apply_fixed_damage` has no `ON_BEFORE_HIT`
emit and no `_land_hit` call, unlike an ordinary hit. Not a gap this port introduces, so left
alone, but it is exactly why the first version of this test (built around Fissure) agreed across
60 seeds while asserting zero saves — a real effect being measured in the one place it doesn't
apply, not a bug. Covered by `test_sturdy_survives_an_otherwise_lethal_hit_from_full_hp`;
vacuity-checked directly, turning seed 0 red (a fainting `DamageDealt` where Sturdy should have
clamped it to `SurvivedAtOneHp`) the moment the clamp was disabled.

**Also done: Skill Link, Serene Grace, Shield Dust, Scrappy, Mind's Eye, Synchronize, Pressure,
Steadfast, Corrosion.** Nine one-liners at sites that already existed, verified against a fresh
read of `moves.py`/`status_apply.py`/`damage.py` rather than the earlier memory-written summary
(which had mis-described Synchronize as a switch-in status copy — it isn't; see below):

- **Skill Link** (`turn::planned_hits`): a multi-hit move always rolls the top of its range,
  drawn from the tape not at all — the same shape `_planned_hits` gives it in the Python.
- **Serene Grace / Shield Dust** (`inline::tune_status_secondary`/`tune_stage_secondary`, called
  from `resolve_move` ahead of the substitute check): Serene Grace doubles a secondary status or
  stage effect's probability, Shield Dust blocks one aimed at its holder outright, no draw at all
  — both ahead of, and independent of, whether the move is behind a substitute.
- **Scrappy / Mind's Eye** (`Pokemon::effective_bypass`, unioned into `identify_bypass` at all
  three sites that read a bypass set: the immunity gate in `resolve_move`, `calculate_hit`'s own
  effectiveness check, and `resolve_future_sight`): a Normal or Fighting move from either ability
  bypasses a Ghost type's usual immunity to both.
- **Pressure** (`resolve_move`'s PP-spend block): a move that faces its holder — `DEFENDER_FACING`,
  the same set Protect already used — spends 2 PP instead of 1.
- **Steadfast** (`turn::apply_volatile`, the instant a `FLINCH` volatile is inserted): raises its
  holder's Speed by one stage on any flinch, not just a contact ability's own retaliation.
- **Synchronize** (`apply_main_status_from`, after the status is logged): a burn/paralysis/
  poison/toxic landing on its holder reflects straight back onto whoever inflicted it — only if
  the inflictor isn't already statused, and without an inflictor of its own on the way back, so
  the reflection can't re-trigger anything (including a second Synchronize).
- **Corrosion** (`apply_main_status_from`'s type-immunity check): the *inflictor's* ability, not
  the target's, lets a poison-family status through a Poison or Steel type's own immunity — the
  only clause the Python's status/type-immunity table has for either status.

Each of the nine now has its own differential test in `test_rust_battles.py`
(`test_skill_link_always_rolls_max_hits_and_agrees`,
`test_serene_grace_and_shield_dust_tune_secondary_chances`,
`test_scrappy_and_minds_eye_hit_ghosts_with_normal_and_fighting`,
`test_pressure_doubles_the_pp_cost_of_a_move_that_faces_it`,
`test_steadfast_gains_speed_from_flinching`,
`test_synchronize_mirrors_a_status_back_onto_its_inflictor`,
`test_corrosion_lets_a_poison_status_through_a_steel_type`), most of them deterministic (a
guaranteed-hit move with a probability forced to 0 or 1, so there is nothing for a seed to vary)
rather than a seed sweep. The Pressure test turned up a harness quirk worth remembering: a
Pokemon's moveset is padded to four slots by repeating its first move when it has fewer, and each
padded slot carries its *own* PP pool — a random chooser mostly rotates between four full pools
instead of ever emptying one, so forcing Struggle for real needs a round-robin chooser that
empties all four in lockstep (`_round_robin_chooser` in the test file).

Vacuity-checked individually — each ability's own code path disabled on its own, release build
rebuilt, only its own test run — and each one turned red on its own: Skill Link and Scrappy/Mind's
Eye as an outright tape divergence (the two engines drawing from the tape differently, or one
seeing `NoEffect` where the other lands a hit); Serene Grace, Shield Dust, Pressure, Steadfast,
Synchronize and Corrosion as a plain digest mismatch on the first turn a seed's random draws
reached them. Confirmed afterward with the full validation suite (`cargo build/test/clippy
--release`, `ruff check`, `pytest -q tests/test_rust_battles.py` — 122 passed, 1 skipped) and two
differential sweeps: 3000 status-slice battles with switching and abilities enabled (2958/3000
agreed, 42 refused — all the same documented Future Sight scope limit, 0 diverged) and 2000
plain-slice battles (2000/2000 agreed, 0 diverged). They're counted in the 144/220 table above and,
per invariant 2, now genuinely belong there.

**Also done: Water Bubble.** A correction to this doc's own earlier scoping, caught the same way
the Synchronize description above was: a fresh read showed it needs no new hook at all. It is a
pure `ON_DAMAGE_CALC` ability — 2x on its own Water moves, 0.5x on Fire damage taken, the same
`abilities::handle` dispatch Blaze/Torrent/Heatproof/Thick Fat already use — plus burn immunity
(`_STATUS_ABILITY_IMMUNITY`), which turned out to already be written into `inline::
ability_blocks_status` for Water Veil's sake and had simply never been reachable because Water
Bubble itself wasn't on a `PORTED` array yet. Two match arms plus one name added to `abilities::
PORTED` (48 → 49) was the entire change. Covered by
`test_water_bubble_doubles_its_own_water_and_halves_fire_taken`; all three clauses (the offense
boost, the defense reduction, the burn immunity) vacuity-checked individually, each turning its own
half of the test into a digest mismatch on its own. 145/220 in the table above now.

**Also done: section 6's `ON_BEFORE_MOVE` hook, and eleven of the twelve abilities it unblocked.**
`hooks::ability_before_move`, called from `resolve_move` right after the accuracy roll and before
the effectiveness/immunity gate — the same spot the Python emits it, for a damaging move or one
that targets the opponent directly (`damaging or move.target in _DEFENDER_FACING_TARGETS`). None of
its handlers draw from the tape; the ability/move-type pair alone decides them. Ported: Volt Absorb,
Water Absorb, Earth Eater (a quarter heal, or `AbsorbBlocked` at full HP), Motor Drive, Lightning
Rod, Storm Drain, Sap Sipper, Well Baked Body (a fixed stat bump, logged only if the stage actually
moved — the raw `change_stat_stage`, not the `apply_stage_changes` wrapper other callers want),
Flash Fire (a flag, not a heal or a boost — see its own paragraph below), Levitate's move-cancelling
half (its grounding half was already written and simply unreachable, same shape as Water Bubble's
burn immunity above), and Soundproof (keyed on the move's own `sound` flag, not its type). Dry Skin
is the one name left out of the twelve — it has two more behaviors this hook doesn't cover (a Fire
vulnerability multiplier and a weather-driven heal/chip), scoped out of this batch rather than
half-ported.

Also required: catching a check the Python has that this hook's first draft didn't — `if not
move.effects and not move.force_switch: MoveFailed` sits one line above where Python emits
`ON_BEFORE_MOVE`, and without it, Electrify (an Electric-type status move whose own effect is
unmodelled — `effects` is empty in the exported data) falsely triggered Motor Drive's boost before
this had a dedicated test. Caught by the pre-existing `test_the_ported_abilities_agree_too` sweep on
the very first run with the new hook live, not by a test aimed at the cluster; the regression test
that followed is `test_effectless_moves_never_falsely_trigger_an_absorber`.

Flash Fire's boost turned up a second, more expensive bug, this one past every dedicated test and
caught only by a 2000-battle plain-slice sweep (seed 1232, turn 30): the boost was first written
pushing 6144 onto `attack_mods_4096`, which reads like the obvious list for "1.5x this Pokemon's own
damage" and is exactly what Water Bubble and Huge Power use — but the Python's own `boost_fire`
pushes onto `pre_screen_mods_4096` instead, folding in much later in the chain (after STAB and the
type multiplier, not onto the attack stat before the base-damage division). The two are not
interchangeable: a Rock/Ground Golem's Mind Blown landed for 36 in Python and 35 here, one point of
rounding apart, purely from which stage of the chain the same nominal 1.5x entered at. Fixed by
moving the push to `pre_screen_mods_4096`; a fresh 5000-battle plain-slice sweep came back clean
afterward. Worth remembering for anything else in the remaining abilities that reads like a plain
attack-stat boost — check which list the Python actually pushes onto before assuming.

Ten differential tests cover the batch: `test_type_absorbing_abilities_cancel_the_move_and_agree`
(the eight-ability table), `test_effectless_moves_never_falsely_trigger_an_absorber` (the Electrify
regression), `test_flash_fire_activates_once_and_boosts_fire_moves_afterward`, and
`test_levitate_cancels_ground_moves_and_soundproof_blocks_sound`. Every one of the eleven abilities'
own code paths, plus the empty-effects guard, was vacuity-checked individually (rename the ability
string or zero the multiplier, rebuild, run only that ability's test, confirm it goes red, revert) —
thirteen checks in total, each a tape divergence or a plain digest mismatch depending on whether the
disabled clause changes what the tape needs or only what a fold computes. Confirmed with the full
validation suite and three differential sweeps: 3000 status-slice (2952/3000 agreed, 48 refused, the
same documented Future Sight limit, 0 diverged), 2000 plain-slice (2000/2000, 0 diverged), and the
5000-battle plain-slice re-check above. 156/220 abilities now.

**Also done: section 6's `ON_FAINT` hook, and all three abilities it unblocks.** A fresh read of the
Python turned up a correction to this doc's own earlier claim (same pattern as Water Bubble,
Synchronize and Magic Bounce above): `ON_FAINT` is not emitted "at every existing `fainted()` check
site" — it is emitted from exactly *one* line in the whole Python, inside `_apply_damage`'s ordinary
variable/listed-power path, right after the defender's own `Fainted` log line and before Destiny
Bond's retaliation or recoil/drain. Residual damage, recoil, confusion self-hits and fixed-damage
moves (`_apply_fixed_damage`, Seismic Toss's own path) never reach it, so a KO from any of those must
not trigger these abilities either. `hooks::ability_on_faint` is called from that one site in
`apply_damage` and nowhere else. Ported: Moxie (+1 Attack), Beast Boost (+1 to whichever of its five
stats is highest — not always Attack, and matched to Python's `max()` tie-break, which keeps the
*first* equal element rather than the last one `Iterator::max_by_key` would), and Soul Heart (+1 Sp.
Atk) — a third name the doc's own list had missed; it binds the identical `_bind_ko_boost` helper
Moxie does; As One (Glastrier) and Chilling Neigh use the same helper too but stay with the Formes
cluster below, since both are locked to a fusion forme this port hasn't built the primitive for yet.

Covered by `test_moxie_beast_boost_and_soul_heart_boost_after_a_ko` and, for the scope correctness
that is the entire point of this hook, `test_ko_boosting_abilities_dont_fire_from_a_fixed_damage_faint`
— a Seismic Toss OHKO that must produce no boost at all, run as its own dedicated regression rather
than left to a sweep to notice by accident. Vacuity-checked directly: each of the three abilities'
own match arm renamed in turn (three checks, each a plain digest mismatch — this engine's own
`StatStageChanged` where Python's silence has nothing to compare it to), and, for the scope check
itself, adding a second call to `ability_on_faint` from `apply_fixed_damage`'s own fainting branch —
the exact mistake this doc's superseded claim would have produced — which turned the regression test
red immediately. Confirmed with the full validation suite and two differential sweeps: 3000
status-slice (2956/3000 agreed, 44 refused, the same Future Sight limit, 0 diverged) and 5000
plain-slice (5000/5000, 0 diverged). 159/220 abilities now.

**Also done: section 6's `ON_SWITCH_OUT` hook, and both abilities it unblocks.** Emitted from one
site, `turn::switch_out` (called from three places: a voluntary switch, `send_out_replacement`'s
faint-forced one, and `force_random_switch`'s Whirlwind/Roar/Dragon Tail one), and — this is the
part worth getting exactly right — *only* when the outgoing Pokemon did not faint. The event's own
Python comment says so directly ("not emitted for fainted switches"), and `switch_out` guards the
call the same way, before any of `withdraw`'s own resets run — so a handler sees the outgoing
Pokemon's HP, status and stages exactly as they stood the moment it left, not zeroed or cleared yet.
`switch_out` needed a `log` parameter threaded through (it previously returned only a nickname
string) to have anywhere to put a handler's own log lines. Ported: Regenerator (a third heal,
`heal_by`'s existing `Healer::Ability` shape — no new helper needed) and Natural Cure (clears
whatever status is active, guarded on there being one to clear). Libero and Protean bind this same
event too, but stay unported: both need a second, stateful hook into `ON_BEFORE_MOVE` (capture the
original type at switch-in, shift to the used move's type once, restore it here) that is its own
piece of design, not a one-liner riding along.

Covered by `test_regenerator_heals_a_third_on_switch_out`, `test_natural_cure_clears_status_on_switch_out`,
and — the scope-correctness test this hook's own history argues for — `test_regenerator_and_natural_cure_dont_fire_on_a_fainted_switch`:
a Regenerator holder KO'd outright must not "heal" its own corpse back to positive HP when the
auto-replacement switches it out next turn. That third test is not academic — vacuity-checking it by
dropping the `!fainted()` guard produced exactly that: a still-standing, healed Golem with positive
HP in the state digest where Python's last word on it was `Fainted`. The other two abilities'
own match arms were vacuity-checked the same way as every other batch, each a plain digest mismatch.
Confirmed with the full validation suite and two differential sweeps: 3000 status-slice (2960/3000
agreed, 40 refused, the same Future Sight limit, 0 diverged) and 5000 plain-slice (5000/5000, 0
diverged). 161/220 abilities now.

**Also done: section 6's `ON_TURN_START` hook, and Protosynthesis/Quark Drive.** The largest single
ability unit ported so far, and the reason it earned its own hook rather than riding along with
`ON_SWITCH_IN`/`ON_RESIDUAL` alone: the same `_paradox_evaluate` binds to *three* events
(`ON_SWITCH_IN`, `ON_TURN_START`, `ON_RESIDUAL` at `ResidualOrder.PARADOX`, right after the field's
own duration tick and before the weather chip) with no event-specific behaviour, so `hooks::
evaluate_paradox` is one function called from all three sites — the existing `on_switch_in`, the new
`ability_on_turn_start` (called once per turn, before `order_actions` reads `effective_speed` —
which is the entire reason `ON_TURN_START` needs to exist at all, not just the residual pass), and a
new call at the top of each side's `residuals()` block. Two new `Pokemon` fields
(`paradox_boost: Option<String>`, `paradox_from_booster: bool`), reset on switch-out alongside Flash
Fire's own flag. The boost itself has three destinations, matching the Python's three separate
sites: a 1.3x offense-or-defense modifier in `abilities::handle` (deliberately not on that file's
own `PORTED` array, for the same reason Flash Fire's boost half isn't — three other events decide
whether it ever has anything to read), a 1.5x Speed multiplier in `effective_speed`, and neither at
all when the boosted stat is something a plain damage roll or turn-order comparison never touches.
`switch_out` needed the same `log` threading as `ON_SWITCH_OUT` did. Booster Energy's own half of
the mechanism (a held-item activation path that outlives the field condition) is written faithfully
but is dead code for now — the item itself is still refused as unported, confirmed directly by
holding one and watching the refusal fire before this code is ever reached — exactly the same shape
Water Bubble's burn immunity and Levitate's grounding clause were before their own abilities joined
a `PORTED` array.

Four differential tests: `test_protosynthesis_and_quark_drive_boost_the_highest_stat_and_agree` (the
1.3x damage case, both abilities, against a Drought/Electric Surge partner whose own switch-in sets
the field before `ON_TURN_START` ever asks), `test_quark_drive_speed_boost_reorders_turns` (Tauros at
110 base Speed loses to a 112-Speed Lycanroc unboosted and wins at a boosted 165 — a strict
comparison, not a coin flip a tie could still lose), and
`test_paradox_boost_fades_silently_when_the_field_ends` (Drought's sun lasts 5 turns; by the last of
7, Machamp's Tackle has quietly reverted to its unboosted amount with no second `ParadoxActivated`
ever logged — the Python does not announce losing the boost the way it announces gaining it).
Vacuity-checked directly: the two activation conditions, the offensive damage multiplier, the Speed
multiplier, and the field-ended clearing, each disabled in turn (five checks), each turning its own
test red — the activation checks as a plain digest mismatch on the very first turn (`ParadoxActivated`
missing entirely), the multipliers and the clearing as damage or turn-order mismatches once the
activation itself was intact. Confirmed with the full validation suite and two differential sweeps:
3000 status-slice (2956/3000 agreed, 44 refused, the same Future Sight limit, 0 diverged) and 5000
plain-slice (5000/5000, 0 diverged). 163/220 abilities now.

**Also done: the nine no-new-hook one-liners.** Sheer Force (`ON_DAMAGE_CALC` 1.3x power plus the
secondary stripped in `inline::tune_status_secondary`/`tune_stage_secondary`, the same site Serene
Grace and Shield Dust already use), Solar Power (`ON_DAMAGE_CALC` 1.5x Special in sun, plus its own
`ON_RESIDUAL` eighth-HP chip at `ResidualOrder.WEATHER` — the sandstorm chip's own band, not the
`WEATHER_ABILITY` band Ice Body and Dry Skin use), Thermal Exchange and Toxic Debris (`ON_AFTER_HIT`,
the same defender-reaction match block Stamina/Justified/Weak Armor/Berserk already share — Toxic
Debris deliberately has no fainted guard, unlike every stat-bump reaction beside it, reproduced as
written rather than brought in line with its neighbours), Cursed Body (`ON_AFTER_HIT`, a 30% chance
to disable the attacker via `turn::start_disable`, now `pub`), Wonder Guard and Wind Rider
(`ON_BEFORE_MOVE`, the hook this doc's own earlier revision hadn't yet noticed both of these
actually need — Wonder Guard reads the *plain* type chart against the defender's raw types, not
`effective_bypass`, which means it reproduces a real quirk: a defender-facing status move whose
type isn't super effective gets blocked here too, not just damaging ones; Wind Rider's own log line
carries `source="justified"` rather than `"wind_rider"`, a copy-paste slip in the Python reproduced
rather than fixed), Liquid Voice (`power::type_override`, checked after the `-ate` abilities and
before every by-name case — no power boost rides along, unlike the `-ate` abilities' own), and
Poison Puppeteer (`apply_main_status_from`, right after the Synchronize-reflection block — confuses
whatever its holder just poisoned). `Move` gained a `wind: bool` field it had never needed to read
before, mirroring `sound`.

This batch also caught and fixed a real ordering bug in the *previous* commit's own `ON_TURN_START`
work, found by this batch's own 3000-battle status-slice sweep (seed 2559, turn 21): Python's
`_apply_residuals` calls `state.bus.emit(ON_RESIDUAL, ...)` once per side, but that emit reaches
*every* handler on the bus, not only the current side's own Pokemon — and Paradox's handler, unlike
Solar Power's, never checks `context.actor is pokemon`. A Quark Drive on side 1 therefore activates
during side 0's own residual emit, before side 0 has taken so much as a weather tick, not during
side 1's own turn through a per-side loop. `residuals()` called `evaluate_paradox` inside the
existing per-side loop, interleaved with that side's own weather/item/status chips, which put a
side-1 activation *after* a side-0 item chip where Python puts it before. Fixed by hoisting both
sides' Paradox evaluation into its own pass, before either side's weather/item/status chips run at
all — matching the emit-reaches-everyone semantics directly instead of approximating them.

Nine differential tests, one per ability except `test_toxic_debris_and_thermal_exchange_react_to_being_hit`
(both share the same `ON_AFTER_HIT` site and setup). Vacuity-checked directly: every ability's own
match arm or check disabled in turn — twelve checks in total, counting Sheer Force's and Solar
Power's two destinations each — each a plain digest mismatch or a tape divergence depending on
whether the disabled clause changes what the tape needs (Sheer Force's nullification, Cursed Body's
draw) or only what a fold computes. The residual-ordering fix was vacuity-checked by reverting to
the old per-side call and re-running the 3000-battle sweep, which reproduced the exact seed
2559 divergence before the fix went back in. Confirmed with the full validation suite and two
5000-battle differential sweeps: status-slice (4908/5000 agreed, 92 refused, the same Future Sight
limit, 0 diverged) and plain-slice (5000/5000, 0 diverged). 172/220 abilities now.

The remaining 48, grouped and roughly ordered by what unblocks the most:

- **Dry Skin**: the one name left over from the `ON_BEFORE_MOVE` cluster above — its own absorb half
  would be a one-line addition to `hooks::ability_before_move` (`"DRY_SKIN" if move_type ==
  "WATER"`, heal-style like Volt Absorb), but the Fire-vulnerability multiplier
  (`ON_DAMAGE_CALC`, a one-liner in `abilities::handle`) and the weather-driven heal/chip
  (`ON_TURN_END`, not built yet — see below) both have to land in the same pass, per invariant 2.
- **Libero / Protean**: bind `ON_SWITCH_OUT` (restore original type) alongside a stateful
  `ON_BEFORE_MOVE` handler (shift to the used move's type, once per switch-in) — see the note above.
- **Ability interaction (9)**: Mold Breaker/Teravolt/Turboblaze (one shared mechanism — an
  attacker-side flag read wherever a target-ability immunity is checked), Neutralizing Gas, Mummy,
  Trace, Imposter, Magician, Pickpocket. Imposter/Trace should reuse the existing Transform/
  ability-copy machinery (Skill Swap etc., section 3) rather than new infrastructure.
- **Bounce / field interaction (8)**: Magic Bounce, Mirror Armor, Good as Gold, Queenly Majesty,
  Dazzling, Guard Dog, Bulletproof, Infiltrator.
- **Activity gating (4)**: Slow Start, Truant, Wimp Out, Emergency Exit (the latter two need an
  `ON_ACTION_RESOLVE`-adjacent forced-switch check, same call site as the existing Magic
  Guard/Life Orb suppression logic).
- **Trapping (3)**: Arena Trap, Shadow Tag, Magnet Pull — affects `legal_actions`, not `step`, and
  `legal_actions` isn't ported yet (section 8). Do this cluster alongside that port rather than
  half-implementing the damage-irrelevant half now.
- **Gen 9 type-power boosters (3)**: Orichalcum Pulse, Hadron Engine, Electromorphosis — same shape
  as the already-ported weather/terrain setters (Drought/Drizzle etc.).
- **Primal weather (3)**: Delta Stream, Primordial Sea, Desolate Land — pair with the primal orb
  items (section 5), since one is inert without the other.
- **Formes (12)**: Stance Change, Zen Mode, Schooling, Shields Down, Power Construct, Zero to Hero,
  Tera Shift, Disguise, Multitype, RKS System, Battle Bond, As One (Glastrier)/Chilling Neigh — "a
  small architecture change" (see "Rough sizing"); build one forme-swap primitive and reuse it for
  all twelve plus Mega Evolution (section 7), rather than reinventing it per ability.
- **The butler**: Nine Lives, and the last-stand machinery in `damage_apply`. Bespoke to this bot,
  no Python event-bus registration to crib from beyond `pokemon.py`'s `_revive`/`_last_stand`.
  Isolated — doesn't block or get blocked by anything else in this list, do it whenever.

**Also done: section 5's opening items batch** — Quick Claw, Custap Berry, Leppa Berry, Chesto
Berry, Lum Berry, the four weather rocks (Damp/Heat/Icy/Smooth Rock), and the four terrain seeds
(Electric/Grassy/Misty/Psychic Seed). 13 items, no new hook needed, following the doc's own
recommended order for section 5.

Quick Claw and Custap Berry both live in `turn::bracket_jump`, the same function Quick Draw already
occupied — Quick Claw checked first (a 20% draw to jump the bracket, taken and consumed whether it
wins or not, unconditionally like Quick Draw's own draw), then Custap Berry (no draw at all: a flat
guarantee once the holder is at or under a quarter of its max HP). Custap's own consumption needed
`bracket_jump` and `order_actions` promoted from `&State`/`&Pokemon` to `&mut State`, which meant
rewriting `order_actions`'s per-side loop to re-fetch `state.sides[side].active_pokemon()` at each
point of use rather than holding one `actor: &Pokemon` borrow across the whole iteration — the
`the_move: &'a Move` returned by `move_in_slot` borrows from `db`, not the Pokemon, so it stays
valid across the re-borrows. Compiled clean on the first attempt. Leppa Berry sits in the PP-spend
block in `resolve_move`: the instant *this* spend brings a slot to zero, it restores up to 10 (a new
`PpRestored` log event), consumes the berry, and logs — a slot already at zero before the spend
takes the Struggle branch instead and never reaches the check. Chesto Berry and Lum Berry both live
in `apply_main_status_from`, checked last, right after the Poison Puppeteer block: Lum cures any
status, Chesto only sleep, both curing on the same turn the status lands and logging
`StatusCleared{clearance: "berry"}`. The four weather rocks extend an ability-set or move-set
weather from 5 turns to 8, via a new `inline::rock_for_weather` helper wired into both sites that
set weather (`hooks::on_switch_in`'s ability arm and `turn.rs`'s `WeatherEffect` handler) — mirroring
the existing Heat Rock/Terrain Extender duration pattern already used elsewhere.

The four terrain seeds turned out to need two independent trigger paths, not one, and finding the
second was the real work of this batch. The seed's own `ON_SWITCH_IN` binding (`EventPriority.ITEM`,
after every ability including Paradox's own) covers a Pokemon switching in after its terrain is
already up, or one that both sets and eats its own seed on the same switch-in (Electric Surge
holding Electric Seed) — this path was implemented first and its own dedicated test passed
immediately. But a 3000-battle status-slice sweep (`--abilities --switches`) turned up 12
divergences, all the same shape: a `StatStageChanged{source: "seed"}` Python's own trace had that
Rust's didn't. A fresh read of `_set_terrain_from_ability` and `_apply_field_effect` (the ability and
move terrain-setters, `abilities.py`/`field_apply.py`) showed why: both sweep *every* side's active
Pokemon and call `consume_terrain_seed` directly, synchronously, the instant terrain actually
changes — completely independent of the event bus. This is what lets a seed fire for a Pokemon that
never switches in at all: an already-standing seed holder, or (the exact shape the sweep found) an
ally auto-replacing a fainted Pokemon on a later turn while a Misty Surge lead's own switch-in sets
the terrain, with nobody else switching in to trigger the seed's own `ON_SWITCH_IN` binding. Fixed
by adding `hooks::consume_terrain_seeds_on_terrain_change`, a sweep of both sides called from both
terrain-setting sites (the ability arm in `hooks::on_switch_in`, and the `TerrainEffect` arm in
`turn::resolve_move`) right after terrain is set and logged. The two paths don't double-consume:
`consume_terrain_seed_for_side`'s own item-match guard is naturally idempotent, and matches Python's
own (a seed already spent by the sweep is simply not `pokemon.item` any more by the time the bus's
lower-priority `ON_SWITCH_IN` handler gets to check it in the same emit).

Six dedicated tests: one each for Quick Claw (60-seed sweep, seed-loop check), Custap Berry
(deterministic — Dragon Rage's fixed 40 damage four times over brings a 180-HP Rhydon to 20, under
the 45-HP quarter-mark), Leppa Berry (a 5-PP move spent from the same slot for exactly 5 turns),
Chesto/Lum Berry together (Spore for the deterministic sleep-only case, a Thunder Wave seed-loop for
Lum's any-status case), weather rocks (both the ability- and move-triggered sites, asserting
`weather_turns_left == 7` one turn after an 8-turn set — the counter itself ticks once in the same
turn it's set, so it reads one below its nominal duration, not the raw duration itself), and terrain
seeds (a same-turn ability+seed case). A seventh test,
`test_terrain_seed_fires_for_a_pokemon_that_never_switched_in`, was added specifically to cover the
cross-side sweep the first terrain-seed test's own scenario couldn't reach (both Pokemon there are
leads switching in together, so the sequential per-side `ON_SWITCH_IN` path alone was enough to pass
it) — a move-set terrain (so the setter's own switch-in predates the terrain by a full move
resolution) against a seed holder that never switches in at all. Vacuity-checked directly, one item
at a time: Quick Claw and Custap Berry (renaming each string in `bracket_jump` — Quick Claw broke
the tape's own draw count, Custap Berry broke turn order), Leppa Berry (renaming its string in the
PP-spend block), Chesto Berry and Lum Berry (renaming each arm separately in the cure match), both
weather-rock call sites independently (forcing `rock` to `None` at each), and the terrain-seed match
arm plus — separately — both calls to `consume_terrain_seeds_on_terrain_change` (commented out
together, since disabling either one alone leaves the other still covering the single-lead test;
the seventh test is what catches this pair). Every one turned its own test red on its own, then
clean again on revert.

Confirmed with the full validation suite (`cargo build/test/clippy --release`, `ruff check`, 150
pytest cases, 1 pre-existing unrelated skip) and two differential sweeps after the terrain-seed fix:
3000 status-slice battles with `--abilities --switches` (2971/3000 agreed, 29 refused — the same
documented Future Sight/Doom Desire scope limit, 0 diverged) and 5000 plain-slice battles (5000/5000
agreed, 0 diverged). Before the fix, the same 3000-battle status-slice sweep showed 12 diverged; the
fix brought that to 0 without changing the refusal count. 46/193 items now.

### 5. Items — 147 left

(This section's header disagreed with the status table in an earlier revision of this doc — 77 vs.
193−33=160. That has since been corrected as items were ported; 147 = 193−46 is current.)

Recommended order, cheapest/most-unblocking first:

1. **Standalone, no new hook needed — done**: Quick Claw and Custap Berry (`turn::bracket_jump`),
   Leppa Berry (the PP-spend block in `resolve_move`), Chesto and Lum Berry
   (`apply_main_status_from`), the four weather rocks (`hooks::on_switch_in`'s ability arm and
   `turn.rs`'s `WeatherEffect` handler), and the four terrain seeds (`hooks::on_switch_in`'s own
   `ON_SWITCH_IN` binding plus the new `consume_terrain_seeds_on_terrain_change` sweep called from
   both terrain-setting sites) — see the batch write-up above. Air Balloon is the one name left in
   this bucket: its grounding check already exists at `field.rs:43`, only the pop-on-hit half is
   new, and needs `ON_TURN_END` or `ON_AFTER_HIT`, both already available once section 6 lands.
2. **Choice trio** (Choice Band/Scarf/Specs, need `choice_locked_move`): verify first whether any
   move-locking already exists generically — invariant 4 already caught one false "Choice Scarf is
   ported" claim, so don't assume.
3. **Focus Sash**, and Endure's other item cousins: unblocked by generalising the existing
   `ON_BEFORE_HIT` survival-clamp dispatch (Sturdy's clamp in `turn.rs`) to also check held items,
   not a new hook.
4. **Life Orb, Eject Button/Pack, Red Card**: `ON_ACTION_RESOLVE`-adjacent, same call site as the
   existing Magic Guard suppression check.
5. **Plates (17) + memories (17) + drives (4)**: pair directly with the abilities that read them —
   Multitype needs plates, RKS System needs memories — so do this batch once those two abilities
   (section 4) land. The drives (Genesect only) have no ability dependency and can ride along.
6. **Mega Stones / Primal orbs / Griseous Orb & Core (86)**: inert until Mega Evolution itself
   exists (section 7) — do as part of that work, not before. Roughly 16 of these are custom/
   fictional stones specific to this project's data set (`RAICHUNITE_X`, `GARCHOMPITE_Z`, etc. —
   see `battle_sim/formes.py:mega_stones()`), not standard Showdown items; treat them identically
   to real ones.
7. **Everything not yet bucketed above**: Heavy-Duty Boots, Light Clay, Terrain Extender, Loaded
   Dice, White Herb, Mental Herb, Mirror Herb, Clear Amulet, Covert Cloak, Shed Shell, Punching
   Glove, Adrenaline Orb, Booster Energy, Power Herb, Ultranecrozium Z — none of these have been
   individually scoped against their hook dependency yet; do that before starting each one rather
   than assuming it's a plain one-liner.

### 6. Hooks still missing

None of these exist as a dispatch point in Rust at all — not even a stub. Rust never ported
Python's generic `EventBus`; each existing hook (`on_switch_in`, `on_after_hit`,
`residual_before_status`/`residual_after_status` in `hooks.rs`) is a bespoke function called from a
fixed point in `turn.rs`, and the five below should follow that same shape rather than introducing
a generic event-bus abstraction — that would be new architecture the rest of the codebase doesn't
use, for a cost the direct-call pattern already pays cheaply elsewhere.

Build in this order — each one sized to unlock the next-biggest ability/item cluster above:

1. **`ON_BEFORE_MOVE` — done.** `hooks::ability_before_move`, called from `resolve_move` right
   after the accuracy roll, before the effectiveness/immunity gate. Unblocked eleven of the twelve
   type-immunity/absorption abilities (Dry Skin held back, see section 4) and Air Balloon's float
   is ready to ride along once items catch up to it. Magic Bounce turned out not to need this hook
   at all — a fresh read placed its bounce check at a separate, earlier site in `moves.py` (before
   the Protect check, not alongside the immunity gate), a correction to this doc's own earlier
   claim caught the same way Water Bubble's and Synchronize's were.
2. **`ON_FAINT` — done.** A correction to this doc's own earlier claim: the Python emits it from
   exactly *one* site, not "every existing `fainted()` check site" — see its write-up in section 4.
   Unblocked Moxie, Beast Boost and Soul Heart (a third name this doc's list had missed). Battle
   Bond's post-KO half stays with the Formes cluster, since the forme-swap it triggers needs the
   primitive section 7 hasn't built yet.
3. **`ON_SWITCH_OUT` — done.** Called from `turn::switch_out`, symmetric to the existing
   `on_switch_in`, and only for a switch that is not a faint replacement. Unblocked Regenerator and
   Natural Cure — see the write-up in section 4. Zero to Hero stays with the Formes cluster; Libero
   and Protean need a second, stateful hook into `ON_BEFORE_MOVE` alongside this one.
4. **`ON_TURN_START` — done.** Called once at the top of the turn, before `order_actions` reads
   `effective_speed`. Unblocked Protosynthesis and Quark Drive — see the write-up in section 4;
   Booster Energy rides along once items catch up, its own logic already written and waiting on the
   item joining a `PORTED` array the same way Levitate's grounding clause once waited.
5. **`ON_TURN_END`** — called in the existing residual pass, symmetric to `field::tick_field`/
   `tick_side`. Lowest priority of the five — nothing in the current ability gap needs it
   standalone; it exists for later item interactions (Air Balloon's pop).

`ON_BEFORE_HIT` (survival clamps) and `ON_ACTION_RESOLVE` (Life Orb, Magic Guard) already have
partial, working dispatch — Sturdy's clamp and Magic Guard's suppression check respectively — that
just needs generalising to cover more abilities/items at the same site, not a hook built from
scratch.

### 7. Z-moves, megas, formes

Decided in scope: a Gen 7 Anything Goes replay corpus (`data/ag_replays/`, feeding
`tools/build_ag_teams.py`) is the intended self-training data, and 89.5% of Pokemon slots across the
teams it builds need Mega Evolution, Primal Reversion, Z-Moves or an Arceus plate — this is not a
dimension the bot can do without.

**Mega Evolution / Primal Reversion / Ultra Burst — done.** Python's `_resolve_mega_evolution` is
one generic table-driven forme-swap (`formes.mega_forme`/`apply_forme`), not 83 special cases —
Mega Stones and Primal orbs share the same `_forme_by_base_and_item()` table, keyed off the vendored
species data's own `requiredItem`/`baseSpecies` fields. `battle_sim/export_data.py` now exports that
table directly (`mega_formes`, `ultra_burst_formes`, plus a `move_gated_formes` fix — it used to
throw the forme name away) and each species' own `regular_ability` in this engine's name, so Rust
carries none of Python's own filtering logic (`_is_transformed_forme`/`_is_playable`/`_source_forme`)
— only one filter Python's export can't supply: whether *this* engine has ported the forme's ability
yet (`rust/src/formes.rs::playable`), so a still-unplayable forme (Sableye-Mega/Magic Bounce) stays
refused rather than silently mega-evolving into an ability with no dispatch. `resolve_forme_changes`
runs once per turn per side, in `turn::step`, in the same slot `ability_on_turn_start` already
occupies, right before `order_actions` reads speed. Desolate Land/Primordial Sea/Delta Stream ride
the existing ability-weather table; Delta Stream's own damage effect (neutralizes a Flying-type's
weaknesses) is ported in `power::strong_winds_negation`, called from both places type effectiveness
feeds a real hit. Shadow Tag/Arena Trap/Magnet Pull were pulled forward from section 4's Trapping
cluster and ported as verified no-ops (Gengar-Mega's ability is Shadow Tag) — confirmed by reading
`battle_sim/` end to end that all three are legal-actions-only, never consulted by `step()`.
Six dedicated tests, all vacuity-checked; a targeted 1300-battle sweep over every playable
mega/primal/ultra-burst pairing plus the still-refused Sableye control (0 diverged); the usual broad
sweeps unaffected. See the commit for the full write-up. **Not done**: Zygarde's Power Construct is
a different forme-swap shape (HP-triggered, no item — belongs with section 4's other HP-forme
abilities, Stance Change/Zen Mode/Schooling/Shields Down, not this one), and Multitype/RKS
System/Arceus-plates/memories (needed for Arceus, extremely common in the same corpus) are next.

**Z-Moves — done.** A Z-move is a flag on the ordinary move action in Python (`z_move: bool` on the
existing `USE_MOVE` action, same slot, same PP), not a new action type, so `step()` needed no
`legal_actions` port to execute one — `Action::Move` gained the flag, and `replay.rs`'s
`parse_action` now builds it for `"zmove:"` the same way it already did for `"move:"`. Z-move data
itself needed a new export: `get_all_z_moves()` is deliberately kept out of `get_all_moves()` (a
Z-move is never an ordinary move slot), so `rust/src/zmoves.rs` reads a new `rust/data/zmoves.json`
(same `Move` shape as `moves.json`, via the same `move_json()`) rather than needing any move data of
its own. `zmoves::z_move_for` mirrors `battle_sim/zmoves.py` exactly: a generic crystal (recognised
by its own placeholder power, not by type — signature crystals share types with generic ones) takes
its power from the fixed 9-band table keyed on the *base* move's power, its category/contact from the
base move, and everything else from the crystal's own template; a signature crystal (15 hand-mapped
pairings, not derivable from data) is just its own template with accuracy forced to never miss —
including any secondary effect it carries (Aloraichium Z's Stoked Sparksurfer always paralyzes).
Status-move Z-effects are not modelled in Python either (preserved, not fixed). `has_used_z_move`,
once per battle per side, never consumes the crystal itself (permanently fused, same as a Mega
Stone) — and `legal_actions` itself stops offering the Z-move variant once used, so the gate is
exercised end to end rather than merely defended against in `step()`. Three dedicated tests
(generic, signature, once-per-battle), all vacuity-checked (DIVERGED, not a quiet pass, with the
substitution disabled); a targeted 1300-battle sweep across every generic type used in the corpus
plus all 15 signature crystals plus a mismatched-type control (0 diverged); broad sweeps unaffected.

**Multitype/RKS System/plates — done.** Simpler than Mega Evolution turned out to be exactly right:
not a forme swap at all, just `hooks::sync_type_from_item` permanently mutating `pokemon.types` in
place — mirroring Python's own `_bind_item_type_shifter` closely enough that it needed the *same*
two call sites Paradox already has (`ON_SWITCH_IN` and the Paradox-priority slot of the residual
pass), not a query-time override the way Roost's is. Judgment's own type-selection (`power::
plate_type`) already existed from an earlier batch; only the ordinary 1.2x same-type boost was new,
and a sweep caught a real gap while adding it: only six of the seventeen plates
(Iron/Earth/Spooky/Pixie/Splash/Stone) have a `_bind_type_boost` binder registered in
`mechanics/items.py` today — the other eleven are live items with a real type-tracking effect but no
damage boost *in Python*, so Rust ported exactly that split rather than "correctly" boosting all
seventeen (`items::PORTED_TYPE_ONLY_PLATES` names the eleven, `hooks::PORTED_RKS_MEMORIES` the
seventeen Memories, which have no boost binder for any of them). Three dedicated tests, all
vacuity-checked separately (the type-sync and the damage boost are independent code paths and each
needed its own check); a targeted 1300-battle sweep across every plate and memory, Arceus and
Silvally alike, including the no-item control (0 diverged).

**Trapping cluster (Arena Trap/Shadow Tag/Magnet Pull) — done, see above; Ghost/Shed-Shell escape
and the rest of `legal_actions`** — genuinely deferred to section 8, since nothing else in this
cluster has any `step()`-level effect to port.

### 8. Integration — after parity

1. Decide the interface. PyO3 in-process is the obvious one (the search calls `step` millions of
   times; a subprocess per call is hopeless). `rust/src/lib.rs` is already a library crate.
2. The search needs `legal_actions` too, which currently lives only in Python
   (`engine/choices.py`) — it reads locks, traps, charges and `needs_switch`.
3. Swap the search over behind a flag, then re-run the AI validation (mirror-match win rate) to
   confirm the engine change did not move play strength.
4. Keep the differential running in CI against the Python for as long as both exist.

## Known gap: cross-side switch-in ability ordering

Found by a broad sweep during the Multitype/RKS System/plates batch (seed 870 of a 8000-battle
status+abilities+switches run) — not caused by that batch, which is otherwise clean, but newly
reachable once it landed (the scenario needed a Pokemon holding an RKS System Memory *not* on an
RKS System Pokemon, previously refused outright as an unported item).

`turn::step`'s `_send_out_leads` (`turn.rs:524-536`) sorts the two leads by descending
`effective_speed` once, then runs each one's entire `on_switch_in` before starting the other's — the
comment there ("the slower weather-setter's weather is the one that stands") is itself the
documented intent. Python's own event bus does not work this way for a switch-in ability: `ON_
SWITCH_IN` handlers from *both* leads sit on the same bus, sorted by `(priority, registered_at)`
globally, not "everything from the faster Pokemon, then everything from the slower one." The sweep's
own case: a slow Golem's Protosynthesis (Booster Energy, no speed dependency) and a fast Tauros's
Electric Surge both fire on turn 0 — Python logs the Golem's `ParadoxActivated` before the Tauros's
`TerrainSetByAbility` (registration order — side 0 registers before side 1), Rust logs them in speed
order instead (Tauros first, since it's faster). Both engines agree on *what* happens, only the
*order* of these two specific log lines differs, so nothing about the ultimate board state actually
diverges here — but `compare()` is event-for-event, correctly refusing to treat that as a pass.

Not fixed here: a real fix needs `on_switch_in`'s per-side dispatch restructured into the same kind
of single, priority-sorted pass across both sides that `apply_damage_calc`'s own two-entry walk
already uses — a own investigation and batch, not a one-line change, and not part of what this
batch's own tests exercise. Flagged so it is not lost, not fixed opportunistically mid-batch.

## Rough sizing

Sections 1–3 are done. Section 6's first four hooks — `ON_BEFORE_MOVE`, `ON_FAINT`,
`ON_SWITCH_OUT`, `ON_TURN_START` — are all done and have already unblocked their clusters outright
(eleven of `ON_BEFORE_MOVE`'s twelve abilities, all three of `ON_FAINT`'s, both of
`ON_SWITCH_OUT`'s, both of `ON_TURN_START`'s — see section 4), and the nine no-new-hook one-liners
are done too. `ON_TURN_END`, the last of the five hooks, is the one left with nothing in the current
ability gap to unlock standalone — it exists for Air Balloon's pop, an item, so it makes more sense
alongside section 5 than ahead of it. Section 4 is the largest remaining block at 48 abilities but
most of it needs pairing with section 5's items (plates, memories, mega stones) or section 7's
forme-swap primitive rather than being one-liners in isolation — call it a session from here.
Section 5 (147 items left) is one, mostly following section 4's abilities in to reuse their
plumbing (Multitype/plates, RKS System/memories, primal weather/primal orbs, Mega Evolution/mega
stones).
Section 7 (Z-moves/megas/formes) is done: Mega Evolution/Primal Reversion/Ultra Burst, Z-Moves,
Multitype/RKS System/plates, and the Shadow Tag/Arena Trap/Magnet Pull no-ops are all in the tree
and verified — see each one's own write-up above, and the cross-side switch-in-ordering gap one of
them surfaced (a pre-existing issue, not fixed as part of any of these batches). Integration
(section 8) is what is left, one session plus whatever the AI re-validation turns up — plus, now,
the ordering gap above if it turns out to matter once `legal_actions` and real AI play exercise
switch-ins far more than this port's own sweeps have.

The tail is not uniform: absorption abilities and formes are each a small architecture change, and
the butler's revival mechanic has no reference outside this codebase.

**Immediate next step for whoever picks this up**: section 5's opening batch (Quick Claw, Custap
Berry, Leppa/Chesto/Lum Berry, the four weather rocks, the four terrain seeds) is done — see its
write-up above. What's left in section 4 needs pairing with items (plates, memories, mega stones) or
the not-yet-built forme-swap primitive, so stay in section 5 rather than jumping back: next up per
its own recommended order is Air Balloon (the last name in the "no new hook" bucket — needs
`ON_TURN_END` or `ON_AFTER_HIT` for its pop-on-hit half, both available since section 6), then the
Choice trio (verify no generic move-locking already exists before assuming it's unported — invariant
4 already caught one false claim here), then Focus Sash and Endure's item cousins (generalising the
existing Sturdy clamp in `turn::apply_damage` to also check held items).
