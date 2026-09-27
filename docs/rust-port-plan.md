# Finishing the Rust port

Written to survive a context compaction. Everything a fresh session needs to carry this to 100% and
then replace the Python engine.

## Where it stands

Branch `feature/vectorised-engine`. Sections 1, 2 and 3 (charges, volatiles, and the remaining
coded moves) are all done — every `CodedMoveKind` is ported, and with it every move in the
database. Section 4 (abilities) is underway: the turn-order cluster (seven abilities), Sturdy, and
a nine-ability one-liner batch (Skill Link, Serene Grace, Shield Dust, Scrappy, Mind's Eye,
Synchronize, Pressure, Steadfast, Corrosion) are in the tree and now verified — see the batch
write-up below. Its dedicated tests, individual vacuity checks (all nine, each confirmed to turn
red on its own), and a confirming differential sweep have all been run since the last update to
this doc; per invariant 2 the batch now belongs on the "done" side.

| | done | total |
|---|---|---|
| moves | 843 | 843 (100%) |
| abilities | 156 | 220 (71%) |
| items | 33 | 193 |
| volatiles | 18 | 18 — every volatile in the database is ported |

The counts above are confirmed against `rust/target/release/replay --coverage rust/data`, run with
a release build for this update:
`{"abilities":{"live":220,"ported":156},"items":{"live":193,"ported":33},"moves":{"playable":843,
"refused_by_cause":{},"total":843}}` — an empty `refused_by_cause` for moves, matching the 100%
row above. The stale code comment at `rust/src/turn.rs:318` ("220 abilities and 110 items are
live") has been corrected to 193 for items.

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

The remaining 64, grouped and roughly ordered by what unblocks the most:

- **Dry Skin**: the one name left over from the `ON_BEFORE_MOVE` cluster above — its own absorb half
  would be a one-line addition to `hooks::ability_before_move` (`"DRY_SKIN" if move_type ==
  "WATER"`, heal-style like Volt Absorb), but the Fire-vulnerability multiplier
  (`ON_DAMAGE_CALC`, a one-liner in `abilities::handle`) and the weather-driven heal/chip
  (`ON_TURN_END`, not built yet — see below) both have to land in the same pass, per invariant 2.
- **Unblocked by `ON_FAINT` (2)**: Moxie, Beast Boost.
- **Unblocked by `ON_SWITCH_OUT` (2)**: Regenerator, Natural Cure.
- **Unblocked by `ON_TURN_START` (2)**: Protosynthesis, Quark Drive.
- **No new hook needed, one-liners at existing sites (~10)**: Sheer Force, Solar Power, Thermal
  Exchange, Toxic Debris, Soul Heart, Wonder Guard, Wind Rider, Liquid Voice, Cursed Body, Poison
  Puppeteer.
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

### 5. Items — 160 left

(This section's header disagreed with the status table in an earlier revision of this doc — 77 vs.
193−33=160. 160 is correct; see the note in "Where it stands" above.)

Recommended order, cheapest/most-unblocking first:

1. **Standalone, no new hook needed**: weather rocks (Damp/Heat/Icy/Smooth Rock — extend the
   existing weather-duration logic), terrain seeds (hook off `ON_SWITCH_IN`, already dispatched),
   status/recovery berries (Chesto, Custap, Leppa, Lum), Air Balloon (its grounding check already
   exists at `field.rs:43`; only the pop-on-hit half is new, and needs `ON_TURN_END` or
   `ON_AFTER_HIT`, both already available once section 6 lands).
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
2. **`ON_FAINT`** — called at every existing `fainted()` check site in `turn.rs`/`field.rs`/
   `damage.rs` (there are several; a single shared call is fine as long as every site calls it).
   Unblocks Moxie, Beast Boost, Battle Bond's post-KO half.
3. **`ON_SWITCH_OUT`** — called from `turn::switch_out` (`turn.rs:673`), symmetric to the existing
   `on_switch_in`. Unblocks Regenerator, Natural Cure, Zero to Hero.
4. **`ON_TURN_START`** — called once at the top of the turn, before actions resolve. Unblocks
   Protosynthesis/Quark Drive (and Booster Energy once items catch up).
5. **`ON_TURN_END`** — called in the existing residual pass, symmetric to `field::tick_field`/
   `tick_side`. Lowest priority of the five — nothing in the current ability gap needs it
   standalone; it exists for later item interactions (Air Balloon's pop).

`ON_BEFORE_HIT` (survival clamps) and `ON_ACTION_RESOLVE` (Life Orb, Magic Guard) already have
partial, working dispatch — Sturdy's clamp and Magic Guard's suppression check respectively — that
just needs generalising to cover more abilities/items at the same site, not a hook built from
scratch.

### 7. Z-moves, megas, formes

`zmove:` actions are refused where actions are parsed. Mega evolution, Z-moves and forme changes are
a whole action/state dimension the port has not touched. **Check what the bot actually uses before
deciding how much is needed** — if the search/team-generation code (`battle_sim/search.py` or
wherever sets are built) never assigns a mega stone or a Z-move-carrying set, refusing this
dimension permanently is a legitimate scope cut, not a gap, and it's cheaper to check that first
than to build it and find out afterward.

If it turns out to be in scope: Mega Evolution needs a forme-swap primitive (stat/type/ability
override, triggered pre-move) — build it once and reuse it for the twelve non-mega forme-change
abilities in section 4 and for Zygarde/Terapagos/Palafin. Z-moves are a separate, smaller unit
(`battle_sim/zmoves.py` is 131 lines) — a generic type-crystal power/effect upgrade plus 15
hand-mapped signature crystals.

### 8. Integration — after parity

1. Decide the interface. PyO3 in-process is the obvious one (the search calls `step` millions of
   times; a subprocess per call is hopeless). `rust/src/lib.rs` is already a library crate.
2. The search needs `legal_actions` too, which currently lives only in Python
   (`engine/choices.py`) — it reads locks, traps, charges and `needs_switch`.
3. Swap the search over behind a flag, then re-run the AI validation (mirror-match win rate) to
   confirm the engine change did not move play strength.
4. Keep the differential running in CI against the Python for as long as both exist.

## Rough sizing

Sections 1–3 are done. Section 6's first hook, `ON_BEFORE_MOVE`, is done and has already unblocked
its cluster (eleven of its twelve abilities — see section 4). The remaining four hooks
(`ON_FAINT`, `ON_SWITCH_OUT`, `ON_TURN_START`, `ON_TURN_END`) are still worth building ahead of
resuming section 4 at large, for the same reason the first one was: small on their own, each turns
a blocked cluster into one-liners. Section 4 is the largest remaining block at 64 abilities but most
of it is one-liners once the rest of section 6 lands — call it a session and a half from here.
Section 5 (160 items) is one, mostly following section 4's abilities in to reuse their plumbing
(Multitype/plates, RKS System/memories, primal weather/primal orbs, Mega Evolution/mega stones).
Section 7 (Z-moves/megas/formes) is a scope question before it's a sizing question — check bot
usage first. Integration (section 8) is last, one session plus whatever the AI re-validation turns
up.

The tail is not uniform: absorption abilities and formes are each a small architecture change, and
the butler's revival mechanic has no reference outside this codebase.

**Immediate next step for whoever picks this up**: `ON_BEFORE_MOVE` is done, along with eleven of
the twelve abilities it unblocked (Dry Skin held back — see its own note in section 4). Build the
next hook, `ON_FAINT`, which unblocks Moxie and Beast Boost — small, and the same "one hook first"
logic applies for the same reason it did last time.
