# Finishing the Rust port

Written to survive a context compaction. Everything a fresh session needs to carry this to 100% and
then replace the Python engine.

## Where it stands

Branch `feature/vectorised-engine`. Sections 1, 2 and 3 (charges, volatiles, and the remaining
coded moves) are all done — every `CodedMoveKind` is ported, and with it every move in the
database. Section 4 (abilities) is underway: its first cluster, the seven that change turn order
rather than a stat, is done — see the batch write-up below.

| | done | total |
|---|---|---|
| moves | 843 | 843 (100%) |
| abilities | 134 | 220 (61%) |
| items | 33 | 193 — see note |
| volatiles | 18 | 18 — every volatile in the database is ported |

The items total jumped from 110 to 193 mid-section: 96 Mega Stones (and Primal orbs, Rusted
Sword/Shield) turned out to have real behaviour — `_resolve_mega_evolution` — that the coverage
export had never been able to see, for reasons worth reading in section 3's own notes below. They
are correctly refused now rather than silently wrong, which is why the *ported* item count did not
move even though total climbed by 83.

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

### 4. Abilities — 86 left

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

The remaining clusters, roughly in order of value:

- **Damage/ordering one-liners at sites that already exist**: Sturdy, Shield Dust, Serene Grace,
  Skill Link, Steadfast, Scrappy/Mind's Eye, Soundproof, Bulletproof, Pressure, Mold Breaker/
  Teravolt/Turboblaze, Corrosion, Synchronize, Natural Cure, Regenerator.
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
