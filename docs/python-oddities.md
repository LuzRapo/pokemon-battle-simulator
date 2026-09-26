# Oddities found in the Python engine while porting it

Everything here was found by making a second engine agree with this one, move by move. Each entry
is something the Rust port **reproduces deliberately** — agreeing with the reference is the job,
and quietly "fixing" one of these would have shown up as a divergence and cost an afternoon.

They are written down because the port is not the place to decide which are bugs. Some plainly
are. Some are deliberate simplifications the Python documents in its own comments. A few are
judgement calls about a format nobody else is playing. Changing any of them changes live battle
outcomes, so each wants its own decision.

Status: nothing here has been changed. The Rust engine matches all of it.

---

## Almost certainly bugs

### Magnitude can never be Magnitude 10

`engine/power.py`, `_magnitude_power`. The roll is `rng.random_integer(1, 20)`, and
`random_integer` is **exclusive** of its upper bound, so it yields 1–19. The threshold table ends
`(19, 110)` with a fallback of 150, and `roll <= 19` is therefore always true.

Magnitude's 150-power outcome is unreachable. Verified by exhausting the roll: the only powers the
move can produce are 10, 30, 50, 70, 90 and 110. Its average power is about 6% lower than intended,
and its ceiling is a third lower.

The fix is `random_integer(1, 21)`. Worth checking the intended distribution at the same time — the
games weight Magnitude 4 through 10, and the current table gives 110 a 2-in-19 share where the
games give Magnitude 9 a 10% share and Magnitude 10 a 5% one.

### Effect Spore can never put anything to sleep

`mechanics/abilities.py`, `_bind_effect_spore`:

```python
status = _SPORE_STATUSES[context.rng.random_integer(0, len(_SPORE_STATUSES) - 1)]
```

`_SPORE_STATUSES` is `(POISON, PARALYSIS, SLEEP)`, so the index range wanted is 0–2. With an
exclusive upper bound, `random_integer(0, 2)` yields 0 or 1 — poison or paralysis, never sleep.
Sleep is the outcome that actually makes Effect Spore threatening.

The fix is `random_integer(0, len(_SPORE_STATUSES))`. Note that `_bind_starf_berry` two files over
writes exactly that and is correct, which is what makes this look like a slip rather than a choice.

### A confused Pokémon that does not hurt itself is announced as unable to act, then acts

`engine/moves.py`, `_confusion_allows_acting`. On the two-thirds where the confusion roll does not
produce a self-hit, the function logs `CantAct(reason="confused")` and then returns `True`, so the
move goes off anyway.

Either the log line is spurious or the return is wrong. Everything downstream behaves as though the
Pokémon acted, and the battle log says it could not — so a player reading the log sees a turn that
did not happen. The mechanic underneath is right (a third of the time you hit yourself); it is the
announcement that is wrong.

### Wise Glasses and Muscle Band are 1.5x, not 1.1x

`mechanics/items.py`:

```python
ITEM_BINDERS[Item.WISE_GLASSES] = _bind_choice_attack_item(Item.WISE_GLASSES, Category.SPECIAL)
ITEM_BINDERS[Item.MUSCLE_BAND] = _bind_choice_attack_item(Item.MUSCLE_BAND, Category.PHYSICAL)
```

They share Choice Band and Choice Specs' binder, which applies a flat `6144/4096` — 1.5x. In the
games both are 1.1x. They also, unlike the Choice items, carry no lock, so as written they are
strictly better than Choice Band with no drawback at all.

This one is live in the bot now and affects every set carrying either item.

---

## Deliberate simplifications the Python already documents

These are in the engine's own comments as known approximations. Listed so the Rust side's matching
behaviour is not mistaken for a porting error.

| Where | What |
|---|---|
| `_friendship_power` | Return and Frustration are both flat 102 — friendship is not modelled, and every real set is built at whichever end its move wants. |
| `_magnitude_power` | (see above — the *approximation* is fine, the bound is not) |
| `_beat_up_power` | One hit carrying the summed power, rather than one hit per healthy ally. |
| `_bind_parental_bond` | A flat 1.25x rather than a real second strike. Total damage is right; breaking a Substitute then striking, rolling secondaries twice, and being blocked twice by Sturdy are all not modelled. |
| `_fixed_amount`, `PSYWAVE` | Taken at its mean (the user's level) rather than rolled between 0.5x and 1.5x, because the function is handed no RNG. |
| `_bind_flower_gift` | The stat boost without the forme change. |
| `_every_other_move_used` | Last Resort's condition read off spent PP rather than a record of which moves have been used. Refuses exactly the cases the real rule refuses at the start of a battle. |
| `_bind_wiki_berry` | No confusion from a disliked nature. |
| `Pokemon.is_grounded` | Intrinsic only — types, item, ability. Gravity would change the answer. |
| `_log_revival` | Marked KNOWN DEFECT in the source: the *log line* can print more revivals than exist and out of order. The mechanic underneath is sound. |

---

## Choices worth a second look, but defensible

### Leaf Guard only blocks statuses that came from a move

`_apply_main_status` takes `field` as an optional argument, and `_ability_immune_to_status` reads
`field is not None` before it will let Leaf Guard block anything. Only the move path passes it —
the contact-status abilities (Static, Flame Body, Poison Point, Effect Spore), Poison Touch, Toxic
Chain and the Toxic and Flame Orbs all call it without one.

So in blazing sun a Leaf Guard holder cannot be put to sleep by Spore, but a Toxic Orb still
poisons it and a Static holder can still paralyse it. That asymmetry does not exist in the games.

It may be deliberate — passing `field` everywhere is a wider change — but nothing says so.

### `payload_overrides` is an if-chain of early returns

`engine/power.py`. A move matching an earlier clause never reaches a later one. The visible
consequence: **Facade carried by an `-ate` ability holder gets `ignore_burn` and not the 1.2x
boost**, because the chain returns at Facade. Body Press, the Psyshock family, Photon Geyser and
Foul Play all sit above Facade and shadow the boost the same way.

Whether that is intended is unclear. It reads like a dispatch table that grew an extra clause.

### Rough Skin and Iron Barbs both report as Rough Skin

`_bind_rough_skin` is shared and hard-codes `ability=Ability.ROUGH_SKIN` in its log entry, so an
Iron Barbs holder's chip damage is announced under the wrong ability's name. Cosmetic.

### Starf Berry's boost is logged with `source="seed"`

`_bind_starf_berry` passes `source="seed"` to `apply_stage_changes`, which is the terrain seeds'
source, not the berry's. Cosmetic, but it makes the log ambiguous.

### Recoil is logged as the amount requested, not the amount dealt

`_apply_recoil_and_drain` and `_bind_rocky_helmet` both log the number they asked for rather than
what `apply_damage` returned. A Pokémon at 3 HP taking 30 recoil is reported as having taken 30.
Compare `_bind_rough_skin`, which logs the dealt amount. Inconsistent between the two.

### Fury Cutter's run is broken by *any* other move, including a failed one

`engine/moves.py:193` resets `rolling_hits` for any move that is not in `ESCALATING_MOVES`, and it
is set where the move is announced — so a move that goes on to fail still ends the run. That is
correct for the games. Noted because the reset's *position* is easy to get wrong and produces a
subtle, rare divergence.

---

## Format decisions, recorded so they are not mistaken for bugs

- **No status clause.** `_CLAUSED_STATUSES` is empty, with a comment explaining that the only
  `|rule|` lines across 5,001 real Gen 7 Anything Goes replays are HP Percentage Mod and Endless
  Battle Clause. The clause machinery is retained but dead.
- **Meowfred's Monocle** is Technician in an item and deliberately not from the games.
