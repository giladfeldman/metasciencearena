## gate fix (2026-09-22) — the two splits are not the same instrument, and the gate now says so

**No task set was minted and no published score changed.** This supersedes the
2026-09-21 entry below, whose fix was refuted before it shipped.

### What the 2026-09-21 attempt got wrong

It held the synthesised `clean` residual to a base-rate tolerance
(`clean_rate_tolerance: 0.02`) and reported the arena green. A three-provider
consult (degraded: 2 seats, 1 provider — Sonnet and Fable; Sol hit a usage limit
mid-read) found two defects, both reproduced here by measurement before being
accepted.

**1. The tolerance could not detect the defect at any value.** The v5 asymmetry
lives entirely in tier 1, where the clean rate is `1.0000` on BOTH splits. Tiers
2–4 are `1.0000` both and tier 5 is `0.0000` both. So the 0.21% global gap was a
pure mixing artifact — strip the two extra tier-1 items and the two rates are
identical to twelve decimal places. A tolerance of 0 would have passed it too.

**2. The evidence used to reject v6 was confounded.** The tier decomposition
pooled 5 players on revealed against 2 on private, and the 3 Claude players exist
on revealed ONLY, so it largely measured "LLMs minus tools". Matched to the two
players that ran both splits, deduplicated against the byte-identical copies in
`runs/v3/_revisions/`:

| | pooled (wrong) | matched |
|---|---:|---:|
| tiers 1–5 | +0.087 | **−0.125** |
| tier 6 | +0.090 | **−0.005** |

The sign reverses. **Do not cite the +0.087 figure**; it appears in the entry
below and in earlier handoffs and is retracted.

### What is actually true

The quantity that makes the two splits different instruments is item **KIND**
composition, not the clean residual — and it comes from a SECOND seed-dependent
draw that the 2026-09-21 analysis never found:

```
generator.py:706   kind = rng.choice(["nhst", "effect_size"]) ...
generator.py:707   if v3 and ... and rng.random() < 0.5:
```

`scorer.py:463` charges every unmatched gold item as a false negative, so a player
that reads only significance tests — statcheck — has its per-task recall capped at
`nhst_items / all_items`. MEASURED on v5 with the real seeds, mean over 36 tasks:

| | revealed | private | gap |
|---|---:|---:|---:|
| NHST-only recall ceiling | **0.6516** | **0.6901** | **0.0385** |

Per-tier `nhst/all`: t1 11/28 vs 11/26 · t2 8/12 vs 7/12 · t3 5/12 vs 7/12 ·
t4 18/18 both · t5 9/12 both · t6 53/81 vs 51/81. **Tiers 2 and 3 have identical
item counts and still differ in composition**, so no count-based or rate-based
check can see this. Worst single cell: `(2,3)` is 5 NHST items revealed against 2
private, same total of 6.

### What changed

`clean_rate_tolerance` is REMOVED. In its place the arena DECLARES the gap and the
gate stops claiming what it cannot verify:

- a third verdict, **`PARITY PARTIAL`**, so a run that leaves something unchecked
  cannot print as a full pass — the headline names it, not just a note;
- `NOT ENFORCED:` lines carrying the arena's declared reason verbatim, plus the
  per-cell item-kind table, so a reader sees the size of the gap rather than a
  verdict;
- declared injected kinds and per-cell task counts stay EXACT and still fail hard
  (24 injected items on every one of 40 seeds — that guarantee is real);
- `independent_holdout` now reports `PARTIAL` too, for the same reason: it already
  said it does not enforce parity, and its headline should have agreed.

**The declaration is scoped to an EXPLICIT version list, `[v3, v4, v5]`.** v6 is
deliberately absent, so the version that fixes the generator gets the full strict
check automatically — nobody edits this file and nobody has to remember. An
exemption phrased as "until the generator is fixed" never expires, because nothing
forces anyone to notice that it has been.

### Still open

`generator.py:706` and `:707` are both unfixed here — v3/v4/v5 are frozen or
superseded and a shared generator is append-only. A separate session is
implementing both behind a `v6` flag. Minting v6 is now justified in a way it was
not on 2026-09-21: two halves that cap a real published tool differently by 0.0385
are a scientific reason to mint a task set, where a 1.2% item-count difference was
not.

**Consequence for anyone publishing meanwhile: do NOT run statcheck on the v5
private split and publish it beside its revealed row.** statcheck is NHST-only, has
no v5 private record, and the ceiling gap runs in its favour on private — the
apparent gain would be composition, not capability. ESCImate reads both kinds and
is far less exposed (measured v5 cross-split delta +0.0008, 95% CI [−0.172, +0.174]).

### Also fixed

`error_categories → abstain_on_checkable` described the scorer backwards. It said a
declined item "is scored as neither and leaves the verification denominator";
`scorer.py:481-483` does `if cell == "abstain": continue` then `v_total += 1`, so a
decline STAYS in the denominator and is charged, and it is the separate `abstain`
cell that leaves it. Found by the session running ESCImate 0.7.10 and verified
here against the arithmetic rather than the comments. It mattered: the single
declined item on `t-tier2-d3-1-s5feceb66` is the ENTIRE published difference between
ESCImate 0.7.6 and 0.7.10 (−0.009259), so our own documentation explained the only
visible difference between two versions of a real tool exactly backwards.

## gate fix (2026-09-21) — SUPERSEDED BY THE 2026-09-22 ENTRY ABOVE, READ THAT FIRST

> **This entry's conclusion was WRONG and its fix never shipped.** The
> `clean_rate_tolerance` it describes was a false green, and the score-gap
> evidence it used to reject v6 was confounded. Both are corrected above. The
> entry is kept because the reasoning that failed is part of the record.

### (original text follows)

## the parity gate had been checking a superseded task set

**No task set was minted and no published score changed.** This entry records a
gate defect, a generator defect found through it, and why only the first was
fixed now.

### The gate was looking at the wrong version

`tools/check_parity.py` hardcoded the literal `"v1"` in both call sites — the
`--all` loop that `sciencearena-qa` Phase 5 runs as a gate, and the single-arena
default. On every arena that had minted a successor it therefore checked a
SUPERSEDED task set and never the one being served, silently, because a
superseded task set parity-matches perfectly well. Fixed 2026-09-17; on the next
run the gate went red here.

`framework/tests/test_parity.py::test_stats_extraction_arena_passes_parity`
carried the SAME hardcoded `"v1"`, so the suite stayed green throughout. A gate
and its only test sharing one blind spot is why the replacement check
(`framework/tests/test_parity_checks_current_version.py`) asserts the gate LOOKS
AT the right version, and carries a mutation control: injecting the historical
`return "v1"` must fail on at least one arena, which it does on three.

### What the gate found, measured across the ladder

| version | parity | clean items revealed / private |
|---|---|---|
| v1 | OK | 132 / 132 |
| v2 | OK | 132 / 132 |
| v3 | **FAIL** | 142 / 137 |
| v4 | **FAIL** | 137 / 136 |
| v5 (current) | **FAIL** | 139 / 137 |

The break enters at v3 (released 2026-08-21) and had been live in the served task
set for 27 days.

**Root cause, `generator.py:707`:**

```python
if v3 and kind == "nhst" and templates.get("nhst_with_es") and rng.random() < 0.5:
```

That coin flip decides whether a density slot yields ONE gold item or TWO, so
per-task item cardinality is seed-dependent. It is gated on `v3`, which is
exactly when the break appears. Measured firings: v1/v2 = 0 on both splits, v4 =
3 revealed vs 2 private, v5 = 4 vs 2 — and the firing delta equals the clean
delta number-for-number. Eighty lines below it the two deterministic tier-1
controls carry the comment *"so both splits carry it equally and parity is
unaffected"*: the author was parity-aware there and missed it here.

### Why the taxonomy is NOT affected

All eight DECLARED deception kinds are in exact parity on every version v1–v5
(24 injected items on both splits at v3/v4/v5). Generating v5 over 40 seeds gives
exactly 24 on every single seed, so `count_tolerance: 0.0` on the declared kinds
is real and stays. Only `clean` drifts — and `clean` is not a declared kind at
all, it is the residual label `framework/parity.py::_mistake_labels` synthesises
for items carrying no injected mistake, so its count is (total gold items − 24)
and it tracks cardinality.

### Why no v6 was minted

v3 is the only version with published records on both splits AND a parity break,
so it is the one place the question is empirically answerable. Decomposing its
cross-split score gap by tier:

| | item counts | revealed − private |
|---|---|---|
| tiers 1–5 (**identical** counts both splits) | matched | **+0.087** |
| tier 6 (the **only** asymmetric tier) | 85 / 80 | **+0.090** |

The same gap. Individual matched tiers swing from +0.407 (tier 2) to −0.229
(tier 4) with identical item counts on both sides. Cross-split disagreement is
driven by which tasks the seed draws, not by the item asymmetry — minting v6
would invalidate 324 records across 9 players and re-run them without touching
the variance that actually matters.

### What changed instead

`framework/parity.py` now separates the residual from the declared kinds. An
arena that knows its cardinality is seed-dependent declares
`clean_rate_tolerance` and the residual is compared as a BASE RATE (clean / all
items); an arena that declares nothing keeps the exact-count behaviour unchanged,
so this is a per-arena opt-in and never a fleet-wide relaxation. Only
stats-extraction-v1 declares it.

The value is derived, not fitted. Over 40 seeds at v5 the clean COUNT ranges
133–144 (sd 2.5, worst pairwise relative gap 7.6%) while the clean BASE RATE
stays inside 0.8471–0.8571 — a worst gap of 1.17%. Our two real seeds sit at
0.8528 and 0.8509, a gap of 0.21%. `clean_rate_tolerance: 0.02` is that measured
envelope with ~1.7× margin, and remains a real gate: at a ~0.853 base rate it
fails the moment either split drops below ~0.836.

The gate prints the raw counts and both base rates on every run, pass or fail. A
tolerance that hides its own inputs is how slow drift becomes invisible.

### DEFERRED — still open

**`generator.py:707` is not fixed.** The coin flip should be index-cycled like
the kind assignment, so cardinality stops depending on the seed. It is NOT
corrected here because v3, v4 and v5 are frozen or superseded, their committed
gold and published records were computed from this generator, and a shared
generator is append-only — editing it in place would rewrite frozen history and
`framework audit`'s drift gate would catch it. The fix belongs in whatever task
set is minted next for a scientific reason; it does not justify one on its own.

### Also fixed in passing

`clean_rate_tolerance` had to be added to
`framework/contract/schemas/arena_manifest.schema.json`, whose `parity` block is
`additionalProperties: false`. Without it, arena discovery raised on every read of
this `arena.yaml` and `generator_fingerprint` reported `*** DRIFT ***` with
`n=0` on all five versions — a broken read wearing a finding's clothes. After the
schema entry, all five fingerprints MATCH at n=72 (36 revealed + 36 private).

## task set v5 (2026-08-29) — the answer key graded a claim the passage never printed

v4 printed the APA inequality (`p < .001`, `p > .999`) whenever the true p falls
beyond the 3-dp grid, and stamped the relation onto gold as `fields.p_rel`.
**Three templates never interpolated that stamp**, so the key could grade a claim
the reader never saw:

| template | v4 line | renders the stamp? |
|---|---|---|
| `tier2_notation.yaml` | `*t*({df1}) = {value}, *p* = {p_no_lead}.` | **no** — hardcodes `p =` |
| `tier2_notation.yaml` | `t({df1}) = {value}, p < {p_threshold}.` | **no** — hardcodes `p <` |
| `tier3_formatting.yaml` | `t({df1})
= {value}, p &lt; {p_threshold}` | **no** — hardcodes `p &lt;` |

The two `{p_threshold}` forms are worse than a wrong relation. `p_threshold` is
the **constant** `".05"` that `_instantiate_template` supplies, so the drawn p is
never rendered at all and the gold asserted a number absent from the input — the
same shape as the ghost CIs `_ci_is_printed` removed in v3.

**MEASURED before anything changed**, with every one of the 112 nhst gold items
located. (An earlier survey located 55 of them and reported its count of 6 as an
explicit lower bound; this is the full count.)

| | v3 | v4 | **v5** |
|---|---:|---:|---:|
| nhst gold items located | 108/108 | 112/112 | 104/104 |
| printed relation ≠ gold relation | 0 | 7 | **0** |
| printed number ≠ gold number | 8 | 11 | **0** |
| printed claim FALSE while key says `ok` | 44 | 5 | **0** |

Private split, last row: v3 41, v4 5, **v5 0**. The v3 column is the control that
makes the v5 zeros readable — the same locator finds the clamp defect v4 was
minted to remove, so a zero is a measurement rather than a broken search.

**The 5 invert the scoring.** `t(44) = -0.02, p < .05` against a true p of .984 is
a decision error, exactly what statcheck exists to find. The key says `ok`, so the
tool that flags it is charged a `false_alarm` and the tool that misses it is
credited a `hit`.

**The rejected repair is the tempting one.** Re-labelling those 5 `inconsistent`
in the gold needs no re-run and leaves the passage printing `p = .001` for a true
p of 7.3e-06 — precisely the manufactured error v4 exists to remove. v5 changes
the TEXT so the printed claim is true and the key grades that claim, which is why
it is a new task set and a full five-player re-run.

**What changed**

- `*p* {p_rel} {p_no_lead}` — the inequality now reaches the reader.
- `p {p_alpha_rel} {p_threshold}` and its `&lt;`-escaped tier-3 twin — the relation
  is measured against alpha (`<` below .05, `>` above, `=` at exactly .05) and the
  gold's `p` becomes the .05 that is actually printed. `p > .05` is ordinary APA
  prose, so the notation variation tier 2 exists to test is preserved rather than
  deleted.
- v5 wording lives under a separate `nhst_v5` key in each template file, the same
  shape as tier 6's `section_titles_v3`.
- One scoring rule moved, and it is a correctness fix: `_label_verification` read
  a `p < X` claim's significance as `X < .05`, which is False for `X == .05` and
  would have read the commonest significance claim in psychology as a null one.
  Now `X <= .05`. Provably inert for v1–v4 — a `<` relation there requires the
  true p below .0005, and `_round_p`'s floor then prints .001, never .05.

**The append-only invariant, proved rather than argued.** All four earlier task
sets regenerate their committed `_ground_truth.json` byte-for-byte — envelope
text AND every gold field — and `framework audit` reports all five fingerprints
OK at 72 tasks each, so both splits are covered.

**Unchanged and checked, not assumed:** 24 deceptions across all 8 kinds, 15
items labelled `inconsistent`, and 0 truthful items with `coherent=False` —
identical to v4. The clean control class is 80 against a floor of 40.

**KNOWN, MEASURED, AND NOT WORTH A TASK SET.** From v4 onward a printed
**inequality** is a *perfect* signal that an item is truthful — v5 revealed has 47
inequality-printing NHST items and none is a deception (private: 49 and none),
because `_apply_apa_p` is called only on truthful items and every deception falls
through to `p = <value>`.

**That sounds worse than it measures.** The cue is perfectly precise and buys
nothing. Over all 36 revealed tasks with a perfect extractor, across all 16
constant strategies the cue's partition admits:

| strategy | primary |
|---|---:|
| best cue-**using** (`inequality → ok`, `else → ok`) | 0.8077 |
| best cue-**free** (`always ok`) | 0.8077 |
| **gain from knowing the cue** | **+0.0000** |

The reason is structural, not a coincidence of this seed: the cue marks items
where `ok` is *correct*, and a player maximising without recomputation already
answers `ok` everywhere because the gold is `ok`-dominated. The cue cannot point
at a deception — it is constant across the bucket that holds all of them.

What it *does* buy is narrower and bounded: it concentrates the deceptions, from
21/104 (20.2%) over all NHST items to 21/57 (36.8%) inside the `=` bucket, a
factor of **1.82**. That is a construct-validity dilution, not a score defect, and
it cannot let a weaker tool outrank a stronger one.

Pinned by `tests/test_printed_relation_is_not_a_shortcut.py` — no cue-based
strategy may beat ignoring the cue, the concentration is bounded at 3×, and a
positive control (a partition on the gold's own truthfulness) must show a clear
gain so a collapsed comparison cannot satisfy the invariant silently.

Found by a three-provider cross-model consult round on the v4 scoring diff (seat:
**Grok 4.6 / xAI** via Cursor CLI, in 946 seconds). Neither other seat found it,
and three sessions of careful measured work on this arena had not.

## scoring fix (2026-08-27) — a non-answer is no longer free; FORMULA_ID -> PxRxV-v2

`verification_score` mispriced two different non-answers, so **refusing to verify
anything was the single best strategy in the arena.** Both defects are in the
verification head only; extraction (precision, recall) is untouched.

**F1 — declining an answerable item was free.** `score()` computed
`(v_hits / v_total) if v_total else 1.0`, and every "unverifiable" answer left
`v_total`. A player that declined every item emptied the denominator and
collected the MAXIMUM verification score. Measured with a perfect extractor over
all 36 revealed v3 tasks, before and after:

| fixed-verdict strategy | verification, before | after |
|---|---:|---:|
| always `unverifiable` | **1.0000** | 0.1232 |
| always `ok` | 0.8078 | 0.8078 |
| always `inconsistent` | 0.2931 | 0.2533 |

`v_total == 0` has two causes and only one was a defect: gold with nothing
answerable (neutral 1.0 is correct — verification drops out of the product) versus
a player that declined everything (not neutral, zero coverage). `_verification_cell`
now returns a separate `declined` cell that stays in the denominator as a non-hit.
It is still **not** a false alarm and still emits no `deception_false_alarm`
finding — the 2026-08-21 rule that an honest "I cannot check this" must never be
scored as a fabrication claim is intact. The genuinely ambiguous case
(`coherent=False`) keeps the old `abstain` cell and still leaves the denominator.

**F2 — `expected_verdict == "unverifiable"` had no branch.** The function handled
`ok`, then `not_a_stat`, then assumed the remainder was `inconsistent`. Gold
really carries it — 7 of the 149 checkable revealed items — so the consequence was
exactly backwards: a player that CORRECTLY said "unverifiable" earned nothing
(the answer left the denominator) while a player that FALSELY accused the same
item of being inconsistent was credited with a **hit**.

**Recomputation control, run before anything changed:** re-scoring every published
v3 record through the then-current scorer reproduced the stored `primary`
identically, 178/178 recomputable records (106 LLM + 72 tool; the other 72 are
held-out records whose output is redacted by design). So the movement below is
caused by this change and nothing else.

**Effect on the published board** (mean composite, revealed split, n=36 each):

| player | published | corrected | delta |
|---|---:|---:|---:|
| claude-opus-4-8-stats | 0.7662 | 0.6649 | −0.1013 |
| claude-sonnet-5-stats | 0.6865 | 0.6449 | −0.0417 |
| claude-haiku-4-5-stats | 0.6201 | 0.5852 | −0.0349 |
| escimate | 0.3352 | 0.2958 | −0.0395 |
| statcheck | 0.3332 | 0.3332 | +0.0000 |

**The LLM order does not change.** No LLM player had a single `v_total == 0` task,
so F1 cannot move them; they fall only through F2. **The one rank change is at
4/5, and it runs against the intuition of the fix**: statcheck overtakes escimate.
The reason is measured and worth recording, because it also corrects how the
defect was originally described. **statcheck never abstains.** Its verdicts across
the arena are `{ok: 28, inconsistent: 45}`. All 34 of its items that the scorer
counted as `abstain` are `said=inconsistent, expected=ok, coherent=False` — flags
forgiven by the deliberate 2026-08-21 incoherence rule, which this change does not
touch. escimate genuinely says "unverifiable" 6 times and is now charged for it.
So the near-tie on the published board is produced by the incoherence forgiveness,
not by the free 1.0; a "46.6% abstention rate" for statcheck describes the
scorer's `abstain` CELL, not the player's behaviour. Whether the forgiveness rule
should also count toward coverage is a separate, open scientific question and is
NOT decided here.

`FORMULA_ID` moves `stats-extraction-PxRxV-v1` -> `stats-extraction-PxRxV-v2`. The
product is still `precision * recall * verification`, but V changed, so the two
are not comparable and records must never be averaged across them.

Tests: `tests/test_abstention_dominance.py` (7 new, 5 watched RED against the
unfixed scorer). `tests/test_verification_head.py::test_unverifiable_is_neither_an_accusation_nor_a_miss`
amended — its "abstention is not an accusation" claim is unchanged and still
asserted; the two lines pinning the free 1.0 pinned the defect. Arena suite 85 -> 92.

## taxonomy broadening (2026-07-01, cycle 5) — +effect_size_rounding, matched clean_es control; Claude tiers wired

Added one more real ESCIcheck-family failure mode from the failure-mode catalog,
with a matched clean control, index-cycled deterministically in tier 5 so both
splits cover it equally (`tools/check_parity.py` stays `[PARITY OK]` at
count_tolerance 0). 7 → 8 deception kinds; task count held at 36.

- **`effect_size_rounding`**: an effect-size point estimate re-rounded to fewer
  decimals so it no longer matches its own reported CI — the CI midpoint
  ((ci_lo+ci_hi)/2) disagrees with the stated estimate by a visible margin
  (≥ 0.05). Only the estimate is corrupted; the CI is the true one. Gold anchors
  on the rendered (inconsistent) estimate.
  - Matched clean control: **`clean_es`** (new truthful tier-1 item, one per task)
    — the same shape (estimate + symmetric 95% CI) but the estimate EQUALS the CI
    midpoint, so a player recomputing the midpoint finds no discrepancy and must
    NOT flag it.

Coverage is now **true + 8 deception kinds** (internal_inconsistency,
fabricated_value, swapped_test_label, statistic_impostor, missing_info_impostor,
nhst_inconsistent, wrong_df, effect_size_rounding). Tests: 37 → 43 (+4
effect_size_rounding emission/inconsistency/control/parity, +2 taxonomy
drift-guards: `test_every_declared_deception_kind_has_a_tier5_template` and
`test_stats_prompt_documents_every_deception_kind` — the 4-place-sync guard, since
the output schema has no deception enum). `_ground_truth.json` regenerated via
`tools/dump_revealed_gold.py` (36 tasks, all public). Player tiers wired with a
dedicated `players/prompts/stats_extraction.txt` (documents all 8 modes):
claude-sonnet-5-stats (0.593) > claude-opus-4-8-stats (0.522) > claude-haiku-4-5-stats
(0.304) on the broadened set. NOTE: escimate/statcheck records are now stale vs the
8-kind gold (F6 infra: R `statcheck` pkg not installed, escimate service down).

## taxonomy broadening (2026-07-01) — +2 statcheck-style deception kinds, matched controls

Added two real meta-science failure modes from the failure-mode catalog
("Stats-extraction & statistical-consistency arenas"), each with a matched clean
control look-alike so the test stays hard and non-synthetic. Index-cycled
deterministically in tier 5 (appended to `deception_kinds.yaml`), so both
benchmark splits cover them equally — `tools/check_parity.py` stays `[PARITY OK]`
at count_tolerance 0.

- **`nhst_inconsistent`** (statcheck's core decision error): a DECISIVELY
  significant statistic (true two-sided p recomputed < .01 from stat+df via
  scipy) reported with a non-significant p. Rendered coherently (correct label +
  df-arity via a code-built `{stat_str}`) so the *only* defect is the p — which
  is what distinguishes it from `swapped_test_label`.
  - Matched clean control: **`nhst_consistent`** (new truthful tier-1 item) — the
    same coherent, significant result reported with its recomputed (small) p, so
    a player that recomputes p finds no discrepancy and must not flag it. This
    closes the staged "TODO item E" (truthful items now have a p consistent with
    their statistic — for this dedicated control, where label↔stat are coherent).
- **`wrong_df`**: a correctly-labelled one-way ANOVA `F(df1, df2)` whose
  denominator df is inconsistent with the stated N and group count (should be
  N − k). Gold records the CORRECT `df2`.
  - Matched clean control: **`clean_anova`** (new truthful tier-4 item, one per
    task) — the same design with `df2 = N − k` consistent.

Span-anchoring (the arena's known gold trap) is preserved: every injected value
still anchors on the statistic AS RENDERED. Verified 0 degenerate / 0
slice-mismatch spans over all 264 gold items across both splits; the committed
revealed `_ground_truth.json` was regenerated via `tools/dump_revealed_gold.py`
(36 tasks, all public — no held-out leak). Player prompt (`players/prompts/claude.txt`)
gained guidance for both kinds. New tests cover emission, control coherence,
df-consistency, parity, scorer behaviour, and a span-integrity invariant over
all kinds. Removed a stale `import re` from the generator and corrected the stale
`n_tasks: 60 -> 36` in `arena.yaml` + manifest.

> Note: existing tier-5 task text changed (new kinds) and tiers 1/4 gained one
> control item each, so stored player run records (`runs/v1/*.jsonl`) are now
> stale for those tasks and need a re-run before the next leaderboard refresh —
> deferred (this change does not run tournaments).

## gold-span + kind fix (2026-06-08) — correctness bug, all players re-scored

Triggered by `communications/2026-06-07-escimate-arena-issues.md`. Root cause was
**ours, not the tools'**: 26% of gold items (28/108 revealed) carried a degenerate
`[0,0]` span, so the scorer forced every player's correct extraction to count as
both a false negative and a false positive.

- **Generator span anchor (A1):** the gold anchor was recomputed with a single
  fixed 2-decimal format (`48.70`) that did not match how the templates actually
  rendered the value (raw `48.7`, comma `48,70`, leading-zero-dropped `.72`, OCR
  `2l.85`). Now anchored on the value *as rendered*, via `_anchor_span` trying
  every rendering with digit-boundary + forward-cursor disambiguation. 0
  degenerate spans on both splits.
- **statistic_impostor value (A2):** stored an unrelated nhst value that never
  appeared in the text; now stores the rendered effect-size value it shows.
- **Gold serving migrated to seed-reproducible** `_GROUND_TRUTH_CACHE`
  (registry-free, per LESSONS 2026-06-06), so the fix needs no eval-only corpus /
  `build_gold` rebuild. `ground_truth()` no longer reads the article-finder view.
- **escimate adapter kind (B)** (`framework/player_adapter.py`): `nhst_stat` vs
  `effect_size` now split by df-presence, so a bare `r = -.85` is an effect_size
  (was mislabelled `nhst_stat` → spurious `kind_mismatch`).
- **Re-scored in place** (`scripts/rescore_stats_extraction.py`) — task text is
  unchanged, so stored outputs stay valid. Revealed composite: claude-opus-4-7
  0.199→0.354, escimate 0.205→0.273, statcheck 0.155→0.228. **Ranking flipped**
  (claude-opus was hurt most by the bug and actually leads). Removed stale
  duplicate `statcheck__poc-r1.jsonl`.
- **Staged (TODO):** D (render notation matching the stat label — no `r = 48.7`)
  and E (truthful items get a p consistent with their statistic) change task text
  and need a full tournament re-run.

## benchmark refresh (2026-06-07) — escimate 0.3.2 → 0.6.2 (+ private split wired)

- Re-ran `escimate` against BOTH splits (revealed seed 0 + private secret seed)
  with effectcheck 0.6.2 via the local R backend. Registry `player_version`
  0.3.2 → 0.6.2.
- `tools/build_gold.py` now materializes BOTH splits into the (git-ignored)
  `arena_taskset_gold` view — previously seed-0 only — so the private split is
  runnable. The private answer key stays local (never committed), like
  `.private_seed`. `tools/check_parity.py` still passes (18/18 cells).
- Composite (mean primary): revealed 0.221 (v0.3.2) → 0.205 (v0.6.2); private
  0.224; combined 0.215 ± 0.062 over 72 tasks, 0 errors. Flat within CI — the
  0.4→0.6 gains (clinical effect sizes, bare-t parsing, binomial/Cohen's h)
  target reporting patterns beyond this arena's tier 1–6 synthetic mix.
- Prior `poc-r1` demo run moved to `runs/v1/_archive/`.

## v1 (2026-06-06) — revealed/private dual benchmark (reference retrofit)
- `generate()` now accepts `split` ("revealed" | "private") and tags each envelope
  with `split` + derived `visibility`. Reference implementation of the
  dual-benchmark framework (contract/README.md "Revealed/private dual benchmark").
- `benchmark_splits` added to arena.yaml (revealed seed 0 = the existing committed
  gold; private seed from gitignored `.private_seed`).
- Deception-KIND assignment in tiers 5–6 is now DETERMINISTIC (cycles through every
  declared kind) so each split covers the full array of injected mistakes equally
  and `tools/check_parity.py` passes at count_tolerance 0. Concrete values/templates
  remain seed-driven, so revealed and private content still differ.
- v1 gold (`_ground_truth.json`, registry) refreshed to reflect deterministic kinds.
  Pre-release 0.x refresh; predates any official tournament (only `poc-r1` demo runs).

## v1 (2026-04-29) — initial task set
- 6 difficulty tiers, ~60 tasks
- Coverage: NHST stats, effect sizes, CI representations, true + 5 deception kinds
- All-procedural generation
