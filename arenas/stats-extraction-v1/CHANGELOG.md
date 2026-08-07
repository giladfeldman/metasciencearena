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

Triggered by `docs/reports/2026-06-07-escimate-arena-issues.md`. Root cause was
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
