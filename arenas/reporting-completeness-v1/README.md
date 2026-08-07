# reporting-completeness-v1

**Challenge:** Given a plain-text results section, flag statistical-reporting
**completeness / precision** defects — *without* recomputing any test statistic
(that is statcheck's job, benchmarked separately in `stats-extraction-v1`). This
arena grades whether each reported test is reported *completely* and *precisely*.
Mirrors metacheck's `stat_effect_size`, `stat_p_exact`, `all_p_values`, and
`stat_p_nonsig` modules.

This arena is built on ScienceArena's revealed/private dual-benchmark framework
(see `contract/README.md` → "Revealed/private dual benchmark").

## Input → output

- **Input** (`schemas/input.schema.json`): `{text}` — the results-section prose.
- **Output** (`schemas/output.schema.json`): `{flags: [{span:{text, char_start,
  char_end}, category, confidence}]}` — one flag per detected defect. A clean,
  complete report should yield an **empty** `flags` array.

## Defect categories (the injected mistake_kinds)

- `missing_effect_size` — a significant test reported with no effect size.
- `imprecise_p` — `p < .05` where an exact p-value is knowable.
- `impossible_p_zero` — `p = .000` (a p-value can never be exactly zero).
- `missing_ci` — an estimate reported with no confidence interval.
- `missing_df` — a test statistic reported without its degrees of freedom.
- `nonsig_as_support` — a `p > .05` result framed as supporting the hypothesis.
- `unspecified_test` — a significance claim with a p-value but **no test statistic**
  named at all (distinct from `missing_df`, which keeps the statistic).
- `no_correction` — many comparisons interpreted at `p < .05` with **no multiplicity
  correction** (Bonferroni/Holm/Tukey/FDR) stated.
- `percent_count_mismatch` — a `<count> of <N> (<pct>%)` whose percentage is
  **arithmetically inconsistent** with count/N.

Each defect has a matched **clean-control look-alike** that must NOT be flagged: a
consistent percent/count clause (`clean_count_match`), a many-comparisons report that
DOES state a correction (`clean_corrected`), and — for `unspecified_test` — the
ordinary complete sentence, which names the statistic. A **clean** sentence reports a
test completely: test statistic + df + exact p + effect size + CI, framed honestly.
See `catalogs/tests.yaml`.

## Difficulty tiers

T1 clean/simple · T2 **complete-but-busy (false-alarm trap)** · T3 single injected
defect · T4 subtle defect · T5 multiple defects · T6 full composition. The hardest
discrimination is T2 vs T3: a dense but *complete* report must not be flagged;
a report *missing* a required component must be.

## Scoring

`composite = detection_f1 × calibration`. Player flags are matched to gold flags by
**category + span overlap** (containment OR IoU ≥ 0.5); detection F1 is over those
matches; calibration is `1 - ECE` over per-flag confidence. Findings: `flag_missed`
(major), `flag_false_alarm` (major), `category_mislabel` (minor).

## Revealed vs private

- **Revealed** (`--split revealed`, seed 0): published gold + full per-task results
  in the Open Benchmark, so developers can reproduce and audit scoring.
- **Private** (`--split private`): secret seed (`task_sets/v1/.private_seed`) + any
  hand-curated real results sections under `_held_out/`; findings redacted, scores
  on the official leaderboard. Same tier matrix and defect-category coverage as
  revealed — enforced by `python tools/check_parity.py reporting-completeness-v1`.

Gold is regenerated from the seed (no committed answer key, no external registry).

## Players

metacheck (tool), a Claude-via-CLI baseline, and a trained human coder. Register
in `players/registry.yaml`.
