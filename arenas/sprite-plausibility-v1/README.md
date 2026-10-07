# sprite-plausibility-v1

**Challenge:** Given a table of reported summary statistics — each a `mean`, `sd`,
and `n` for a measure scored on an integer response scale `[scale_min, scale_max]`
— judge, per statistic, whether the reported `(mean, SD)` pair is *granularity/range
plausible*: can ANY sample of `n` integers in that range actually produce that mean
and SD? This is SPRITE-style summary-stat plausibility checking; the reference tool
is the R package [`rsprite2`](https://cran.r-project.org/package=rsprite2). This is
about whether the numbers are *achievable*, not about data parsing or table layout.

This arena is built on Meta Science Arena's revealed/private dual-benchmark framework
(see `contract/README.md` → "Revealed/private dual benchmark").

## Input → output

- **Input** (`schemas/input.schema.json`):
  `{statistics: [{stat_id, label, mean, sd, n, scale_min, scale_max, decimals}]}`.
- **Output** (`schemas/output.schema.json`):
  `{records: [{stat_id, issue_kind, flagged, confidence}]}` — one record per
  statistic (`issue_kind` is null when `flagged=false`).

## Injected impossibility kinds

`impossible_mean` (the reported mean falls strictly OUTSIDE `[scale_min, scale_max]`
— no sample of in-range integers can do that), `impossible_sd` (an in-range mean but
a reported SD whose square exceeds the theoretical maximum variance for bounded data
with that mean, `max_var = (scale_max - mean) * (mean - scale_min)`). Clean
statistics have neither. See `catalogs/scales.yaml` for the response scales drawn
from.

**Why gold is never a false flag.** Clean stats are *computed*: the generator draws
`n` integers in range, takes the real sample mean and SD (ddof=1), and rounds to
`decimals` — achievable by construction. `impossible_mean` puts the mean past a
bound. `impossible_sd` reports `sd = ceil(sqrt(max_var × k))` with `k > 1`, rounded
up so `sd²` strictly exceeds `max_var` even after rounding. Every condition is a
rigorous *sufficient* condition for impossibility (or for achievability); tests
re-verify each flagged/clean stat against it.

## Difficulty tiers

T1 clean/simple · T2 **controls-only (false-alarm trap)** · T3 single injected issue
· T4 subtle issue (an SD that only *barely* exceeds the variance ceiling) · T5
multiple issues · T6 full composition. The hardest discrimination is T2 vs T3: a
statistic with an SD near its theoretical maximum is extreme but genuinely possible,
whereas one just over the ceiling is impossible.

## Scoring

`composite = detection_f1 × kind_accuracy × calibration`. Detection F1 is over
flagged vs not-flagged across statistics; `kind_accuracy` is the fraction of
correctly-flagged statistics given the right `issue_kind`; calibration is `1 - ECE`
over confidences. Findings: `sprite_missed`, `sprite_false_alarm`, `kind_mislabel`.

## Revealed vs private

- **Revealed** (`--split revealed`, seed 0): published gold + full per-task results
  in the Open Benchmark, so developers can reproduce and audit scoring.
- **Private** (`--split private`): secret seed (`task_sets/v1/.private_seed`) + any
  hand-curated real summary-stat tables under `_held_out/`; findings redacted, scores
  on the official leaderboard. Same tier matrix and impossibility-kind coverage as
  revealed — enforced by `python tools/check_parity.py sprite-plausibility-v1`.

Gold is regenerated from the seed (no committed answer key, no external registry).

## Players

The `rsprite2` reference tool, a Claude-via-CLI baseline, and a trained human coder.
Register in `players/registry.yaml`.
