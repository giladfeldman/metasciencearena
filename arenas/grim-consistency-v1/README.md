# grim-consistency-v1

**Challenge:** Given a table of reported summary statistics, judge — per statistic —
whether the reported value is *mathematically impossible* given the sample size and
its granularity. Two statistic types appear, distinguished by `stat_type`:

- `mean` — a mean of N integer responses on a bounded single-item scale, so it sits
  on a **1/N** grid.
- `percent` — a percentage of respondents (k of N), so it sits on a **100/N** grid.

This is GRIM checking (granularity of a reported central tendency), not parsing or
recomputation from raw data.

This arena is **GRIM-only**. GRIMMER / SD-granularity is deliberately out of scope;
SD range-plausibility (variance impossibilities) is covered by the separate
`sprite-plausibility-v1` arena. The reported SD here is descriptive context only and
is never the subject of an injected issue. (A prior attempt to inject GRIMMER here
shipped a gold bug — see `CHANGELOG.md` → "Fixed".)

This arena is built on ScienceArena's revealed/private dual-benchmark framework
(see `contract/README.md` → "Revealed/private dual benchmark").

## The mathematics

**GRIM** (Granularity-Related Inconsistency of Means): a mean of N integer
responses on a bounded scale (here `n_items=1`, so each response is one integer) can
only equal `total/N` for some integer `total`, and therefore sits on a 1/N
granularity grid. A reported mean is GRIM-*inconsistent* if no achievable integer
total rounds (to the reported number of decimals) to it.

The same logic applies to a reported **percentage**: it is `k/N × 100` for an
integer count k, so it sits on a 100/N grid and is GRIM-inconsistent if no
achievable count rounds to it. This is `scrutiny::grim(percent = TRUE)`.

The reference implementation is the R package `scrutiny` (`grim()`); this arena
re-derives the same truth in `generator.py` and re-verifies it independently in the
tests, and the `scrutiny-grim` tool player scores **1.00 on all 16 revealed tasks**
against the gold (cross-validation, re-confirmed 2026-08-04 with scrutiny 0.6.1).

> **Running the R players:** R is often installed without being on `PATH`. Set
> `RSCRIPT_BINARY` to the full `Rscript` path (e.g.
> `C:/Program Files/R/R-4.4.0/bin/Rscript.exe`) or every R tool records as an
> errored task.

## Input → output

- **Input** (`schemas/input.schema.json`): `{statistics: [...]}` where each entry is
  either `{stat_id, label, stat_type: "mean", mean, sd, n, n_items, scale_min, scale_max, decimals}`
  or `{stat_id, label, stat_type: "percent", percent, n, decimals}`.
- **Output** (`schemas/output.schema.json`):
  `{records: [{stat_id, issue_kind, flagged, confidence}]}` — one record per
  statistic (`issue_kind` is null when `flagged=false`).

## Injected issue kinds

- `grim_inconsistent` — the reported **mean** cannot arise from any achievable
  integer total at the stated 1/N granularity.
- `grim_percent_inconsistent` — the reported **percentage** equals no achievable
  count `k/N × 100` at the stated 100/N granularity. At N=63, for instance, 42.9%
  (27/63) and 44.4% (28/63) are achievable but 43.0% is impossible.

Each has a matched clean control that must NOT be flagged. See
`catalogs/scales.yaml`.

## Difficulty tiers

T1 clean/simple · T2 **controls-only (false-alarm trap, including
achievable-but-odd percentages)** · T3 single GRIM error, one task per injected
kind · T4 subtle: one-ULP mean, an impossible percentage hidden among clean
percentages, and a mixed mean/percentage table · T5 multiple errors with the kinds
interleaved · T6 maximum density. The hardest discrimination is T2 vs T3/T4: a value
that *looks* odd (small N, many decimals, an arbitrary-looking percentage) is not
necessarily impossible; an off-granularity value *is*. Percentages are the sharper
trap — neither blanket-flagging nor blanket-ignoring them scores.

## Scoring

`composite = detection_f1 × kind_accuracy × calibration`. Detection F1 is over
flagged vs not-flagged across all statistics; `kind_accuracy` is the fraction of
correctly-flagged statistics given the right `issue_kind` (an impossible percentage
must be labelled `grim_percent_inconsistent`, not `grim_inconsistent`); calibration
is `1 - ECE` over confidences. Findings: `grim_missed`, `grim_false_alarm`,
`kind_mislabel`.

## Revealed vs private

- **Revealed** (`--split revealed`, seed 0): published gold + full per-task results
  in the Open Benchmark, so developers can reproduce and audit scoring.
- **Private** (`--split private`): secret seed (`task_sets/v1/.private_seed`) + any
  hand-curated real summary-stat tables under `_held_out/`; findings redacted,
  scores on the official leaderboard. Same tier matrix and issue-kind coverage as
  revealed — enforced by `python tools/check_parity.py grim-consistency-v1`.

Gold is regenerated from the seed (no committed answer key, no external registry).

## Players

A GRIM tool (`scrutiny::grim()` via `players/adapters/grim_scrutiny.R`, brings its
own dependencies), a Claude-via-CLI baseline, and a trained human coder. Register in
`players/registry.yaml`.
