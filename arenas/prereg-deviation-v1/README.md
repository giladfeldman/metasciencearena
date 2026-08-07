# prereg-deviation-v1

**Challenge:** Given a preregistration and the eventual paper, identify — per
comparison dimension — whether the paper *deviates* from what was preregistered,
and what *kind* of deviation it is. Modeled on
[regcheck](https://github.com/JamieCummins/regcheck).

This is the flagship arena built on ScienceArena's revealed/private dual-benchmark
framework (see `contract/README.md` → "Revealed/private dual benchmark").

## Input → output

- **Input** (`schemas/input.schema.json`): `{preregistration, paper, dimensions}`.
  `dimensions` is the closed set of dimension ids to assess.
- **Output** (`schemas/output.schema.json`): `{deviations: [{dimension, deviation,
  deviation_kind, confidence, registered_summary?, paper_summary?}]}` — one record
  per dimension.

## Dimensions (regcheck coverage)

General preregistration: sample size, data source, inclusion/exclusion criteria,
missing-data handling, hypotheses (HARKing), manipulated variables, measured
variables (outcome switching), transformations, statistical models. Clinical-trial
extensions: randomisation & allocation, primary-outcome timepoint. See
`catalogs/dimensions.yaml`.

## Difficulty tiers

T1 verbatim-consistent · T2 **paraphrase-consistent (false-alarm trap)** · T3 single
clear deviation · T4 subtle deviation · T5 multiple deviations · T6 full composition.
The hardest discrimination is T2 vs T3: rewording is *not* a deviation, a changed
commitment *is*.

## Scoring

`composite = detection_f1 × kind_accuracy × calibration`. Detection F1 is over
deviation vs no-deviation across dimensions; `kind_accuracy` is the fraction of
correctly-flagged deviations given the right `deviation_kind`; calibration is
`1 - ECE` over confidences. Findings: `deviation_missed`, `deviation_false_alarm`,
`kind_mislabel`, `dimension_missed`, `unknown_dimension`.

## Revealed vs private

- **Revealed** (`--split revealed`, seed 0): published gold + full per-task results
  in the Open Benchmark, so developers can reproduce and audit scoring.
- **Private** (`--split private`): secret seed (`task_sets/v1/.private_seed`) + any
  hand-curated real prereg/paper pairs under `_held_out/`; findings redacted, scores
  on the official leaderboard. Same tier matrix and deviation-kind coverage as
  revealed — enforced by `python tools/check_parity.py prereg-deviation-v1`.

Gold is regenerated from the seed (no committed answer key, no external registry).

## Players

regcheck (tool, brings its own LLM dependency), a Claude-via-CLI baseline, and a
trained human coder. Register in `players/registry.yaml`.
