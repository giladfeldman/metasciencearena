# publication-bias-v1

**Challenge:** Given a meta-analytic dataset — a set of studies, each with an
effect-size estimate `yi` and its standard error `sei` — judge whether the set as a
whole shows **small-study / publication bias** (funnel-plot asymmetry). One verdict
per task: `bias_detected` (boolean) + a confidence.

This arena is built on ScienceArena's revealed/private dual-benchmark framework
(see `contract/README.md` → "Revealed/private dual benchmark").

## The statistics

A symmetric funnel plot (effect vs precision) is the signature of an unbiased
literature: small and large studies scatter symmetrically around the pooled effect.
**Small-study / publication bias** breaks that symmetry — small (high-`sei`) studies
that found small or wrong-signed effects are selectively missing, so the surviving
small studies over-represent large effects. The classic detector is **Egger's
regression test**: regress the effect on its standard error (weighted by 1/variance)
and test the slope; a significant slope means asymmetry. **Trim-and-fill** estimates
how many studies are "missing" to restore symmetry.

The reference implementation is the R package **metafor** (`regtest()` for Egger,
`trimfill()`). This arena re-derives Egger's test in pure Python inside
`generator.py` and re-verifies it independently in the tests; the `metafor-pubbias`
tool player scores ≈1.0 against the gold (cross-validation).

## Computed gold

Every dataset is **constructed** with a known label, deliberately far from the
p ≈ 0.05–0.10 grey zone so the deterministic tool agrees regardless of tiny numerical
differences:

- **unbiased** (`bias_detected=false`): `k` studies drawn around a common true effect
  `mu` with symmetric sampling error (optional between-study heterogeneity `tau`) and
  NO censoring → symmetric funnel → Egger p well above threshold.
- **biased** (`bias_detected=true`): an explicit positive coupling between `yi` and
  `sei` (small studies report inflated effects) PLUS censoring of the small-N
  under-shooters → strongly asymmetric funnel → Egger p ≪ 0.01.

The label is known by construction; the generator ALSO computes Egger's test and the
self-consistency test asserts the Python Egger verdict matches the constructed label
for every task in both splits.

## Input → output

- **Input** (`schemas/input.schema.json`):
  `{k?, studies: [{yi, sei}, ...]}` — ≥10 studies per task.
- **Output** (`schemas/output.schema.json`):
  `{bias_detected, confidence, egger_p?, n_missing_trimfill?}` — one verdict per task.
  Optional keys are omitted (not null) when not computed.

## Difficulty tiers

T1 small-k clean vs strong-biased · T2 **controls-only (heterogeneous-but-clean
false-alarm trap)** · T3 null-effect clean vs biased (k=20) · T4 heterogeneous clean
vs biased (k=24) · T5 large-k (k=32) · T6 largest-k (k=40). The hardest
discrimination is T2/T4: high between-study heterogeneity looks noisy but is NOT
funnel asymmetry.

## Scoring

`primary = 0.85 · correct + 0.15 · calibration`, where `correct` is 1.0 iff
`bias_detected` matches the computed gold and `calibration = 1 - |confidence -
correct|`. A correct, fully-confident player scores 1.0. Findings: `bias_missed`
(gold biased, said unbiased), `bias_false_alarm` (gold unbiased, flagged bias).

## Revealed vs private

- **Revealed** (`--split revealed`, seed 0): published gold + full per-task results in
  the Open Benchmark, so developers can reproduce and audit scoring.
- **Private** (`--split private`): secret seed (`task_sets/v1/.private_seed`) + any
  hand-curated real meta-analytic datasets under `_held_out/`; findings redacted,
  scores on the official leaderboard. Same tier matrix and label coverage as revealed.

Gold is regenerated from the seed (no committed answer key, no external registry).

## Players

A metafor tool (`metafor::regtest()` Egger + `trimfill()` via
`players/adapters/metafor_pubbias.R`, brings its own dependencies) and a
Claude-via-CLI baseline. Register in `players/registry.yaml`.
