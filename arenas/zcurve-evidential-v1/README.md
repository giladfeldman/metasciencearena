# zcurve-evidential-v1

**Challenge:** Given a SET of statistically-significant test results (their
z-scores, all `|z|>1.96`), classify whether the set reflects genuine underlying
effects (**evidential value**) or is consistent with selection-for-significance
over null/weak effects (**no evidential value / p-hacked**). The reference tool
is the R package [`zcurve`](https://cran.r-project.org/package=zcurve) (z-curve
2.0), which estimates the expected discovery / replication rate from the
distribution of significant z-scores.

Built on Meta Science Arena's revealed/private dual-benchmark framework (see
`contract/README.md` → "Revealed/private dual benchmark").

## Input → output

- **Input** (`schemas/input.schema.json`): `{study_set_id, z_scores:[>1.96,...],
  n_studies}` — the significant results of a set of studies, as z-scores.
- **Output** (`schemas/output.schema.json`): `{has_evidential_value,
  expected_discovery_rate?, confidence}` — a single verdict for the set.
  `expected_discovery_rate` is optional and not scored (reporting/auditing only).

## Gold principle — the arena does NOT run zcurve

Gold is the **generating regime** the arena controlled, not a zcurve output:

- **`evidential`** — z-scores drawn from studies with REAL effects and high true
  power: `z = |Normal(mean=ncp, sd=1)|` kept only when `z>1.96`, with `ncp`
  chosen for high per-study power. Such a set genuinely has evidential value →
  `has_evidential_value=true`.
- **`non_evidential`** — z-scores drawn from NULL effects selected for
  significance: a half-standard-normal truncated to `z>1.96` with `ncp≈0` (only
  false positives that cleared the bar). No evidential value →
  `has_evidential_value=false`.

Harder mixed tiers (low true power, small sets, a little cross-contamination)
still take their gold label from the **dominant** generating regime.

## Difficulty tiers

T1 clearly evidential (high power, large set) · T2 clearly non-evidential (pure
nulls, large set) · T3 evidential, moderate power · T4 **non-evidential, subtle**
(a few weak real effects mixed into nulls, still dominated by selection) · T5
small-set evidential (harder) · T6 small-set non-evidential (harder). Tiers
encode the difficulty of the SAME binary task — set size and effect strength set
the difficulty, and both regimes appear across the tiers.

## Scoring

`primary = 1.0` if `has_evidential_value` matches the generating regime, else
`0.0`. `calibration = 1 - |confidence - correctness|` (a confident-correct or
unconfident-wrong call scores well; a confidently-wrong call is punished).
Findings: `misclassified_evidential` (said non-evidential but the set was
evidential), `misclassified_non_evidential` (said evidential but the set was
p-hacked/null).

## Revealed vs private

- **Revealed** (`--split revealed`, seed 0): published gold + full per-task
  results in the Open Benchmark, so developers can reproduce and audit scoring.
- **Private** (`--split private`): secret seed (`task_sets/v1/.private_seed`) +
  any hand-curated real significant-result sets under `_held_out/`; findings
  redacted, scores on the official leaderboard. Same tier matrix and regime
  coverage as revealed — enforced by
  `python tools/check_parity.py zcurve-evidential-v1`.

Gold is regenerated from the seed (no committed answer key, no external registry).

## Players

`zcurve` (the R reference tool), a Claude-via-CLI baseline, and a trained human
coder. Register in `players/registry.yaml`.
