# power-reporting-v1

**Challenge:** Given a methods/power excerpt, (1) detect whether it reports a
power analysis, (2) classify its *kind* — a-priori / sensitivity / post-hoc —
and (3) extract its structured fields. A **field-map** arena modeled on
[metacheck](https://github.com/JamieCummins/metacheck)'s `power` module.

Built on Meta Science Arena's revealed/private dual-benchmark framework (see
`contract/README.md` → "Revealed/private dual benchmark").

## Input → output

- **Input** (`schemas/input.schema.json`): `{text}` — a plain-text methods/power
  section excerpt.
- **Output** (`schemas/output.schema.json`): `{has_power_analysis, kind,
  fields:{test, sample, alpha, power, effect_size, software}, confidence}`.
  Omit a field from `fields` entirely if the excerpt does not report it — do not
  invent values.

## Injected mistakes (cycled deterministically)

- **`posthoc_as_apriori`** — the headline trap. A *post-hoc* power analysis
  (power computed from the **observed** effect, after data collection) **worded
  as if it were a-priori**. The true `kind` is `posthoc`; a naive reader is lured
  into `apriori` by the framing.
- **`missing_fields`** — a genuine power analysis with some structured fields
  absent from the text (the player must not hallucinate them).
- **`no_power_analysis`** — the excerpt reports no power analysis at all.

Clean variants (`mistake_kinds == []`): a complete a-priori, a complete
sensitivity, and a complete **correctly-labelled** post-hoc analysis.

## Difficulty tiers

T1 clean labelled analyses · T2 **false-alarm trap** (suspicious but clean:
sample-size/alpha mentions with no power analysis, and correctly-labelled
post-hoc/sensitivity that must not be relabelled) · T3 single injected mistake ·
T4 **post-hoc-as-a-priori** trap + subtle missing-fields · T5 multiple mistakes ·
T6 full composition. The hardest discrimination is T4: a-priori *framing* over a
post-hoc *computation*.

## Scoring

`composite = mean(detection, kind, field_f1) × calibration`.

- **detection** — 1.0 if `has_power_analysis` matches gold.
- **kind** — kind-classification accuracy when a power analysis is present (1.0
  when gold has none, so a correct "no PA" call is not double-penalised).
- **field_f1** — F1 of the extracted field-map (precision over reported fields,
  recall over gold fields; values compared case/whitespace-insensitively).
- **calibration** — `1 - ECE` over the player's confidence in the detection call.

Findings: `power_missed`, `power_false_alarm`, `kind_mislabel`, `field_wrong`.

## Revealed vs private

- **Revealed** (`--split revealed`, seed 0): published gold + full per-task
  results in the Open Benchmark, so developers can reproduce and audit scoring.
- **Private** (`--split private`): secret seed (`task_sets/v1/.private_seed`) +
  any hand-curated real power-section excerpts under `_held_out/`; findings
  redacted, scores on the official leaderboard. Same tier matrix and
  mistake-kind coverage as revealed — enforced by
  `python tools/check_parity.py power-reporting-v1`.

Gold is regenerated from the seed (no committed answer key, no external registry).

## Players

metacheck (tool, brings its own LLM dependency), a Claude-via-CLI baseline, and a
trained human coder. Register in `players/registry.yaml`.
