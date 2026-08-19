# significance-language-v1

**Challenge:** Given a manuscript excerpt, flag interpretive-language problems —
*marginal-significance* phrasing for p in (.05, .10), *spin / overclaim* not
supported by the reported result, and *causal* language used without a randomised
design. Modeled on
[metacheck](https://github.com/JamieCummins/metacheck)'s `marginal` and
`causal_claims` modules.

Built on Meta Science Arena's revealed/private dual-benchmark framework (see
`contract/README.md` → "Revealed/private dual benchmark").

## Input → output

- **Input** (`schemas/input.schema.json`): `{text}` — the excerpt to scan.
- **Output** (`schemas/output.schema.json`): `{flags: [{span: {text, char_start,
  char_end}, category, confidence}]}` — one flag per problem, localised by
  character range. A correctly-written passage yields an empty `flags` list.

## Injected mistake kinds (metacheck coverage)

- **marginal_significance** — "marginally significant", "trend toward
  significance", "approaching significance" applied to p in (.05, .10).
- **spin_overclaim** — a strong claim ("clearly improved", "highly effective")
  not supported by the reported (null / weak) result.
- **causal_overclaim** — causal language ("caused", "led to", "drives") in a
  cross-sectional / observational / correlational design with no randomisation.

See `catalogs/sentences.yaml`.

## Clean controls (the T2 false-alarm trap)

Legitimate hedging ("did not reach significance"), exact correctly-qualified
significant claims (p < .001), and causal claims that DO cite randomisation
("in this randomised controlled trial, the drug caused…"). These must **not** be
flagged — flagging them is the dominant failure mode.

## Difficulty tiers

T1 clean/simple · T2 **controls-only paraphrase (false-alarm trap)** · T3 single
injected mistake · T4 subtle (mistake beside its look-alike clean control) · T5
multiple mistakes · T6 full composition. The hardest discrimination is T4: a
marginal phrasing next to a legitimate hedge, or a causal overclaim next to a
randomised causal claim.

## Scoring

`composite = span_f1 × calibration`. `span_f1` is the F1 of flagged-span
localisation (a predicted flag matches a gold flag when their character ranges
overlap); `category_accuracy` (reported, not in the composite) is the fraction of
correctly-localised flags given the right category; `calibration` is `1 - ECE`
over per-flag confidences. Findings: `flag_missed`, `flag_false_alarm`,
`category_mislabel`.

## Revealed vs private

- **Revealed** (`--split revealed`, seed 0): published gold + full per-task
  results in the Open Benchmark, so developers can reproduce and audit scoring.
- **Private** (`--split private`): secret seed (`task_sets/v1/.private_seed`) +
  any hand-curated real excerpts under `_held_out/`; findings redacted, scores on
  the official leaderboard. Same tier matrix and mistake-kind coverage as
  revealed — enforced by `python tools/check_parity.py significance-language-v1`.

Gold is regenerated from the seed (no committed answer key, no external registry).

## Players

metacheck (tool, brings its own LLM dependency), a Claude-via-CLI baseline, and a
trained human coder. Register in `players/registry.yaml`.
