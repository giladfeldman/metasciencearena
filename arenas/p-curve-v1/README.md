# p-curve-v1

**Challenge:** Given a SET of independent statistically-significant findings
(p < .05), each supplied as a test statistic, judge whether the set as a whole has
**evidential value** by the **p-curve** method (Simonsohn, Nelson & Simons, 2014). One
verdict per task: `evidential_value` (boolean) + a confidence.

This arena is built on ScienceArena's revealed/private dual-benchmark framework
(see `contract/README.md` → "Revealed/private dual benchmark").

## The method — p-curve (Simonsohn, Nelson & Simons, 2014)

> Simonsohn, U., Nelson, L. D., & Simmons, J. P. (2014). *P-curve: A key to the
> file-drawer.* Journal of Experimental Psychology: General, 143(2), 534–547.
> https://doi.org/10.1037/a0033242

p-curve is the distribution of statistically-significant p-values (p < .05) for a set
of independent findings. Its shape is diagnostic:

- **Right-skewed** (more p ≈ .01 than p ≈ .04): the findings reflect a real,
  well-powered effect → the set has **evidential value**.
- **Flat** (uniform on [0, .05]): consistent with selection-for-significance over
  **null** effects (false positives) → **no** evidential value.
- **Left-skewed** (mass bunched just under .05): the signature of intense
  **p-hacking** → **no** evidential value.

**Algorithm (implemented identically in `generator.py` and `players/adapters/pcurve.R`):**
for each significant finding,

1. Compute the exact two-sided p-value from its test statistic (`t` with df;
   `F(1, df2)` — p-curve only uses F with df1 = 1; `z`; `chi2(1)`; `r` with `n`).
2. Keep only p < .05. Compute the **right-skew pp-value** — the probability under the
   null of a p at least this extreme, conditional on significance: `pp_i = p_i / .05`.
   z-transform: `z_i = qnorm(pp_i)`.
3. **Right-skew test (evidential value)** via Stouffer's method:
   `Z = sum(z_i) / sqrt(k)`, `right_skew_p = pnorm(Z)`. **Evidential value is present
   iff `right_skew_p < .05`** (the full-curve right-skew test).
4. **Flatness test (optional diagnostic):** compare the curve against the pp-values
   expected under 33% power (`flatness_p`). Reported but NOT used for the verdict, so
   the verdict stays deterministic.

The reference player `players/adapters/pcurve.R` is a **faithful pure-R
implementation** of this published p-curve. (`dmetar`, which wraps p-curve, is not
available for R 4.4, and `puniform` implements p-uniform\* — a sibling, not p-curve —
so the algorithm is implemented directly, as it is fully specified in the paper.)

## How this DIFFERS from `zcurve-evidential-v1`

Both arenas classify a set of significant findings, but they are **different methods**:

| | **p-curve-v1** (this arena) | **zcurve-evidential-v1** |
|---|---|---|
| What it models | the **right-skew** of the significant **p-values** | an **EM mixture** over the significant **z-values** |
| Test statistic | Stouffer's Z over pp-value z-scores | fitted expected discovery / replication rate |
| Null it tests | flat (uniform) p-curve under H0 | mixture-implied EDR/ERR |
| Verdict | right-skew present (`right_skew_p < .05`) | EDR/ERR above a replicability threshold |
| Reference | faithful R p-curve (Simonsohn et al. 2014) | `zcurve` R package (EM) |

p-curve asks *"are the significant p-values skewed toward 0?"*; z-curve asks *"what
discovery/replication rate does the mixture of significant z-values imply?"*. They can
even disagree on edge cases — which is exactly why both are worth benchmarking.

## Computed gold

Every set is **constructed** with a known label, deliberately far from the right-skew
decision boundary so the deterministic reference agrees regardless of tiny numerical
differences:

- **evidential** (`evidential_value=true`): findings drawn from a real non-null effect
  with decent power (true d ≈ 0.5–0.7, n ≈ 60–100) → significant p-values cluster near
  0 → strong right skew → `right_skew_p ≪ .01`.
- **no-evidential** (`evidential_value=false`): either **true-null** significant
  results (false positives, p uniform on [0, .05] → flat curve) OR intense
  **p-hacking** (results bunched just under .05 → left skew) → `right_skew_p ≫ .5`.
  No-evidential sets are rejection-sampled so `right_skew_p ≥ 0.55` on the final
  rounded findings (never borderline).

The label is known by construction; the generator ALSO runs the full Python p-curve
and the self-consistency test asserts the computed verdict matches the constructed
label for every task in both splits.

## Input → output

- **Input** (`schemas/input.schema.json`):
  `{findings: [{type, value, df1?, df2?, n?}, ...]}` — only significant results, ≥5 per
  task. `type ∈ {t, F, z, chi2, r}`; p-curve uses F only with df1 = 1; `r` requires `n`.
- **Output** (`schemas/output.schema.json`):
  `{evidential_value, confidence, right_skew_p?, flatness_p?}` — one verdict per task.
  Optional keys are omitted (not null) when not computed.

## Difficulty tiers

T1 small-k (k=5) evidential vs true-null (t) · T2 controls (clearly-evidential F vs
clearly-flat null z) · T3 **p-hacked left-skew trap** (z) · T4 chi2(1) evidential vs
null (k=20) · T5 correlations r evidential vs p-hacked (k=25) · T6 largest-k (k=30) t.
Statistic type is varied across tiers so t / F / z / chi2 / r are all exercised. The
hardest discrimination is T3/T5: a left-skewed p-hacked curve must NOT be mistaken for
the right-skew of evidential value.

## Scoring

`primary = 0.85 · correct + 0.15 · calibration`, where `correct` is 1.0 iff
`evidential_value` matches the computed gold and `calibration = 1 - |confidence -
correct|`. A correct, fully-confident player scores 1.0. Findings:
`evidential_missed` (gold evidential, said no), `evidential_false_alarm` (gold
no-evidential, claimed evidential).

## Revealed vs private

- **Revealed** (`--split revealed`, seed 0): published gold + full per-task results in
  the Open Benchmark, so developers can reproduce and audit scoring.
- **Private** (`--split private`): secret seed (`task_sets/v1/.private_seed`) + any
  hand-curated real significant-findings sets under `_held_out/`; findings redacted,
  scores on the official leaderboard. Same tier matrix and label coverage as revealed.

Gold is regenerated from the seed (no committed answer key, no external registry).

## Players

A faithful R p-curve tool (`players/adapters/pcurve.R`, the deterministic reference)
and a Claude-via-CLI baseline (`claude-haiku-4-5-pcurve`). Register in
`players/registry.yaml`.
