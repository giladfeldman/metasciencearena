# publication-bias-v1

**Challenge (task set v2, current):** Given a meta-analytic dataset — a set of
studies, each with an effect-size estimate `yi` and its standard error `sei` —
compute the statistics of the standard publication-bias methods: Egger's regression
test, Begg and Mazumdar's rank correlation, trim-and-fill, PET-PEESE, the
three-parameter selection model (3PSM) and robust Bayesian meta-analysis (RoBMA).
**No verdict is scored.**

Why no verdict: turning these statistics into "biased / not biased" is an unresolved
methodological question (owner decision 2026-09-17), so a verdict score measures an
answer nobody can defend. v2 asks the question that does have a right answer: did
you compute each method correctly?

Task set **v1** (superseded, still published) asked for the verdict instead — see
"v1" at the end.

This arena is built on Meta Science Arena's revealed/private dual-benchmark framework
(see `contract/README.md` → "Revealed/private dual benchmark").

## The statistics (v2)

Each statistic is the default of a named R function, so a correct implementation of
the nine deterministic statistics reproduces the gold to rounding; RoBMA's two are
MCMC estimates and reproduce only to within sampler noise. The prompt (`players/prompts/publication_bias_v2.txt`)
spells each one out.

| method | statistic(s) | gold computed by |
|---|---|---|
| Egger | `egger.p` — two-sided p of the `sei` slope | `lm(yi ~ sei, weights = 1/vi)` |
| Begg | `begg.tau`, `begg.p` (exact) | `metafor::ranktest(yi, sei = sei)` |
| trim-and-fill | `trimfill.k0`, `trimfill.estimate` | `metafor::trimfill(rma(yi, vi, method = "REML"))` |
| PET-PEESE | `pet.estimate`, `peese.estimate` | intercepts of `lm(yi ~ sei)` / `lm(yi ~ vi)`, weights `1/vi` |
| 3PSM | `three_psm.estimate`, `three_psm.lrt_p` | `metafor::selmodel(rma(yi, vi), type = "stepfun", steps = .025, alternative = "greater")` |
| RoBMA | `robma.bf_bias`, `robma.mu` | `RoBMA::RoBMA(d = yi, se = sei)` defaults, without the two machine-dependent limits (see below) |

Implementation notes, each measured 2026-09-30:

- **REML** fits use `control = list(stepadj = 0.5, maxiter = 1000)`: metafor's default
  Fisher scoring does not converge on some datasets (1 of the 12 v1 revealed sets).
  With the remedy, tau² agrees with a brute-force maximisation of the REML likelihood
  to within 2e-5 on all 12 v1 sets.
- **Trim-and-fill's side** follows metafor's rule: fit a REML random-effects
  meta-regression of yi on sei (`rma(yi, vi, mods = sqrt(vi))`); a negative slope
  imputes on the right, otherwise on the left. The `meta` package uses a different
  rule and chose the other side on 2 of the 24 v2 datasets (both private, slope near
  zero; on one of them this changed k0 by 1). With the side aligned the two agree
  exactly on k0 and to within 6e-6 on the estimate on all 24.
- **3PSM uses `metafor::selmodel`, not `weightr`.** weightr's optimiser crashed on 2
  of the 12 v2 revealed datasets (both with tau² near 0), and on one private dataset
  it stopped early (its adjusted model's log-likelihood came out BELOW the unadjusted
  model it contains, which a true maximum cannot do), moving the LRT p by 0.035. The
  `selmodel` gold moves by at most 0.0002 (p) and 0.0001 (estimate) across five
  optimisers on all 24 datasets.
- **RoBMA is MCMC**, so its gold is one draw (seed 20260930). Its tolerance bands are
  sized from the measured seed-to-seed spread (`task_sets/v2/robma_stability.json`,
  pinned by `test_robma_full_credit_band_covers_measured_mcmc_noise`).
- **RoBMA runs without two of its defaults that depend on the machine** (2026-10-06,
  debt d-125b6a): the 60-minute wall-clock limit on autofit chain extension, and the
  silent restart of a sub-model whose JAGS run errors (inits from seed + i, then the
  sub-model is dropped). The script stops on any failed sub-model instead. Whenever
  neither limit fires the result is identical to the defaults. Set
  `PUBBIAS_ROBMA_DIAG_DIR` to log per-sub-model samples, log marginal likelihoods and
  errors for every fit. Why: on 2026-10-01 the reference returned a different seed-1
  answer on 2 of 12 sets (|Δ ln BF| up to 0.25), which no later refit reproduced; see
  `runs/v2/_archive/2026-10-06-robma-not-reproducible-v2c6/README.md`.

## Gold (v2)

Datasets come from `generator.py` (stdlib only). Trim-and-fill, 3PSM and RoBMA exist
only in R, so the gold is computed once by `tools/build_gold_v2.py`, which runs
`tools/pubbias_v2_stats.R` over every dataset:

- revealed: `task_sets/v2/_r_gold.json` (committed)
- private: `task_sets/v2/_held_out/_r_gold.json` (gitignored)

Every entry records the sha256 of the exact dataset it was computed on. The generator
refuses a gold entry whose hash does not match the dataset it generates now, so a
stale gold file fails loudly instead of scoring players against another dataset's
answers. Regenerating the gold needs R 4.4 with metafor 5.0.1 and RoBMA 3.6.1 (and
JAGS 4.3.1), and takes hours: RoBMA fits 36 models by MCMC for each dataset.

`tools/crosscheck_gold_v2.py` recomputes every deterministic statistic with code
that shares nothing with the gold (numpy/scipy, the `meta` package, `weightr`) and
prints the largest gaps.

## Datasets (v2)

Six tiers by number of studies (k = 10, 16, 20, 24, 32, 40), two datasets per tier,
each built under one of three conditions: **no selection**, **p-value selection**
(non-significant studies published with probability w — what 3PSM and RoBMA
model), or a **small-study coupling** (effect grows with `sei` — what Egger and PET
model), all at moderate strength so the statistics spread across their range. The
condition is a difficulty descriptor and is never scored. Every dataset has at least
three studies on each side of one-sided p = .025, so the selection model is
identified.

## Input → output

- **Input** (`schemas/input.schema.json`, both task sets): `{k?, studies: [{yi, sei}, ...]}`.
- **Output, v2** (`schemas/v2/output.schema.json`):
  `{egger {p}, begg {tau, p}, trimfill {k0, estimate}, pet {estimate}, peese {estimate}, three_psm {estimate, lrt_p}, robma {bf_bias, mu}}`.
  Every key is optional; omit (never null) what you did not compute.

## Scoring (v2)

Each statistic earns credit in [0, 1]: full credit within a small band of the gold,
falling linearly to 0 at a wider band.

| statistic | full credit within | zero at |
|---|---|---|
| p-values (Egger, Begg, 3PSM LRT) and Kendall's tau | 0.005 | 0.05 |
| effect estimates (trim-and-fill, PET, PEESE, 3PSM) | 0.002 | 0.02 |
| `trimfill.k0` | exact | off by 1 |
| `robma.bf_bias` (on the natural-log scale) | 0.05 | 0.50 |
| `robma.mu` | 0.01 | 0.05 |

RoBMA's bands come from its measured MCMC noise: refitting all 12 revealed sets under
a second seed moved ln BF by at most 0.013 and mu by at most 0.0029
(`task_sets/v2/robma_stability.json`). The reference player's seed-1 fits (run v2c7)
sit at most 0.014 (ln BF) and 0.0031 (mu) from the gold. Full credit sits at about four
times that noise.

The bands are sized for rounding, not approximation: a correct implementation
matches to about 1e-5. Measured against a player that computes nothing (the pooled
mean for every adjusted estimate, p = 0.5, tau = 0, k0 = 0), these bands give it
0.20 on the revealed sets with RoBMA's measured bands and the final method weights
(0.39 under the first draft's 0.01/0.10 bands); the remaining 0.20 is real partial correctness (when trim-and-fill
imputes nothing, the plain random-effects mean is its answer).

`primary` = a weighted mean over the **six methods** (PET and PEESE form one) of the
mean over each method's statistics. A statistic that is not
reported scores 0, exactly like a wrong one: in v1, where reporting was optional,
only 45 of 115 reported Egger p-values were within 0.01 of the correct value (33 of
103 without the reference tool), so reporting is not evidence of computing. Findings: `statistic_off` (reported, outside the
full-credit band) and `statistic_missing`.

**Method weights** (arena owner's decision, 2026-10-01: more weight on the current
methods than on the mostly outdated ones):

| method | weight |
|---|---|
| RoBMA (most comprehensive) | 0.30 |
| PET-PEESE | 0.225 |
| three-parameter selection model | 0.225 |
| Egger | 0.10 |
| Begg (included for coverage) | 0.075 |
| trim-and-fill | 0.075 |

The modern methods carry 0.75 of the score. The split inside each group follows the
owner's ranking (Egger among the weakest, trim-and-fill weaker still); the exact
numbers are this arena's reading of it. Per-method credit is published alongside, so
any other weighting can be recomputed from the records.

## Revealed vs private

- **Revealed** (`--split revealed`, seed 0): published gold and per-task results.
- **Private** (`--split private`): secret seed (`task_sets/<v>/.private_seed`),
  findings redacted. Same tier and condition matrix as revealed.

## Players

The reference tool `metafor-robma-pubbias` runs the same script that computes the
gold, so on the nine deterministic statistics its 1.0 is a DETERMINISM check, not an
independent validation of the gold (that is `tools/crosscheck_gold_v2.py`). It uses
RoBMA seed 1 where the gold uses 20260930, so its RoBMA credit measures sampler noise
against the band.

Tool access is part of what is measured: a player with no code execution (the
hermetically contained CLI players, for instance) cannot fit a selection model or run
MCMC, and will score accordingly. The arena scores the answer, not the route to it;
each record carries its containment label. LLM players read `players/prompts/publication_bias_v2.txt`. The
prompt every v1 record was produced with (`prompt_template_sha256`
`c2ded7861e854256`, all 72 published v1 records) is archived at
`players/prompts/_archive/publication_bias.c2ded7861e854256.txt`.

## v1 (superseded 2026-09-30, still published)

One verdict per dataset: `primary = 0.85 · correct + 0.15 · calibration`, where
`correct` is 1.0 iff `bias_detected` matched a constructed label (`unbiased`: a
symmetric funnel; `biased`: an explicit small-study coupling plus censoring), with the
datasets built far from the Egger decision boundary. `egger_p` and
`n_missing_trimfill` were optional and unscored. Output schema:
`schemas/output.schema.json`. Reference tool: `metafor-pubbias`
(`players/adapters/metafor_pubbias.R`). NOT comparable with v2.
