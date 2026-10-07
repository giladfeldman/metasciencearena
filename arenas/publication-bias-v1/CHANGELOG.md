## v2 (2026-09-30) — score the statistics, not a verdict {#v2}

- **Why.** Owner decision 2026-09-17: how to turn publication-bias statistics into a
  "biased / not biased" verdict is an unresolved methodological question, so v1's
  verdict score measured an answer nobody can defend. And v1 told players the
  statistics were OPTIONAL, so scoring them retroactively would punish obedience —
  v1 is not rescored; v2 is a new task set with a new prompt.
- **What is scored.** 11 statistics from six methods: Egger (p), Begg (tau, p),
  trim-and-fill (k0, estimate), PET-PEESE (both estimates), the three-parameter
  selection model (estimate, LRT p) and RoBMA (inclusion BF for bias, model-averaged
  mu). Each earns band credit; `primary` = weighted mean over the six methods —
  RoBMA 0.30, PET-PEESE 0.225, 3PSM 0.225, Egger 0.10, Begg 0.075, trim-and-fill
  0.075 (owner decision 2026-10-01: modern methods 0.75). Omitted = 0. Formula id
  `publication-bias-STATS6-MODERNW-v2`.
- **v1 is DEPRECATED** (owner decision 2026-10-01): its verdict score rests on an
  unresolved methodological question. It stays published under `versions/v1/`,
  labelled deprecated; nothing is deleted.
- **Gold** from R (metafor 5.0.1, RoBMA 3.6.1, JAGS 4.3.1, R 4.4.0) via
  `tools/build_gold_v2.py`, bound to each dataset by sha256. REML with `stepadj`
  (default Fisher scoring fails on some sets); 3PSM from `metafor::selmodel` because
  `weightr` crashed on 2 of 12 revealed sets.
- **Datasets.** New generator path (`_generate_v2`, task ids `pubbias2-*`), branching
  before any v1 rng call: v1 generation is byte-identical (pinned by
  `test_v1_generation_is_byte_identical_to_its_committed_gold` and the v1
  fingerprint). Conditions: no selection, p-value selection, small-study coupling, at
  moderate strength so p-values spread (v1's biased sets had Egger p down to 1e-13).
- **Found on the way.** v1's pure-Python Egger p-value is wrong when |t| is small:
  `_betai` runs its continued fraction without the symmetry transformation, so it
  converges badly as x → 1 (df = 30, t = 0.01: 0.511 instead of 0.992). On v1's
  revealed split this moved one gold `egger_p` (pubbias-t5-0-s0: 0.937 vs the correct
  0.972). v1 never scored `egger_p` — it fed only the diagnostic `egger_p_abs_err` and
  the acceptance tests for drawing the sets — so no published v1 score moves.
  MEASURED on all 24 v1 datasets (both splits) against scipy: largest gap 0.0356,
  0 verdict flips at alpha 0.10, and every set still satisfies its construction
  threshold (clean p >= 0.25, biased p <= 0.02) under the correct p. It is NOT fixed
  in v1: `egger_p` drives v1's rejection sampling, so correcting it would redraw v1's
  datasets and break the frozen fingerprint. The handoff's "two valid Egger implementations disagree by
  0.0356" was this bug, not a disagreement between methods. v2 computes no p-value in
  Python.
- **NOT comparable with v1.** Different datasets, question and score. v1's records
  and page stay published under `versions/v1/`; nothing is retracted.
- **Reference reproducibility (2026-10-06, debt d-125b6a).** The reference run v2c6
  returned a different seed-1 RoBMA answer on 2 of 12 sets (|Δ ln BF| 0.088 and 0.264
  against the gold, outside the full-credit band); no later refit reproduced it, and the
  cause is not identified. Tested and ruled out: one dropped sub-model, one silently
  restarted sub-model (each of 36), package drift, BLAS threading, input differences.
  The 60-minute limit cannot explain t4-0 either, although that fit spanned 381 min (5.4 h
  of it in Modern Standby): the limit is consulted only when a sub-model needs extra
  samples, and at seed 1 none of t4-0's sub-models does (all stop at 5000, measured in
  four refits with and without autofit). t1-1 deviated after 11 min with no standby.
  The reference script now runs RoBMA without its 60-minute wall-clock autofit limit and
  without silent JAGS restarts, stops on any failed sub-model, and can log per-sub-model
  diagnostics (`PUBBIAS_ROBMA_DIAG_DIR`). Under the new settings all 24 gold fits (both
  splits) reproduce the committed gold exactly, and the reference re-run v2c7
  reproduces its earlier seed-1 output on 12 of 12 sets (primary 1.000). Because the cause
  is unknown, this does not prove a recurrence is impossible (2 deviations in the 31 reference
  fits up to v2c6; 0 in the 47 full fits since, 23 at seed 1 and 24 at the gold seed): a recurring deviation in a reference record fails
  `test_robma_full_credit_band_covers_measured_mcmc_noise`, and the gold cannot carry
  one, since all 24 gold fits reproduced exactly. Gold, bands,
  weights and the other five players' records are unchanged. v2c6 reference record
  archived under `runs/v2/_archive/2026-10-06-robma-not-reproducible-v2c6/`.

## v1 (2026-06-08) — initial task set
- Computed-gold meta-analysis arena: per-set small-study / publication-bias detection
  (funnel-plot asymmetry). One verdict per task (bias_detected + confidence).
- 2 constructed labels (unbiased / biased), 6 difficulty tiers, 12 tasks
  (T1:2, T2:2 controls-trap clean-only, T3:2, T4:2, T5:2, T6:2; k ranges 10->40).
- Reference tool: metafor (Egger's regtest + trimfill), run as the `metafor-pubbias`
  player. Gold cross-validated: metafor agrees with the computed label at mean
  primary ≈1.0 on the revealed split.
- Computed gold by construction (we know which sets were biased) AND independently via
  an in-generator pure-Python Egger test; the self-consistency test asserts the Egger
  verdict matches the constructed label for every task in both splits.
- Datasets built FAR from the Egger boundary (biased p ≪ 0.01, unbiased p ≥ ~0.2) so
  the deterministic tool agrees regardless of tiny numerical differences — the
  cross-validation guarantee for a computed-gold arena.
- Dual-benchmark from day one: revealed (committed seed 0) + private (secret seed +
  real holdout). Deterministic label assignment per tier => parity by construction.
- Registry-free (gold regenerated from seed); Claude baseline via the Claude Code CLI
  (Claude Max, no API key).
