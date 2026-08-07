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
