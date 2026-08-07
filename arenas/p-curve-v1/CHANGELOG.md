## v1 (2026-06-08) — initial task set
- Computed-gold p-curve arena: per-set evidential-value classification of a set of
  significant findings via the right-skew test of Simonsohn, Nelson & Simons (2014).
  One verdict per task (evidential_value + confidence).
- 2 constructed labels (evidential / no-evidential), 6 difficulty tiers, 12 tasks
  (T1:2 small-k, T2:2 controls, T3:2 p-hacked trap, T4:2 chi2, T5:2 correlations,
  T6:2 largest-k; k ranges 5->30). Statistic type varied across tiers (t/F/z/chi2/r).
- Reference player: a FAITHFUL pure-R implementation of the Simonsohn et al. (2014)
  p-curve (players/adapters/pcurve.R), run as the `pcurve` player. dmetar (which wraps
  p-curve) is unavailable for R 4.4 and puniform implements p-uniform* (a sibling), so
  the published algorithm is implemented directly. Gold cross-validated: the R p-curve
  agrees with the computed label at mean primary ≈1.0 on the revealed split.
- DISTINCT from zcurve-evidential-v1 (z-curve EM mixture for expected
  discovery/replicability rate); p-curve tests the right-skew of the significant
  p-values. README spells out the distinction.
- Computed gold by construction (we know which sets were drawn from real effects vs
  null/p-hacked) AND independently via an in-generator pure-Python p-curve; the
  self-consistency test asserts the computed verdict matches the constructed label for
  every task in both splits.
- Sets built FAR from the right-skew boundary (evidential right_skew_p ≪ .01,
  no-evidential right_skew_p ≥ .55 via rejection sampling) so the deterministic tool
  agrees regardless of tiny numerical differences — the cross-validation guarantee for
  a computed-gold arena.
- Dual-benchmark from day one: revealed (committed seed 0) + private (secret seed +
  real holdout). Deterministic label assignment per tier => parity by construction.
- Registry-free (gold regenerated from seed); Claude baseline via the Claude Code CLI
  (Claude Max, no API key).
