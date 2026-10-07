## v1 (2026-06-08) — initial task set
- Computed-gold conversion arena: convert one effect size from one metric to another
  across the closed set {d, r, OR, eta2, f}. One conversion per task
  (converted + confidence).
- 8 context-free conversion identities + 2 context-requiring (d<->r with group sizes),
  6 difficulty tiers, 22 tasks (T1:2, T2:2 round-trip controls, T3:8 all-pairwise,
  T4:2 needs-context, T5:4 extreme magnitudes, T6:4 small-magnitude mix).
- Reference tool: effectsize / esc, run as the `effectsize-convert` player. Gold
  cross-validated: the tool agrees with the canonical formula at mean primary ≈1.0 on
  the revealed split.
- Canonical formula set (matched function-for-function against effectsize): d<->r via
  r=d/sqrt(d^2+h) with h=(n1+n2-2)(1/n1+1/n2) (h=4 default); d<->OR via OR=exp(d*pi/sqrt(3));
  eta2<->f via f=sqrt(eta2/(1-eta2)); d<->f via f=d/2. Constants verified against
  effectsize to double precision BEFORE committing.
- Computed gold by the canonical formula AND independently re-derived in the
  self-consistency tests (alternative Python recomputation + round-trip closure) for
  every task in both splits.
- Values kept FAR from degenerate boundaries (|r|<~0.93, eta2<~0.85, OR in a sane
  range) so rounding cannot flip agreement — the cross-validation guarantee.
- Dual-benchmark from day one: revealed (committed seed 0) + private (secret seed +
  real holdout). Deterministic conversion-kind assignment per tier => parity by
  construction.
- Registry-free (gold regenerated from seed); Claude baseline via the Claude Code CLI
  (Claude Max, no API key).
