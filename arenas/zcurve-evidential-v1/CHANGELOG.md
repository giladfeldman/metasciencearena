## v1 (2026-06-07) — initial task set
- Single-verdict arena: classify a SET of significant z-scores as having
  evidential value (real effects) or not (selection-for-significance over nulls).
  Reference tool: the R package `zcurve` (z-curve 2.0).
- Gold = the generating REGIME the arena controlled (evidential = real effects
  with high true power; non_evidential = null effects selected for significance).
  The arena does NOT run zcurve to label.
- 2 regimes, 6 difficulty tiers (set size + effect strength + contamination set
  difficulty), 18 tasks (3 per tier).
- Dual-benchmark from day one: revealed (committed seed 0) + private (secret seed
  + real holdout). Deterministic (index-driven) regime assignment per tier → both
  regimes appear in both splits → parity by construction.
- Gold regenerated from seed (registry-free).
