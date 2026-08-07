## v1 (2026-06-06) — initial task set
- metacheck-style arena: flag interpretive-language problems in manuscript prose.
- 3 injected mistake kinds (marginal_significance, spin_overclaim, causal_overclaim)
  + 3 matched clean-control flavours, 6 difficulty tiers, 17 tasks.
- Dual-benchmark from day one: revealed (committed seed 0) + private (secret seed
  + real holdout). Deterministic mistake-kind assignment → parity by construction;
  only sentence wording is seed-driven so the splits differ in content.
- Span-localisation F1 × calibration scoring; gold regenerated from seed
  (registry-free).
