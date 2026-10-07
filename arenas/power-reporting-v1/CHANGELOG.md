## v1 (2026-06-06) — initial task set
- metacheck-style FIELD-MAP arena: detect + classify + extract a power analysis
  from a methods/power excerpt.
- 6 structured fields (test, sample, alpha, power, effect_size, software),
  3 clean kinds (apriori / sensitivity / posthoc), 3 injected mistake kinds
  (posthoc_as_apriori, missing_fields, no_power_analysis), 6 difficulty tiers,
  19 tasks.
- Headline trap: post-hoc power analysis worded as a-priori (T4). T2 is the
  false-alarm trap (suspicious-but-clean excerpts).
- Dual-benchmark from day one: revealed (committed seed 0) + private (secret seed
  + real holdout). Deterministic mistake-kind assignment → parity by construction.
- Gold regenerated from seed (registry-free).
