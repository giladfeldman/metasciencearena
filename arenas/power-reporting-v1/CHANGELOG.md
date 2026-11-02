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

## v2 (2026-10-23) - DEVELOPMENT, not declared in arena.yaml
- The PowerBench extraction contract (per-analysis fields, verbatim quotes with
  offsets, explicit missing), built by `arenas/power_review_v4.py`. Integrates the
  Codex draft arenas `power-review-extraction-v1..v3`, which never had their own
  served identity; their attempts keep their original labels.
- Formula `power-reporting-EXTRACT-SPAN-v2`; output schema `schemas/v2/`.
  NOT comparable with v1. v1's generator path, tasks and formula are unchanged
  (pinned by `tests/test_power_review_integration.py`).
- See `DEVELOPMENT_v2.md` for why this is a task set and how to promote it.
