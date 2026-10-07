## v1 (2026-06-07) — initial task set
- SPRITE-style arena: per-statistic summary-stat plausibility checking — can a sample
  of N integers in [scale_min, scale_max] produce the reported (mean, SD)? Reference
  tool: the R package rsprite2.
- 2 injected impossibility kinds (impossible_mean, impossible_sd), 6 difficulty tiers,
  15 tasks.
- Gold COMPUTED with rigorous sufficient conditions: clean stats are real sample
  statistics (achievable by construction); impossible_mean puts the mean past a scale
  bound; impossible_sd reports an SD whose square strictly exceeds the maximum variance
  (scale_max - mean)*(mean - scale_min). Never a false flag.
- Dual-benchmark from day one: revealed (committed seed 0) + private (secret seed +
  real holdout). Deterministic issue-kind assignment → parity by construction.
- Gold regenerated from seed (registry-free).
