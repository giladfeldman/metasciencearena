## v1 (2026-07-01) — broadened defect taxonomy (in-place)
- Added 3 real, less-synthetic completeness/precision defect kinds, each with a
  matched CLEAN CONTROL look-alike (the honest version that must NOT be flagged):
  - `unspecified_test` — a significance claim with a p-value but NO test statistic
    named (vs the ordinary complete sentence, which names the statistic).
  - `no_correction` — many comparisons interpreted at p < .05 with NO multiplicity
    correction (vs `clean_corrected`: the same report that DOES state a Bonferroni/
    Holm/Tukey/FDR correction). From AbusingPreReg module 4 (Specification).
  - `percent_count_mismatch` — a "<count> of <N> (<pct>%)" whose % is arithmetically
    inconsistent with count/N (vs `clean_count_match`: a consistent clause).
- 9 defect kinds now (was 6). T3/T4 cycle all 9 (range(n_kinds)); T2 trap and T4
  subtle now carry the new clean-control look-alikes (consistent-% / corrected-report)
  so the false-alarm discrimination is harder. n_tasks 23 -> 30 (T3 6->9, T4 6->9,
  T6 2->3). Assignment stays index-driven (no rng) so revealed/private parity holds
  at count_tolerance 0. Schema enum + player prompt updated with the 3 new kinds.

## v1 (2026-06-06) — initial task set
- New arena: statistical-reporting completeness / precision detection on results-section
  text (no statistic recomputation — that is stats-extraction-v1).
- 6 test-report templates, 6 injected defect categories (missing_effect_size,
  imprecise_p, impossible_p_zero, missing_ci, missing_df, nonsig_as_support),
  6 difficulty tiers, 23 tasks.
- Dual-benchmark from day one: revealed (committed seed 0) + private (secret seed
  + real holdout). Deterministic defect-category assignment → parity by construction.
- Span-aware scoring: flags matched by category + span overlap (containment or
  IoU ≥ 0.5); composite = detection_f1 × calibration.
- Gold regenerated from seed (registry-free).
