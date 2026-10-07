## v1 (2026-08-04, cycle 7) — percentage GRIM added in place
- +1 injected issue kind: `grim_percent_inconsistent`. A reported PERCENTAGE of a
  count is k/N*100 for an integer k, so it sits on a 100/N granularity grid — the
  same GRIM test applied to a proportion (`scrutiny::grim(percent = TRUE)`). Still
  GRIM, still in scope; GRIMMER/SD remains out of scope.
- +matched clean control `clean_percent`: achievable-but-odd-looking percentages
  that must NOT be flagged. Without it, "flag every percentage" would win.
- Statistics now carry a `stat_type` discriminator (`mean` | `percent`). Input
  schema declares both shapes with per-type conditional requirements.
- Tier ladder: T2 gains a percentage false-alarm trap; T4 gains two mixed tasks
  (an impossible % among clean %, and a mixed mean/% table); T5/T6 interleave both
  kinds so no player can specialise in one detector. 13 -> 16 tasks.
- Tests 13 -> 30, including the four-places taxonomy drift guards (schema enum,
  player prompt, input-schema field coverage, reference-adapter dispatch).
- Cross-validated: `scrutiny-grim` scores **1.00 on all 16 revealed tasks** against
  the broadened gold — the reference implementation agrees with every new
  injection and every new clean control.

### Rejected — do NOT re-add: `grim_subgroup_n_mismatch`
Cycle 7 also prototyped a "wrong denominator" kind: a mean achievable at the full
sample N, printed against a smaller subgroup n where it is impossible. The
arithmetic is sound (verified against scrutiny: "4.24" is consistent at n=80,
inconsistent at n=27) and the error is real in the literature — but it is
**observationally identical** to a plain off-grid mean. The obvious discriminator
("the value is achievable at some larger n") is equally true of ordinary
`grim_inconsistent` means: measured over the revealed split it labelled 9 of 18
correctly and 9 incorrectly, i.e. chance. No player could infer the kind from the
input, so it would deflate `kind_accuracy` for reasons unrelated to skill. It
could only be made legitimate by putting the discriminating evidence IN the input
(e.g. rendering the subgroup row beside the total row it was copied from) — a
cross-row task shape, not a drop-in kind.

## v1 (2026-06-07) — GRIM-only task set
- metacheck-style arena: per-statistic GRIM granularity-consistency checking of
  reported MEANS (granularity of means). Single-item scores throughout (n_items=1),
  so the mean has granularity 1/N — the exact convention `scrutiny::grim()` models.
- 1 injected issue kind (grim_inconsistent), 6 difficulty tiers, 13 tasks
  (T1:3 clean, T2:3 controls-trap, T3:1 single, T4:1 subtle-one-ULP, T5:3 multi,
  T6:2 max-density).
- GRIMMER / SD-granularity is OUT OF SCOPE: SD range-plausibility (variance
  impossibilities) is covered by the separate sprite-plausibility-v1 arena. The
  reported SD here is descriptive context only.
- Cross-validated: the reference tool `scrutiny::grim()` (run as the `scrutiny-grim`
  player) agrees with the gold at mean composite ≈1.0 on the revealed split.
- Dual-benchmark from day one: revealed (committed seed 0) + private (secret seed +
  real holdout). Deterministic issue-kind assignment → parity by construction.
- Gold computed rigorously and re-verified independently in tests; clean means built
  from real integer data, injected means provably off-grid (GRIM search).
  Registry-free (gold regenerated from seed).

### Fixed (prior gold disagreed with scrutiny at ~0.61)
- Removed the n_items granularity mismatch: the old generator modelled scores as the
  SUM of n_items integer items (granularity 1/(N·n_items) vs scrutiny's 1/N), so
  scrutiny missed injected GRIM errors on multi-item scales. All scales are now
  single-item.
- Removed grimmer_inconsistent: the old "GRIMMER" injection set an SD above the
  max-variance RANGE bound — a SPRITE/range impossibility, not a GRIMMER granularity
  violation — so scrutiny's grimmer() correctly did not flag it. Dropped from the
  generator, schema, adapter, tests, and docs.
