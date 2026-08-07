## v1 (2026-07-01) — broaden taxonomy: invalid_doi, predatory_source, tortured_phrase
- Added 3 real, less-synthetic injected issue kinds from the citation-integrity
  failure-mode catalog (CitationGuard/scimeto, referencecheck, torturedcheck):
  - `invalid_doi`: the DOI is structurally malformed / non-resolvable (dropped
    "10." prefix, letter-O-for-zero, truncated suffix). Clean twin: a VALID DOI
    that legally contains parentheses / angle-brackets / colons (e.g. an
    old-style Wiley SICI DOI) — must NOT be flagged.
  - `predatory_source`: the venue is a predatory / hijacked journal. Clean twin:
    a legitimate but small / broad-scope / mega-journal (Heliyon, PLOS ONE).
  - `tortured_phrase`: the title carries a paper-mill synonym-swapped paraphrase
    ("irregular timberlands" for "random forests"). Clean twin: genuine domain
    jargon / eponyms ("The Hungarian method", "Idiot's Bayes").
- Issue kinds 6 → 9; tasks 19 → 25 (T3 now cycles all 9 kinds; T4 adds 3 subtle
  defect-beside-its-confusable-clean-twin tasks). All assignment stays
  index-cycled/deterministic, so revealed & private remain in parity (categories
  match, count_tolerance 0). New kinds added to output schema enum AND the player
  prompt; a drift-guard test asserts both. Every reference now carries a `venue`
  field (added to input schema) so predatory_source is a same-shaped comparison.
- New catalog: `catalogs/clean_controls.yaml` (the confusable honest-twin pools).

## v1 (2026-06-06) — initial task set
- metacheck-style arena: per-reference integrity checking (retraction, metadata
  mismatch, dangling cited/uncited, missing replication, miscitation).
- 6 injected issue kinds, 6 difficulty tiers, 19 tasks.
- Dual-benchmark from day one: revealed (committed seed 0) + private (secret seed
  + real holdout). Deterministic issue-kind assignment → parity by construction.
- Gold regenerated from seed (registry-free).
