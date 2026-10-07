## v1 (2026-07-01) — broaden integrity-abuse taxonomy (in-place)
- Added 3 real preregistration-integrity failure modes from the AbusingPreReg
  Tier-D taxonomy, each with a matched confusable CLEAN look-alike:
  - viewonly_instead_of_doi (module 13): registration linked only via an
    anonymized OSF view-only URL (osf.io/<id>/?view_only=<token>) instead of the
    canonical link. Gold: found=True, platform osf, canonical osf.io link, fields
    present. Clean twin: the canonical OSF registration DOI
    (doi.org/10.17605/OSF.IO/<ID>).
  - embargoed_at_publication (module 12): registration referenced but still
    embargoed at publication. Gold: found=True, platform osf, but all four fields
    unrecoverable (null). Clean twin: an embargo that lifted before publication
    (fields public).
  - withdrawn_still_cited (module 17): a withdrawn tombstone registration still
    cited as pre-registered. Gold: found=False (no usable plan) despite the text
    claiming pre-registration and showing a dead link. Clean twin: a live
    registration cited correctly.
- Injected kinds 3 -> 6; clean controls now include doi_clean / embargo_lifted /
  live_cited (added to T1 and paired beside their abuse in T4).
- n_tasks 28 -> 40 (T1 5, T2 4, T3 6, T4 7, T5 7, T6 11). Form placement stays
  index-cycled (seed-free), so revealed/private parity is EXACT — count_tolerance
  tightened 0.2 -> 0.
- Player prompt extended with the three judgment calls (withdrawn->false;
  embargoed->found-but-null-fields; OSF DOI & view-only links are still platform
  "osf", recover the canonical link). Drift-guarded by a test. No output-schema
  enum change (the player never emits the abuse kind; platform stays osf/aspredicted).
- Scorer unchanged: the four existing categories (prereg_missed,
  prereg_false_alarm, platform_mislabel, field_wrong) already cover the new modes.

## v1 (2026-06-06) — initial task set
- Field-map arena: find a paper's preregistration link (OSF/AsPredicted) and
  extract {hypotheses, design, sample_size, analysis_plan}.
- 6 studies, 6 difficulty tiers, 28 tasks.
- Injected mistakes: no_prereg, prereg_plaintext, wrong_platform_schema; clean
  variants are a structured OSF reg and a structured AsPredicted reg.
- Dual-benchmark from day one: revealed (committed seed 0) + private (secret seed
  + real holdout). Deterministic mistake-kind assignment → parity by construction.
- Gold regenerated from seed (registry-free).
