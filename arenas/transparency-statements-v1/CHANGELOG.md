## v1 (2026-07-01) — broadened injected-mistake taxonomy (in-place)
- Added three real, less-synthetic failure modes (from the transparency/citation
  failure-mode catalog), each with a matched CLEAN CONTROL look-alike:
  - `false_open_claim` — an open-practices field (data/code/materials) ASSERTS
    openness ("All data are openly available in a public repository.") but quotes
    NO link. Gold: available=False, on_request=False, url=None. Look-alike: a genuine
    openness claim WITH a real link (the only differentiator is link vs no-link).
  - `false_prereg_claim` — "This study was preregistered prior to data collection."
    with NO registry link/ID. Gold: prereg available=False, url=None. Look-alike: a
    real registration link.
  - `funding_on_request` — a funding line that defers disclosure ("funding details
    are available on request") instead of naming a source: NOT a real funding
    statement. Gold: present=False. Look-alike: a genuine "no external funding"
    declaration (present=True), now seeded into the T2 false-alarm trap.
- MISTAKE_KINDS grows 4 → 7; assignment stays index-cycled / deterministic
  (seed-independent structure), so revealed & private hit the IDENTICAL matrix —
  parity is now enforced at count_tolerance 0.
- n_tasks 23 → 33 (T1:3 + T2:3 + T3:7 + T4:13 + T5:5 + T6:2). n_mistakes axis max
  4 → 5 (T6 task 0 injects five field-disjoint kinds). T6 restructured so the two
  tasks' kinds are field-disjoint (missing_funding vs funding_on_request never
  co-occur) and their union spans all seven kinds.
- No scorer change: the new kinds are expressed purely through existing field
  states (a fooled player marks the field present/available and is penalised with
  statement_false_positive + a field-accuracy drop). The player prompt
  (players/prompts/transparency_statements.txt) was extended to teach the three
  modes (untestable-kind guard). No output-schema enum exists to extend.

## v1 (2026-06-06) — initial task set
- metacheck-style field-map arena: transparency-statement detection/extraction.
- Six transparency fields (COI, funding, data, code, materials, prereg), 6
  difficulty tiers, 23 tasks.
- Four injected mistake kinds: missing_coi, missing_funding,
  data_on_request_not_real, placeholder_url. Cycled deterministically (index-driven)
  so both splits cover every kind.
- Dual-benchmark from day one: revealed (committed seed 0) + private (secret seed +
  real holdout). Deterministic mistake-kind assignment → parity by construction.
- Gold regenerated from seed (registry-free).
