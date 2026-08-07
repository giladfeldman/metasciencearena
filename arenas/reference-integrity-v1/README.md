# reference-integrity-v1

**Challenge:** Given a paper's reference list and the set of in-text citation
markers, judge — per reference — whether it has an *integrity* problem, and what
*kind*. This is about INTEGRITY, not parsing or linking (see
`pdf-reference-parsing-v1` / `pdf-citation-matching-v1` for those). Modeled on
[metacheck](https://github.com/)'s `ref_*` integrity modules.

This arena is built on ScienceArena's revealed/private dual-benchmark framework
(see `contract/README.md` → "Revealed/private dual benchmark").

## Input → output

- **Input** (`schemas/input.schema.json`):
  `{references: [{reference_id, authors, year, title, doi, venue, cited_in_text}],
  in_text_marker_ids: [...]}`.
- **Output** (`schemas/output.schema.json`):
  `{records: [{reference_id, issue_kind, flagged, confidence}]}` — one record per
  reference (`issue_kind` is null when `flagged=false`). A `dangling_missing`
  marker is flagged on its `in_text_marker_id`.

## Injected issue kinds

`retracted` (the DOI carries a retraction marker the arena injected — the arena
KNOWS, since it injected it; no live database), `metadata_mismatch` (an
author/year/title altered away from the canonical value the generator stores),
`dangling_uncited` (a reference listed but never cited in the body),
`dangling_missing` (an in-text marker id with no matching reference),
`replication_uncited` (an original whose known replication is absent from the
list), `miscitation` (a reference whose listed attribution is a known-wrong
year/title), `invalid_doi` (a structurally malformed / non-resolvable DOI),
`predatory_source` (a predatory/hijacked venue swapped for the reputable one),
`tortured_phrase` (a paper-mill synonym-swapped title). Clean references have
none. Every new kind ships a **confusable clean twin** (a valid-but-unusual DOI,
an obscure-but-reputable venue, legitimate jargon) in `catalogs/clean_controls.yaml`
that a good player must NOT flag. See `catalogs/references.yaml`.

## Difficulty tiers

T1 clean/simple · T2 **controls-only (false-alarm trap)** — includes the
confusable honest twins (unusual-but-valid DOI, obscure-but-reputable venue,
legitimate jargon) · T3 single injected issue (cycles all 9 kinds) · T4 subtle
issue placed beside its confusable clean twin · T5 multiple issues · T6 full
composition. The hardest discrimination is the clean twin vs the real defect: a
reference that *looks* odd is not necessarily flawed; an altered attribution,
malformed DOI, predatory venue, or tortured title *is*.

## Scoring

`composite = detection_f1 × kind_accuracy × calibration`. Detection F1 is over
flagged vs not-flagged across references; `kind_accuracy` is the fraction of
correctly-flagged references given the right `issue_kind`; calibration is
`1 - ECE` over confidences. Findings: `integrity_missed`,
`integrity_false_alarm`, `kind_mislabel`.

## Revealed vs private

- **Revealed** (`--split revealed`, seed 0): published gold + full per-task
  results in the Open Benchmark, so developers can reproduce and audit scoring.
- **Private** (`--split private`): secret seed (`task_sets/v1/.private_seed`) +
  any hand-curated real reference lists under `_held_out/`; findings redacted,
  scores on the official leaderboard. Same tier matrix and issue-kind coverage as
  revealed — enforced by `python tools/check_parity.py reference-integrity-v1`.

Gold is regenerated from the seed (no committed answer key, no external
registry).

## Players

metacheck (tool, brings its own dependencies), a Claude-via-CLI baseline, and a
trained human coder. Register in `players/registry.yaml`.
