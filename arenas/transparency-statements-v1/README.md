# transparency-statements-v1

**Challenge:** Given a manuscript transparency section (plain text), extract the
structured transparency **field map** — conflict-of-interest and funding
statements (present/absent + text), and open-practices availability for
**data / code / materials / preregistration**, distinguishing a *real repository
link* from an *"available on request"* non-statement and from a *placeholder/broken
URL* presented as a repo. Mirrors
[metacheck](https://github.com/quest-bih/metacheck)'s `coi_check`, `funding_check`,
`open_practices` and `all_urls` modules.

This arena is built on ScienceArena's revealed/private dual-benchmark framework
(see `contract/README.md` → "Revealed/private dual benchmark").

## Input → output

- **Input** (`schemas/input.schema.json`): `{text}` — the transparency-section text.
- **Output** (`schemas/output.schema.json`):
  `{coi:{present,statement}, funding:{present,statement},
    data:{available,on_request,url}, code:{…}, materials:{…},
    prereg:{available,url}, confidence}`.
  `available` is true **only** when the artifact is openly available at a real
  repository link — not on request, not behind a placeholder/broken URL.

## Injected mistake kinds

Cycled deterministically (so both splits cover all of them):

- `missing_coi` — the competing-interests statement is omitted.
- `missing_funding` — the funding statement is omitted.
- `data_on_request_not_real` — an open-practices field says "available on request"
  instead of giving a real repository link (`available=false, on_request=true`).
- `placeholder_url` — a fake/broken URL is presented as a real repository
  (`available=false`, `url` non-null).

The **clean** variant has a real COI + funding statement and real repository URLs
for data/code/materials/prereg.

## Difficulty tiers

T1 clean/simple · T2 **paraphrase-consistent (false-alarm trap)** · T3 single
injected mistake · T4 subtle mistake (one URL mistake hidden amid paraphrased
fields) · T5 multiple mistakes · T6 full composition. The hardest discrimination
is T2 vs T3: a reworded-but-real statement is **not** a mistake; a missing
statement or a fake/on-request link **is**.

## Scoring

`composite = field_accuracy × url_judgement × calibration`.
`field_accuracy` is macro accuracy over the present/absent (and on_request) flags
across the six fields; `field_f1` is statement-present-vs-absent F1 (the T2 trap
punishes false positives here); `url_judgement` is the fraction of URL
availability claims judged correctly (real repo vs on-request/placeholder);
calibration is `1 - ECE` over the player's confidence. Findings:
`statement_missed`, `statement_false_positive`, `url_misjudged`.

## Revealed vs private

- **Revealed** (`--split revealed`, seed 0): published gold + full per-task results
  in the Open Benchmark, so developers can reproduce and audit scoring.
- **Private** (`--split private`): secret seed (`task_sets/v1/.private_seed`) + any
  hand-curated real transparency sections under `_held_out/`; findings redacted,
  scores on the official leaderboard. Same tier matrix and mistake-kind coverage as
  revealed — enforced by `python tools/check_parity.py transparency-statements-v1`.

Gold is regenerated from the seed (no committed answer key, no external registry).

## Players

metacheck (tool), oddpub / rtransparent (tools), a Claude-via-CLI baseline, and a
trained human coder. Register in `players/registry.yaml`.
