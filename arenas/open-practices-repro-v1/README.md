# open-practices-repro-v1

**Challenge:** Given a MOCKED snapshot of a referenced data/code repository (file
names, types, and contents — no live OSF/GitHub fetch), flag reproducibility
defects per target. Modeled on
[metacheck](https://github.com/JamieCummins/metacheck)'s `repo_check` /
`code_check` modules.

This is the hardest arena in the metacheck family: it requires distinguishing
genuine defects from files that merely *look* suspicious.

## Input → output

- **Input** (`schemas/input.schema.json`): `{repo_url, files: [{name, type,
  content}], targets}`. `targets` is the closed set of candidate targets to
  assess — every file name plus the `repo_url` — so scoring is well-defined.
- **Output** (`schemas/output.schema.json`): `{records: [{target, flagged,
  issue_kind, confidence}]}` — one record per target. `target` is a file name,
  or the `repo_url` for a `broken_link`.

## Injected issue kinds (= mistake_kinds, cycled deterministically)

- **absolute_path** — code with a hard-coded absolute path (`C:/Users/...` or
  `/home/...`) instead of a relative one.
- **uncommented_code** — a code file with no comment lines at all.
- **missing_file_load** — code that `load()`s / `read_csv()`s a file that is NOT
  present in `files[]`.
- **broken_link** — `repo_url` is a placeholder / 404-style link (target = the
  `repo_url`, a repo-level defect).
- **dead_data_link** — a data/code availability statement (e.g.
  `DATA_AVAILABILITY.md`) claims open data/code AT A URL that is a placeholder /
  404-style link (target = the statement doc).
- **available_upon_request** — an availability statement offers data/materials
  only "from the authors upon request" instead of an open public link/DOI (NOT an
  open practice; target = the statement doc).
- **materials_claim_no_link** — a statement claims study materials are "available"
  but gives no link/DOI and no materials file is present (target = the statement
  doc).

Clean files / controls: commented code with relative paths loading files that ARE
present; a real `repo_url`; an availability statement that links a **genuine
resolvable DOI**; a materials statement that links a **real materials repository**
with the `materials/` file actually present. The open-practices clean look-alikes
must NOT be flagged.

## Difficulty tiers

T1 clean/simple · T2 **false-alarm trap** (files that mention absolute paths /
renames in their *comments* but actually use relative paths to present files —
a good player must NOT flag them) · T3 single injected defect (cycles every one
of the 7 kinds) · T4 subtle defect buried among trap files (cycles the code
defects AND each open-practices statement defect next to its confusable clean
look-alike) · T5 multiple defects · T6 full composition. The hardest
discrimination is T2/T4: a suspicious-looking comment is *not* a defect and a
genuine DOI is *not* a dead link; a hard-coded path *is* a defect and a
placeholder data URL *is* a false open claim.

## Scoring

`composite = detection_f1 × kind_accuracy × calibration`. Detection F1 is over
defect vs no-defect across all candidate targets; `kind_accuracy` is the fraction
of correctly-flagged defects given the right `issue_kind`; calibration is
`1 - ECE` over per-target confidences. Findings: `repro_issue_missed` (major),
`repro_issue_false_alarm` (major), `kind_mislabel` (minor).

## Revealed vs private

- **Revealed** (`--split revealed`, seed 0): published gold + full per-task
  results in the Open Benchmark, so developers can reproduce and audit scoring.
- **Private** (`--split private`): secret seed (`task_sets/v1/.private_seed`) plus
  any hand-curated real repository snapshots under `_held_out/`; findings
  redacted, scores on the official leaderboard. Same tier matrix and issue-kind
  coverage as revealed — enforced by
  `python tools/check_parity.py open-practices-repro-v1`.

Gold is regenerated from the seed (no committed answer key, no external
registry).

## Players

metacheck (tool, brings its own LLM dependency), a Claude-via-CLI baseline, and a
trained human coder. Register in `players/registry.yaml`.
