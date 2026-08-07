# prereg-extraction-v1

**Challenge:** Given a paper's text, find its preregistration link (OSF or
AsPredicted) and extract the registration into a structured field-map. This is a
FIELD-MAP arena: the player both *detects* a prereg and *recovers its content*.

Built on ScienceArena's revealed/private dual-benchmark framework (see
`contract/README.md` → "Revealed/private dual benchmark").

## Input → output

- **Input** (`schemas/input.schema.json`): `{text}` — paper text that may embed an
  OSF/AsPredicted link plus the registration content.
- **Output** (`schemas/output.schema.json`):
  `{prereg_found: bool, platform: "osf"|"aspredicted"|null, link: string|null,
  fields: {hypotheses, design, sample_size, analysis_plan}, confidence}`.

## Injected mistakes (parity-covered)

Six injected mistake kinds, cycled deterministically so both splits cover all of
them. The first three are field-map mistakes; the last three are real OSF
preregistration-integrity failures mined from the AbusingPreReg Tier-D taxonomy
(modules 12/13/17), each paired with a confusable CLEAN look-alike.

- **no_prereg** — no registration is embedded; the correct answer is
  `prereg_found=false`.
- **prereg_plaintext** — a real registration that is *unstructured plain text*
  (link present, fields unlabelled), so field-mapping is hard.
- **wrong_platform_schema** — AsPredicted's numbered-question fields sitting under
  an OSF link, or OSF's prose labels under an AsPredicted link.
- **viewonly_instead_of_doi** (module 13) — the paper links only an *anonymized
  OSF view-only* URL (`osf.io/<id>/?view_only=<token>`) rather than the canonical
  registration link. The registration is real and readable, so `prereg_found=true`,
  `platform=osf`, fields present; the player must recover the canonical `osf.io/<id>`
  link (dropping the view-only token). Clean twin: the canonical OSF registration
  **DOI** (`doi.org/10.17605/OSF.IO/<ID>`) — also OSF, also found.
- **embargoed_at_publication** (module 12) — the registration is referenced but
  still under embargo at publication: `prereg_found=true`, `platform=osf`, but the
  four fields are unrecoverable (`null`). Clean twin: an embargo that **lifted**
  before publication, so the fields are public and must be extracted.
- **withdrawn_still_cited** (module 17) — the paper still cites a **withdrawn**
  (tombstone) registration; the plan no longer exists, so there is no usable
  preregistration → `prereg_found=false`, even though the text claims
  pre-registration and shows a dead link. Clean twin: a live registration cited
  correctly (`found=true`, fields present).

Clean variants (no injected mistake): a clean structured OSF registration, a clean
structured AsPredicted registration, and the three integrity look-alikes above
(`doi_clean`, `embargo_lifted`, `live_cited`).

## Difficulty tiers

T1 clean structured regs (incl. the DOI / embargo-lifted / live-cited
look-alikes) · T2 **decoy false-alarm trap** (text discusses prereg but embeds
none) · T3 single injected mistake (one per kind) · T4 **subtle** — each integrity
abuse placed beside its confusable clean twin, plus the plain-text reg · T5
multiple injected mistakes · T6 full composition. The hardest discriminations are
T2 (and withdrawn) vs a real reg — discussing or *having claimed* preregistration
is *not* the same as a usable one — and view-only/DOI links vs a wrong platform.

## Scoring

`composite = detection × platform_acc × field_f1 × calibration`. `detection` is
1.0 when `prereg_found` matches gold; `platform_acc` is the osf-vs-aspredicted
label accuracy when a reg is present; `field_f1` is the mean token-overlap F1 of
the four extracted fields; calibration is `1 - ECE` over the detection
confidence. Findings: `prereg_missed`, `prereg_false_alarm`, `platform_mislabel`,
`field_wrong`.

## Revealed vs private

- **Revealed** (`--split revealed`, seed 0): published gold + full per-task results
  in the Open Benchmark.
- **Private** (`--split private`): secret seed (`task_sets/v1/.private_seed`) + any
  hand-curated real papers under `_held_out/`; findings redacted, scores on the
  official leaderboard. Same tier matrix and mistake-kind coverage as revealed —
  enforced by `python tools/check_parity.py prereg-extraction-v1`.

Gold is regenerated from the seed (no committed answer key, no external registry).
