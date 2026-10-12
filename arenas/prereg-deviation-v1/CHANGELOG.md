## v2 — UNRELEASED (built 2026-10-08) — optional summaries may be null
- Held off the version ladder: prereg-deviation is the human-evaluation arena, and
  releasing v2 changes the tasks human evaluators see and blocks scoring of v1 human
  answers. Owner decision pending.
- `registered_summary` / `paper_summary` are optional and unscored, but v1's schema
  typed them as plain strings and the prompt never said they were optional. A
  model that wrote null for "the preregistration says nothing here" had its whole
  answer rejected (35 records of cohere-command-r7b-prereg scored 0.0). v2's
  `schemas/v2/output.schema.json` allows null for both, and
  `players/prompts/prereg_deviation_v2.txt` says they are optional (owner decision
  2026-10-07, debt d-603203; Sonnet consult in
  docs/consult/2026-10-07-prereg-r7b-null-summary/).
- Scorer and dimensions unchanged. v2 task ids are `pdv2-…`; the private split's
  ids carry no seed (`…-private`).
- NOT comparable with v1: a fresh task draw and a changed prompt. v1 stays published.

## v1 (2026-06-29) — AbusingPreReg taxonomy broadening (in-place)
- +4 dimensions / deviation_kinds drawn from the AbusingPreReg detection modules,
  taking the arena beyond the "planned X vs reported Y" value-swap family into
  procedural-integrity failures the real detectors flag:
  - `posthoc_results_leak` (module 2, Temporal Anomaly): results-language leakage
    ("as expected", "confirmed our hypothesis") in what should be a forward plan.
  - `unresolved_contingency` (module 5): an "if X then Y" whose trigger/action is
    vague ("if assumptions are violated we will use an appropriate alternative").
  - `nondirectional_as_confirmatory` (module 6, Hypothesis Quality): an
    exploratory/non-directional aim reported as a confirmed directional hypothesis.
  - `maineffect_to_interaction` (module 9, Reg-Pub Deviation): the focal test is
    reframed from a pre-specified main effect to an interaction.
  Each ships a confusable clean control (resolved contingency, plainly-predicted
  direction, disclosed exploratory interaction) so the T2/T4 traps stay hard.
- 16 dimensions, 6 difficulty tiers, 46 tasks (was 12 dims / 38 tasks). Parity
  holds at count_tolerance 0 (every kind in both splits). Gold regenerates from
  seed; the deterministic generator is the gold oracle for this AI-only arena.

## v1 (2026-06-06) — initial task set
- Flagship regcheck-style arena: prereg↔paper deviation detection.
- 12 dimensions (10 general + 2 clinical-trial), 6 difficulty tiers, 38 tasks.
- Dual-benchmark from day one: revealed (committed seed 0) + private (secret seed
  + real holdout). Deterministic deviation/kind assignment → parity by construction.
- Gold regenerated from seed (registry-free).
