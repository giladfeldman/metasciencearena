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
