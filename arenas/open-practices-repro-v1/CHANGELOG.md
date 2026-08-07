## v1 (2026-07-01) — open-practices-reporting expansion (in-place)
- Broadened the injected-mistake taxonomy from 4 to **7 kinds**, adding three
  REAL open-practices-reporting failure modes (catalog `false_open_claim` family),
  each with a matched CLEAN look-alike that must NOT be flagged:
  - **dead_data_link** — a data/code availability statement claims open data at a
    DEAD/placeholder URL. Clean control: a genuine resolvable OSF/Zenodo/figshare DOI.
  - **available_upon_request** — data offered only "from the authors upon reasonable
    request" (not an open practice). Clean control: data openly available at a real
    public link/DOI.
  - **materials_claim_no_link** — a materials claim with NO link AND no materials
    file present. Clean control: a real materials repository with the materials/
    file actually present in files[].
- New availability-statement doc targets (`DATA_AVAILABILITY.md`, `MATERIALS.md`)
  carried by every task; clean controls appear in T1/T2/T4, defects in T3/T4/T5/T6.
- Tasks **14 → 20** (T3 now cycles all 7 kinds: 7 tasks; T4 cycles 2 code + 3
  open-practices kinds: 5 tasks). Kind assignment stays index-cycled/deterministic
  → revealed/private parity holds exactly (`count_tolerance: 0`).
- Output schema `issue_kind` now carries an explicit `enum` (drift-guarded); player
  prompt teaches the three new kinds and their clean look-alikes. Scorer unchanged
  (the new kinds reuse the existing flag/kind matching).

## v1 (2026-06-06) — initial task set
- metacheck-style arena: repo/code reproducibility defect detection on MOCKED
  repository snapshots (no live OSF/GitHub fetch).
- 4 injected issue kinds (absolute_path, uncommented_code, missing_file_load,
  broken_link), 6 difficulty tiers, 14 tasks.
- T2 is a false-alarm trap: files that MENTION absolute paths / renames in their
  comments but actually use relative paths to present files — clean, must not be
  flagged.
- Dual-benchmark from day one: revealed (committed seed 0) + private (secret seed
  + real holdout). Deterministic issue-kind assignment → parity by construction.
- Gold regenerated from seed (registry-free).
