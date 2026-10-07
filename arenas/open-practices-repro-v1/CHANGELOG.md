## Catalog custody (2026-09-27) — no new task set; v1's pool leaves the public catalog

- **Why.** v2 stopped *generating* tasks around the real personal directory, but
  the v1 pool that contains it still sat in `catalogs/repo_templates.yaml`, which
  is mirrored to the public repository.
- **What changed.** The v1 pool moved, unedited, to
  `catalogs/abs_dirs_v1_as_published.yaml`, tracked in the private repository only
  (`framework.publish.PRIVATE_ARENA_PATHS` keeps it out of the mirror).
  `_load_catalog()` folds it in as `abs_dirs_v1` when present. The public
  catalog's `abs_dirs` carries an invented placeholder of the same shape in its
  place.
- **No task set changed.** In the private repository v1 still regenerates its
  committed gold byte-for-byte and its `generator_fingerprint` matches across all
  40 tasks; v2 is untouched. Without the private file — the public mirror — v1
  renders the placeholder: same task shape and gold kinds, different input hashes
  from the published v1, and the generator emits a `UserWarning` saying so. The
  three tests that need the exact published v1 skip there with that reason.
- **New gate.** `framework.publish.scan_for_leaks` now refuses any mirrored file
  containing a home-directory path or email that identifies the person building
  the mirror (read from the machine at run time, so no name is written down).

## v2 (2026-09-17) — new task set: one placeholder was not a placeholder

- **Why.** The `absolute_path` defect renders a script whose data path points at
  "the author's own machine", drawn from a pool of four strings in
  `catalogs/repo_templates.yaml`. They are meant to be fabricated. One was not: it
  is a real personal directory belonging to the repository owner, and it reached a
  public site. Not a credential and not a security hole, but the wrong thing for a
  benchmark about reproducible repositories to be publishing.
- **What changed.** A NEW pool, `abs_dirs_v2` — same length, same platform mix,
  same flavour of path, one string replaced by a fabricated sibling. `abs_dirs` is
  untouched and append-only; `generator._abs_dirs_for()` resolves the pool per
  version and refuses one of a different length (a different length changes which
  element each `rng.choice` lands on, i.e. task SHAPE).
- **Also fixed.** `_assemble()` hardcoded `task_set_version: "v1"` into every
  envelope. Any v2 run would have written records labelled v1, and the leaderboard
  builds one task set per arena, so those records would have been invisible or
  mis-attributed. The envelope now carries the version it was generated as.
- **Shape unchanged, measured on BOTH splits:** identical task_ids, tier
  histogram, n_issues histogram, injected-kind counts, per-task file counts,
  per-task gold-record counts.
- **NOT comparable with v1.** `_seed_int` hashes the task-set version into every
  cell's rng, so v2 is a fresh draw, not a re-render: 19 of 20 revealed and 19 of
  20 private tasks differ in text. v1 and v2 scores are not on the same
  instrument.
- **v1 is untouched and still reproduces**: its committed `_ground_truth.json`
  regenerates byte-for-byte and its `generator_fingerprint` is unchanged across
  all 40 tasks of both splits. Its 27 published records carrying the string are
  deliberately NOT retracted.
- **This removes nothing already published.** v1 is `superseded`, not deleted, and
  `build-data.mjs` keeps building a superseded version that has runs into
  `public/data/<arena>/versions/v1/` — records and gold included. v2 is a fix
  going forward. Until players are re-run against v2 the arena has 0 records on
  the current task set and `index.json` reports it as `design` rather than `live`.
- Pinned by `tests/test_task_set_v2.py`, which derives the replaced string from
  the two pools rather than spelling it (this arena's tests are published), and
  carries a positive control on every "not present" assertion.

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
