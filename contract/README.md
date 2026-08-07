# Arena Contract (C-level)

> **First trial — to be re-evaluated with collaborators.**
>
> See the design spec: `docs/superpowers/specs/2026-04-29-sciencearena-taxonomy-and-contract-design.md` (§4).

This document specifies what every Meta Science Arena arena must provide so that the framework can run it. The contract is **C-level**: contamination guards, task-set versioning, and run provenance are mandatory, not optional. See also [`/how-it-works`](https://metasciencearena.app/how-it-works) for a user-facing explanation of arenas, tasks, players, and the revealed/private dual-benchmark concept.

## Arena directory shape

Every arena lives in its own directory under `arenas/`:

```
arenas/<arena-id>/
  arena.yaml              ← manifest; conforms to framework/contract/schemas/arena_manifest.schema.json
  generator.py            ← yields tasks
  scorer.py               ← scores player outputs against ground truth
  schemas/
    input.schema.json     ← what the player receives (inner shape of the task envelope's `input`)
    output.schema.json    ← what the player must return
  difficulty.yaml         ← human-readable difficulty axis docs (mirrors arena.yaml)
  task_sets/
    v1/
      manifest.yaml       ← list of {task_id, difficulty, generator_seed}
      held_out/           ← never published; gitignored; framework-only
      public/             ← published examples for player development
  README.md               ← human-readable challenge description
  CHANGELOG.md            ← every task-set version + what changed
```

See `contract/arena.example.yaml` for an annotated `arena.yaml`.

## Required interfaces

### `generator.py`

Must expose a callable:

```python
def generate(task_set_version: str, seed: int, split: str = "revealed") -> Iterable[TaskEnvelope]: ...
```

Each yielded `TaskEnvelope` conforms to `framework/contract/schemas/task_envelope.schema.json`. The generator is deterministic given `(task_set_version, seed)` so the framework can reproduce held-out tasks without storing them in the public repo.

The optional `split` argument (`"revealed"` | `"private"`) selects which benchmark suite to emit; see **Revealed/private dual benchmark** below. A generator that ignores `split` (legacy single-suite arena) is still valid — the runner only passes `split` when the generator's signature accepts it.

### `scorer.py`

Must expose a callable:

```python
def score(player_output: Any, ground_truth: Any) -> Score: ...
```

Where `Score = {"primary": float in [0, 1], "breakdown": dict}`. The `breakdown` is arena-specific and may be empty.

### `schemas/input.schema.json` and `schemas/output.schema.json`

JSON Schemas constraining the inner `input` of the task envelope and the player's response. These are the **only** player-facing contract. Humans (filling a form), AI models (called via API), and platforms (HTTP wrappers) all consume the same JSON.

## C-level commitments

### Contamination guards

- `task_sets/<v>/held_out/` is gitignored (see top-level `.gitignore`).
- Arena authors commit only the generator + seeds. The held-out task pool is reproducible by the framework's runner but invisible to players and to anyone reading the public repo.
- New leaderboard submissions are scored on held-out tasks; only the public set is available for player development.

### Revealed/private dual benchmark (added 2026-06-06)

To let developers verify our scoring and improve against a real benchmark, an arena may
publish a **revealed** suite alongside its **private** one. Both are produced from the *same*
generator and *same* injected-mistake catalog over the *same* difficulty × mistake matrix, so
they parallel each other in difficulty by construction.

- **`revealed`** — committed seed (`arena.yaml#benchmark_splits.revealed.seed`); gold and full
  per-task results are disclosed (the leaderboard's "Open Benchmark" section). Tasks carry
  `visibility: public` and `split: revealed`.
- **`private`** — secret seed (gitignored `task_sets/<v>/.private_seed` or
  `SCIENCEARENA_PRIVATE_SEED`) **plus** any hand-curated real-world holdout
  (`benchmark_splits.private.real_holdout_dir`); gold gitignored, findings redacted at write
  time. Tasks carry `visibility: held_out` and `split: private`. This is the official
  leaderboard suite.

`framework/parity.py` (CLI: `python tools/check_parity.py <arena-id>`) generates both splits and
**fails loud** unless every injected-mistake category appears in both and per-cell difficulty
counts match within `benchmark_splits.parity.count_tolerance`. So the parity checker can read
the injected-mistake labels uniformly across arena gold shapes, each task's gold should declare
them: either a top-level `mistake_kinds: [...]` list (preferred; `[]` or `["clean"]` for a clean
control — works for field-map gold too), or a per-item `*_kind` field (`deception_kind`,
`deviation_kind`, `mistake_kind`, `injected_mistake`). This is the technical guarantee
that the revealed and private suites stay equally hard. Arenas that declare `benchmark_splits`
must keep this check green (wired into `sciencearena-qa`).

### Task-set versioning

- Once a `task_sets/vN/` directory is committed, it is **immutable**.
- Bug fixes or improvements bump to `vN+1` and trigger optional re-runs of all prior players.
- Leaderboards always display "score on task-set vN" so cross-version comparisons are explicit.

### Run provenance

- The framework writes a `RunRecord` (see `framework/contract/schemas/run_record.schema.json`) for every task played by every player.
- Arenas must NOT write run records themselves.
- Provenance fields: `run_id`, `arena_id`, `task_set_version`, `task_id`, `player_id`, `player_version`, `player_type`, `input_hash` (SHA-256), `output`, `score`, `timestamp_utc`, optional `cost_usd`, `latency_ms`, `task_visibility`, and `split` (`revealed` | `private`).

### Tool feedback reports (added 2026-05-06)

Meta Science Arena reports per-player feedback bundles so tool authors can improve from concrete failure analysis. The contract supports this via three additions:

1. **`arena.yaml#error_categories`** *(optional)* — failure modes the scorer may report. Drives the categorical histogram in tool reports.
2. **`task_envelope.visibility`** *(optional, default `held_out`)* — generators tag each task `public` or `held_out`. Public tasks may appear in report drilldown with full input + gold; held-out tasks only contribute aggregated category counts.
3. **`score.findings`** *(optional)* — array of structured per-task failure annotations conforming to `framework/contract/schemas/findings.schema.json`. Each finding's `category` must match an id declared in the manifest's `error_categories`.

**Held-out redaction is enforced at runner-write time.** A held-out task's findings have all content-bearing fields (`anchor`, `evidence`, `correct_value`, `examples`) stripped before the run record is written to disk. Even a leak of the JSONL file does not reveal held-out gold.

**`score.breakdown` is preserved verbatim on held-out tasks** because it is meant for aggregate numerics (e.g. `levenshtein_similarity: 0.41`). **Arena scorers MUST NOT place gold-leaking strings or spans into `breakdown`.** Use `findings` for any content-bearing diagnosis.

**Severity levels** for `error_categories` (used in report rendering):
- `minor` — output is mostly correct, a small number of details are wrong.
- `major` — output is partially wrong in a way that affects downstream usefulness.
- `critical` — output is unusable as-is.

Reports are derived artifacts. The CLI command `python -m framework report --arena <id> --task-set vN --player <p> --player-version <v>` regenerates the bundle under `arenas/<arena>/runs/<task_set>/reports/<player>@<version>/`. See the design spec at `docs/superpowers/specs/2026-05-06-arena-feedback-reports-design.md` for the full data flow and Decisions Log.

### Player-agnostic I/O

- A "player" is anything that consumes a task envelope and returns an output conforming to `output.schema.json`.
- Players are registered with the framework via a manifest with `player_type` drawn from the same controlled vocabulary as the taxonomy's `existing_solutions.type` field: `platform`, `tool`, `human-baseline`, `ai-model`. Exact registration-manifest shape is TBD in the framework session.

## What is NOT in the contract

These live in the framework (`framework/`) or the leaderboard app (`leaderboard-app/`), not in the contract itself:

- The runner that orchestrates generator → player → scorer → store (`framework/runner.py`).
- The storage backend for run records (`arenas/*/runs/`).
- The leaderboard rendering / website (`leaderboard-app/`).
- The player registration UI.
- Longitudinal tracking dashboards.
- Cost / rate-limiting against AI model calls.
- Anti-gaming measures beyond held-out task pools (see Open Question 5 in the design spec).

## Validation

The schemas live at `framework/contract/schemas/` — INSIDE the package, not at
the repo root. They are consumed at runtime by `framework.discovery` and
`framework.storage`, so a wheel that did not carry them raised
`FileNotFoundError` on the first `load_arena`. Resolve one with
`framework.paths.schema_path(name)`; never compose a path from `__file__`.
This file remains the prose specification.

They are JSON Schema Draft 2020-12. They can be validated independently:

```bash
python -c "
import json, jsonschema
for p in ['framework/contract/schemas/arena_manifest.schema.json',
          'framework/contract/schemas/task_envelope.schema.json',
          'framework/contract/schemas/run_record.schema.json']:
    jsonschema.Draft202012Validator.check_schema(json.load(open(p)))
print('all schemas OK')
"
```

A contract-level validator that checks an arena directory against this full shape (analogous to `taxonomy/scripts/validate.py` but for arenas) remains a future task. In the meantime, `sciencearena-qa` (the project skill) runs pytest + parity checks across all arenas on every build.
