# metasciencearena

**Reproducible benchmark arenas for meta-science tools** (Version 0.4.0).
Leaderboard: <https://metasciencearena.app>

Meta-science has plenty of tools — statcheck, GRIM, z-curve, regcheck, oddpub, metacheck — and
almost no way to say which one is better at what. This package is the measurement layer: task
generation, scoring, contamination control, statistical ranking and leaderboard aggregation,
published so that any score on the site can be recomputed by someone who does not trust us.

## Contents

- [Concepts](#concepts) · [Installation](#installation) · [Quickstart](#quickstart)
- [The public arenas](#the-public-arenas)
- [Command-line interface](#command-line-interface)
- [Running the built-in players](#running-the-built-in-players)
- [Writing your own player](#writing-your-own-player) · [Writing an arena](#writing-an-arena)
- [Output: run records and reports](#output-run-records-and-reports)
- [How scores and ranks are computed](#how-scores-and-ranks-are-computed)
- [Contamination control, stated precisely](#contamination-control-stated-precisely)
- [Environment variables](#environment-variables)
- [Limitations and failure modes](#limitations-and-failure-modes)
- [Citing](#citing) · [Licence](#licence) · [Contributing](#contributing)

The complete reference — every CLI flag, every environment variable, every adapter class, every
run-record field and the Python API — is in
[`docs/reference.md`](https://github.com/giladfeldman/metasciencearena/blob/main/docs/reference.md).

## Concepts

| Term | Meaning |
|---|---|
| **arena** | one task family with known ground truth (e.g. GRIM consistency, statistics extraction): a directory with `arena.yaml`, `generator.py`, `scorer.py` and JSON schemas |
| **task set** | a versioned, frozen set of tasks for an arena (`v1`, `v2`, …); a score is only meaningful against the task set it was produced on |
| **split** | `revealed` (public tasks and gold, generated from a committed seed) or `private` (held-out tasks from a secret seed or a withheld corpus) |
| **task envelope** | one task as the player sees it: `task_id`, `arena_id`, `task_set_version`, `difficulty`, `visibility`, `split`, `input` |
| **player** | anything that answers tasks — a tool, a platform, a language model — declared in a registry YAML and driven by an **adapter** class |
| **run record** | one validated JSON line per (player, task, trial): the answer, its score, cost, latency and full provenance |

## Installation

```bash
pip install "metasciencearena @ git+https://github.com/giladfeldman/metasciencearena"
```

This installs the current source (0.4.0). PyPI (`pip install metasciencearena`) carries an
earlier release, 0.3.0. Python 3.11 or newer.

Optional extras:

| Extra | Adds | Needed for |
|---|---|---|
| `[inspect]` | `inspect-ai` | running tasks emitted by `metasciencearena export-inspect` |
| `[kaggle]` | `kaggle-benchmarks` | running modules emitted by `metasciencearena export-kaggle` |
| `[pdf-docling]` | Docling (pulls torch, ~2 GB) | the Docling PDF players |
| `[pdf-liteparse]` | liteparse | the liteparse PDF players |
| `[dev]` | pytest | running the test suite |

**Arena data is not bundled.** The package is the scoring machinery; the arenas are the data it
operates on. Get them by cloning this repository and point `SCIENCEARENA_ARENAS_ROOT` (or
`--arenas-root`) at its `arenas/` directory. If the root is missing or wrong the CLI exits 2
with an explanation — it never prints an empty leaderboard and exits 0.

## Quickstart

```bash
git clone --depth 1 https://github.com/giladfeldman/metasciencearena
export SCIENCEARENA_ARENAS_ROOT=$PWD/metasciencearena/arenas
# PowerShell: $env:SCIENCEARENA_ARENAS_ROOT = "$PWD\metasciencearena\arenas"

metasciencearena arenas list
metasciencearena arenas versions --arena grim-consistency-v1
metasciencearena players list
```

Then play an arena with a player of your own — here a 20-line GRIM checker — and score it with
exactly the code the leaderboard uses:

```python
import math
from pathlib import Path

from framework.leaderboard import render_leaderboard
from framework.paths import arenas_root
from framework.player_adapter import PlayerAdapter, register_adapter_class
from framework.runner import run_tournament
from framework.storage import read_records


def grim_possible(value, n, decimals, scale):
    """Can `value` (a mean, or a percentage with scale=100) arise from k/n for integer k?"""
    step = 10 ** -decimals
    lo, hi = (value - step / 2) * n / scale, (value + step / 2) * n / scale
    return math.floor(hi + 1e-9) >= math.ceil(lo - 1e-9)


class MyGrimAdapter(PlayerAdapter):
    """A complete player: read the task envelope, return the arena's output schema."""

    def play_task(self, envelope, *, timeout_s):
        records = []
        for s in envelope["input"]["statistics"]:
            if s["stat_type"] == "mean":
                ok = grim_possible(s["mean"], s["n"] * s.get("n_items", 1), s["decimals"], 1)
                kind = "grim_inconsistent"
            else:
                ok = grim_possible(s["percent"], s["n"], s["decimals"], 100)
                kind = "grim_percent_inconsistent"
            records.append({"stat_id": s["stat_id"], "flagged": not ok,
                            "issue_kind": None if ok else kind, "confidence": 1.0})
        return {"records": records}


register_adapter_class("MyGrimAdapter", MyGrimAdapter)
Path("my_registry.yaml").write_text(
    "- player_id: my-grim\n"
    "  player_type: tool\n"
    '  player_version: "0.1.0"\n'
    "  adapter_class: MyGrimAdapter\n"
    "  confidence_strategy: implicit-1.0\n"
    "  deterministic: true\n",
    encoding="utf-8",
)

n = run_tournament(
    arena_dir=arenas_root() / "grim-consistency-v1",
    task_set_version="v1",
    registry_path=Path("my_registry.yaml"),
    player_ids=["my-grim"],
    output_path=Path("runs/my-grim.jsonl"),
    trials=1,
    public_only=True,      # the CLI implies this for --split revealed
    max_tasks=20,
    overwrite=True,
)
print(f"{n} run records written")
print(render_leaderboard(read_records(Path("runs/my-grim.jsonl"))))
```

It prints `16 run records written` and a one-row leaderboard (the v1 revealed split of
`grim-consistency-v1` has 16 public tasks). Every line of `runs/my-grim.jsonl` is a complete run
record; `--registry my_registry.yaml` makes the same player available to the CLI once its
adapter module is importable (see [Writing your own player](#writing-your-own-player)).

## The public arenas

| Arena | Task |
|---|---|
| `code-translation-r-v1` | translate SPSS / Stata syntax to R (scored by running it) |
| `effect-size-conversion-v1` | convert an effect size to another metric |
| `grim-consistency-v1` | judge whether reported means / percentages are GRIM-possible |
| `open-practices-repro-v1` | assess a mocked data/code repository snapshot |
| `p-curve-v1` | evidential value from a set of significant results (p-curve; Simonsohn, Nelson & Simmons, 2014) |
| `power-reporting-v1` | detect and extract a power analysis from a methods excerpt |
| `prereg-deviation-v1` | find deviations between a preregistration and the paper |
| `prereg-extraction-v1` | find a paper's preregistration link |
| `publication-bias-v1` | publication-bias verdict on a meta-analytic dataset |
| `reference-integrity-v1` | reconcile a reference list with in-text citations |
| `reporting-completeness-v1` | flag reporting gaps in a results section |
| `significance-language-v1` | flag marginal-significance and spin language |
| `sprite-plausibility-v1` | plausibility of reported mean / SD / n |
| `stats-extraction-v1` | extract test statistics, effect sizes and CIs (six difficulty tiers) |
| `transparency-statements-v1` | extract items from a transparency section |
| `zcurve-evidential-v1` | z-curve on a set of significant results |

Each ships its generator, scorer, schemas, catalogs and tests; `arenas/<id>/README.md` describes
the task and scoring. Six further arenas are private because their source PDFs are not ours to
republish; their prompt templates still ship (see below).

## Command-line interface

`metasciencearena` (or `python -m framework` in a checkout). Global options, before the command:
`--arenas-root DIR` (overrides `SCIENCEARENA_ARENAS_ROOT`) and `--registry FILE` (overrides
`SCIENCEARENA_REGISTRY`).

| Command | What it does |
|---|---|
| `metasciencearena arenas list` | every arena under the root |
| `metasciencearena arenas versions [--arena ID]` | task-set ladder: status, task count, release date, comparability |
| `metasciencearena players list` | every registry entry with its adapter class |
| `metasciencearena run --arena ID --task-set V --players P [P ...]` | play tasks, append run records to `arenas/<id>/runs/<V>/<players>__<split>__<tag>.jsonl` |
| `metasciencearena retry-failed --arena ID --task-set V --player P` | re-play only the tasks a flaky player errored on, merging results in |
| `metasciencearena leaderboard --arena ID --task-set V` | aggregate every run file for that task set (exits 1 if there are none) |
| `metasciencearena report --arena ID --task-set V --player P --player-version X` | per-player feedback bundle (JSON, Markdown, findings, reproduce envelopes) |
| `metasciencearena audit` | fairness audits: `--versions`, `--symmetry`, `--fresh`, `--identity` (all four by default); exit 1 on any failure |
| `metasciencearena export-inspect --arena ID` | emit an Inspect AI task for the public split |
| `metasciencearena export-kaggle --arena ID` | emit a Kaggle Benchmarks task for the public split |

Key `run` options: `--split revealed|private` (default `revealed`, which implies
`--public-only`), `--trials N` (default 1), `--timeout S` (default 120), `--max-tasks N`,
`--seed N`, `--tag NAME`, `--overwrite` (replace instead of append — re-running without it
doubles the records), `--include-held-out` / `--held-out-only`. Every flag of every command is
in the reference.

To run a public arena inside a standard harness instead:

```bash
pip install "metasciencearena[inspect]"
metasciencearena export-inspect --arena grim-consistency-v1 --out ./inspect
inspect eval ./inspect/grim_consistency_v1.py --model <provider>/<model>
```

## Running the built-in players

`players/registry.yaml` ships ~300 entries: R tools (statcheck, metacheck, …), extraction tools
(docpluck, GROBID, Docling, liteparse, CERMINE, AnyStyle, pdftotext), platforms, hosted models
via OpenAI-compatible endpoints, local GGUF models, and agentic CLIs (Claude Code, Codex, Gemini,
Cursor, Antigravity). **The measuring instrument is published**: the registry, the adapters and
all prompt templates, including those of the private arenas. Each run record carries
`provenance.prompt_template_sha256`; superseded templates are kept under
`players/prompts/_archive/` so every published score resolves to readable text.

```bash
# from the root of the cloned repository (registry paths are repository-relative)
metasciencearena run --arena stats-extraction-v1 --task-set v5 --players statcheck --max-tasks 3
metasciencearena leaderboard --arena stats-extraction-v1 --task-set v5
```

A superseded task set refuses new records and names the current one
(`metasciencearena arenas versions` lists them).

What each kind of player needs (a missing dependency is a clear error at `prepare`, never a
wrong score):

| Adapter | Needs |
|---|---|
| `RCliAdapter` | R on PATH (or `RSCRIPT_BINARY`) with the tool's package installed |
| `OpenAIChatCompletionsAdapter` | the base-URL and key variables the entry names, e.g. `GROQ_BASE_URL` + `GROQ_API_KEY` |
| `SubprocessCliAdapter`, `LlmCliPdfAdapter` | the CLI on PATH, logged in with its own account |
| `LocalGgufAdapter` | a running llama-server at the entry's `endpoint` |
| `GrobidTextAdapter` and the other GROBID adapters | a GROBID server (default `http://localhost:8070`) |
| `DoclingTextAdapter`, `LiteparseTextAdapter` | the `[pdf-docling]` / `[pdf-liteparse]` extra |

Credentials are only ever environment-variable **names** in the registry (`openai_api_key_env`,
`api_key_env`, …), never values. Agentic CLIs run **hermetically**: a scratch working
directory outside the checkout, our own `SCIENCEARENA_*` variables stripped from their
environment, and containment flags recorded in `provenance`.

## Writing your own player

Subclass `PlayerAdapter` (from `framework.player_adapter`), implement
`play_task(self, envelope, *, timeout_s) -> dict` returning the arena's output schema, and
register the class. Optional overrides: `prepare()`, `cleanup()`, `resolved_tool_version()`.
To make it available to the CLI without vendoring anything, publish it under the adapter entry
point:

```toml
[project.entry-points."metasciencearena.adapters"]
my_tool = "my_package.adapters"
```

Every module in `my_package.adapters` is imported at start-up and registers its classes with
`register_adapter_class`; then add a registry entry naming your `adapter_class`. A plugin that
fails to import is logged and skipped. Scoring is identical to ours because it is this code.

## Writing an arena

An arena is a directory with `arena.yaml`, a `generator.py` yielding task envelopes, a
`scorer.py`, and `schemas/input.schema.json` + `schemas/output.schema.json`. The manifest and
envelope are validated against schemas shipped in this package. `contract/README.md` in this
repository is the prose specification; `contract/arena.example.yaml` is an annotated manifest.

## Output: run records and reports

Each run record (schema `framework/contract/schemas/run_record.schema.json`) carries
`run_id`, `arena_id`, `task_set_version`, `task_id`, `player_id`, `player_version`,
`player_type`, `resolved_tool_version`, `input_hash` (SHA-256 of the canonical envelope),
`output`, `score`, `task_visibility`, `split`, `timestamp_utc`, `cost_usd`, `latency_ms`,
`provenance`, `usage`, `schema_version`, `instrument`, `adherence`, `player_identity`,
`response_meta` and, only when the output schema refused the answer, `rejected_output`. Field meanings are in the reference. Held-out records are redacted before any
public artifact is written.

## How scores and ranks are computed

Each arena's `scorer.py` turns one answer into a per-task score in [0, 1] (the arena README states
the formula). The leaderboard reports the mean with a 95% interval **across tasks**, which
describes how task difficulty varies, not how precisely two players differ. Ranks are therefore
decided separately (`framework.ranking`): every ranked player plays the identical task set, so
each pair is compared with an exact paired sign-flip permutation test on the per-task
differences (10,000 resamples, seeded per arena and task set), Holm-corrected across all pairs;
players not separated by it share a rank. `metasciencearena audit --symmetry` enforces the
identical-task-set precondition.

## Contamination control, stated precisely

Every arena has a **revealed** split (published: tasks, gold, scoring formula, injected-mistake
list) and a **held-out** split. For seed-based arenas the held-out tasks come from a private seed
that has never been in git; for real-paper arenas the corpus itself is withheld.
`framework.holdout` is the one place that defines what "held-out" means, and every writer of a
tracked or published artifact goes through it:

> For any held-out task, no tracked file and no public artifact may carry its gold answer, its
> reconstructable player output, its per-task score breakdown, or its `input_hash` membership
> oracle.

**What we do not claim.** We cannot claim a model has never seen a *published* paper, and we do
not characterise any provider's retention. The benchmark is contamination-**resistant**, not
contamination-**free**. A `--split revealed` run refuses to send held-out tasks to a cloud player
unless `SCIENCEARENA_ALLOW_HELDOUT_EGRESS=1` is set; see `DATA_HANDLING.md`.

## Environment variables

| Variable | Purpose |
|---|---|
| `SCIENCEARENA_ARENAS_ROOT` | arenas directory (required off-repo; `--arenas-root` overrides) |
| `SCIENCEARENA_REGISTRY` | player registry YAML (default: the bundled `players/registry.yaml`) |
| `SCIENCEARENA_SYNTHETIC_ONLY` | `1` = allow a missing held-out corpus and emit synthetic tasks only |
| `SCIENCEARENA_ALLOW_HELDOUT_EGRESS` | `1` = allow held-out tasks to reach a cloud player |
| `SCIENCEARENA_PRIVATE_SEED` | private-split seed for arenas with `seed_source: env` |
| `SCIENCEARENA_HOST` | label recorded in `provenance.host_fingerprint` |
| `RSCRIPT_BINARY` | Rscript used by R players |

Provider credentials, the regcheck, Docling, SciMeto and model-proxy settings are listed in the
reference.

## Limitations and failure modes

- **The held-out half is not reproducible from this package** — private seeds and withheld
  corpora never ship; that is what keeps the private split meaningful.
- **`metasciencearena leaderboard` on a fresh clone exits 1**: run records are not published in
  this repository; produce your own with `run`, or use the leaderboard site.
- **Many players need external software or accounts** (R packages, GROBID, provider keys,
  logged-in CLIs); without them `prepare` fails before any task is played.
- **The library default of `run_tournament(trials=...)` is 3**, while the CLI default is 1;
  pass `trials` explicitly when calling the Python API.
- **Registry paths are repository-relative** (`players/prompts/...`, `players/adapters/*.R`):
  run built-in players from the root of a checkout, or an R player fails on every task.
- **On Windows, run R players from PowerShell or cmd.** `Rscript` launched from Git Bash can
  crash (exit code 0xC0000005), which is recorded as an errored task.
- **Scores depend on the task set.** Task sets are versioned and broadened over time; compare
  scores only within one `arena_id` + `task_set_version`.

## Citing

See [`CITATION.cff`](CITATION.cff). If you cite a specific score, cite the arena id and task-set
version alongside it.

## Licence

MIT. The private repository holds the web application, the held-out corpora and the private
seeds; this package holds the verifiable logic.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). This repository is published from the maintainer's
development tree; pull requests are reviewed here, applied there, and mirrored back.
