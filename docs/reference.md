# metasciencearena reference

The complete public surface of the `metasciencearena` package: every command and flag, every
environment variable, every adapter class, the data schemas and the Python API. The overview,
installation and quickstart are in the [README](../README.md).

## Command-line interface

Entry points: the `metasciencearena` console script, or `python -m framework` in a checkout.

```text
metasciencearena [--arenas-root DIR] [--registry FILE] <command> ...
```

| Global flag | Meaning |
|---|---|
| `--arenas-root DIR` | directory of arena subdirectories; sets `SCIENCEARENA_ARENAS_ROOT` for this process and any subprocess player |
| `--registry FILE` | player registry YAML; sets `SCIENCEARENA_REGISTRY` |

Exit codes: 0 success; 1 a command-level failure (no runs to aggregate, a failed audit, a retry
that retired tasks, an uncaught error); 2 a missing or wrong data root.

### `metasciencearena arenas list`

Prints `arena_id  path` for every arena under the root.

### `metasciencearena arenas versions`

| Flag | Default | Meaning |
|---|---|---|
| `--arena ID` | all | one arena only |

Prints each task-set version with status (`current`, `superseded`, …), task count, release
date, whether runs exist, whether its generator is fingerprinted, and its comparability note.

### `metasciencearena players list`

Prints every registry entry: `player_id`, adapter class, confidence strategy, determinism.

### `metasciencearena run`

| Flag | Default | Meaning |
|---|---|---|
| `--arena ID` | required | arena to play |
| `--task-set V` | required | task-set version; a superseded version refuses new records |
| `--players P [P ...]` | required | registry `player_id`s |
| `--trials N` | `1` | plays per task (deterministic players collapse to 1) |
| `--timeout S` | `120` | per-task timeout in seconds |
| `--tag NAME` | `run` | part of the output file name |
| `--split revealed\|private` | `revealed` | which suite to generate |
| `--seed N` | from `arena.yaml` | override the generator seed |
| `--max-tasks N` | all | stop after N tasks |
| `--tasks-file FILE` | all | play only the task_ids listed in FILE (one per line, `#` comments); exits 2 if a listed id produced no record |
| `--public-only` | implied by `--split revealed` | skip tasks whose `visibility` is not public |
| `--include-held-out` | off | opt out of the safe default and play held-out tasks in a revealed run |
| `--held-out-only` | off | keep only held-out tasks |
| `--overwrite` | off | replace the output file instead of appending |

Writes `arenas/<arena>/runs/<task-set>/<players>__<split>__<tag>.jsonl` and prints the path. A
`private` split refuses to fall back to a development seed.

### `metasciencearena retry-failed`

Re-plays only the tasks one player errored on or missed, in rounds, merging fresh OK results
back into the target file; sleeps between rounds so provider rate limits recover. Idempotent.

| Flag | Default | Meaning |
|---|---|---|
| `--arena ID`, `--task-set V` | required | as for `run` |
| `--player P` | required | one `player_id` |
| `--tag NAME` | `run` | must match the target file |
| `--split`, `--seed` | `revealed`, arena seed | as for `run` |
| `--timeout S` | sized from the player's declared `max_tokens`, at least `300` | per-task timeout |
| `--max-rounds N` | `6` | retry rounds this invocation |
| `--cooldown S` | `150` | seconds between rounds |
| `--include-truncated` | off | also re-play answers cut off at the output ceiling (`finish_reason = "length"`) |
| `--resample-model-output-errors` | off | also re-play an errored task whose call returned the model's answer (schema-rejected, or unparseable with its raw response kept). Off by default since 2026-10-19: re-playing an answer re-samples the model until its format parses, which flatters its score, so by default only calls that got no answer (429, 5xx, timeouts) are re-played |
| `--keep-model-output-errors` | — | accepted for older scripts; it is now the default and does nothing |

Exits 1 when tasks are retired as hopeless or remain truncated.

### `metasciencearena leaderboard`

`--arena ID --task-set V` (both required). Reads every `runs/<V>/*.jsonl` of the arena and prints
the leaderboard table; exits 1 when there are no runs.

### `metasciencearena report`

| Flag | Default | Meaning |
|---|---|---|
| `--arena ID`, `--task-set V` | required | |
| `--player P`, `--player-version X` | required | the focal player |
| `--tier ID` | none | data tier to report on; required for a task set that lists tiers in `task_sets/<v>/tiers.jsonl` (scores are per tier, never pooled) |
| `--rebuild-tasks` | off | re-run the generator so the bundle can include difficulty breakdowns, reproduce envelopes and gold |
| `--seed N` | `0` | generator seed for `--rebuild-tasks` |
| `--split` | `revealed` | suite to rebuild |

Writes a bundle under `arenas/<arena>/runs/<timestamp>/reports/<player>@<version>/`:
`tool_report.json`, `tool_report.md`, per-task findings (held-out ones redacted), reproduce
envelopes and a `README.md`.

### `metasciencearena audit`

| Flag | Meaning |
|---|---|
| `--arena ID` | limit to one arena (default all) |
| `--task-set V` | default: each arena's current version |
| `--versions` | declared vs installed tool versions |
| `--symmetry` | every ranked player ran the identical task set |
| `--fresh` | each record's `input_hash` still matches what the generator emits today |
| `--identity` | one instrument never covers two inference configurations, and names are slug-safe |
| `--write-fingerprints` | record the generator fingerprint of every frozen task set lacking one (never overwrites) |

With none of the four check flags, all four run. Exits 1 if any fails.

### `metasciencearena contributed add`

```text
metasciencearena contributed add ARENA VERSION [--dry-run]
```

| Argument / flag | Meaning |
|---|---|
| `ARENA` | arena id |
| `VERSION` | a NEW task-set version the arena's generator has just minted; it must already exist under `task_sets/` and in `arena.yaml` |
| `--dry-run` | report what would be added; write nothing |

Adds accepted community-contributed tasks to that version and prints how many were added and
the sha256 of the contributed set. It never runs implicitly, and it refuses frozen versions,
versions that already have results, and study arenas.

### `metasciencearena export-inspect` / `metasciencearena export-kaggle`

| Flag | Default | Meaning |
|---|---|---|
| `--arena ID` | required | |
| `--task-set V` | `v1` | |
| `--out DIR` | `build/inspect` / `build/kaggle` | output directory |
| `--prompt FILE` | built-in | prompt template containing `{{INPUT_TEXT}}` |

Emit an Inspect AI task module (the format OpenBench and Epoch AI ingest) or a Kaggle
Benchmarks `@kbench.task` module for the arena's public split, plus its tasks as JSON.

## Environment variables

### Framework

| Variable | Default | Meaning |
|---|---|---|
| `SCIENCEARENA_ARENAS_ROOT` | `arenas/` beside a checkout | arenas directory; a missing root exits 2 |
| `SCIENCEARENA_REGISTRY` | bundled `players/registry.yaml` | player registry |
| `SCIENCEARENA_SYNTHETIC_ONLY` | unset | `1`: allow a missing held-out corpus and emit synthetic tasks only (otherwise a missing corpus raises) |
| `SCIENCEARENA_ALLOW_HELDOUT_EGRESS` | unset | `1` / `true` / `yes`: allow held-out tasks to reach a cloud player |
| `SCIENCEARENA_PRIVATE_SEED` | unset | private-split seed for arenas declaring `seed_source: env` |
| `SCIENCEARENA_HOST` | derived from host facts | label written to `provenance.host_fingerprint` |
| `SCIENCEARENA_ALLOW_REMOTE` | unset | `1`: let the SciMeto adapter call a non-loopback URL |
| `RSCRIPT_BINARY` | `Rscript` on PATH | Rscript for R players (a registry `rscript_binary` wins) |
| `ARENA_RSCRIPT` | `Rscript` on PATH | Rscript used by the `code-translation-r-v1` scorer |
| `DOCLING_ARTIFACTS_PATH` | Docling's cache | pre-fetched Docling model weights |
| `SCIENCEARENA_MINERU_HOME` | `<temp>/sciencearena-mineru-home` | MinerU's server home for the MinerU players (outside the repository: it stores parses) |
| `SCIENCEARENA_PDFFORGE_CACHE` | `%LOCALAPPDATA%/sciencearena/pdfforge_cache` (else `~/.cache/...`) | cache of generated synthetic PDFs for the PDF-universe arenas (outside the repository on purpose) |
| `SCIENCEARENA_PDFU_PRIVATE_SEED` | `arenas_private/pdf-universe/.private_seed` | integer seed of the private PDF-universe split; without either, only the revealed split can be built |
| `DOCPLUCK_INTERNAL_TOKEN` | `tmp/docpluck_local_token` | token for the LOCAL docpluck service, used by `StatsPipelineAdapter` when its extractor is docpluck |
| `METACHECK_DEV_RLIB` | a per-user R library path | R library holding metacheck's development build, used by `MetacheckDevAdapter` |
| `R_LIBS` | the R default | prepended with `METACHECK_DEV_RLIB` by `MetacheckDevAdapter` when it starts Rscript |
| `SCIENCEARENA_TOOL_ENV_<TOOL>` | `players/tool_envs/<tool>/.venv` | interpreter for an out-of-process tool worker (`DOCLING`, `MARKER`, `MINERU`) |

### Provider credentials (OpenAI-compatible players)

Each `OpenAIChatCompletionsAdapter` entry names three variables in the registry fields
`openai_base_url_env`, `openai_api_key_env` and `openai_available_models_env`. The URL and key
must be set or the player fails at `prepare`; if the models variable is set, the entry's model
must be listed in it.

| Provider | Base URL | Key | Models |
|---|---|---|---|
| Cohere | `COHERE_BASE_URL` | `COHERE_API_KEY` | `COHERE_MODELS` |
| NVIDIA NIM | `NVIDIA_NIM_BASE_URL` | `NVIDIA_NIM_API_KEY` | `NVIDIA_NIM_MODELS` |
| OpenRouter | `OPENROUTER_BASE_URL` | `OPENROUTER_API_KEY` | `OPENROUTER_MODELS` |
| Groq | `GROQ_BASE_URL` | `GROQ_API_KEY` | `GROQ_MODELS` |
| Gemini | `GEMINI_BASE_URL` | `GEMINI_API_KEY` | `GEMINI_MODELS` |
| Mistral | `MISTRAL_BASE_URL` | `MISTRAL_API_KEY` | `MISTRAL_MODELS` |
| Hugging Face | `HF_BASE_URL` | `HF_API_KEY` | `HF_MODELS` |

`GeminiGenerateContentAdapter` reads the key from the variable named by `gemini_api_key_env`
(usually `GEMINI_API_KEY`).

`ScimetoReplicationApiAdapter` reads the variables named by `api_url_env` and `api_key_env`
(`SCIMETO_API_URL`, `SCIMETO_API_KEY`; the key is optional, the URL must be loopback unless
`SCIENCEARENA_ALLOW_REMOTE=1`).

### Other adapters

| Variable | Used by | Meaning |
|---|---|---|
| `REGCHECK_DIR`, `REGCHECK_PYTHON` | `RegcheckShimAdapter` | a regcheck clone and a Python with its dependencies (both required) |
| `REGCHECK_CLIENT` | `RegcheckShimAdapter` | `openai` (default, via the local shim), `groq` or `deepseek` |
| `REGCHECK_PARSER_CHOICE` | `RegcheckShimAdapter` | `pymupdf` (default) or `dpt2` |
| `REGCHECK_GROQ_MODEL`, `REGCHECK_DEEPSEEK_MODEL` | `RegcheckShimAdapter` | model when the entry names none |
| `REGCHECK_SHIM_CODEX`, `REGCHECK_SHIM_CODEX_MODEL`, `REGCHECK_SHIM_CODEX_TIMEOUT_S` | regcheck shim | codex binary (default `codex`), model, timeout (default 600) |
| `REGCHECK_SHIM_PORT` | regcheck shim | port (default 8765) |
| `REGCHECK_SHIM_NO_ST` | regcheck shim | set: skip sentence-transformers, use hashing embeddings |
| `MODEL_PROXY_URL`, `MODEL_PROXY_API_KEY`, `LLMS_AVAILABLE` | `ModelProxyAdapter` (Kaggle proxy; no registry entry) | proxy URL, token, and the funded-model list |
| `CLAUDE_CODE_OAUTH_TOKEN` | `AgenticContainerAdapter` | the Claude subscription token, passed into the container by name only (never an API key; `ANTHROPIC_API_KEY` and friends are removed from the container's environment) |

Agentic CLI players use their own login; our `SCIENCEARENA_*` variables are stripped from their
environment.

## Adapter classes

The `adapter_class` values used by the bundled registry:

| Class | What it drives |
|---|---|
| `OpenAIChatCompletionsAdapter` | hosted OpenAI-compatible `/chat/completions` endpoints |
| `AnthropicApiAdapter` | the Anthropic Messages API, no tools or system prompt; effort required per entry and recorded in `response_meta.sampling`; live, batch-plan and batch-replay modes behind a local spend ledger |
| `GeminiGenerateContentAdapter` | Google Gemini's native `generateContent` API (seed, thinking level, served `modelVersion` recorded) |
| `LocalGgufAdapter` | a local llama-server; records the GGUF file's SHA-256 |
| `SubprocessCliAdapter` | an agentic CLI (Claude Code, Codex, Gemini, Cursor): templated prompt in, JSON out, hermetically contained |
| `AntigravityCliAdapter` | the Antigravity `agy` CLI |
| `AgenticContainerAdapter` | Claude Code with Bash/R/Python inside a locked Docker container, one fresh container per task (`framework.agentic_container`); counts as cloud for the held-out egress gate |
| `LlmCliPdfAdapter`, `LlmCliPdfSectionsAdapter`, `LlmCliPdfTablesAdapter` | an LLM CLI reading a PDF for text / sections / tables |
| `LlmPdfReferencesAdapter`, `LlmPdfCitationsAdapter` | an LLM reading a PDF for references / citations |
| `RCliAdapter` | an R script via Rscript, JSON on stdout (statcheck, metacheck, …) |
| `StatsPipelineAdapter` | a PDF extractor (pypdf, pdftotext, GROBID, docpluck, ...) feeding an R statistics checker (statcheck, effectcheck, metacheck); each (extractor, checker) pair is one registry entry and every tool version is recorded |
| `MetacheckDevAdapter` | the metacheck player run against metacheck's development build (needs `METACHECK_DEV_RLIB`) |
| `HttpAdapter` | a local HTTP service (e.g. the ESCImate checker) |
| `ScimetoReplicationApiAdapter` | a local SciMeto API |
| `RegcheckShimAdapter` | regcheck through a local OpenAI-compatible shim |
| `DocpluckLibraryAdapter`, `DocpluckSectionsAdapter`, `DocpluckTablesAdapter` | the docpluck library |
| `GrobidTextAdapter`, `GrobidTablesAdapter`, `GrobidSectionsAdapter`, `GrobidReferencesAdapter`, `GrobidCitationsAdapter` | a GROBID server |
| `DoclingTextAdapter`, `DoclingTablesAdapter`, `DoclingSectionsAdapter` | Docling (`[pdf-docling]`, or a pinned out-of-process venv via the registry's `tool_env`) |
| `MarkerTextAdapter`, `MarkerSectionsAdapter`, `MarkerTablesAdapter` | Marker (`marker-pdf`), out of process in its own pinned venv; fast mode, no OCR, no LLM |
| `MineruTextAdapter`, `MineruSectionsAdapter`, `MineruTablesAdapter` | MinerU (flash tier), out of process with its own local server; telemetry disabled |
| `LiteparseTextAdapter`, `LiteparseTablesHeuristicAdapter`, `LiteparseSectionsHeuristicAdapter` | liteparse (`[pdf-liteparse]`) |
| `PdftotextSubprocessAdapter` | poppler `pdftotext` |
| `CermineReferencesAdapter`, `AnystyleReferencesAdapter` | CERMINE / AnyStyle reference parsers |
| `XlatFixtureAdapter` | replays recorded tool outputs for the code-translation arena |
| `FredFridaOnlyAdapter` | a baseline with no I/O |

Unknown classes are treated as cloud players for the held-out egress guard. Third-party
adapters register through the `metasciencearena.adapters` entry-point group (README, "Writing
your own player").

## Data schemas

Shipped as package data under `framework/contract/schemas/`.

### Task envelope (`task_envelope.schema.json`)

| Field | Meaning |
|---|---|
| `task_id`, `arena_id`, `task_set_version` | identity (required) |
| `difficulty` | map of difficulty-axis id to value; keys match the arena's `difficulty_axes` (required) |
| `visibility` | `public` or `held_out`; absent means `held_out` (fail-safe) |
| `split` | `revealed` or `private` |
| `input` | the arena-specific payload, validated by the arena's `schemas/input.schema.json` (required) |

### Run record (`run_record.schema.json`)

| Field | Meaning |
|---|---|
| `run_id` | unique record id |
| `arena_id`, `task_set_version`, `task_id` | the task played |
| `player_id`, `player_version`, `player_type` | the player as declared |
| `resolved_tool_version` | the tool version detected at run time, or null |
| `input_hash` | SHA-256 of the canonical envelope (redacted for held-out tasks in public artifacts) |
| `output` | the answer, in the arena's output schema |
| `rejected_output` | the answer as returned, present only when the output schema refused it (`output` is then `{}`); dropped for held-out tasks |
| `score` | `primary` (0–1), `breakdown`, and `findings` (per-item error categories) |
| `task_visibility`, `split` | copied from the envelope |
| `timestamp_utc`, `latency_ms`, `cost_usd` | when, how long, and the list-price cost |
| `usage` | provider token counts, when metered |
| `provenance` | how the record was produced: command (sanitised), seed, prompt-template SHA-256, host fingerprint, containment |
| `schema_version` | record shape version |
| `instrument` | which scoring formula computed the number |
| `adherence` | whether the output schema accepted the answer (separate from whether it was right) |
| `player_identity` | the inference configuration that produced the answer |
| `response_meta` | provider metadata such as `finish_reason` and the served model |
| `env_sha` | (v12) SHA-256 of the run environment stored once at `run_env/envs/<env_sha>.json`: runner image, its full `pip freeze`, egress proxy, network policy, resource limits, orchestrator, generator fingerprint; or `isolation: host-exempt` with the named exemption. Required on every v12 record |
| `call_config_sha256` | (v12) SHA-256 of the call settings stored at `run_env/call_configs/<sha>.json`: every knob that can change a model's answer, read off the request actually sent (`n/a` where a provider has none); null when no model request was made |

From v12 the provider's raw response is kept too: `<run file>.raw.jsonl.gz` beside the run file
for public tasks, and in article custody for held-out tasks (it can quote the real document).

### Arena manifest (`arena_manifest.schema.json`)

Required: `arena_id`, `taxonomy_leaf_id`, `version` (semver of the arena code), `owner`,
`license`, `input_artifact`, `output_to_judge`, `difficulty_axes`, `task_set_versions` (the
version ladder; which one is live is decided by each entry's status, not its position),
`display_columns` (leaderboard columns projected from run records) and `tier_pivot`
(per-tier panel semantics). Optional: `title`, `min_ranked_n` (smallest common task set
below which a ranking is flagged insufficient), `benchmark_splits` (the revealed/private policy),
`unlisted` (a reason string; the arena is kept off the public arenas index, e.g. an owner-gated study not yet registered), `data_tiers` (board display per data tier for a task set that lists tiers; each tier is its own board, never pooled), `error_categories` (failure modes `scorer.score()` may report), `runner` (`container-required`
refuses any player that cannot run in the pinned runner image; `exemptions-allowed`, the
default, lets a named exempt adapter class run on the host) and `pass_bars` (pre-registered
pass criteria, committed before any player is scored). `contract/README.md` is
the prose specification.

## Python API

The package imports as `framework` (engine) and `players` (instrument). Modules with a declared
public API:

**`framework.paths`** — data roots, resolved in one place: `arenas_root()`, `registry_path()`,
`schema_path(name)`, `installed_repo_root()`, `RootNotFoundError`, and the variable names
`ARENAS_ROOT_ENV`, `REGISTRY_PATH_ENV`.

**`framework.ranking`** — which differences between players are real: `rank_players(...)`
(ranks, tie groups, pairwise verdicts), `paired_signflip_p` (two-sided exact paired permutation
p), `holm` (Holm-adjusted p-values), `seed_for(arena_id, task_set_version)`, `PairVerdict`,
and the constants `DEFAULT_ALPHA`, `DEFAULT_RESAMPLES`, `MIN_TASKS_FOR_TESTING`.

**`framework.adherence`** — structural adherence (did the output schema accept the answer):
`outcome(record)` returning `CONFORMANT`, `VIOLATION` or `UNATTRIBUTED`; `summarise(records)`;
`classify_record`, `classify_message`, `primary_class`, `violation_reason`,
`enum_values_in_schema`; constants `VIOLATION_MARKER`, `POLICY_ZERO_FLAG`, `CLASS_PRECEDENCE`.

**`framework.hermetic`** — the containment boundary for agentic CLI players:
`harden_argv`, `hermetic_env`, `hermetic_cwd`, `spawn_kwargs`, `cli_version`,
`config_fingerprint`, `player_instrument_label`, `is_agentic_cli`, `normalise_binary`,
`effective_binary`, `takes_stdin_prompt`, `needs_fresh_session`, `prompt_goes_in_a_file`,
`HermeticProfile`, `HERMETIC_PROFILES`, `INERT_LOCAL_TOOLS`, `WRAPPER_TOOLS`,
`PROFILE_VERSION`, `UnhardenedCliError`.

**`framework.publish`** — the rule deciding what may be published (auditable, not asserted):
`public_manifest(repo_root)`, `published_arena_ids(arenas_root)`, `classify_arenas`,
`ArenaPublishability`, `scan_for_leaks(repo_root, files)`, `held_out_shingles`, `code_only`,
`imports_a_private_package`, and the rule tables `PUBLIC_TOP_LEVEL`, `PUBLIC_ARENA_GLOBS`,
`MIRROR_RENAMES`, `PRIVATE_GOLD_MARKERS`, `ARENA_EXTRA_GOLD_MARKERS`, `PRIVATE_PACKAGES`, `PRIVATE_TEST_FILES`,
`PUBLISHED_ARENAS_FILE`.

Other modules used in the README examples: `framework.runner` (`run_tournament`,
`ADAPTER_ENTRY_POINT_GROUP`), `framework.player_adapter` (`PlayerAdapter`,
`register_adapter_class`, `build_adapter`), `framework.storage` (`read_records`,
`RunRecordWriter`), `framework.leaderboard` (`aggregate`, `render_leaderboard`),
`framework.report` (`generate_report`), `framework.discovery` (`load_arena`,
`discover_arenas`), `framework.registry` (`load_registry`), `framework.holdout`
(`is_held_out`, `redact_held_out_record`), `framework.pricing` (`cost_usd`).
