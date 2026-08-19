# Running **regcheck** on `prereg-deviation-v1` with NO OpenAI API key

`regcheck` (JamieCummins/regcheck) is a prereg-vs-paper deviation checker that
talks to OpenAI through the `openai` Python SDK. The SDK honours
`OPENAI_BASE_URL`, so we run it against a **local OpenAI-compatible shim**
(`openai_shim.py`) that translates each request into a `codex exec` call (the
owner's ChatGPT subscription — **no API key, no API cost**). regcheck's
`OPENAI_API_KEY` is a dummy pointing at the shim.

This is the owner-reproducible runbook. The committed `RegcheckShimAdapter`
automates the same flow for the framework runner (see "Automated run" below).

---

## RECOMMENDED PATH (used for the committed records): Groq chat + shim embeddings

The pure-codex path works but is impractically slow (codex's per-call latency makes
regcheck's own OpenAI client time out / reset mid-task — ~9-30 min per task). The
**committed run records were produced via Groq** for the LLM chat (fast, free tier)
while the local shim still serves the **embeddings** (Groq has no embeddings API), so
**no OpenAI key is ever needed**. Per-task latency drops to ~20-25s; the full 38-task
revealed split runs in ~15 min.

```bash
export REGCHECK_DIR="C:/Users/filin/regcheck_run/regcheck"
export REGCHECK_PYTHON="C:/Users/filin/regcheck_run/venv/Scripts/python.exe"
export REGCHECK_CLIENT="groq"                       # native Groq for chat
export REGCHECK_GROQ_MODEL="llama-3.3-70b-versatile"
export GROQ_API_KEY="gsk_…"                          # YOUR key — env only, NEVER committed
rm -f arenas/prereg-deviation-v1/runs/v1/regcheck__revealed__run.jsonl
PYTHONPATH=. python -m framework run --arena prereg-deviation-v1 --task-set v1 \
  --players regcheck --split revealed --trials 1 --timeout 540
```

The adapter (`REGCHECK_CLIENT=groq`) starts the shim for `/v1/embeddings` only and
points regcheck's chat at Groq. The shim is a **ThreadingHTTPServer** — required, since
regcheck's httpx client pools keep-alive connections and a single-threaded server
resets them (`WinError 10053/10054`).

**Caveat — degraded embeddings:** `sentence-transformers` is not installed, so the shim
uses a deterministic hashing pseudo-embedding fallback. regcheck's retrieval is slightly
weaker than with a real embedding model, but verdicts are produced normally. To upgrade:
`"$REGCHECK_PYTHON" -m pip install sentence-transformers` (large download) before running.

---

## Spike findings (2026-06-08)

| Check | Result |
|---|---|
| **(a) codex round-trip** | PASS. `codex exec --skip-git-repo-check -o <file> "<prompt>"` writes ONLY the final assistant answer to `<file>`. (stdout also ends with the answer, used as a fallback.) |
| **(b) regcheck offline install + non-GROBID parser** | PASS. The pinned `requirements.txt` will NOT build on Python 3.14 (`pandas==2.2.2` has no 3.14 wheel and fails to compile). The **`general` CLI path needs only a minimal subset**, all of which have 3.14 wheels — see "Install" below. `--parser-choice **pymupdf**` (and plain `.txt`) avoid GROBID/Java entirely (there is no Java on this host). `read_file` / `_extract_paper_sections` accept `.txt` for both prereg and paper, so we feed the arena's plain-text fields directly. |
| **(c) embeddings needed?** | YES. `backend.cli general` → `general_preregistration_comparison` → `run_comparison` calls `build_corpus` + `get_embedding` → `/v1/embeddings`. So the shim implements **both** `/v1/chat/completions` and `/v1/embeddings`. |

regcheck git SHA at time of writing: **`bdb961d`** (`git rev-parse HEAD` in the clone).

---

## One-time setup (NOT committed — heavy clone + venv stay out of the repo)

```bash
# 1. Clone regcheck somewhere OUTSIDE the Meta Science Arena repo (use forward slashes on Windows).
git clone https://github.com/JamieCummins/regcheck "C:/path/to/regcheck"

# 2. Create a venv and install ONLY the CLI-path deps (the pinned requirements.txt
#    fails on Python 3.14; this minimal set installs cleanly with 3.14 wheels).
python -m venv "C:/path/to/regcheck_venv"
"C:/path/to/regcheck_venv/Scripts/python.exe" -m pip install \
  "openai>=2.6" groq "pydantic>=2.7" python-dotenv numpy PyMuPDF tiktoken \
  python-docx fpdf httpx fastapi starlette uvicorn python-multipart Jinja2 \
  itsdangerous redis boto3 requests
# (fastapi/redis/boto3 are imported eagerly by backend/__init__.py even though the
#  `general` path doesn't use them; they are pure-python / have 3.14 wheels.)

# Optional, for BETTER embeddings than the hashing fallback (large download):
#   "C:/path/to/regcheck_venv/Scripts/python.exe" -m pip install sentence-transformers
```

Point the adapter / runbook at them:

```bash
export REGCHECK_DIR="C:/path/to/regcheck"
export REGCHECK_PYTHON="C:/path/to/regcheck_venv/Scripts/python.exe"
```

---

## Manual run (one task, by hand)

```bash
# A. Start the shim (ephemeral or fixed port). It prints the base URL.
python -m players.regcheck_shim.openai_shim --port 8765
#   -> regcheck_shim listening on http://127.0.0.1:8765/v1

# B. In another shell, point regcheck at the shim with a DUMMY key.
export OPENAI_BASE_URL="http://127.0.0.1:8765/v1"
export OPENAI_API_KEY="dummy-local-shim"     # never a real key
export GROQ_API_KEY="dummy-local-shim"       # regcheck constructs a Groq client at import
# Optional: pick the codex model the shim uses:
#   export REGCHECK_SHIM_CODEX_MODEL=gpt-5.5

# C. Write the task's prereg/paper text + a dimensions CSV (one `dimension` column),
#    then run regcheck's general comparison with the pymupdf parser and JSON output.
cd "$REGCHECK_DIR"
"$REGCHECK_PYTHON" -m backend.cli general \
  --preregistration /tmp/prereg.txt \
  --paper /tmp/paper.txt \
  --dimensions-csv /tmp/dimensions.csv \
  --client openai \
  --parser-choice pymupdf \
  --output-format json \
  --output /tmp/result.json

# D. Convert regcheck's result.json into the arena output schema.
python -m players.regcheck_shim.regcheck_to_runrecords /tmp/result.json --out /tmp/output.json
```

`dimensions.csv` is just:

```
dimension
sample_size
data_source
...
```

(the 12 ids in `arenas/prereg-deviation-v1/catalogs/dimensions.yaml`).

---

## Automated run (framework runner — produces committed run records)

`RegcheckShimAdapter` (`players/adapters/regcheck_shim.py`) does all of A–D per
task: starts the shim in-process on an ephemeral port, writes the temp files,
shells `backend.cli general`, and maps the result. `input_hash` + `provenance`
come from the framework runner, so records are directly comparable to the Claude
baselines.

### Three providers = three distinct players (2026-07-02)

regcheck is an LLM-backed comparator, so it runs under THREE LLM providers, each a
**distinct player** on the leaderboard (registry ids `regcheck-groq`,
`regcheck-deepseek`, `regcheck-openai`). ONE adapter class serves all three; the
per-player registry field `regcheck_client` pins the provider (and
`regcheck_model` the model). In every mode the EMBEDDINGS go to the local shim
(neither Groq nor DeepSeek exposes an embeddings API), so **no real OpenAI key is
ever needed**. Provider keys are read from the live environment ONLY — never
committed, never on argv, never in the sanitized provenance command. Note:
regcheck's `backend/services/comparisons.py` constructs a `Groq()` client at
MODULE IMPORT, so `GROQ_API_KEY` must be present in the env for *every* provider
(it is unused by the deepseek/openai players, but the import fails without it).

| player | `regcheck_client` | model | key(s) needed | speed | free-tier note |
|---|---|---|---|---|---|
| `regcheck-groq` | groq | `llama-3.1-8b-instant` | `GROQ_API_KEY` | ~30–40s/task | free tier; **use 8b-instant** — `llama-3.3-70b-versatile` exhausts the daily token cap (TPD 100k) after ~4–7 tasks. 429-lost tasks are infra throttle → excluded from scoring. |
| `regcheck-deepseek` | deepseek | `deepseek-chat` | `DEEPSEEK_API_KEY` (+`GROQ_API_KEY` for import) | ~50–70s/task | **paid** API (no free-tier TPD wall) → reliably completes all 46; the recommended lead provider. |
| `regcheck-openai` | openai (codex shim) | codex default (ChatGPT subscription) | none real (+`GROQ_API_KEY` for import) | ~6 min/task | no token cap, but SLOW — a full 46-task split is ~4.5h of codex latency. Run as ONE backgrounded serial script. |

```bash
export REGCHECK_DIR="C:/path/to/regcheck"
export REGCHECK_PYTHON="C:/path/to/regcheck_venv/Scripts/python.exe"
# provider keys — env ONLY, never committed. GROQ_API_KEY is required for ALL three
# (regcheck imports a Groq client at module load); DEEPSEEK_API_KEY only for deepseek.
export GROQ_API_KEY="gsk_…"
export DEEPSEEK_API_KEY="sk-…"

# The runner auto-names runs/<task-set>/<player>__<split>__<tag>.jsonl and, with
# --overwrite, replaces the target file (without it, records APPEND → double).
# Run each provider as its own player; DeepSeek first (paid, reliably full):
for P in regcheck-deepseek regcheck-groq regcheck-openai; do
  python -m framework run --arena prereg-deviation-v1 --task-set v1 \
    --players "$P" --split revealed --tag c6 --trials 1 --timeout 300 --overwrite
done
#   --max-tasks N      # run only the first N tasks (a representative subset)
```

(Run `python -m framework run --help` for the full flag list; the runner CLI is
`framework/cli.py`. `--trials 1` is the intended cost-bounded mode — regcheck is
`deterministic:false` in the LLM sense.)

### Coverage note

The **paid DeepSeek** player reliably completes the whole revealed split (all 46
tasks), which is what makes prereg-deviation-v1 pass the per-split symmetry gate.
The **free Groq** player is best-effort: it may lose some tasks to the free-tier
per-minute/daily throttle (`429`), which are infra errors (not wrong answers) and
are excluded from scoring — a partial Groq run is documented, not fabricated. The
**codex-shim OpenAI** player is slow; run it unattended (a full split is hours of
codex latency). Every committed record is a REAL provider-backed regcheck run.

---

## Mapping (regcheck → arena output schema)

regcheck emits per dimension a `deviation_judgement` of `yes` / `no` / `missing`.
`regcheck_to_runrecords.py` maps:

| regcheck `deviation_judgement` | arena `deviation` | `confidence` |
|---|---|---|
| `yes` | `true` | 0.9 |
| `no` | `false` | 0.9 |
| `missing` (or blank/unexpected) | `false` | 0.3 |

`deviation_kind` is not produced by regcheck in the arena's label vocabulary, so
when a deviation is flagged we look up the dimension's canonical kind from
`arenas/prereg-deviation-v1/catalogs/dimensions.yaml` (a deterministic 1:1 map):
regcheck decides *whether* a dimension deviates; the arena's closed dimension set
fixes *what kind*.

## Embedding fidelity note

If `sentence-transformers` is importable the shim uses `all-MiniLM-L6-v2`
(real semantic embeddings). Otherwise it falls back to a **deterministic
hashing-based pseudo-embedding** (`REGCHECK_SHIM_NO_ST=1` forces this). The
fallback only affects regcheck's internal chunk RETRIEVAL (which excerpts it
shows the model), not the final yes/no judgement directly; retrieval quality is
degraded but the pipeline runs fully offline.
