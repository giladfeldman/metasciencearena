# Data handling — what leaves your machine when you run players

Meta Science Arena's leaderboard and public API are static and self-contained; nothing
about a benchmark run is sent anywhere by the site. **But some *players*
(benchmark competitors) are thin adapters around third-party cloud services**,
and running those players transmits the task input — including held-out real
papers — to those services. This note exists so an operator can make an informed
choice before running them. (This concerns *local benchmark execution only*; it
does not affect the published bundle, whose held-out contamination boundary is
documented in [`README.md`](README.md#contamination-posture-for-tool-feedback-reports)
and enforced by `framework/holdout.py`.)

## What we control, and what we don't

Being precise about this matters more than sounding strong:

- **We control what we hand over.** For every player, on every split, the scorer
  receives the **task** and never the **outcome**. Held-out gold, the output that
  reconstructs it, per-task score breakdowns, and `input_hash` never reach a
  tracked file, the published bundle, or a player.
- **We do not control what a provider already has.** Many arenas are built on
  published papers — exactly the material large models train on. A paper sitting
  in our held-out set says nothing about whether it was in someone's training
  corpus, and we make no claim either way.
- **We do not control what a provider does with an API call.** Whether a request
  is retained, or influences later responses, is outside our visibility.

So the guarantee is scoped and honest: **best effort to give scorers the task and
never the outcome, plus an explicit statement of the limits.** Held-out numbers
are contamination-*resistant*, not contamination-*free*. Where the distinction
carries weight, the procedurally generated arenas are the stronger evidence —
their task instances are genuinely novel, not merely unpublished.

## Players that send data to third-party cloud services

| Player family | Adapter(s) | Sends to | What is sent |
|---|---|---|---|
| LLM-as-player (CLI) | `players/adapters/llm_pdf*.py` (`llm_pdf`, `llm_pdf_sections`, `llm_pdf_tables`, `llm_pdf_citations`, `llm_pdf_references`) | Anthropic / Google / OpenAI (via the `claude` / `gemini` / `codex` CLIs) | The **full task PDF**, including held-out real PMC/APA papers (identifiable authors, affiliations, full text; some APA PDFs are copyrighted) |
| OpenAI-compatible model endpoints | `players/adapters/openai_compatible.py` | The configured provider or gateway (NVIDIA NIM, OpenRouter, Groq, Mistral, Hugging Face, OmniRoute named backend, etc.) | The task input rendered into the arena prompt; for PDF/text arenas this may include full paper text |
| ↳ **routers are a chain, not a destination** | same adapter, any `OPENROUTER_*` player | OpenRouter **and whichever backend it forwards to** — measured 2026-08-13, one `openai/gpt-oss-120b` id resolves to CoreWeave, DeepInfra, Novita, Amazon Bedrock or Google depending on routing | Same as above. Naming the gateway does not name the recipient, so a router adds an *unbounded* set of sub-processors unless pinned. The registry pins one backend (`provider.order` + `allow_fallbacks: false`) and every record carries `response_meta.served_by`, so the actual recipient of each call is recoverable from the run artifact rather than assumed |
| regcheck shim | `players/adapters/regcheck_shim.py` | Groq (if the shim is configured to use a Groq-hosted model) | Manuscript / preregistration text for the prereg-deviation arena |
| scimeto (remote) | `players/adapters/scimeto_*.py` | The configured `SCIMETO_API_URL` endpoint | Citation / matching / replication-lookup queries |

Tool/library players that run **entirely locally** — docpluck, liteparse,
pdftotext, GROBID (your own server), Docling, statcheck, escimate, the R tools — send
**nothing** off-machine.

## What this means

- **PMC Open Access** content generally permits downstream use, but you are
  still transmitting it to a commercial provider; check that provider's data-use
  and retention terms.
- **APA-journal held-out PDFs are copyrighted.** Only run the LLM-CLI players on
  the held-out (private) split if your data-handling obligations and the
  providers' terms permit sending copyrighted full text to them.
- The held-out gold answers are **not** sent (the player only ever sees the
  input PDF, never the gold). The concern is the *input papers* themselves.

## The egress gate (enforced, not advisory)

`framework run` **refuses** to send held-out tasks to a cloud-backed player. The
run aborts before the first task with a message naming the offending player(s).
To proceed anyway you must set an environment variable explicitly:

```bash
SCIENCEARENA_ALLOW_HELDOUT_EGRESS=1 framework run --arena … --split private --held-out-only --players claude-…
```

Two guards, on purpose:

1. **CLI default** — `--split revealed` now implies `--public-only`. Opt out with
   `--include-held-out` (which the gate below will still stop).
2. **Runner gate** — `framework/runner.py::assert_heldout_egress_allowed`, which
   also covers callers that bypass the CLI (scripts, notebooks, `retry-failed`).

An adapter counts as cloud by class-name prefix, and **unknown adapters are
treated as cloud**: a false positive costs one env var, a false negative is
irreversible disclosure.

> **Why this is enforced rather than documented.** On 2026-08-04 a
> `--split revealed` run of `claude-sonnet-5-sections` transmitted **9 held-out
> real PMC papers** to Anthropic before the operator noticed the record count was
> 15 instead of 6. `--split` only selects the generator seed — it never filtered
> by visibility, and the PDF arenas' generators emit both visibilities from one
> call. The advice that used to sit here ("run the LLM players on the revealed
> split only") was therefore *wrong*, and following it caused the leak.

## How to avoid the egress

- Run only the local tool/library players (omit `claude-*` / `gemini-*` /
  `gpt-*` from `--players`, and don't configure scimeto/regcheck remote
  endpoints). The local players fully exercise every arena, held-out included.
- Or pass `--public-only` explicitly. Do **not** rely on `--split revealed`
  alone as a safety mechanism — it is a seed selector. (It now implies
  `--public-only`, but be explicit in scripts that must stay safe if the default
  ever changes.)
