"""The single, class-level contamination boundary for held-out tasks.

Meta Science Arena's headline integrity guarantee is that a held-out task's *gold
answer* and the *player output* that reconstructs it never leak — not into a
tracked source file, not into the published static bundle. Historically that
redaction lived in three separate, divergent code paths (the per-arena
`dump_ground_truth.py` scripts, the runner's findings stripper, and
build-data's `publicOnly`/`redactHeldOutRecords`), and a 2026-06-28 deep review
found the *committed* artifacts — the tracked `_ground_truth.json` golds and the
tracked run JSONL `output` — were never covered, so the gold answer key for 84
held-out real-paper tasks sat in git (DR-0007).

This module is the fix as a *class*, not per-symptom: one place that defines what
"held-out" means and what must be stripped, consumed by every writer of a
tracked or published artifact, and asserted by one guard test
(`framework/tests/test_holdout_contamination.py`) over every tracked file.

The invariant, stated once:

    For any held-out task, NO tracked file and NO public artifact may carry
    its gold answer (any ``ground_truth`` field whose key starts with ``gold``),
    its reconstructable player ``output``, its per-task ``score.breakdown``,
    or its ``input_hash`` membership oracle.

Visibility, difficulty axes, and non-revealing aggregate counts (e.g.
``section_count``) are intentionally *kept* — the build needs them to profile
difficulty and to know a task is held-out at all.
"""
from __future__ import annotations

from typing import Any

HELD_OUT = "held_out"

# Held-out records keep an ``input_hash`` STRING (the run-record schema requires
# one) but every held-out record gets this same constant, so the field can no
# longer act as a per-task membership oracle. A constant is preferable to
# dropping the key (which would break schema validation) or to an empty string
# (less self-documenting in a leaked file).
REDACTED_INPUT_HASH = "<redacted>"
#: Replaces `provenance.seed` on a held-out record. The private seed regenerates the
#: entire held-out split (tasks AND gold), so it must never reach a tracked or
#: published artifact.
REDACTED_SEED = "<redacted>"

# A ground_truth field is "gold" — the answer key — iff its name starts with
# this prefix. Every PDF arena follows the convention (gold_sections,
# gold_full_text, gold_tables, gold_ascii_greek, ...); enforcing the prefix is
# what lets one rule cover all arenas and any future one. Non-gold GT fields
# (doc_id, source_kind, *_count, *_diversity, *_complexity) are non-revealing
# aggregates kept for difficulty profiling.
GOLD_FIELD_PREFIX = "gold"


def is_held_out(envelope: dict | None) -> bool:
    """True iff an envelope marks its task held-out.

    Defaults to held-out when visibility is absent: the safe direction is to
    over-redact an unlabeled task, never to under-redact one.
    """
    if not isinstance(envelope, dict):
        return True
    return envelope.get("visibility", HELD_OUT) == HELD_OUT


def gold_keys(ground_truth: dict | None) -> list[str]:
    """The gold (answer-key) field names present in a ground_truth dict."""
    if not isinstance(ground_truth, dict):
        return []
    return [k for k in ground_truth if str(k).startswith(GOLD_FIELD_PREFIX)]


def strip_gold(ground_truth: dict | None) -> dict:
    """Return ``ground_truth`` with every gold (answer-key) field removed.

    Keeps non-gold metadata/aggregate fields untouched. Non-mutating.
    """
    if not isinstance(ground_truth, dict):
        return {}
    return {k: v for k, v in ground_truth.items() if not str(k).startswith(GOLD_FIELD_PREFIX)}


def redact_ground_truth_entry(entry: dict) -> dict:
    """Redact one ``{envelope, ground_truth}`` task entry for a TRACKED dump.

    For a held-out task the gold answer fields are stripped from
    ``ground_truth`` (visibility/difficulty metadata is preserved). Public and
    legacy (no-visibility) entries pass through unchanged. Non-mutating: returns
    a new dict, so callers can map over a generator's output safely.
    """
    if not isinstance(entry, dict):
        return entry
    envelope = entry.get("envelope")
    if not is_held_out(envelope):
        return dict(entry)
    out = dict(entry)
    out["ground_truth"] = strip_gold(entry.get("ground_truth"))
    out["envelope"] = _strip_held_out_input(envelope)
    return out


# Envelope input fields that ARE the held-out secret and must never reach a
# tracked dump. For a document arena the bytes are already omitted upstream; for
# code-translation-r-v1 the held-out SOURCE SCRIPT is itself the thing players
# must not have seen, so publishing it would defeat the split entirely.
# Difficulty/visibility metadata and the required-statistic NAMES stay: they are
# non-revealing and the leaderboard needs them to render the split.
_SECRET_INPUT_FIELDS = ("source_code", "text", "document_bytes_b64", "content")
REDACTED_INPUT = "<redacted: held-out>"


def _strip_held_out_input(envelope: dict | None) -> dict:
    """Blank the revealing input fields of a held-out envelope. Non-mutating."""
    if not isinstance(envelope, dict):
        return {}
    out = dict(envelope)
    inp = out.get("input")
    if isinstance(inp, dict):
        red = dict(inp)
        for key in _SECRET_INPUT_FIELDS:
            if red.get(key):
                red[key] = REDACTED_INPUT
        out["input"] = red
    return out


def sanitize_error(message: Any) -> str:
    """Reduce an error string to its exception TYPE, dropping the message body.

    Why: for a held-out task the ``error`` marker is deliberately preserved (it
    is operational, and aggregation excludes errored records from the mean), but
    the message itself is free text produced by a scorer or an external tool. It
    is an uncontrolled channel out of a held-out task — an assertion such as
    ``AssertionError: expected '<the gold text>'`` would pass every other check
    in this module, because the ``error`` key exempts the breakdown entirely.

    Flagged by a cross-model review (codex, 2026-08-07) and reproduced: a record
    with ``{"error": "gold=SECRET"}`` returned no leak reasons and survived
    redaction verbatim. Live data held 47 such records — all genuinely
    operational (RuntimeError/HTTPError/TimeoutError), so latent rather than a
    breach, but nothing bounded what could land there next.

    The type alone keeps the signal a maintainer actually uses (timeout vs HTTP
    vs scorer crash) and carries no task content.
    """
    text = str(message or "").strip()
    if not text:
        return "Error"
    head = text.split(":", 1)[0].strip()
    # Exception type names only; anything else collapses to a constant so a bare
    # free-text message cannot squeeze through as a "type".
    return head if head.isidentifier() else "Error"


def redact_held_out_record(record: dict) -> dict:
    """Redact one run record so a TRACKED or PUBLISHED JSONL carries no held-out leak.

    For a held-out record (``task_visibility == "held_out"``):
      * ``output`` -> ``{}``        (player output reconstructs the gold doc text)
      * ``input_hash`` -> ``REDACTED_INPUT_HASH`` (was a membership oracle; the
        run-record schema requires the key, so we constant it rather than drop it)
      * ``score.breakdown`` -> ``{}`` UNLESS it carries an ``error`` marker
        (which is operational, not gold — e.g. ``{"error": "TimeoutError: ..."}``);
        ``score.primary`` is preserved so aggregate ranking still works.
      * ``usage`` -> dropped. ``prompt_tokens`` is a near-exact proxy for the
        length of the held-out input document, so publishing it hands out a
        continuous measurement of a corpus we do not distribute.

    Findings are already redacted to ``{category, count}`` at write time by the
    runner; this closes the remaining channels. Non-mutating.
    """
    if not isinstance(record, dict):
        return record
    # Fail SAFE on a missing marker, per the run-record schema's documented
    # default and to match `held_out_leak_reasons`. Over-redacting an unlabelled
    # record is recoverable; publishing a held-out answer is not.
    if record.get("task_visibility", HELD_OUT) != HELD_OUT:
        return dict(record)
    out = dict(record)
    if out.get("output"):
        out["output"] = {}
    # Always set the sentinel: held-out records MUST carry an input_hash string
    # (schema-required) but never the real per-task hash. Unconditional so the
    # redaction is self-healing — a record whose hash was previously dropped
    # gets the sentinel back rather than staying schema-invalid.
    out["input_hash"] = REDACTED_INPUT_HASH
    # Token counts are a length oracle. `prompt_tokens` tracks the input document
    # almost linearly, so a published held-out row would quantify a PDF we
    # deliberately do not ship. The run-record schema has promised this redaction
    # since 2026-08-13; nothing performed it until 2026-08-14, which was harmless
    # only because no record carried `usage` yet.
    out.pop("usage", None)
    score = out.get("score")
    if isinstance(score, dict):
        score = dict(score)
        breakdown = score.get("breakdown")
        if isinstance(breakdown, dict) and breakdown.get("error"):
            # Keep the operational error marker, but only its TYPE — the message
            # body is free text from a scorer or external tool. See sanitize_error.
            score["breakdown"] = {"error": sanitize_error(breakdown["error"])}
        elif "breakdown" in score:
            score["breakdown"] = {}
        out["score"] = score
    # provenance.seed IS the secret. Arena tasks and their gold are a deterministic
    # function of it, so a committed private record carrying it verbatim lets anyone
    # with repo access regenerate the whole held-out split and score perfectly. This
    # is the second half of the original stats-extraction-v1 exposure — the first
    # (the seed embedded in task_id strings) was closed by task set v2's hashed
    # discriminator, and the very first v2 private run promptly republished the
    # freshly-rotated seed through this channel. Nothing legitimately reads the raw
    # seed back out of a record: reproducing a private run needs `.private_seed`
    # regardless. The key stays so the record shape does not change for consumers.
    provenance = out.get("provenance")
    if isinstance(provenance, dict) and "seed" in provenance:
        provenance = dict(provenance)
        provenance["seed"] = REDACTED_SEED
        out["provenance"] = provenance
    return out


def held_out_leak_reasons(entry_or_record: dict, *, kind: str) -> list[str]:
    """Return human-readable reasons a held-out artifact leaks (empty == clean).

    ``kind`` is ``"ground_truth"`` (a ``{envelope, ground_truth}`` entry) or
    ``"record"`` (a run record). Used by the guard test to produce precise
    failure messages; safe to call on any entry (returns [] for non-held-out).
    """
    reasons: list[str] = []
    if not isinstance(entry_or_record, dict):
        return reasons
    if kind == "ground_truth":
        if not is_held_out(entry_or_record.get("envelope")):
            return reasons
        leaked = gold_keys(entry_or_record.get("ground_truth"))
        if leaked:
            reasons.append(f"ground_truth carries gold field(s): {sorted(leaked)}")
        return reasons
    if kind == "record":
        # Fail SAFE, matching envelopes and matching what the run-record schema
        # already promises: task_visibility "Defaults to 'held_out' (fail-safe)
        # when the envelope did not declare visibility."
        #
        # This used to read `!= HELD_OUT`, which returned "clean" for a record
        # with NO marker at all — so an unmarked record was invisible to the
        # leak check no matter what it carried, and the code contradicted its
        # own published contract. 136 tracked records were in that state (107
        # with a non-empty output) when this was found on 2026-08-07; none
        # mapped to a held-out task, so it was latent rather than a breach, but
        # nothing stopped the next one from being real.
        if entry_or_record.get("task_visibility", HELD_OUT) != HELD_OUT:
            return reasons
        if entry_or_record.get("output"):
            reasons.append("output is non-empty")
        ih = entry_or_record.get("input_hash")
        if ih and ih != REDACTED_INPUT_HASH:
            reasons.append("input_hash present (membership oracle)")
        if entry_or_record.get("usage"):
            # prompt_tokens ≈ input document length. A per-task length readout
            # over a corpus we do not publish is a leak, not metadata.
            reasons.append(
                f"usage present (token counts are a length oracle): "
                f"{sorted(entry_or_record['usage'])}"
            )
        breakdown = (entry_or_record.get("score") or {}).get("breakdown")
        if isinstance(breakdown, dict) and breakdown:
            err = breakdown.get("error")
            if not err:
                reasons.append(f"score.breakdown carries keys: {sorted(breakdown)}")
            else:
                # The error marker is allowed, but only in its sanitized form —
                # otherwise it is an unbounded free-text channel (see sanitize_error).
                if set(breakdown) != {"error"}:
                    reasons.append(
                        f"score.breakdown carries keys beside 'error': "
                        f"{sorted(set(breakdown) - {'error'})}"
                    )
                if str(err) != sanitize_error(err):
                    reasons.append(
                        f"score.breakdown.error carries a message body, not just its "
                        f"type: {str(err)[:60]!r}"
                    )
        return reasons
    raise ValueError(f"unknown kind {kind!r} (expected 'ground_truth' or 'record')")


# ---------------------------------------------------------------------------
# Corpus presence — "missing" must never read as "empty"
# ---------------------------------------------------------------------------
# The held-out corpora are gitignored: they live in the working tree only, and
# they hold the task INPUT (the PDFs players are scored on), not just the gold.
# Every PDF arena's real-paper generator opened with the same three lines:
#
#     if not HELD_OUT_PMC_DIR.exists():
#         return
#
# which turns "the corpus is gone" into "this arena has no real-paper tasks" —
# reported as a successful, merely smaller, benchmark. On 2026-08-08 a
# `git stash --all` removed all 123 gitignored corpus files and every one of the
# six PDF arenas silently dropped to synthetic-only. Nothing failed; 137 tests
# skipped; the run would have published.
#
# That is the same false-green class as a suite quietly collecting 1023 of 1143
# tests, and the same one `scan_for_leaks` already refuses to commit: a check
# that cannot see its subject must not certify it.
#
# So absence now RAISES — unless the caller has DECLARED that it wants a
# synthetic-only task set, which makes the reduced benchmark a stated choice
# rather than an accident.

#: Set to "1" to declare that a synthetic-only task set is intended.
SYNTHETIC_ONLY_ENV = "SCIENCEARENA_SYNTHETIC_ONLY"


class HeldOutCorpusMissing(RuntimeError):
    """A generator's held-out corpus is absent and no synthetic-only opt-in was declared."""


def require_corpus(path, *, arena_id: str, kind: str):
    """Return `path` if the corpus is present; otherwise fail loudly.

    Returns ``None`` (after a stderr warning) only when the caller has declared
    a synthetic-only run via ``SCIENCEARENA_SYNTHETIC_ONLY=1``. Callers should
    treat ``None`` as "emit no real-paper tasks".
    """
    import os
    import sys
    from pathlib import Path

    path = Path(path)
    if path.exists():
        return path

    if os.environ.get(SYNTHETIC_ONLY_ENV) == "1":
        print(
            f"[holdout] {arena_id}: {kind} corpus absent at {path} — emitting a "
            f"SYNTHETIC-ONLY task set because {SYNTHETIC_ONLY_ENV}=1 was set. "
            "Scores from this run are not comparable with a full run.",
            file=sys.stderr,
        )
        return None

    raise HeldOutCorpusMissing(
        f"{arena_id}: the {kind} held-out corpus is missing at {path}.\n"
        "These files are GITIGNORED — they exist only in the working tree, and "
        "`git stash -a`, a branch switch, or a manual tidy-up all remove them "
        "without git noticing.\n"
        "Generating anyway would silently produce a SMALLER benchmark and report "
        "it as a success, so this raises instead.\n"
        "Fix: restore the corpus (see TODO.md P0-0), or set "
        f"{SYNTHETIC_ONLY_ENV}=1 to declare that a synthetic-only task set is "
        "what you actually want."
    )
