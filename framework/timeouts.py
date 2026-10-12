"""A TRANSPORT TIMEOUT is our clock, not the player's answer — one definition.

OWNER'S DECISION, 2026-10-07 (d-545c62), on the recommendation given 2026-10-06:

  1. A transport timeout is NEVER scored. `report.apply_unscoreable_policy` zeroes
     an unscoreable record when another player scored the task; a timeout is
     excluded instead, like a crash, because it measures the wall we set and not
     the model. Zeroing it would publish our mistake as the player's.
  2. Any timeout in a live run file is a hard landing gate:
     `scripts/qa_coverage_checks.py` C11 fails until the tasks are re-run.
  3. `framework run` / `retry-failed` size their clock from the player's DECLARED
     `max_tokens` when `--timeout` is not given, so the default clock can always
     express the output ceiling the registry configures.

WHY ONE MODULE. The same question ("is this error our clock?") was answered by
three private marker lists: the runner's run summary, `scripts/run_health.py`, and
— implicitly, by not asking it — the scoring policy. Two of them already
disagreed (the runner knew `exited 124` / `exited 137`; run_health knew
`timed out`). A scoring rule keyed on a list that drifts from the gate's list is
how a timeout gets zeroed on the board while the gate calls it a timeout.

`leaderboard-app/scripts/lib/unscoreable.mjs` mirrors `TRANSPORT_TIMEOUT_MARKERS`
and `is_transport_timeout`; `unscoreable-parity.test.ts` reads this file and fails
if the two lists differ.
"""
from __future__ import annotations

#: Lower-case markers that mean OUR wall expired. Where each may match (see
#: `is_transport_timeout`): the bare "timeout" ONLY in the exception TYPE
#: (TimeoutExpired, ReadTimeout, TimeoutError, TransportTimeout) — in free text it
#: is as likely to be "invalid timeout value"; the free-text markers "timed out"
#: (urllib/socket wording), "exited 124" / "exited 137" (GNU `timeout` exiting, and
#: SIGKILL after `-k` escalation) in the type or the first message segment.
TRANSPORT_TIMEOUT_MARKERS = ("timeout", "timed out", "exited 124", "exited 137")

#: An answer the arena's schema rejected is the model's output, whatever words it
#: contains — a violation message quoting a field named "timeout" is not our clock.
_MODEL_OUTPUT_PREFIX = "output_schema_violation"

# Measured decode floor for the local GGUF players on this machine: gpt-oss-20b Q8_0
# on the Vulkan iGPU sustains 7.3-8.3 tok/s, and drops to ~0.45 tok/s only under memory
# pressure (observed 2026-08-21, and that is a RAM verdict, not a budget to plan for).
# 6.0 is deliberately below the healthy range so a normal run never trips the ceiling.
DECODE_FLOOR_TOK_S = 6.0
# Prefill plus scheduling slack. Arena prompts here are 247-1553 tokens at ~80 tok/s.
TIMEOUT_OVERHEAD_S = 180


def is_transport_timeout(error: object) -> bool:
    """True when a record's error text says OUR wall expired, not the model's answer.

    ANCHORED, not a substring over the whole message (Sonnet consult 2026-10-07,
    reproduced): adapters embed model output and tool stderr AFTER the second
    ": " -- `ValueError: no parseable JSON in CLI output (len=40): '<model text>'`
    -- so a model that wrote "the request timed out" would have been excused as
    our clock. Only the exception TYPE (before the first ": ") and the first
    message segment (before the second) are read. Split on ": " rather than ":"
    because a Windows command path (`C:/.../claude.EXE exited 124`) carries a
    bare colon. Every real timeout in the archives (39 records) still matches on
    its type alone: TimeoutExpired, ReadTimeout, TimeoutError.
    """
    if not isinstance(error, str) or not error:
        return False
    low = error.lower().strip()
    if low.startswith(_MODEL_OUTPUT_PREFIX):
        return False
    parts = low.split(": ", 2)
    if "timeout" in parts[0]:
        return True
    # The bare word "timeout" counts only in the TYPE: in free text it is as likely
    # to be "invalid timeout value in config" (review 2026-10-08, reproduced).
    anchored = " ".join(parts[:2])
    return any(m in anchored for m in TRANSPORT_TIMEOUT_MARKERS if m != "timeout")


def record_is_transport_timeout(rec: dict) -> bool:
    breakdown = (rec.get("score") or {}).get("breakdown")
    err = breakdown.get("error") if isinstance(breakdown, dict) else None
    return is_transport_timeout(err)


def timeout_for_cap(max_tokens: int) -> int:
    """Seconds a run must allow if it declares `max_tokens`.

    A timeout smaller than this cannot express the output ceiling the registry
    configures, so the run silently measures the clock instead of the model.
    """
    return int(max_tokens / DECODE_FLOOR_TOK_S) + TIMEOUT_OVERHEAD_S


def declared_max_tokens(entry: dict) -> int | None:
    """The output ceiling a registry entry declares, or None if it declares none.

    Measured 2026-10-07: 74 of 313 entries declare one, all under
    `openai_extra_body.max_tokens`. Since 2026-10-08 a `claude` CLI player can
    declare `cli_max_output_tokens` (d-60b28c); without it here the clock stayed
    at the 120 s default and every long v2 answer timed out.
    """
    for cap in ((entry.get("openai_extra_body") or {}).get("max_tokens"),
                entry.get("cli_max_output_tokens")):
        if isinstance(cap, int) and not isinstance(cap, bool) and cap > 0:
            return cap
    return None


def timeout_for_entry(entry: dict, fallback: int) -> int:
    """The clock for one player when the operator gave no `--timeout`.

    Sized from the declared ceiling when there is one; `fallback` (the command's
    historical default) when the player declares none. Never below `fallback`.
    """
    cap = declared_max_tokens(entry)
    return max(fallback, timeout_for_cap(cap)) if cap else fallback
