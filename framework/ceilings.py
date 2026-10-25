"""Whose output ceiling stopped this answer — ours, or the model's own?

ONE definition, imported by BOTH gates. `scripts/qa_coverage_checks.py` (C6) and
`scripts/run_health.py` each answered this question, and on 2026-09-12 they
answered it DIFFERENTLY about the same three records: C6 passed them as
attributable to the model while run_health failed them as "stopped exactly at OUR
ceiling, so it is binding" and prescribed "raise openai_extra_body.max_tokens" —
a remedy the provider refuses, because the ceiling in question already WAS the
provider's maximum. A gate that prints an unexecutable remedy sends the next
session to do something impossible, and a gate that calls the model's limit ours
is the "publishing OUR max_tokens as THEIR incapability" failure running in
reverse.

They disagreed because run_health had only half the inputs: it read the ceiling
off the record correctly and then had no way to notice that ours IS theirs. The
fix is not a second copy of the comparison inside run_health — that is how the
two drifted in the first place. It is this module, and a test in each caller
asserting neither re-implements it.

WHAT WAS ACTUALLY WRONG, stated precisely, because the conflation will otherwise be
re-introduced by the next reader. run_health's OBSERVATION was TRUE: the completion
did stop at the ceiling, and the ceiling WAS binding. Two other things were false:

  * the ATTRIBUTION — calling 4096 "OUR ceiling" implies a configuration we chose
    freely, when the boundary probe says it is the most the provider will accept;
  * the REMEDY — "raise openai_extra_body.max_tokens" is refused with HTTP 400 at
    the provider maximum, so the retry re-truncates forever.

So do not weaken the observation. A truncation at the provider's maximum is still
reported, still carries the task ids, and the SCORE STILL STANDS — a model that
cannot finish inside the most it is allowed to emit has a real capability limit,
and a low score for that task is an honest benchmark result rather than an
artifact. Only the attribution, the remedy and the severity change.

The two facts deliberately come from different places:

  * what ceiling the RECORD says was sent -- `provenance.request_max_tokens`
    (schema v9) or `response_meta.max_tokens` (the local llama.cpp path);
  * the model's MEASURED maximum, from `framework/contract/provider_ceilings.json`,
    which records a live boundary probe (largest accepted / smallest refused).

Never compare a registry value against a registry value. The ceiling we send comes
from `players/registry.yaml`; if the reference came from there too, the check would
be asking whether a number equals itself and could not fail.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CEILINGS_PATH = REPO / "framework" / "contract" / "provider_ceilings.json"

#: `response_meta.finish_reason` values meaning "stopped at the output ceiling".
#: OpenAI-compatible providers say "length"; Anthropic (API and Claude CLI) says
#: "max_tokens". Checking only "length" hid every Claude truncation (2026-10-23).
TRUNCATED_FINISH_REASONS = frozenset({"length", "max_tokens"})


def is_truncated(meta: dict | None) -> bool:
    return (meta or {}).get("finish_reason") in TRUNCATED_FINISH_REASONS

#: A truncation whose ceiling matched the provider's measured maximum. The model
#: ran to the most it is allowed to emit and still did not finish: a real property
#: of the model, not of our configuration.
MODEL_LIMIT = "model_limit"
#: A truncation below the provider's maximum. We throttled it; the score measures
#: our configuration. Raising the ceiling here is executable and is the remedy.
OUR_CEILING = "our_ceiling"
#: The record does not say what ceiling was sent, so the question cannot be
#: decided from it. NOT the same as "fine" — every pre-v9 hosted record is here.
UNKNOWN = "unknown"
#: The record's ceiling exceeds the measured maximum, or the model has no measured
#: row at all. The ledger is stale or wrong; fix the ledger, not the record.
NO_REFERENCE = "no_reference"


def sent_ceiling(rec: dict) -> int | None:
    """The ceiling this record SAYS was sent, or None when it does not say.

    A missing ceiling is None, never 0. Measured 2026-09-11: of 11,709 records the
    OpenAI-compatible path wrote neither field, so `None` is the common case for a
    hosted player and reading it as a zero would make every comparison spurious.
    """
    prov = rec.get("provenance") or {}
    for value in (prov.get("request_max_tokens"),
                  (rec.get("response_meta") or {}).get("max_tokens")):
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def stopped_at_sent_ceiling(rec: dict) -> bool | None:
    """Did the completion stop at EXACTLY the ceiling this record says we sent?

    True/False when both numbers are known, None when either is missing.

    This is the OBSERVATION, and it is separable from the ATTRIBUTION. Whether the
    ceiling was ours or the provider's needs a measured reference; whether it was
    BINDING needs only the record. Keeping the two apart is the whole point of this
    module: the message `run_health` used to print conflated them, asserting "OUR
    ceiling" from an observation that could only support "a ceiling". Dropping the
    observation when the reference is missing is the same mistake with the signs
    reversed, and it is what this helper exists to prevent — see the NO_REFERENCE
    branch of `verdict()`, and
    `framework/tests/test_run_health.py::test_truncation_will_not_claim_our_ceiling_it_cannot_see`,
    which caught exactly that loss in the full suite on 2026-09-12 after both of this
    module's own test files passed.
    """
    sent = sent_ceiling(rec)
    used = (rec.get("usage") or {}).get("completion_tokens")
    if sent is None or not isinstance(used, int) or isinstance(used, bool):
        return None
    return used == sent


def measured_ceilings(path: Path | None = None) -> tuple[dict, str]:
    """(model -> measured row, staleness complaint or "").

    A ledger that cannot be read returns an EMPTY mapping and a complaint, never a
    silent pass: an absent reference must make every truncation undecidable rather
    than making them all look attributable.
    """
    src = path or CEILINGS_PATH
    try:
        doc = json.loads(src.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, f"{src.name} is unreadable ({exc}) — no truncation can be adjudicated"
    checked = doc.get("checked_on") or ""
    max_age = doc.get("max_age_days")
    stale = ""
    if checked and isinstance(max_age, int):
        try:
            age = (dt.date.today() - dt.date.fromisoformat(checked)).days
            if age > max_age:
                stale = (f"{src.name} was measured {checked} ({age}d ago, limit {max_age}d) "
                         "— re-probe the boundaries before trusting it")
        except ValueError:
            stale = f"{src.name} has an unparseable checked_on={checked!r}"
    return doc.get("measured") or {}, stale


def verdict(rec: dict, measured: dict | None = None) -> tuple[str, str]:
    """(verdict constant, human sentence) for a record that truncated.

    Call only when `response_meta.finish_reason == "length"`; this does not check
    that, so a caller can report on whatever set it has already selected.
    """
    rows = measured if measured is not None else measured_ceilings()[0]
    model = rec.get("resolved_tool_version")
    sent = sent_ceiling(rec)
    cap = (rows.get(str(model)) or {}).get("max_output_tokens")

    if sent is None:
        return UNKNOWN, (
            "the record does not say what ceiling was sent, so whether it measures "
            "our limit or the model's cannot be decided from it"
        )
    if not isinstance(cap, int):
        # THE OBSERVATION SURVIVES THE MISSING REFERENCE. Without a measured row we
        # cannot say WHOSE limit this was — but we can still say whether the ceiling
        # we sent is what ended the completion, because that needs only the record.
        # Reporting the gap without the observation was a real regression: it told a
        # reader "no ledger row" and silently dropped "and the ceiling was binding",
        # which is the actionable half.
        at_cap = stopped_at_sent_ceiling(rec)
        if at_cap is True:
            observed = (f"the completion stopped at exactly the max_tokens={sent} we sent, "
                        "so that ceiling IS binding, but ")
        elif at_cap is False:
            observed = (f"the completion stopped BELOW the max_tokens={sent} we sent, so "
                        "something other than that ceiling ended it, and ")
        else:
            observed = ""
        return NO_REFERENCE, (
            f"{observed}no measured provider ceiling for model {model!r} in "
            f"{CEILINGS_PATH.name} — so whether it is ours or the model's cannot be "
            "decided; probe the boundary and add a row"
        )
    if sent < cap:
        return OUR_CEILING, (
            f"sent max_tokens={sent} is BELOW the measured maximum {cap} for {model} "
            "— the score measures our limit, not the model"
        )
    if sent > cap:
        return NO_REFERENCE, (
            f"sent max_tokens={sent} EXCEEDS the measured maximum {cap} for {model} "
            f"— {CEILINGS_PATH.name} is stale or names the wrong model"
        )
    return MODEL_LIMIT, (
        f"sent max_tokens={sent} IS the model's measured maximum for {model}, so the "
        "answer ran to the most it is allowed to emit and still did not finish — the "
        "model's own limit, not our configuration. THE SCORE STANDS: this is a real "
        "capability limit and a low score for the task is an honest benchmark result, "
        "not an artifact to be repaired"
    )


def remedy(verdict_name: str) -> str:
    """What to actually do. Never prescribe something the provider will refuse."""
    if verdict_name == OUR_CEILING:
        return ("raise openai_extra_body.max_tokens for this player, then "
                "`framework retry-failed --include-truncated`, which retires a task that "
                "re-truncates at the raised ceiling instead of reporting it fixed")
    if verdict_name == MODEL_LIMIT:
        return ("do NOT raise the ceiling — it is already the provider's measured maximum "
                f"(see {CEILINGS_PATH.name}, which records the accepted/refused boundary), "
                "so a higher value is refused with HTTP 400 and a retry re-truncates "
                "forever. This is a reportable property of the model: it did not finish "
                "inside the most it can emit. Leave the score, and say so in the writeup.")
    if verdict_name == UNKNOWN:
        return ("re-run this task so the record carries provenance.request_max_tokens "
                "(schema v9); until it does, the question cannot be answered from the record")
    return (f"add or refresh the model's row in {CEILINGS_PATH.name} by probing the live "
            "boundary — record the largest accepted and the smallest refused max_tokens")


#: Severity per verdict. The ship-coordinating seat ruled on 2026-09-12 that a
#: truncation AT the provider's measured maximum is a WARNING while every other
#: verdict stays a failure.
#:
#: This is discrimination, not a lowered floor, and the distinction is worth stating
#: because "we downgraded a gate" is exactly what a rot-detector should be suspicious
#: of. A truncation at a ceiling WE chose too low is our defect and must keep biting.
#: A truncation at the provider's own maximum is a property of the model. Those have
#: always been different events deserving different severities, and the gate used to
#: conflate them. Nothing a reader of the leaderboard sees moves either way: the score
#: is 0.0 in both cases.
_SEVERITY = {
    OUR_CEILING: "fail",    # our defect — the remedy is executable, keep it red
    UNKNOWN: "fail",        # a gap in the evidence, not a finding. Must not read as fine
    NO_REFERENCE: "fail",   # the ledger is stale or wrong — fix the ledger
    MODEL_LIMIT: "warn",    # the model's own limit; reported, score stands
}


def severity(verdict_name: str) -> str:
    """"fail" or "warn" for a verdict. Unknown names fail closed."""
    return _SEVERITY.get(verdict_name, "fail")
