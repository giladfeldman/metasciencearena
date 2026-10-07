"""STRUCTURAL ADHERENCE — can the player fill in the form, as opposed to answer it.

THE QUESTION THIS SEPARATES OUT
-------------------------------
Every arena asks two things at once, and until now published one number for
both: *did the player understand the science*, and *did the player emit JSON the
arena's output schema accepts*. An answer the schema rejects scores 0.0 by
`report.apply_unscoreable_policy`, and that 0.0 sits in the mean next to a 0.0
earned by reading a t-test wrongly. A reader cannot tell them apart.

They are not the same failure and they do not have the same remedy. The one is
capability; the other is formatting, and formatting is fixable by a wrapper
nobody has written yet.

WHAT MADE IT WORTH A METRIC
---------------------------
Measured on the University of Innsbruck desktop marathon, 2026-09-04, across
2,325 local-GGUF records: EVERY scoring error its models produced was
`output_schema_violation`, and none was a conceptual failure. 17 of 35 placed
`not_a_stat` — a legal value of `stats-extraction-v1`'s `verdict` enum — into
`kind`. The rest were nesting: a bare extraction object where
``{"extractions": [...]}`` was required, or a flattened `span`. On the SAME
prompt and the SAME schema the rate ran from 0.0 % (deterministic tools) to
25-33 % (`Qwen3.6-35B-A3B`, thinking disabled).

Reproduced independently against THIS repo's 10,389 committed records before any
of this was written: 59 `output_schema_violation`s, and the marathon's claim that
the Claude players sit at 0.0 % is not quite right — `claude-haiku-4-5-stats`
made exactly the `not_a_stat`-into-`kind` swap on task set v3. The mechanism is
therefore not a local-model artifact, which is the point of measuring it rather
than assuming it.

DERIVED, NOT STAMPED — AND WHY THAT IS THE SAFER HALF
-----------------------------------------------------
Everything here is computed from a record that already exists. Nothing in this
module writes, mutates or re-scores anything, so no published number can move
(`test_adherence_never_touches_score_primary`). That also gives the metric full
retrospective coverage: it reads the 5,386 records that predate `schema_version`
just as well as tomorrow's.

The runner additionally STAMPS the class onto new records (`record["adherence"]`,
schema v8), because a held-out record's error message is reduced to its type by
`holdout.sanitize_error` — so for those, derivation can recover the VERDICT but
never the CLASS. Both paths call `classify_message` on the same string, so the
stamped and the derived answer cannot drift apart.

WHAT THIS METRIC FIRES ON, AND HOW OFTEN IT IS WRONG
----------------------------------------------------
A metric published about other people's tools does not ship without its corpus
FIRING COUNT and how many of those firings were correct. Measured 2026-09-07
over the published (non-archive) corpus, all 22 arenas:

    firings (records counted as a VIOLATION)          13
      ... carrying a genuine jsonschema wording       13   (100%)
      ... uncorroborated (redacted / unclassified)     0
    suppressed as ORPHAN (would have fired before)     1
    classes: missing_required 9, extra_property 2, enum_swap 1, envelope_shape 1

    conformant records RE-VALIDATED against the arena's own schema   4,524
      of those whose stored output actually FAILS validation             0
    skipped: 738 (held-out, whose output is redacted to {} by design)

**CAN A CORRECT SUBMISSION SCORE NON-ADHERENT? On the testable side, no: 0 of
4,524.** The direction that can be tested completely is the false-NEGATIVE one,
because a conformant record still has its output on disk. The false-POSITIVE
direction cannot be re-validated at all — `runner._play_one` sets
`output_to_store = {}` on a rejection, so the offending output is GONE. That is a
LIMIT of this measurement, not a clean bill: what corroborates the 13 firings is
that each carries jsonschema's own wording, and that the marker is written by the
validator rather than by anything here.

The one firing that is a scientifically CORRECT answer in the wrong shape — a
model that identifies a non-statistic and writes `not_a_stat` into `kind` — is
counted deliberately. Adherence measures conformance, never correctness; keeping
that answer in the numerator is what the two axes exist to separate.

The gate test for a rule that fires on correct input — *if every surrounding word
and number were replaced with garbage, would the same evidence still justify the
output?* — passes here by construction. The evidence is a structural JSON-Schema
rejection, so replacing every word and number leaves the verdict identical. That
is precisely what distinguishes this from a content rule.

A SHARPER STATEMENT OF THE LIMIT, after mutation-testing the guard rather than
asserting it: a SWALLOWED violation is invisible in BOTH directions. It cannot be
re-validated as a firing (the output is gone) and it cannot be caught as a
non-firing either, because a record with an empty output is skipped. Measured on
power-reporting-v1: 6 violating records, 0 with a non-empty stored output. The
0-of-4,524 above therefore says the metric does not misread outputs it CAN see;
it does not say a rejection can never be lost.

`framework/tests/test_adherence.py::test_no_conformant_record_would_fail_its_own_schema`
is what keeps the testable half true as the corpus grows. Mutation-proved red on
a schema edit; mutation-proved INERT on a swallowed rejection, and its docstring
says so.

Re-measure before quoting these numbers; they are a fact about a moment.

THE MARKER MOVES — READ BOTH NAMES
----------------------------------
`runner._play_one` writes the reason to ``score.breakdown.error``.
`report.apply_unscoreable_policy` then RENAMES it to
``score.breakdown.unscoreable_reason`` (because `_is_errored` keys on `error`,
and leaving it would make the record simultaneously "scored 0.0" and "errored").
Every report-time consumer sees the renamed form. A classifier reading only
`error` would report a flawless 100 % adherence on precisely the surface this
metric is published on.
"""
from __future__ import annotations

import re
from collections import Counter

__all__ = [
    "CONFORMANT",
    "VIOLATION",
    "UNATTRIBUTED",
    "VIOLATION_MARKER",
    "POLICY_ZERO_FLAG",
    "CLASS_PRECEDENCE",
    "classify_message",
    "classify_record",
    "enum_values_in_schema",
    "outcome",
    "primary_class",
    "summarise",
    "violation_reason",
]

#: The player answered and the arena's schema accepted the answer. Says nothing
#: about whether the answer was RIGHT — that is `score.primary`'s job.
CONFORMANT = "conformant"
#: The schema rejected the output. Attributable to the player.
VIOLATION = "violation"
#: Anything else that stopped a score existing — a timeout, an HTTP 429, a CLI
#: quota refusal, an excluded task. NOT attributable in either direction, so it
#: leaves the denominator and is reported on its own.
#:
#: This bucket is deliberately generous, and that is a stated LIMIT of the
#: metric rather than a claim about it: `ValueError: no parseable JSON in CLI
#: output (len=0): ''` and `JSONDecodeError: Expecting value: line 1 column 1`
#: are indistinguishable, from the record alone, between "the model emitted prose
#: instead of JSON" (a real structural failure) and "the transport returned
#: nothing" (not the model's doing). Measured across this repo's 10,389 records
#: the ambiguous forms number 21 and every one of them came from a player whose
#: sibling records failed with an explicit quota or network error, so they are
#: transport. Counting them as violations would have published our network as
#: their sloppiness. They are counted HERE, in the open, instead.
UNATTRIBUTED = "unattributed"

#: The prefix `runner._play_one` puts in front of every jsonschema rejection.
VIOLATION_MARKER = "output_schema_violation"

#: Marks a 0.0 that `report.apply_unscoreable_policy` assigned rather than the
#: scorer computing it. Duplicated from `report.UNSCOREABLE_ZERO_FLAG` rather
#: than imported, so this module stays free of a circular import and the public
#: mirror can ship it standalone; `test_the_policy_zero_flag_matches_report_py`
#: fails if the two ever drift.
POLICY_ZERO_FLAG = "scored_zero_by_policy"

#: Most specific diagnosis first. When one rejection carries several messages,
#: the violation is counted once, under the earliest class present here.
#:
#: The ordering is by how narrowly the class names what the player did wrong, not
#: by severity: `enum_swap` says "you used the right vocabulary in the wrong
#: field", which is actionable; `unclassified` says nothing at all.
CLASS_PRECEDENCE = (
    "enum_swap",
    "envelope_shape",
    "enum_value",
    "missing_required",
    "extra_property",
    "type_error",
    "redacted",
    "unclassified",
)

# jsonschema's own message wordings. Pinned here rather than reconstructed,
# because they are the only structured thing a stored record retains: the runner
# joins `e.message` and drops `e.path` and `e.validator`, so the path a rejection
# occurred at is not recoverable from any record written before today.
_ENUM_RE = re.compile(r"^(.*?) is not one of \[", re.S)
_TYPE_RE = re.compile(r" is not of type '([a-z]+)'")
_REQUIRED_RE = re.compile(r" is a required property")
_ADDITIONAL_RE = re.compile(r"Additional properties are not allowed")

#: A rejection naming one of these as the EXPECTED type is a nesting error: the
#: schema wanted a container and got something else. That is exactly the
#: marathon's second failure family — a bare extraction object where
#: ``{"extractions": [...]}`` belongs, and a `span` flattened into an array.
_CONTAINER_TYPES = frozenset({"object", "array"})


def violation_reason(record: dict) -> str | None:
    """The schema-rejection message on `record`, under EITHER name it can carry.

    Returns None when the record carries no schema violation. See the module
    docstring: reading only `score.breakdown.error` reports 100 % adherence on
    every report-time surface, because `apply_unscoreable_policy` has already
    renamed the field by then.
    """
    if not isinstance(record, dict):
        return None
    breakdown = (record.get("score") or {}).get("breakdown")
    if not isinstance(breakdown, dict):
        return None
    for key in ("error", "unscoreable_reason"):
        value = breakdown.get(key)
        if isinstance(value, str) and value.startswith(VIOLATION_MARKER):
            return value
    return None


def outcome(record: dict) -> str:
    """One of CONFORMANT / VIOLATION / UNATTRIBUTED for a single run record."""
    if violation_reason(record) is not None:
        return VIOLATION
    score = (record or {}).get("score") or {}
    if "breakdown" in score and not isinstance(score["breakdown"], dict):
        # PRESENT but unreadable. Distinct from ABSENT, which is normal on a
        # clean scored record and stays conformant below. We cannot tell what
        # happened, and "cannot tell" must never resolve to the flattering
        # answer: this used to fall through to CONFORMANT and silently INFLATE
        # the published rate.
        return UNATTRIBUTED
    breakdown = score.get("breakdown")
    if isinstance(breakdown, dict):
        # KEY PRESENCE, not truthiness. `breakdown.get("error")` treated an
        # empty string and a 0 as "no error", so a rejection we could not parse
        # counted as a clean answer — the one direction that matters for a
        # published number.
        if "error" in breakdown or "unscoreable_reason" in breakdown:
            # Some other failure — transport, quota, adapter crash — or a marker
            # we cannot read. Either way, not adherence evidence.
            return UNATTRIBUTED
        if breakdown.get(POLICY_ZERO_FLAG):
            # A zero the POLICY assigned, carrying no reason at all. That is an
            # EXCLUDED record (primary was null, no error) which shared a task
            # with a scorer, so `apply_unscoreable_policy` zeroed it and had no
            # `error` to rename. There was nothing to grade, so there is nothing
            # to conform to — and reading it as `conformant` inflates the rate.
            #
            # NOT the same as a scorer-produced 0.0: a player that answered in
            # the right shape and got the science wrong conformed perfectly, and
            # collapsing those two would fold the metric back into the score it
            # exists to separate from. Found by GROK 4.6 in the 2026-09-05
            # cross-model round — the one finding both anthropic seats missed.
            return UNATTRIBUTED
    if score.get("primary") is None:
        # The arena excluded the task; the player was never really asked.
        return UNATTRIBUTED
    return CONFORMANT


def enum_values_in_schema(node) -> set[str]:
    """Every string an `enum` anywhere in a JSON Schema declares as legal.

    This is the arena's whole controlled VOCABULARY, collapsed across fields —
    which is the point: an `enum_swap` is a value that is legal *somewhere* in
    this schema being offered *here*. Walking only the sibling enums would miss
    the `verdict` -> `kind` confusion entirely, since those two live at different
    depths.
    """
    found: set[str] = set()
    if isinstance(node, dict):
        raw = node.get("enum")
        if isinstance(raw, list):
            found |= {v for v in raw if isinstance(v, str)}
        for value in node.values():
            found |= enum_values_in_schema(value)
    elif isinstance(node, list):
        for value in node:
            found |= enum_values_in_schema(value)
    return found


def classify_message(message: str, *, enum_values: set[str] | frozenset[str] = frozenset()) -> list[str]:
    """Sorted, de-duplicated classes present in one joined rejection message.

    `enum_values` is the arena's vocabulary from `enum_values_in_schema`. Without
    it every rejected enum value is reported as `enum_value` rather than
    `enum_swap` — the CONSERVATIVE direction, because `enum_swap` is the specific
    claim the rename investigation turns on and it must not be assumed into
    existence.
    """
    text = str(message or "")
    if text.startswith(VIOLATION_MARKER):
        text = text[len(VIOLATION_MARKER):].lstrip(": ").strip()
    if not text:
        # A held-out record: `holdout.sanitize_error` reduced the message to its
        # type. We know a violation happened and cannot know which kind. Saying
        # so is the honest answer; picking a class would be an invention.
        return ["redacted"]

    classes: set[str] = set()
    for part in text.split(";"):
        part = part.strip()
        if not part:
            continue
        enum_match = _ENUM_RE.match(part)
        if enum_match:
            offered = enum_match.group(1).strip().strip("'\"")
            classes.add("enum_swap" if offered in enum_values else "enum_value")
            continue
        type_match = _TYPE_RE.search(part)
        if type_match:
            classes.add("envelope_shape" if type_match.group(1) in _CONTAINER_TYPES
                        else "type_error")
            continue
        if _REQUIRED_RE.search(part):
            classes.add("missing_required")
            continue
        if _ADDITIONAL_RE.search(part):
            classes.add("extra_property")
            continue
        classes.add("unclassified")
    return sorted(classes)


def classify_record(record: dict, *,
                    enum_values: set[str] | frozenset[str] = frozenset()) -> list[str]:
    """Classes for a record, or [] when it carries no schema violation."""
    reason = violation_reason(record)
    if reason is None:
        return []
    return classify_message(reason, enum_values=enum_values)


def primary_class(classes: list[str]) -> str:
    """The single class a multi-class violation is counted under."""
    for candidate in CLASS_PRECEDENCE:
        if candidate in classes:
            return candidate
    return "unclassified"


def summarise(records, *,
              enum_values: set[str] | frozenset[str] = frozenset(),
              unattributable_tasks: set[str] | frozenset[str] = frozenset()) -> dict:
    """Structural adherence over a set of run records. PURE — mutates nothing.

    Keys:
      * ``structural_adherence`` — conformant / (conformant + violations), or
        **None** when nothing attributable was produced. None, never 1.0: a
        player whose every record timed out has produced no evidence, and
        publishing a perfect score for it would rank it above a model that
        answered 99 of 100 tasks and fumbled one brace. A green from an empty
        input is the portfolio's founding false-green class.
      * ``n_conformant`` / ``n_violations`` / ``n_attributable`` — the rate's parts.
      * ``n_unattributed`` — records outside the denominator, reported so a
        thin denominator is visible rather than implied.
      * ``violation_classes`` — each violation counted ONCE, under
        `primary_class`. Sums to ``n_violations``.

    ``unattributable_tasks`` is the ORPHAN set from
    ``report.apply_unscoreable_policy`` — tasks NO player could score. A rejection
    there is moved to ``unattributed``, because the policy has already ruled that
    a universal failure indicts the task rather than the field, and charging the
    same failure in the adherence while excusing it in the score is a
    contradiction off one record.

    CONFIRMED DEFECT, cross-model consult 2026-09-05 (Sonnet 5). Measured live
    before the fix: `local-gpt-oss-20b-q8-replication` published adherence 0.9412
    on `replication-target-lookup-v1` v1, whose single rejection sits on one of
    the four tasks no player could score. Honest value: 1.0000.

    The default is EMPTY, and deliberately so. A caller with no arena-wide view
    cannot justify dropping records, so the conservative direction is to
    attribute. Every real call site passes the set, pinned by
    `test_every_engine_passes_the_orphan_set`.
    """
    orphans = frozenset(unattributable_tasks or ())
    n_conformant = n_violations = n_unattributed = 0
    classes: Counter[str] = Counter()
    for record in records or ():
        verdict = outcome(record)
        if verdict == VIOLATION and isinstance(record, dict)                 and record.get("task_id") in orphans:
            verdict = UNATTRIBUTED
        if verdict == CONFORMANT:
            n_conformant += 1
        elif verdict == VIOLATION:
            n_violations += 1
            classes[primary_class(classify_record(record, enum_values=enum_values))] += 1
        else:
            n_unattributed += 1
    n_attributable = n_conformant + n_violations
    return {
        "structural_adherence": (n_conformant / n_attributable) if n_attributable else None,
        "n_conformant": n_conformant,
        "n_violations": n_violations,
        "n_attributable": n_attributable,
        "n_unattributed": n_unattributed,
        "violation_classes": dict(sorted(classes.items())),
    }
