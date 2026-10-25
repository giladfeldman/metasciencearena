import json
from pathlib import Path
from jsonschema import Draft202012Validator
from framework.paths import schema_path

SCHEMA = json.loads(schema_path("run_record.schema.json").read_text(encoding="utf-8"))
V = Draft202012Validator(SCHEMA)

BASE = {
    "run_id": "r1", "arena_id": "a", "task_set_version": "v1", "task_id": "t1",
    "player_id": "p", "player_version": "1.0", "player_type": "tool",
    "input_hash": "h", "output": {}, "score": {"primary": 1.0},
    "timestamp_utc": "2026-06-07T00:00:00Z",
}

def test_legacy_record_without_provenance_validates():
    assert list(V.iter_errors(BASE)) == []

def test_record_with_provenance_validates():
    rec = dict(BASE, provenance={
        "tested_at_utc": "2026-06-07T00:00:00Z", "host": "win11-local",
        "adapter_class": "RCliAdapter", "command": "Rscript players/adapters/metacheck.R",
        "tool_version_detail": "metacheck 0.1.0 @abc1234", "trials": 1, "seed": 0, "split": "revealed",
    })
    assert list(V.iter_errors(rec)) == []

def test_provenance_rejects_unknown_field():
    rec = dict(BASE, provenance={"api_key": "leak"})
    assert list(V.iter_errors(rec)) != []


# --- shape-change / version-bump guard ---------------------------------------
#
# WHY THIS EXISTS (2026-08-20). The schema's own `schema_version` description says
# it is "bumped whenever fields are added", and on 2026-08-20 nine properties were
# added to `response_meta` -- a CLOSED object, so a real validation-contract change
# -- with the producer constant left at 3. Nothing caught it: no test asserted a
# version, and grep found NO consumer of `schema_version` anywhere in the repo or
# the website. A write-only field cannot fail loudly on its own, so 58 run records
# were written claiming a shape they did not have.
#
# The fix for the SYMPTOM was to bump and re-stamp. This is the fix for the CLASS:
# a fingerprint over the schema's STRUCTURE, pinned to the version it belongs to.
# Add, remove or retype a property anywhere in the tree and this test fails until
# someone bumps RECORD_SCHEMA_VERSION and re-pins the digest deliberately.
#
# Descriptions are deliberately EXCLUDED. Prose gets edited constantly and a
# fingerprint that churns on wording is one people learn to update without reading,
# which would defeat the guard exactly when it matters.

import hashlib

from framework.runner import RECORD_SCHEMA_VERSION

_PROSE_KEYS = {"description", "title", "$comment", "examples", "default"}


def _shape(node):
    """Canonical structure of a schema node, with all prose stripped."""
    if isinstance(node, dict):
        return {k: _shape(v) for k, v in sorted(node.items()) if k not in _PROSE_KEYS}
    if isinstance(node, list):
        return [_shape(v) for v in node]
    return node


def schema_shape_fingerprint(schema: dict) -> str:
    canonical = json.dumps(_shape(schema), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


#: version -> fingerprint of the record SHAPE that version promises.
#: Bump RECORD_SCHEMA_VERSION and add a row when you change the shape on purpose.
SHAPE_FINGERPRINT_BY_VERSION = {
    4: "2b8b0cfa0f910e461b75b5ddbc3ab322aa26d7678894aaf4896dd9b13e7e3218",
    5: "8fb9efe3f6892a5e6d36e6da6292345d2fea4f326e3bafde52d3ee94fab60bfc",
    6: "8effc312a5c074a9dfd6482af881bf28669d3e8b9d4d3d44a9234cfa061418d2",
    # v7 (2026-08-25) added the top-level `instrument` block and the
    # schema_version>=7 conditional that requires it. Deliberately NOT a
    # top-level required[] entry -- see the schema_version description.
    7: "f34b62e8d0762cd16745d13cb61e2a918c12b7cce20a495f07a8f26d4a7eb53a",
    # v8 (2026-09-05) added the top-level `adherence` block: STRUCTURAL
    # ADHERENCE, i.e. did the arena's output schema ACCEPT this answer, kept
    # apart from whether the answer was right. Derivable from
    # score.breakdown.error for a public record; stamped because
    # holdout.sanitize_error destroys the diagnosis on a held-out one. Both its
    # fields are CLOSED enums, so it can cross the held-out boundary that the
    # free-text message cannot.
    8: "208db872efeae00bf46fdb9e08b661a2346496f6da341a9f1e1d28bb8eb68661",
    # v9 (2026-09-11) added `provenance.request_max_tokens`: the output ceiling
    # WE sent, read off the finished request body so a player's
    # `openai_extra_body.max_tokens` override is captured rather than the
    # adapter default a registry re-read would report. `provenance` is
    # additionalProperties:false, so adding one key IS a shape change --
    # which is the whole reason this fingerprint exists. `null` for every
    # adapter that sends no ceiling; it deliberately survives held-out
    # redaction, being a number we chose rather than one the document implies.
    9: "0d951f2d2e42266d02f6082288f3df89dc259f22d50de0c8de59e191a9522c95",
    # v10 (2026-09-17) added the top-level `player_identity` block: WHICH
    # INSTRUMENT produced the answer, as distinct from the sibling `instrument`
    # block, which is about the SCORER. Until v10 nothing on a record said which
    # instrument produced it -- `player_version` labels WEIGHTS, so the same GGUF
    # with thinking on and thinking off declared one identical string while
    # differing by +0.24 to +0.28 mean score. Carries the DECLARED instrument
    # from players/registry.yaml beside the inference settings actually sent and
    # a digest of them, so the declaration can be audited rather than believed.
    # Like `instrument` and `adherence` it is NOT in top-level required[]; the
    # requirement is a schema_version>=10 conditional, because every one of the
    # published records predates it and its ABSENCE means "predates the field".
    10: "a9cb6321e0683dd0a4daba88192953e41d75de250cb4d1b8573261548eecdffd",
    # v11 (2026-10-08) added the top-level `rejected_output`: the answer the
    # arena's output schema refused, kept so the violation can be diagnosed
    # (conductor F-0082). Optional; held-out redaction drops it.
    11: "bc5fed60870a018776e567a27aaa2294223a6bc674713ba24d7de5cf42098ff0",
    # v12 (2026-10-14) added top-level env_sha (content-addressed run
    # environment, required non-null by a schema_version>=12 conditional and by
    # RunRecordWriter) and call_config_sha256 (content-addressed call settings,
    # null for players that make no model request). CC8 runner image.
    12: "5c55eab3f4afaa57ad6932cb86155faeb0cda3523dc76236227af6bdeb833d98",
}


def test_schema_shape_change_requires_a_version_bump():
    assert RECORD_SCHEMA_VERSION in SHAPE_FINGERPRINT_BY_VERSION, (
        f"RECORD_SCHEMA_VERSION is {RECORD_SCHEMA_VERSION} but no shape fingerprint "
        "is pinned for it. Add one to SHAPE_FINGERPRINT_BY_VERSION."
    )
    expected = SHAPE_FINGERPRINT_BY_VERSION[RECORD_SCHEMA_VERSION]
    actual = schema_shape_fingerprint(SCHEMA)
    assert actual == expected, (
        f"run_record.schema.json's SHAPE changed but RECORD_SCHEMA_VERSION is still "
        f"{RECORD_SCHEMA_VERSION}.\n"
        f"  pinned for v{RECORD_SCHEMA_VERSION}: {expected}\n"
        f"  actual now:                {actual}\n"
        "A record's schema_version is how a reader tells 'this run predates the "
        "field' from 'not applicable to this run'. Leaving it behind makes every "
        "gap look the same. Bump RECORD_SCHEMA_VERSION, document the new version in "
        "the schema's schema_version description, and pin the new fingerprint here."
    )


def test_the_fingerprint_ignores_prose_but_not_structure():
    """The guard must be sensitive to shape and deaf to wording, or it is useless.

    A fingerprint that churns on every description edit gets updated reflexively;
    one that misses an added property does not guard anything.
    """
    import copy

    reworded = copy.deepcopy(SCHEMA)
    reworded["properties"]["score"]["description"] = "totally different prose"
    assert schema_shape_fingerprint(reworded) == schema_shape_fingerprint(SCHEMA), \
        "a description edit moved the fingerprint; it would churn and be ignored"

    added = copy.deepcopy(SCHEMA)
    added["properties"]["response_meta"]["properties"]["sneaky_new_field"] = {"type": "string"}
    assert schema_shape_fingerprint(added) != schema_shape_fingerprint(SCHEMA), \
        "an ADDED property did not move the fingerprint; the guard is blind"

    retyped = copy.deepcopy(SCHEMA)
    retyped["properties"]["latency_ms"] = {"type": "string"}
    assert schema_shape_fingerprint(retyped) != schema_shape_fingerprint(SCHEMA), \
        "a RETYPED property did not move the fingerprint; the guard is blind"


# --- v7 instrument: the ratchet, expressed in the schema ----------------------

def _v7_record():
    return {
        "run_id": "r-7", "arena_id": "fake-arena", "task_set_version": "v1", "task_id": "t1",
        "player_id": "p", "player_version": "1", "player_type": "tool",
        "input_hash": "0" * 64, "output": {"label": "ok"},
        "score": {"primary": 1.0, "breakdown": {}},
        "timestamp_utc": "2026-08-25T12:00:00Z",
        "schema_version": 7,
        "instrument": {"formula_id": "fake-arena-v1", "scorer_sha256": "a" * 64},
    }


def test_v7_record_without_an_instrument_block_is_invalid():
    """The whole point of the version bump: new records must say what scored them.

    Watched FAIL first by deleting the allOf conditional from the schema, which
    made this record validate cleanly -- i.e. the bump would have been decorative.
    """
    rec = _v7_record()
    rec.pop("instrument")
    errors = list(V.iter_errors(rec))
    assert errors, (
        "a schema_version 7 record with no instrument block validated. The "
        "conditional in the schema's allOf is missing or wrong, and the version "
        "bump records an intention nothing enforces."
    )


def test_records_predating_v7_stay_valid_without_an_instrument_block():
    """The ratchet must not condemn history.

    5,386 of 8,748 tracked records carry no schema_version at all and 816 carry v6.
    Requiring `instrument` of them would assert retroactively that most of the
    corpus was always malformed, and would make RunRecordWriter refuse to rewrite
    any historical record (measured 2026-08-25: 72/72 valid -> 0/72 under a
    top-level required[] entry). Both shapes below must remain valid.
    """
    v6 = _v7_record()
    v6["schema_version"] = 6
    v6.pop("instrument")
    assert list(V.iter_errors(v6)) == [], "a v6 record was invalidated by the v7 rule"

    gen0 = _v7_record()
    gen0.pop("schema_version")
    gen0.pop("instrument")
    assert list(V.iter_errors(gen0)) == [], "a pre-schema_version record was invalidated"


def test_instrument_rejects_a_digest_that_is_not_a_digest():
    """A field that accepts anything is a field that proves nothing."""
    rec = _v7_record()
    rec["instrument"]["scorer_sha256"] = "not-a-sha"
    assert list(V.iter_errors(rec)), "instrument.scorer_sha256 accepted a non-digest"
