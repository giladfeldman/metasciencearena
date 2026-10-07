"""Task set v3 pins the three defects it was minted to fix (2026-08-21).

Each test states the MEASURED v2 value it is moving away from, so a future reader
can tell whether the property is holding or has silently drifted back. Every one
was watched fail against v2 before v3 existed.

Full write-up: docs/reports/2026-08-21-stats-arena-coverage-gaps.md
"""
from pathlib import Path
import importlib.util
import json

import pytest
import re
import sys

ARENA_DIR = Path(__file__).resolve().parents[1]


def _load(mod_name, filename):
    spec = importlib.util.spec_from_file_location(mod_name, ARENA_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


generator = _load("_stats_extraction_v1_generator_v3", "generator.py")

_STAT = re.compile(r"\b(t|F|r|chi2|z|U|W|H|Q|BF10)\s*[(=]")


def _tasks(version="v3", seed=0, split="revealed"):
    return list(generator.generate(version, seed, split))


# --- defect 1: truthful NHST items were real statcheck decision errors -------

def test_no_truthful_nhst_item_crosses_alpha():
    """v2 measured 37 of 84. The player prompt orders players to recompute p and
    flag exactly this, so every one of those items punished a correct answer."""
    crossing = []
    for t in _tasks():
        for it in generator.ground_truth(t["task_id"])["items"]:
            if it["kind"] != "nhst_stat" or not it.get("truthful", True):
                continue
            f = it["fields"]
            tp = generator._two_sided_p(f["test_type"], f["value"], f["df1"], f["df2"])
            if tp is None:
                continue
            if (f["p"] < 0.05) != (tp < 0.05):
                crossing.append((t["task_id"], f["test_type"], f["value"],
                                 f["df1"], f["df2"], f["p"], round(tp, 6)))
    assert not crossing, f"{len(crossing)} truthful NHST items cross alpha: {crossing[:5]}"


def test_the_clean_class_is_not_empty():
    """Guards the opposite failure: 'no crossings' is trivially true if there are
    no truthful checkable NHST items left at all."""
    n_ok = sum(1 for t in _tasks()
               for it in generator.ground_truth(t["task_id"])["items"]
               if it["kind"] == "nhst_stat" and it.get("truthful", True)
               and it.get("expected_verdict") == "ok")
    assert n_ok >= 40, f"only {n_ok} genuinely coherent truthful NHST items"


# --- defect 2: tier 6 advertised deception and had none ----------------------

def test_tier6_actually_contains_deception():
    """v2: `sub_tier = (i % 5) + 1` over FOUR titles never reached 5, so the
    branch was dead and tier 6 held 66 of 144 gold items with zero deception."""
    n = sum(1 for t in _tasks() if t["difficulty"]["tier"] == 6
            for it in generator.ground_truth(t["task_id"])["items"]
            if not it.get("truthful", True))
    assert n > 0, "tier 6 still contains no deception - is sub_tier == 5 reachable?"


def test_declared_deception_rate_matches_what_is_emitted():
    """The stamp must be COMPUTED from the items, never asserted beside them.
    Every published per-player report is keyed off `difficulty_axes`."""
    bad = []
    for t in _tasks():
        declared = t["difficulty"]["deception_rate"]
        items = generator.ground_truth(t["task_id"])["items"]
        realised = sum(1 for it in items if not it.get("truthful", True))
        if declared > 0 and realised == 0:
            bad.append((t["task_id"], declared, realised))
        if declared == 0 and realised > 0:
            bad.append((t["task_id"], declared, realised))
    assert not bad, f"declared deception_rate disagrees with the items: {bad[:5]}"


# --- defect 3: no effect size ever appeared with its test statistic ----------

def test_some_effect_sizes_are_reported_with_their_statistic():
    """v2 measured 0 of 51 - and probed against live effectcheck 0.7.6, all six
    bare forms return zero rows while all three paired forms parse and PASS. The
    arena was scoring a verification tool only on what it cannot answer."""
    paired = 0
    for t in _tasks():
        txt = t["input"]["text"]
        for it in generator.ground_truth(t["task_id"])["items"]:
            if it["kind"] != "effect_size":
                continue
            cs, ce = it["span"]["char_start"], it["span"]["char_end"]
            starts = [m.end() for m in generator._SENTENCE_BREAK.finditer(txt[:cs])]
            st = starts[-1] if starts else 0
            m = generator._SENTENCE_BREAK.search(txt[ce:])
            en = ce + m.end() if m else len(txt)
            if _STAT.search(txt[st:en]):
                paired += 1
    assert paired >= 10, f"only {paired} effect sizes sit beside a test statistic"


def test_bare_effect_sizes_still_exist():
    """Do not over-correct. A bare effect size is a legitimate EXTRACTION target
    even though it is not a verification target, and dropping the form entirely
    would trade one blind spot for another."""
    bare = 0
    for t in _tasks():
        txt = t["input"]["text"]
        for it in generator.ground_truth(t["task_id"])["items"]:
            if it["kind"] != "effect_size":
                continue
            cs, ce = it["span"]["char_start"], it["span"]["char_end"]
            starts = [m.end() for m in generator._SENTENCE_BREAK.finditer(txt[:cs])]
            st = starts[-1] if starts else 0
            m = generator._SENTENCE_BREAK.search(txt[ce:])
            en = ce + m.end() if m else len(txt)
            if not _STAT.search(txt[st:en]):
                bare += 1
    assert bare >= 10, f"only {bare} bare effect sizes left - over-corrected"


# --- the append-only invariant ----------------------------------------------

@pytest.mark.skipif(
    not all((ARENA_DIR / "task_sets" / v / "_ground_truth.json").is_file()
            for v in ("v1", "v2")),
    reason="committed _ground_truth.json is NEVER_PUBLISH — a public-mirror "
           "checkout cannot run the byte-for-byte comparison")
def test_v1_and_v2_task_text_is_untouched():
    """A shared generator is APPEND-ONLY. v1 and v2 gold are committed artifacts
    whose published scores were computed against exactly that text; pinning
    labels for every version once left only 7 of 36 v1 tasks matching their gold.
    This is the test that would have caught it."""
    for version in ("v1", "v2"):
        committed = json.loads(
            (ARENA_DIR / "task_sets" / version / "_ground_truth.json").read_text(encoding="utf-8"))
        changed = [t["task_id"] for t in _tasks(version)
                   if committed[t["task_id"]]["envelope"]["input"]["text"] != t["input"]["text"]]
        assert not changed, f"{version}: {len(changed)} task(s) drifted from committed gold"


def test_every_v3_gold_item_carries_a_verification_label():
    for t in _tasks():
        for it in generator.ground_truth(t["task_id"])["items"]:
            assert "expected_verdict" in it and "checkable" in it, (t["task_id"], it)


def test_no_gold_item_claims_a_confidence_interval_the_text_never_prints():
    """v2 carried 5 ghost CIs: in the answer key, absent from the input. Nothing
    read them, which is what made the class invisible."""
    ghosts = []
    for t in _tasks():
        txt = t["input"]["text"]
        for it in generator.ground_truth(t["task_id"])["items"]:
            f = it.get("fields", {})
            if f.get("ci_low") is None:
                continue
            if not generator._ci_is_printed(txt, it):
                ghosts.append((t["task_id"], f.get("value")))
    assert not ghosts, f"gold claims {len(ghosts)} CI(s) the text never prints: {ghosts[:5]}"


# --- the prompt and the schema must agree -----------------------------------

def test_the_output_schema_accepts_what_the_prompt_asks_for():
    """A contract cannot call a field deprecated in one document and require it
    in another.

    Introduced and caught on 2026-08-22. The prompt was rewritten to say
    `flagged_suspicious` is "deprecated; set it to (verdict != 'ok') if you emit
    it at all", while `output.schema.json` still listed it in `required`. Every
    model that took the prompt at its word failed validation: **36 of 36 haiku
    records and 9 opus records**, all `output_schema_violation`, all reading as
    total incapability. Sonnet happened to emit the field anyway and scored fine,
    which is why it looked like a haiku problem.
    """
    import json as _json
    import jsonschema
    schema = _json.loads((ARENA_DIR / "schemas" / "output.schema.json").read_text(encoding="utf-8"))
    base = {"span": {"text": "x", "char_start": 0, "char_end": 1},
            "kind": "nhst_stat", "fields": {"value": 1.0}, "confidence": 1.0}

    # Draft202012Validator(...).validate, NOT jsonschema.validate: the module-level
    # helper runs `check_schema` on every call, which resolves the draft
    # META-schema and — measured 2026-09-05, 436 DeprecationWarnings from this one
    # test — fetches it OVER THE NETWORK. That made the test nondeterministic: it
    # passes alone and failed a full-suite run with
    # `_WrappedReferencingError: Unresolvable: meta/core`, i.e. a red gate caused
    # by a remote host rather than by the code under test. This is the same
    # constructor `framework/runner.py` and `framework/discovery.py` already use,
    # so it is also what actually validates player output in production.
    _validator = jsonschema.Draft202012Validator(schema)

    def _validate(extraction):
        _validator.validate({"extractions": [extraction], "player_strategy_notes": ""})

    # what the prompt now asks for
    _validate({**base, "verdict": "ok"})
    # the legacy shape, still accepted for one version
    _validate({**base, "flagged_suspicious": False})
    _validate({**base, "verdict": "ok", "flagged_suspicious": False})

    # but NEITHER must be rejected: with both absent the scorer's
    # `_player_verdict` falls through to "ok", so a missing answer would be
    # silently read as "this result is fine".
    try:
        _validate(base)
    except jsonschema.ValidationError:
        pass
    else:
        raise AssertionError(
            "an extraction with no verdict AND no flagged_suspicious validated - "
            "the scorer would read it as 'ok' rather than as a missing answer")
