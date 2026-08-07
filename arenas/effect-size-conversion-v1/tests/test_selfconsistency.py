"""Self-consistency + scorer tests for effect-size-conversion-v1.

The arena's gold is COMPUTED: every task's gold must equal an INDEPENDENT Python
recomputation of the canonical conversion, every round-trip must close, and every value
must sit inside the valid range for its metric. These tests prove that across both
splits and several seeds, and cover the scorer's agreement/calibration blend and the
conversion_error category.
"""
import importlib.util
import math
import sys
from collections import Counter
from pathlib import Path

ARENA_DIR = Path(__file__).resolve().parents[1]


def _load(mod_name, filename):
    spec = importlib.util.spec_from_file_location(mod_name, ARENA_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


generator = _load("_esconv_generator", "generator.py")
scorer = _load("_esconv_scorer", "scorer.py")

PI = math.pi
SQRT3 = math.sqrt(3.0)


# --------------------------------------------------------------------------- #
# An INDEPENDENT recomputation of the canonical conversions (intentionally NOT a
# call into generator.convert — it re-derives each identity from scratch so a typo
# in the generator would be caught).
# --------------------------------------------------------------------------- #
def _indep_h(context):
    if context and "n1" in context and "n2" in context:
        n1, n2 = float(context["n1"]), float(context["n2"])
        m = n1 + n2 - 2.0
        return m / n1 + m / n2
    return 4.0


def _indep_convert(value, frm, to, context):
    if (frm, to) == ("d", "r"):
        return value / (value * value + _indep_h(context)) ** 0.5
    if (frm, to) == ("r", "d"):
        return (_indep_h(context) ** 0.5) * value / (1.0 - value * value) ** 0.5
    if (frm, to) == ("d", "OR"):
        return math.e ** (value * PI / SQRT3)
    if (frm, to) == ("OR", "d"):
        return math.log(value) * SQRT3 / PI
    if (frm, to) == ("eta2", "f"):
        return (value / (1.0 - value)) ** 0.5
    if (frm, to) == ("f", "eta2"):
        return value ** 2 / (1.0 + value ** 2)
    if (frm, to) == ("d", "f"):
        return value / 2.0
    if (frm, to) == ("f", "d"):
        return 2.0 * value
    raise AssertionError(f"unhandled {frm}->{to}")


# --------------------------------------------------------------------------- #
# Generator structure / determinism.
# --------------------------------------------------------------------------- #
def test_generate_is_deterministic():
    a = list(generator.generate("v1", seed=3))
    b = list(generator.generate("v1", seed=3))
    assert [t["task_id"] for t in a] == [t["task_id"] for t in b]
    assert [t["input"] for t in a] == [t["input"] for t in b]


def test_envelopes_have_required_fields_and_tags():
    tasks = list(generator.generate("v1", seed=0, split="revealed"))
    assert tasks
    for t in tasks:
        assert {"task_id", "arena_id", "task_set_version", "difficulty", "input"} <= t.keys()
        assert t["arena_id"] == "effect-size-conversion-v1"
        assert t["split"] == "revealed"
        assert t["visibility"] == "public"
        inp = t["input"]
        assert set(inp.keys()) <= {"value", "from", "to", "context"}
        assert {"value", "from", "to"} <= inp.keys()
        assert inp["from"] in generator.METRICS
        assert inp["to"] in generator.METRICS
        assert inp["from"] != inp["to"]


def test_private_split_is_held_out():
    tasks = list(generator.generate("v1", seed=9, split="private"))
    assert all(t["split"] == "private" and t["visibility"] == "held_out" for t in tasks)


def test_all_six_tiers_present():
    tiers = {t["difficulty"]["tier"] for t in generator.generate("v1", seed=0)}
    assert tiers == {1, 2, 3, 4, 5, 6}


def test_ground_truth_missing_task_raises_keyerror():
    import pytest
    list(generator.generate("v1", seed=0))
    with pytest.raises(KeyError):
        generator.ground_truth("esconv-tX-9-sDOES_NOT_EXIST")


def test_splits_share_tier_matrix_but_differ_in_content():
    rev = list(generator.generate("v1", seed=0, split="revealed"))
    priv = list(generator.generate("v1", seed=12345, split="private"))

    def cells(tasks):
        return Counter((t["difficulty"]["tier"], t["input"]["from"], t["input"]["to"])
                       for t in tasks)

    # Identical (tier, from, to) matrix on both splits: conversion kinds are
    # deterministic functions of (tier, idx).
    assert cells(rev) == cells(priv)
    # ...but the concrete values differ.
    assert [t["input"]["value"] for t in rev] != [t["input"]["value"] for t in priv]


def test_every_conversion_kind_appears():
    kinds = set()
    for t in generator.generate("v1", seed=0):
        kinds.add((t["input"]["from"], t["input"]["to"], "context" in t["input"]))
    expected = {(c[0], c[1], c[2]) for c in generator.CONVERSIONS}
    assert kinds == expected


# --------------------------------------------------------------------------- #
# THE cross-validation invariant: computed gold equals an INDEPENDENT recomputation
# of the canonical conversion, for EVERY task in BOTH splits and several seeds.
# --------------------------------------------------------------------------- #
def test_gold_matches_independent_recomputation():
    for split, seed in (("revealed", 0), ("private", 777), ("private", 5), ("private", 42)):
        for t in generator.generate("v1", seed=seed, split=split):
            gt = generator.ground_truth(t["task_id"])
            inp = t["input"]
            expect = _indep_convert(inp["value"], inp["from"], inp["to"], inp.get("context"))
            assert abs(gt["converted"] - expect) < 1e-6, (
                f"{t['task_id']}: gold={gt['converted']} indep={expect} "
                f"({inp['from']}->{inp['to']} value={inp['value']})"
            )


def test_round_trips_close():
    """Converting out then back must return the original value (to rounding)."""
    inverse = {
        ("d", "r"): ("r", "d"),
        ("d", "OR"): ("OR", "d"),
        ("eta2", "f"): ("f", "eta2"),
        ("d", "f"): ("f", "d"),
    }
    for split, seed in (("revealed", 0), ("private", 5)):
        for t in generator.generate("v1", seed=seed, split=split):
            inp = t["input"]
            key = (inp["from"], inp["to"])
            if key not in inverse:
                continue
            forward = generator.ground_truth(t["task_id"])["converted"]
            back = generator.convert(forward, key[1], key[0], inp.get("context"))
            assert abs(back - inp["value"]) < 1e-4, (
                f"{t['task_id']}: round-trip {key} did not close: "
                f"{inp['value']} -> {forward} -> {back}"
            )


def test_values_are_in_valid_ranges():
    for split, seed in (("revealed", 0), ("private", 5), ("private", 99)):
        for t in generator.generate("v1", seed=seed, split=split):
            inp = t["input"]
            v = inp["value"]
            m = inp["from"]
            gold = generator.ground_truth(t["task_id"])["converted"]
            if m == "r":
                assert -0.95 < v < 0.95
            if m == "eta2":
                assert 0.0 < v < 0.85
            if m == "OR":
                assert v > 0.0
            # The target value must also be valid for its metric.
            tm = inp["to"]
            if tm == "r":
                assert -1.0 < gold < 1.0
            if tm == "eta2":
                assert 0.0 < gold < 1.0
            if tm == "OR":
                assert gold > 0.0


def test_context_only_on_needs_context_tier():
    for t in generator.generate("v1", seed=0):
        has_ctx = "context" in t["input"]
        assert has_ctx == t["difficulty"]["needs_context"]
        if has_ctx:
            assert {"n1", "n2"} == set(t["input"]["context"].keys())
            assert t["input"]["context"]["n1"] != t["input"]["context"]["n2"]


def test_group_sizes_actually_change_d_to_r():
    """The d<->r-with-context tasks must use group sizes that move the answer away
    from the h=4 default (otherwise the context would be vacuous)."""
    seen = 0
    for t in generator.generate("v1", seed=0):
        inp = t["input"]
        if "context" in inp and (inp["from"], inp["to"]) == ("d", "r"):
            seen += 1
            with_ctx = generator.ground_truth(t["task_id"])["converted"]
            without_ctx = generator.convert(inp["value"], "d", "r", None)
            assert abs(with_ctx - without_ctx) > 1e-4
    assert seen >= 1


# --------------------------------------------------------------------------- #
# Scorer.
# --------------------------------------------------------------------------- #
def _gt(tier, frm=None, to=None):
    for t in generator.generate("v1", seed=0):
        g = generator.ground_truth(t["task_id"])
        if t["difficulty"]["tier"] == tier and (frm is None or g["from"] == frm) and \
           (to is None or g["to"] == to):
            return g
    raise AssertionError(f"no task with tier={tier} from={frm} to={to}")


def test_perfect_confident_player_scores_one():
    g = _gt(1, "d", "r")
    out = {"converted": g["converted"], "confidence": 1.0}
    s = scorer.score(out, g)
    assert s["primary"] == 1.0
    assert s["breakdown"]["within_tol"] is True
    assert s["findings"] == []


def test_within_tolerance_passes():
    g = _gt(1, "d", "r")
    out = {"converted": g["converted"] + 0.005, "confidence": 1.0}  # inside 0.01 floor
    s = scorer.score(out, g)
    assert s["breakdown"]["within_tol"] is True
    assert s["primary"] == 1.0


def test_conversion_error_finding_and_decay():
    g = _gt(1, "d", "r")
    out = {"converted": g["converted"] + 0.5, "confidence": 0.9}  # way off
    s = scorer.score(out, g)
    assert s["breakdown"]["within_tol"] is False
    assert any(f["category"] == "conversion_error" for f in s["findings"])
    assert s["breakdown"]["agreement"] < 1.0


def test_missing_converted_scores_low():
    g = _gt(1, "d", "r")
    s = scorer.score({"confidence": 0.5}, g)
    assert s["breakdown"]["within_tol"] is False
    assert s["breakdown"]["agreement"] == 0.0


def test_calibration_penalises_overconfident_wrong():
    g = _gt(1, "d", "r")
    confident_wrong = scorer.score({"converted": g["converted"] + 5.0, "confidence": 1.0}, g)
    humble_wrong = scorer.score({"converted": g["converted"] + 5.0, "confidence": 0.0}, g)
    assert humble_wrong["primary"] > confident_wrong["primary"]


def test_relative_tolerance_for_large_targets():
    """For a large OR target, the relative band (1% of gold) should exceed the abs floor."""
    g = _gt(5, "OR", "d")  # OR->d target is a d; pick a large-magnitude task instead
    # Use a constructed gold to exercise the relative band directly.
    big = {"converted": 10.0, "from": "d", "to": "OR", "value": 1.0, "context": None}
    s_ok = scorer.score({"converted": 10.05, "confidence": 1.0}, big)   # within 1% (0.1)
    s_bad = scorer.score({"converted": 12.0, "confidence": 1.0}, big)   # outside
    assert s_ok["breakdown"]["within_tol"] is True
    assert s_bad["breakdown"]["within_tol"] is False


def test_primary_in_unit_interval_for_random_outputs():
    g = _gt(3)
    for cv in (g["converted"], g["converted"] + 0.3, g["converted"] - 2.0):
        for c in (0.0, 0.3, 0.7, 1.0):
            s = scorer.score({"converted": cv, "confidence": c}, g)
            assert 0.0 <= s["primary"] <= 1.0
