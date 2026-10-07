"""The task TEXT and the ANSWER KEY must make the same claim about p.

The version-agnostic form of the v5 invariant. `test_task_set_v5.py` pins what
v5 measured and holds v3/v4 still as controls; THIS file asks the same question
of whatever task set `arena.yaml` currently marks `status: current`, so a v6
minted next year inherits the guard without anyone remembering to add it.

That distinction is the whole point. The defect this file exists for was
introduced by v4 -- a version whose own test file was careful, thorough, and
asked a different question. `test_task_set_v4.py` checks that gold `p_rel` agrees
with a recomputation of the gold; it never checks that `p_rel` reaches the
reader. Both can be true while the passage prints something else entirely.

FOUND 2026-08-28 by a three-provider cross-model consult round on the v4 scoring
diff (seat: Grok 4.6 / xAI via Cursor CLI, in 946 seconds). Three sessions of
careful measured work on this arena had not found it, and neither other seat did.

MEASURED, revealed split:

                                          v3     v4     v5
    printed relation != gold relation        0      7      0
    printed number   != gold number          8     11      0
    printed claim FALSE while key says ok   44      5      0

Mechanism and the rejected repair: `generator._is_v5_plus`.
Write-up: docs/reports/2026-08-28-v4-answer-key-p-relation-defect.md
"""
from pathlib import Path
import importlib.util
import re
import sys

import pytest
import yaml

ARENA_DIR = Path(__file__).resolve().parents[1]


def _load(mod_name, filename):
    spec = importlib.util.spec_from_file_location(mod_name, ARENA_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


generator = _load("_stats_extraction_v1_generator_pc", "generator.py")


def _current_version():
    """The task set the board is built from, read the way the framework reads it.

    `status: current` decides, never position in the list -- the same rule
    `framework/discovery.py::resolve_task_set_version` and `build-data.mjs`
    follow. Hardcoding the newest version here would silently stop guarding the
    live task set the moment a version was appended without being promoted.
    """
    manifest = yaml.safe_load((ARENA_DIR / "arena.yaml").read_text(encoding="utf-8"))
    current = [v["version"] for v in manifest["task_set_versions"]
               if v.get("status") == "current"]
    assert len(current) == 1, f"arena.yaml has {len(current)} current task sets: {current}"
    return current[0]


# A printed p-claim. Tolerates italics, HTML-escaped relations, comma decimals,
# and the tier-3 OCR corruption (`_ocr_corrupt` maps 0->O and 1->l).
_CLAIM = re.compile(r"\*?p\*?\s*(<|>|=|&lt;|&gt;)\s*([0-9OolI]*[.,][0-9OolI]+|[0-9OolI]+)")


def _rel(raw):
    return {"&lt;": "<", "&gt;": ">"}.get(raw, raw)


def _val(raw):
    t = raw.replace("O", "0").replace("l", "1").replace("I", "1").replace(",", ".")
    try:
        return float(t)
    except ValueError:
        return None


def _printed_claim(text, item):
    """The p-claim the reader actually sees for this item.

    Anchored on the item's own span, NOT on text order. Text order is wrong: the
    tier-4 dual template prints a second statistic that has no gold item, so a
    positional zip drifts and manufactures defects that are not there (it invented
    two on t-tier6-d1-1 while this file was being written).
    """
    end = (item.get("span") or {}).get("char_end")
    if end is None:
        return None
    m = _CLAIM.search(text, end, end + 45)
    if not m:
        return None
    return _rel(m.group(1)), _val(m.group(2)), m.group(0)


def _claim_is_true(rel, val, true_p):
    if val is None or true_p is None:
        return None
    if rel == "<":
        return true_p < val
    if rel == ">":
        return true_p > val
    dec = len(f"{val}".split(".")[1]) if "." in f"{val}" else 0
    return round(true_p, dec) == round(val, dec)


def _truthful_checkable_nhst(version, split="revealed"):
    """(task_id, text, item) for every item this invariant covers.

    Clears the gold cache first: task_id is NOT unique across task-set versions
    (v2..v5 share all 36 ids) and `_GROUND_TRUTH_CACHE` is keyed on task_id
    alone, so a cache left over from another version serves that version's gold.
    """
    generator._GROUND_TRUTH_CACHE.clear()
    envelopes = list(generator.generate(version, 0, split=split))
    assert envelopes, f"generator produced no {version}/{split} tasks — the probe is broken"
    out = []
    for env in envelopes:
        text = env["input"]["text"]
        gt = generator.ground_truth(env["task_id"])
        for item in gt.get("items", []):
            if item.get("kind") != "nhst_stat" or not item.get("checkable"):
                continue
            if not item.get("truthful", True):
                continue  # a planted deception prints a wrong p ON PURPOSE
            out.append((env["task_id"], text, item))
    return out


@pytest.mark.parametrize("split", ["revealed"])
def test_every_item_has_a_findable_printed_claim(split):
    """Two-sided control. Without it a zero from either test below is meaningless —
    a regex that matches nothing looks exactly like an arena with no defects. The
    first survey of this defect located 55 of 112 items and said so."""
    items = _truthful_checkable_nhst(_current_version(), split)
    assert len(items) >= 60, (
        f"only {len(items)} truthful checkable NHST items found on "
        f"{_current_version()}/{split} — the probe is looking at the wrong thing")
    unmatched = [t for t, text, it in items if _printed_claim(text, it) is None]
    assert not unmatched, f"no printed p-claim found for {len(unmatched)} item(s): {unmatched[:5]}"


def test_printed_p_claim_matches_the_gold_p_claim():
    """The direct assertion: the text and the answer key must say the same thing."""
    version = _current_version()
    bad = []
    for tid, text, item in _truthful_checkable_nhst(version):
        rel, val, raw = _printed_claim(text, item)
        f = item.get("fields") or {}
        g_rel, g_p = f.get("p_rel", "="), f.get("p")
        gp = float(g_p) if g_p is not None else None
        if rel != g_rel or (val is not None and gp is not None and abs(val - gp) > 1e-9):
            bad.append(f"{tid}: text says {raw!r}, gold grades 'p {g_rel} {g_p}'")
    assert not bad, (
        f"{version}: {len(bad)} item(s) where the printed p-claim is not the claim "
        "the answer key grades:\n  " + "\n  ".join(bad))


def test_no_item_with_a_false_printed_claim_is_graded_ok():
    """The scientific invariant, and the one that actually corrupts scores.

    An item whose printed claim is FALSE of its own recomputed p is a genuine
    reporting error. Grading it "ok" inverts the scoring: the tool that catches it
    is charged a false alarm, the tool that misses it is credited with a hit.
    """
    version = _current_version()
    bad = []
    for tid, text, item in _truthful_checkable_nhst(version):
        rel, val, raw = _printed_claim(text, item)
        true_p = item.get("p_recomputed")
        if _claim_is_true(rel, val, true_p) is False and item.get("expected_verdict") == "ok":
            decision = (val is not None and true_p is not None
                        and ((val <= 0.05 if rel == "<" else val < 0.05) != (true_p < 0.05)))
            bad.append(f"{tid}: text asserts {raw!r} but true p = {true_p}"
                       + ("  [DECISION ERROR]" if decision else ""))
    assert not bad, (
        f"{version}: {len(bad)} item(s) print a FALSE claim about p yet are graded "
        "'ok', so a tool that correctly flags them is scored a false alarm:\n  "
        + "\n  ".join(bad))


def test_the_invariant_would_still_catch_the_v4_defect():
    """CONTROL. Every assertion above is a zero, and a zero proves nothing unless
    the same code finds the defect where it is known to live. v4 is frozen and
    still carries it: 5 items print a claim that is false while the key says ok."""
    caught = []
    for tid, text, item in _truthful_checkable_nhst("v4"):
        rel, val, raw = _printed_claim(text, item)
        if (_claim_is_true(rel, val, item.get("p_recomputed")) is False
                and item.get("expected_verdict") == "ok"):
            caught.append(tid)
    assert len(caught) == 5, (
        f"the invariant finds {len(caught)} of the 5 known v4 defects — either it "
        "has stopped working, or frozen v4 has been rewritten")
