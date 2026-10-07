"""The printed p-relation must not be worth anything to a player that does no work.

WHY THIS FILE EXISTS, AND THE CLAIM IT CORRECTS.

From task set v4 onward the generator prints an APA inequality (`p < .001`,
`p > .999`, and from v5 also `p < .05` / `p > .05`) whenever the true p supports
it. `_apply_apa_p` is called ONLY on truthful items -- `_build_tier5_task` never
calls it -- so every deception falls through to `_instantiate_template`'s
`derived.setdefault("p_rel", "=")` and prints `p = <value>`.

The consequence LOOKS alarming and was written up that way on 2026-08-29:

    v4 revealed: 50 items print an inequality -- 50 truthful, 0 deceptions
    v5 revealed: 47 items print an inequality -- 47 truthful, 0 deceptions

"printed inequality => the item is truthful" is therefore 100% precise on ~45% of
the NHST items, with no recomputation. That was reported as "a shortcut past
exactly the competence this arena measures".

**MEASURED, that conclusion is wrong, and this file is the measurement.** The cue
is perfectly precise and WORTH NOTHING. Scored over all 36 revealed v5 tasks with
a perfect extractor, every one of the 16 constant strategies the cue's partition
admits:

    best cue-USING surface strategy (inequality -> ok, else -> ok)   0.8077
    best cue-FREE  constant strategy (always ok)                     0.8077
    gain from knowing the cue                                        +0.0000

The structural reason, which is why this is an invariant and not a coincidence:
the cue identifies items where `ok` is the RIGHT answer, and a player maximising
without recomputation already answers `ok` everywhere because the gold is
`ok`-dominated. Knowing where `ok` is safe adds nothing to a strategy that says
`ok` regardless. The cue cannot point at the deceptions -- it is constant across
the bucket that contains all of them.

WHAT THE CUE DOES BUY, stated honestly, is narrower: it concentrates the
deceptions. The prior rises from 21/104 (20.2%) over all NHST items to 21/57
(36.8%) inside the `=` bucket, a factor of 1.82. That is a construct-validity
dilution -- the arena can be partly navigated by a surface feature -- not a score
defect, and it does not let a weaker tool outrank a stronger one.

THE INVARIANT THIS FILE PINS is the part that would matter if it ever changed:
**no strategy built on the printed relation may beat the best strategy that
ignores it.** If a future task set shifts the class balance so that the cue starts
paying, this goes red and the leak becomes worth a task set of its own.
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


generator = _load("_stats_extraction_v1_generator_shortcut", "generator.py")
scorer = _load("_stats_extraction_v1_scorer_shortcut", "scorer.py")

_P_CLAIM = re.compile(r"[,;:\s]\s*\*?p\*?\s*(&lt;|&gt;|<|>|=)\s*")
_HTML = {"&lt;": "<", "&gt;": ">"}
_VERDICTS = ("ok", "inconsistent", "unverifiable", "not_a_stat")


def _current_version():
    manifest = yaml.safe_load((ARENA_DIR / "arena.yaml").read_text(encoding="utf-8"))
    current = [v["version"] for v in manifest["task_set_versions"]
               if v.get("status") == "current"]
    assert len(current) == 1, f"arena.yaml has {len(current)} current task sets: {current}"
    return current[0]


def _printed_relation(text, item):
    """`<`, `>`, `=` or None — the relation the reader sees for this gold item."""
    span = item.get("span") or {}
    if span.get("char_end", 0) <= span.get("char_start", 0):
        return None
    m = _P_CLAIM.search(text, span["char_end"])
    return _HTML.get(m.group(1), m.group(1)) if m else None


@pytest.fixture(scope="module")
def tasks():
    generator._GROUND_TRUTH_CACHE.clear()
    out = [(env["input"]["text"], generator.ground_truth(env["task_id"]))
           for env in generator.generate(_current_version(), 0, "revealed")]
    assert len(out) == 36, f"expected 36 revealed tasks, got {len(out)} — the probe is broken"
    return out


def _score(tasks, verdict_fn):
    """Mean primary for a PERFECT extractor answering by `verdict_fn`.

    Perfect extraction on purpose: it holds precision and recall at 1.0 so the
    composite varies only with the verification head, which is the only place the
    printed relation could pay.
    """
    total = 0.0
    for text, gt in tasks:
        out = {"extractions": [{
            "span": dict(it.get("span") or {}),
            "kind": it.get("kind"),
            "fields": dict(it.get("fields") or {}),
            "confidence": 1.0,
            "flagged_suspicious": False,
            "verdict": verdict_fn(it, text),
        } for it in gt.get("items", [])]}
        total += scorer.score(out, gt)["primary"]
    return total / len(tasks)


def test_the_harness_discriminates(tasks):
    """TWO-SIDED CONTROL. Every number below is a comparison between strategies,
    and a harness that scored them all alike would satisfy the invariant while
    measuring nothing."""
    oracle = _score(tasks, lambda it, text: it.get("expected_verdict", "ok"))
    refuse = _score(tasks, lambda it, text: "unverifiable")
    assert oracle == pytest.approx(1.0), (
        f"a perfect extractor answering from the gold scored {oracle}, not 1.0 — "
        "the harness is measuring itself, not the strategies")
    assert refuse < 0.2, (
        f"refusing everything scored {refuse}; the spread between strategies has "
        "collapsed and no comparison below is meaningful")


def test_the_printed_relation_is_perfectly_precise(tasks):
    """The finding itself, pinned so the numbers in the write-ups stay checkable.

    Not a defect on its own — it is the premise of the test below. If it ever
    stops holding, the severity argument has to be re-derived rather than
    inherited.
    """
    ineq_truthful = ineq_deception = 0
    for text, gt in tasks:
        for it in gt["items"]:
            if it.get("kind") != "nhst_stat":
                continue
            if _printed_relation(text, it) in ("<", ">"):
                if it.get("truthful", True):
                    ineq_truthful += 1
                else:
                    ineq_deception += 1
    assert ineq_truthful >= 30, (
        f"only {ineq_truthful} items print an inequality — the arena has stopped "
        "emitting the APA form v4 was minted to introduce")
    assert ineq_deception == 0, (
        f"{ineq_deception} deception(s) now print an inequality. That is the "
        "REPAIR for this leak, not a regression — but the severity measured in "
        "test_knowing_the_printed_relation_buys_no_score is derived from a bucket "
        "with none, so re-measure before trusting it.")


def test_knowing_the_printed_relation_buys_no_score(tasks):
    """THE INVARIANT. A surface cue must not out-earn ignoring it.

    Exhaustive over the cue's partition: every constant answer for the
    inequality cell crossed with every constant answer for the `=` cell, which is
    the ceiling of any strategy built on the relation that does no recomputation.
    Measured 2026-08-29 on v5: the best of the 16 is `ok`/`ok`, which IS
    always-ok, and the gain is exactly zero.
    """
    cue_free = max(_score(tasks, lambda it, text, v=v: v) for v in _VERDICTS)
    best, best_pair = -1.0, None
    for a in _VERDICTS:
        for b in _VERDICTS:
            s = _score(tasks, lambda it, text, a=a, b=b:
                       a if _printed_relation(text, it) in ("<", ">") else b)
            if s > best:
                best, best_pair = s, (a, b)
    assert best <= cue_free + 1e-9, (
        f"knowing the printed p-relation is now worth {best - cue_free:+.4f}: the "
        f"best cue-using strategy (inequality->{best_pair[0]}, else->{best_pair[1]}) "
        f"scores {best:.4f} against {cue_free:.4f} for the best strategy that "
        "ignores it. The relation has become a scoring shortcut and the arena is "
        "no longer measuring recomputation on those items.")


def test_the_comparison_would_detect_a_cue_that_DID_pay(tasks):
    """POSITIVE CONTROL for the invariant above, and the reason it is not vacuous.

    `best <= cue_free` passing tells you nothing unless the same machinery reports
    a GAIN when the partition is genuinely informative. So partition by a cue that
    obviously pays -- the gold's own truthfulness, which no reader can see -- and
    require a clear gain. Without this, a bug that made every strategy score the
    same would satisfy the invariant permanently.
    """
    cue_free = max(_score(tasks, lambda it, text, v=v: v) for v in _VERDICTS)
    cheating = _score(tasks, lambda it, text:
                      "ok" if it.get("truthful", True) else "inconsistent")
    assert cheating > cue_free + 0.05, (
        f"a partition on the gold's own truthfulness scored {cheating:.4f} against "
        f"{cue_free:.4f} for the best blind strategy. The comparison cannot tell a "
        "paying cue from a worthless one, so the invariant above proves nothing.")


def test_the_deception_concentration_is_bounded(tasks):
    """What the cue DOES buy, kept honest and kept small.

    The relation cannot point at a deception, but it can narrow where to look:
    every deception sits in the `=` bucket, so the prior rises there. Measured on
    v5: 21/104 (20.2%) overall against 21/57 (36.8%) inside the bucket, 1.82x.
    Bounded at 3x — beyond that the cue is doing most of the work of finding the
    error class, and this stops being a dilution and becomes a defect.
    """
    n = n_dec = eq = eq_dec = 0
    for text, gt in tasks:
        for it in gt["items"]:
            if it.get("kind") != "nhst_stat":
                continue
            deception = not it.get("truthful", True)
            n += 1
            n_dec += deception
            if _printed_relation(text, it) == "=":
                eq += 1
                eq_dec += deception
    assert n_dec and eq, "no deceptions or no `=` items — the probe is broken"
    lift = (eq_dec / eq) / (n_dec / n)
    assert lift < 3.0, (
        f"the `=` bucket now concentrates deceptions {lift:.2f}x over the base "
        f"rate ({eq_dec}/{eq} against {n_dec}/{n}). The printed relation has "
        "become a usable detector for the error class.")
