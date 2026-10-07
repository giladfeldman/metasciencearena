"""RED-first tests for the verification head's pricing of non-answers (2026-08-27).

Written against the UNFIXED scorer and watched fail before the fix, per the
portfolio rule that a regression test which has never failed is decoration.

TWO defects, both in the same place, both MEASURED on the published v3 records
before a line was changed.

F1 - universal abstention was strictly dominant. `score()` read
`verification_score = (v_hits / v_total) if v_total else 1.0`, and abstentions
leave `v_total`. A player that answered "unverifiable" to every item therefore
emptied the denominator and collected the MAXIMUM verification score. Measured
with a perfect extractor over all 36 revealed v3 tasks:

    always "unverifiable"  -> verification 1.0000   <- the best strategy
    always "ok"            -> verification 0.8078
    always "inconsistent"  -> verification 0.2931

Live footprint: 11 escimate tasks and 15 statcheck tasks reached `v_total == 0`
and took the free 1.0. No LLM player had a single such task, which is why no
LLM score can move when this is repaired.

F2 - `expected_verdict == "unverifiable"` fell through to the `inconsistent`
branch, because `_verification_cell` handled "ok", "not_a_stat" and then assumed
whatever was left was "inconsistent". Gold really does carry it: 7 of the 149
checkable revealed items. The consequence was exactly backwards - a player that
CORRECTLY said "unverifiable" scored nothing (the answer left the denominator),
while a player that FALSELY accused the same item of being inconsistent was
credited with a hit.
"""
from pathlib import Path
import importlib.util
import sys

ARENA_DIR = Path(__file__).resolve().parents[1]


def _load(mod_name, filename):
    spec = importlib.util.spec_from_file_location(mod_name, ARENA_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


scorer = _load("_stats_extraction_v1_scorer_ad", "scorer.py")
generator = _load("_stats_extraction_v1_generator_ad", "generator.py")


def _item(**kw):
    item = {
        "kind": "nhst_stat",
        "span": {"text": "5.41", "char_start": 10, "char_end": 14},
        "fields": {"value": 5.41},
        "truthful": True,
        "deception_kind": None,
        "checkable": True,
        "expected_verdict": "ok",
        "coherent": True,
    }
    item.update(kw)
    return item


def _ext(verdict, value=5.41):
    return {
        "span": {"text": "5.41", "char_start": 10, "char_end": 14},
        "kind": "nhst_stat",
        "fields": {"value": value},
        "confidence": 1.0,
        "flagged_suspicious": verdict == "inconsistent",
        "verdict": verdict,
    }


# --- F1: declining an answerable item is not free ---------------------------

def test_abstaining_on_every_checkable_item_is_not_a_perfect_score():
    """The whole defect in one assertion.

    Three coherent, checkable items the gold can answer. The player declines all
    three. Before the fix this returned 1.0 - a better verification score than a
    player who answered all three and got two right.
    """
    gt = {"items": [_item(), _item(), _item()]}
    res = scorer.score({"extractions": [_ext("unverifiable") for _ in range(3)]}, gt)
    assert res["breakdown"]["verification_score"] == 0.0, (
        "declining every answerable item must not score as verifying them all: "
        f"got {res['breakdown']['verification_score']}"
    )
    assert res["breakdown"]["n_verifiable"] == 3, (
        "the three declined items must stay in the denominator, not vanish from it"
    )


def test_declining_is_worse_than_answering_and_being_partly_wrong():
    """Dominance, stated as an ordering rather than a constant.

    A player that answers three items and gets two right must beat a player that
    declines all three. Before the fix the decliner won, 1.0 to 0.667.
    """
    gt = {"items": [_item(), _item(), _item(expected_verdict="inconsistent")]}
    answered = scorer.score(
        {"extractions": [_ext("ok"), _ext("ok"), _ext("ok")]}, gt)
    declined = scorer.score(
        {"extractions": [_ext("unverifiable") for _ in range(3)]}, gt)
    assert (answered["breakdown"]["verification_score"]
            > declined["breakdown"]["verification_score"]), (
        f"answering ({answered['breakdown']['verification_score']}) must beat "
        f"declining ({declined['breakdown']['verification_score']})"
    )


def test_a_task_with_nothing_answerable_is_still_neutral():
    """The control that keeps the fix honest.

    `v_total == 0` has TWO causes and only one of them is a defect. When the gold
    itself has nothing checkable, there is no verification evidence either way and
    1.0 (neutral, verification drops out of the product) is correct. This must not
    become 0.0 as collateral.
    """
    gt = {"items": [_item(checkable=False), _item(checkable=False)]}
    res = scorer.score({"extractions": [_ext("unverifiable"), _ext("unverifiable")]}, gt)
    assert res["breakdown"]["n_verifiable"] == 0
    assert res["breakdown"]["verification_score"] == 1.0


def test_flagging_an_incoherent_item_still_leaves_the_denominator():
    """The other legitimate exclusion, pinned so the F1 fix cannot swallow it.

    An item whose reported p disagrees with the recomputed p at the reported
    precision but does not cross alpha is genuinely ambiguous: flagging it is
    right at a finer granularity than `expected_verdict` asks about. Measured
    2026-08-21: 74 of 84 truthful NHST items differ at the reported precision
    while only 37 cross alpha, and without this rule statcheck took 31 false
    alarms for being right.
    """
    gt = {"items": [_item(coherent=False)]}
    res = scorer.score({"extractions": [_ext("inconsistent")]}, gt)
    assert res["breakdown"]["n_verifiable"] == 0
    assert res["breakdown"]["verification_score"] == 1.0


# --- F2: gold that says "this genuinely cannot be checked" ------------------

def test_correctly_saying_unverifiable_is_credited_as_a_hit():
    """When the gold's own expected verdict is `unverifiable`, saying so is the
    RIGHT answer, not a non-answer. Before the fix it earned nothing."""
    gt = {"items": [_item(expected_verdict="unverifiable")]}
    res = scorer.score({"extractions": [_ext("unverifiable")]}, gt)
    assert res["breakdown"]["n_verifiable"] == 1, "a correct answer must be scored"
    assert res["breakdown"]["verification_score"] == 1.0
    assert res["breakdown"]["verification_hits"] == 1


def test_accusing_a_genuinely_unverifiable_item_is_a_false_alarm_not_a_hit():
    """The same defect from the other side, and the sharper half: the code's
    final line assumed `expected == "inconsistent"`, so an accusation levelled at
    an item the gold says CANNOT be checked was credited as a hit."""
    gt = {"items": [_item(expected_verdict="unverifiable")]}
    res = scorer.score({"extractions": [_ext("inconsistent")]}, gt)
    assert res["breakdown"]["verification_hits"] == 0, (
        "claiming to have found an inconsistency in something that cannot be "
        "checked is not a correct verification"
    )
    assert res["breakdown"]["verification_false_alarms"] == 1
    assert "deception_false_alarm" in [f["category"] for f in res["findings"]]


# --- the arena-level acceptance test ----------------------------------------

def test_no_fixed_verdict_strategy_beats_every_other_on_the_real_gold():
    """Acceptance, on the real generator rather than a hand-built fixture.

    A perfect extractor that answers one fixed verdict to everything. Before the
    fix "unverifiable" scored 1.0000 and won outright; nothing a real verifier
    could do would beat it, so the arena rewarded refusing to work.
    """
    def _synthetic(gt, verdict):
        return {"extractions": [{
            "span": dict(item.get("span") or {}),
            "kind": item.get("kind"),
            "fields": dict(item.get("fields") or {}),
            "value": item.get("value"),
            "confidence": 1.0,
            "flagged_suspicious": verdict == "inconsistent",
            "verdict": verdict,
        } for item in gt.get("items", [])]}

    envelopes = list(generator.generate("v3", 0, split="revealed"))
    assert envelopes, (
        "generator produced no revealed v3 tasks - the probe, not the arena, is broken")

    means = {}
    for verdict in ("unverifiable", "ok", "inconsistent"):
        scores = []
        for env in envelopes:
            gt = generator.ground_truth(env["task_id"])
            scores.append(
                scorer.score(_synthetic(gt, verdict), gt)["breakdown"]["verification_score"])
        means[verdict] = sum(scores) / len(scores)

    assert means["unverifiable"] < means["ok"], (
        "refusing to verify anything must not outscore even a trivial guesser: "
        f"unverifiable={means['unverifiable']:.4f} ok={means['ok']:.4f}"
    )
    assert means["unverifiable"] < 0.5, (
        f"a universal abstainer scored {means['unverifiable']:.4f}; gold marks only "
        f"7 of 149 checkable revealed items genuinely unverifiable, so the ceiling "
        f"for that strategy is low"
    )


# --- F3: the same defect, surviving in the incoherent branch -----------------
# Found by cross-model consult (seat: Sol / GPT-5.6 via Codex CLI, OpenAI) on
# 2026-08-28 and reproduced locally before these tests were written.
#
# F1 above charged a decline on a COHERENT item. It left the incoherent branch
# alone: `_verification_cell` still read
#
#     if said == "unverifiable": return "declined" if coherent else "abstain"
#
# so a decline aimed at an INCOHERENT item was still forgiven, still left
# `v_total`, and on a task where every checkable item is incoherent still emptied
# the denominator into the neutral 1.0. That is F1's exact shape, one branch over.
#
# MEASURED 2026-08-28 on the real generator, counted item = one gold task:
#   v3: 3 tasks have every checkable item incoherent -- t-tier2-d1-1-s5feceb66,
#       t-tier3-d1-0-s5feceb66, t-tier4-d3-1-s5feceb66 (Sol named all three).
#   v4: 0 such tasks; the clamp fix removed the incoherent class outright
#       (v3 has 45 `coherent=False` items, v4 has 0).
#
# And the fact that decides the SEMANTICS, measured over both task sets: in real
# gold `coherent=False` occurs ONLY with `expected_verdict == "ok"` -- never with
# `inconsistent` or `not_a_stat`. So the ambiguity forgiveness is doing exactly
# one job: protecting a player that ACCUSES an item whose reported p disagrees
# with the recomputed p at the reported precision without crossing alpha. An
# accusation can be right at a finer granularity than `expected_verdict` asks
# about. A REFUSAL cannot be. "I cannot check this" is not a sharper reading of
# an answerable item, it is the absence of a reading, and the gold can answer it.


def test_declining_every_incoherent_item_is_not_a_perfect_score():
    """The F3 defect in one assertion, mirroring F1's.

    Three checkable items the gold answers `ok`, all three incoherent. The player
    declines all three. Before the fix every cell was `abstain`, `v_total` stayed
    0, and the head fell through to the neutral 1.0 -- the same free perfect score
    F1 removed for coherent items.
    """
    gt = {"items": [_item(coherent=False) for _ in range(3)]}
    res = scorer.score({"extractions": [_ext("unverifiable") for _ in range(3)]}, gt)
    assert res["breakdown"]["verification_score"] != 1.0, (
        "declining every answerable item must not score as verifying them all, "
        "and an item being incoherent does not make it unanswerable: got "
        f"{res['breakdown']['verification_score']}"
    )
    assert res["breakdown"]["n_verifiable"] == 3, (
        "the three declined items must stay in the denominator, not vanish from "
        "it -- incoherence forgives an ACCUSATION, not a REFUSAL"
    )


def test_a_perfect_extractor_that_declines_does_not_tie_a_real_verifier():
    """The exploit stated as an ordering, on the real v3 gold.

    On `t-tier2-d1-1-s5feceb66` every checkable item is incoherent. A PERFECT
    extractor that declines every one of them took `primary == 1.0` before the
    fix -- identical to a player that extracted perfectly AND verified correctly.
    No real player is a perfect extractor that also abstains, which is why no
    published score moved; the path being reachable at all is the defect.
    """
    def _synthetic(gt, verdict):
        return {"extractions": [{
            "span": dict(item.get("span") or {}),
            "kind": item.get("kind"),
            "fields": dict(item.get("fields") or {}),
            "value": item.get("value"),
            "confidence": 1.0,
            "flagged_suspicious": verdict == "inconsistent",
            "verdict": verdict,
        } for item in gt.get("items", [])]}

    # `ground_truth()` serves from the cache `generate()` fills, so generate the
    # split FIRST. Without this the test passes only when some earlier test in
    # the file happens to have generated v3 already — an order dependency that
    # bit this very test when it was written (KeyError in isolation, green in a
    # full-file run).
    envelopes = list(generator.generate("v3", 0, split="revealed"))
    assert envelopes, "generator produced no revealed v3 tasks — the probe is broken"

    gt = generator.ground_truth("t-tier2-d1-1-s5feceb66")
    checkable = [i for i in gt["items"] if i.get("checkable")]
    assert checkable, "probe is broken: the task has no checkable items to decline"
    assert all(not i.get("coherent", True) for i in checkable), (
        "probe is broken: this task is supposed to be entirely incoherent")

    declined = scorer.score(_synthetic(gt, "unverifiable"), gt)
    answered = scorer.score(_synthetic(gt, "ok"), gt)
    assert declined["primary"] < answered["primary"], (
        "a perfect extractor that declines every item must not tie one that "
        f"answers them: declined={declined['primary']} answered={answered['primary']}"
    )
    assert declined["breakdown"]["verification_score"] != 1.0


def test_flagging_an_incoherent_item_is_still_forgiven():
    """The control that keeps the F3 fix from overreaching.

    This is the ONE job `coherent=False` exists to do, and the F3 fix must not
    touch it. Measured 2026-08-21: 74 of 84 truthful NHST items differ at the
    reported precision while only 37 cross alpha; without this rule statcheck
    took 31 false alarms for being right. Duplicates the F1-era control above on
    purpose -- if a fix to the decline path ever collapses the whole incoherent
    branch, two tests go red instead of one.
    """
    gt = {"items": [_item(coherent=False)]}
    res = scorer.score({"extractions": [_ext("inconsistent")]}, gt)
    assert res["breakdown"]["n_verifiable"] == 0, (
        "an accusation against an incoherent item must still leave the denominator")
    assert res["breakdown"]["verification_score"] == 1.0
