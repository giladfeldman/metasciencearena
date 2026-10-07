"""A finding category every player shares is a GOLD constant, not a tool defect.

Regression test for the defect measured on pdf-text-fidelity-v1, 2026-08-21:
`tools/fetch_pmc.py` hardcoded `"footnotes": []` into all 30 real-paper golds, so
every deterministic tool ended up with **exactly 56 `footnote_missed` findings
over 40 tasks** -- docpluck, GROBID, liteparse and pdftotext alike. The feedback
report then told each of them, individually:

    "Reduce `footnote_missed`: 90% of your findings, 40 tasks affected."

Nothing any tool did had ever moved that number. `_improvement_priorities` ranked
purely on within-player share, so a constant with a large share became everyone's
number-one thing to work on.

Written against the UNFIXED engine first and watched fail (the failure was:
`footnote_missed` appeared as priority #1 for a player whose count was identical
to every other player's).
"""
from __future__ import annotations

from framework.report import _improvement_priorities, field_wide_constant_categories


def _rec(player_id: str, task_id: str, categories: list[str], primary: float = 0.5) -> dict:
    return {
        "player_id": player_id,
        "task_id": task_id,
        "task_set_version": "v1",
        "status": "ok",
        "score": {
            "primary": primary,
            "breakdown": {},
            "findings": [{"category": c} for c in categories],
        },
    }


def test_a_category_identical_across_every_player_is_reported_as_constant():
    records = []
    for pid in ("tool-a", "tool-b", "tool-c"):
        for i in range(4):
            # Two `gold_constant` findings per task for EVERY player, plus a
            # varying one that genuinely differs between them.
            extra = ["real_defect"] if pid == "tool-a" and i < 2 else []
            records.append(_rec(pid, f"t{i}", ["gold_constant", "gold_constant", *extra]))

    constant = field_wide_constant_categories(records)
    assert constant == {"gold_constant": 8}, constant
    assert "real_defect" not in constant


def test_a_single_player_arena_declares_nothing_constant():
    """With one player every category is trivially constant.

    Suppressing all of them would empty the report rather than clean it, so the
    rule needs at least two players before it can say anything.
    """
    records = [_rec("tool-a", f"t{i}", ["anything"]) for i in range(3)]
    assert field_wide_constant_categories(records) == {}


def test_errored_records_do_not_make_a_category_look_constant():
    """A player whose runs all errored contributes no findings.

    Counting it as "0 in every category" would make a genuinely varying category
    look non-constant (or an absent one look constant) for the wrong reason.
    Errored records are simply not scored, so they are skipped entirely.
    """
    records = [_rec("tool-a", f"t{i}", ["shared"]) for i in range(3)]
    records += [_rec("tool-b", f"t{i}", ["shared"]) for i in range(3)]
    # The real shape of an errored record: run_record.schema.json requires
    # `score` to be an object, with `primary: null` and an `error` in breakdown.
    # (`score: null` is schema-forbidden, so testing that shape would be testing
    # a record the runner cannot write.)
    records.append({"player_id": "tool-c", "task_id": "t9", "task_set_version": "v1",
                    "score": {"primary": None, "breakdown": {"error": "timeout"}}})
    assert field_wide_constant_categories(records) == {"shared": 3}


def test_a_constant_category_is_not_published_as_an_improvement_priority():
    """THE regression. `footnote_missed` at 90% share must not be priority #1."""
    public_hist = [
        {"category": "footnote_missed", "severity": "major",
         "share_of_findings": 0.90, "n_tasks_affected": 40},
        {"category": "scope_reference_leak", "severity": "major",
         "share_of_findings": 0.10, "n_tasks_affected": 12},
    ]
    kwargs = dict(
        public_hist=public_hist, held_out_hist=[], public_mean=0.66,
        held_out_mean=None, diff_breakdown={}, competitors=[],
        cat_examples={}, competitor_examples={},
    )

    # Without the constant set, the old behaviour: the constant ranks first.
    unfiltered = _improvement_priorities(**kwargs)
    assert unfiltered[0]["evidence"]["category"] == "footnote_missed"

    # With it, the priority list is about something the tool can actually change.
    filtered = _improvement_priorities(**kwargs, constant_categories={"footnote_missed": 56})
    cats = [i["evidence"].get("category") for i in filtered if i["kind"] == "failure_mode"]
    assert "footnote_missed" not in cats
    assert "scope_reference_leak" in cats


def test_suppression_does_not_empty_the_priorities_when_everything_is_constant():
    """All categories constant -> no failure_mode items, and that is correct.

    The report still carries its other priority kinds and its histogram. An empty
    failure-mode list is the honest answer to "what should this tool fix?" when
    every finding it has is one the gold hands to everyone equally.
    """
    public_hist = [{"category": "only_one", "severity": "major",
                    "share_of_findings": 1.0, "n_tasks_affected": 40}]
    out = _improvement_priorities(
        public_hist=public_hist, held_out_hist=[], public_mean=0.5,
        held_out_mean=None, diff_breakdown={}, competitors=[],
        cat_examples={}, competitor_examples={},
        constant_categories={"only_one": 40})
    assert [i for i in out if i["kind"] == "failure_mode"] == []
