"""Tests for the leaderboard fairness audits (framework/audit.py).

Covers Finding 1 (tool-version drift) and Finding 2 (task-set symmetry) of the
2026-06-12 accuracy handoff.
"""
from framework.audit import (
    DriftRow,
    _numeric,
    per_split_symmetry_ok,
    task_set_symmetry,
    version_drift_from_registry,
)


def _rec(player_id, task_id, visibility="public"):
    return {
        "player_id": player_id, "task_id": task_id, "task_visibility": visibility,
        "score": {"primary": 1.0, "breakdown": {}},
    }


# ---- Finding 2: task-set symmetry -----------------------------------------

def test_symmetry_ok_when_all_players_share_task_set():
    records = [
        _rec("a", "t1"), _rec("a", "t2"),
        _rec("b", "t1"), _rec("b", "t2"),
    ]
    rep = task_set_symmetry(records, arena_id="x")
    assert rep.ok
    assert rep.intersection_size == 2
    assert rep.offenders == []


def test_symmetry_flags_asymmetric_player():
    # docpluck-style: 'a' ran 3 tasks, 'b' only 2 — the classic Finding 2 case.
    records = [
        _rec("a", "t1"), _rec("a", "t2"), _rec("a", "t3"),
        _rec("b", "t1"), _rec("b", "t2"),
    ]
    rep = task_set_symmetry(records, arena_id="x")
    assert not rep.ok
    assert "a" in rep.offenders
    assert rep.intersection_size == 2
    assert rep.union_size == 3


def test_symmetry_respects_visibility_split():
    records = [
        _rec("a", "t1", "public"), _rec("a", "p1", "held_out"),
        _rec("b", "t1", "public"),  # b never ran the held-out task
    ]
    # Revealed split is symmetric (both ran t1)...
    assert task_set_symmetry(records, visibility="public").ok
    # ...but the held-out split has only 'a', so a single-player split is trivially OK.
    held = task_set_symmetry(records, visibility="held_out")
    assert held.per_player == {"a": 1}


def test_symmetry_single_player_is_ok():
    rep = task_set_symmetry([_rec("a", "t1"), _rec("a", "t2")], arena_id="x")
    assert rep.ok
    assert rep.offenders == []


# ---- per-split gate (Finding 2 exit-code semantics) -----------------------

def test_per_split_gate_passes_when_a_tool_covers_both_splits():
    # escimate/PDF-parser shape: a deterministic tool runs BOTH splits while the
    # AI players run revealed only. Each split is internally symmetric, so the
    # per-split gate must PASS even though the pooled "[all]" view double-counts
    # the tool and would flag it as asymmetric.
    records = [
        # revealed: tool + both AI players cover t1,t2
        _rec("tool", "t1", "public"), _rec("tool", "t2", "public"),
        _rec("ai1", "t1", "public"), _rec("ai1", "t2", "public"),
        _rec("ai2", "t1", "public"), _rec("ai2", "t2", "public"),
        # private: only the tool runs it (single-player split = trivially OK)
        _rec("tool", "p1", "held_out"), _rec("tool", "p2", "held_out"),
    ]
    # Pooled scope IS asymmetric (tool has 4 task_ids vs AI players' 2)...
    assert not task_set_symmetry(records, arena_id="x").ok
    # ...but the per-split gate passes (fairness is within-split).
    assert per_split_symmetry_ok(records, arena_id="x")


def test_per_split_gate_fails_on_within_split_asymmetry():
    # A genuine unfairness: within the revealed split, ai2 skipped t2.
    records = [
        _rec("tool", "t1", "public"), _rec("tool", "t2", "public"),
        _rec("ai1", "t1", "public"), _rec("ai1", "t2", "public"),
        _rec("ai2", "t1", "public"),  # <-- missing t2 within revealed
    ]
    assert not per_split_symmetry_ok(records, arena_id="x")


def test_per_split_gate_passes_when_all_symmetric():
    records = [
        _rec("a", "t1", "public"), _rec("a", "t2", "public"),
        _rec("b", "t1", "public"), _rec("b", "t2", "public"),
    ]
    assert per_split_symmetry_ok(records, arena_id="x")


def test_per_split_gate_tolerates_best_effort_partial():
    # regcheck shape: deepseek + 2 AI players cover all 3 tasks; the free-tier
    # groq provider only got 1 (throttled). Marked best_effort → gate PASSES.
    records = [
        _rec("regcheck-deepseek", "t1", "public"), _rec("regcheck-deepseek", "t2", "public"),
        _rec("regcheck-deepseek", "t3", "public"),
        _rec("ai1", "t1", "public"), _rec("ai1", "t2", "public"), _rec("ai1", "t3", "public"),
        _rec("regcheck-groq", "t1", "public"),  # throttled after 1 task
    ]
    # Without the exemption it fails (groq is an offender)...
    assert not per_split_symmetry_ok(records, arena_id="x")
    # ...with groq marked best_effort, the gate tolerates the partial coverage.
    assert per_split_symmetry_ok(records, arena_id="x", best_effort={"regcheck-groq"})


def test_per_split_gate_still_fails_on_non_best_effort_offender():
    # A best-effort partial is fine, but a NON-best-effort player that skips a task
    # is still a real failure — the exemption must not mask genuine unfairness.
    records = [
        _rec("regcheck-deepseek", "t1", "public"), _rec("regcheck-deepseek", "t2", "public"),
        _rec("ai1", "t1", "public"),  # ai1 missing t2 — NOT best_effort
        _rec("regcheck-groq", "t1", "public"),
    ]
    assert not per_split_symmetry_ok(
        records, arena_id="x", best_effort={"regcheck-groq", "regcheck-deepseek"})


def test_per_split_report_excludes_best_effort_like_the_gate():
    """The printed per-split report must agree with the gate it sits above.

    Observed 2026-08-04 on prereg-deviation-v1: the CLI printed
    "[revealed] SYMMETRY FAIL" while "[GATE] ... PASS" appeared one line below,
    because the report called task_set_symmetry() over ALL records while the gate
    correctly excluded best-effort players. A report that cries foul on a passing
    arena sends someone chasing a non-bug — and worse, trains the reader to
    ignore SYMMETRY FAIL lines that DO matter.

    This asserts the underlying invariant the CLI now relies on: over the
    non-best-effort subset, the per-split report is symmetric exactly when the
    gate passes.
    """
    records = [
        _rec("regcheck-deepseek", "t1", "public"), _rec("regcheck-deepseek", "t2", "public"),
        _rec("ai1", "t1", "public"), _rec("ai1", "t2", "public"),
        _rec("regcheck-groq", "t1", "public"),  # throttled, best_effort
    ]
    best_effort = {"regcheck-groq"}

    # The gate passes: the two full-coverage players agree.
    assert per_split_symmetry_ok(records, arena_id="x", best_effort=best_effort)

    # Reporting over ALL records disagrees with the gate (this is the old bug)...
    all_rep = task_set_symmetry(records, arena_id="x", visibility="public")
    assert not all_rep.ok

    # ...so the CLI reports over the non-best-effort subset, which agrees.
    primary = [r for r in records if r["player_id"] not in best_effort]
    gate_rep = task_set_symmetry(primary, arena_id="x", visibility="public")
    assert gate_rep.ok, "per-split report must match the gate's verdict"


# ---- Finding 1: tool-version drift ----------------------------------------

def test_numeric_extracts_dotted_version():
    assert _numeric("docpluck-2.4.84") == "2.4.84"
    assert _numeric("poppler-pdftotext-24.08.0") == "24.08.0"
    assert _numeric("liteparse-2.0.0+arena-heuristic-sections") == "2.0.0"
    assert _numeric(None) is None
    assert _numeric("no-version-here") is None


class _FakeAdapter:
    def __init__(self, resolved):
        self._resolved = resolved

    def resolved_tool_version(self):
        return self._resolved


def test_version_drift_detects_stale_declared(monkeypatch):
    import framework.audit as audit

    registry = [
        {"player_id": "stale", "player_version": "docpluck-2.4.79-academic"},
        {"player_id": "fresh", "player_version": "docpluck-2.4.84-academic"},
        {"player_id": "llm", "player_version": "claude-haiku-4-5"},
    ]
    resolved = {"stale": "docpluck-2.4.84", "fresh": "docpluck-2.4.84", "llm": None}

    def fake_build(entry):
        return _FakeAdapter(resolved[entry["player_id"]])

    monkeypatch.setattr("framework.player_adapter.build_adapter", fake_build)
    rows = version_drift_from_registry(registry)
    by_id = {r.player_id: r for r in rows}
    # LLM resolves None -> not reported (its version IS the declared model id).
    assert "llm" not in by_id
    assert by_id["stale"].drift is True
    assert by_id["fresh"].drift is False
