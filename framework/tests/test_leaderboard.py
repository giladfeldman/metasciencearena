"""Tests for the leaderboard renderer + intersection-aware aggregation (§A)."""
from framework.leaderboard import render_leaderboard, aggregate


def _record(player_id, primary, task_id, *, error=None, latency=None, version="0.0.1"):
    score = {"primary": primary, "breakdown": {}}
    if error is not None:
        score = {"primary": 0.0, "breakdown": {"error": error}}
    rec = {
        "run_id": f"r-{player_id}-{task_id}", "arena_id": "fake-arena",
        "task_set_version": "v1", "task_id": task_id, "player_id": player_id,
        "player_version": version, "player_type": "tool", "input_hash": "0" * 64,
        "output": {}, "score": score,
        "timestamp_utc": "2026-04-29T12:00:00Z",
    }
    if latency is not None:
        rec["latency_ms"] = latency
    return rec


def test_aggregate_groups_by_player():
    # Both players ran the SAME two tasks -> symmetric, both ranked over {t0,t1}.
    records = [
        _record("a", 1.0, "t0"), _record("a", 0.5, "t1"),
        _record("b", 0.25, "t0"), _record("b", 0.25, "t1"),
    ]
    rows = aggregate(records)
    assert {r["player_id"] for r in rows} == {"a", "b"}
    a = next(r for r in rows if r["player_id"] == "a")
    assert a["n_ranked"] == 2
    assert abs(a["primary_mean"] - 0.75) < 1e-9
    assert a["bucket"] == "ranked"
    assert a["n_missing"] == 0


def test_render_leaderboard_returns_markdown_table():
    records = [_record("a", 1.0, "t0"), _record("b", 0.5, "t0")]
    md = render_leaderboard(records)
    assert "Player" in md
    assert "| a |" in md
    assert "| b |" in md


def test_aggregate_latency_mean_excludes_errored():
    records = [
        _record("p", 1.0, "t0", latency=100),
        _record("p", 0.5, "t1", latency=200),
        _record("p", None, "t2", error="boom", latency=9999),
    ]
    rows = aggregate(records)
    assert len(rows) == 1
    assert rows[0]["latency_ms_mean"] == 150  # (100 + 200) / 2; error record excluded


# ---- §A: intersection-aware ranking -------------------------------------

def test_aggregate_ranks_over_common_intersection():
    # A ran {t1,t2,t3}, B ran {t1,t2}. Both are ranked over the common {t1,t2};
    # A's extra t3 is dropped from the mean but counted in n_ran_total. B is
    # flagged partial because it skipped a task (t3) in the co-ranked universe.
    records = [
        _record("A", 0.8, "t1"), _record("A", 0.6, "t2"), _record("A", 0.4, "t3"),
        _record("B", 0.2, "t1"), _record("B", 0.4, "t2"),
    ]
    rows = {r["player_id"]: r for r in aggregate(records)}
    a, b = rows["A"], rows["B"]
    # Mean is over the COMMON {t1,t2} for BOTH (comparable), not A's full 3.
    assert a["n_ranked"] == 2 and abs(a["primary_mean"] - 0.7) < 1e-9
    assert b["n_ranked"] == 2 and abs(b["primary_mean"] - 0.3) < 1e-9
    assert a["n_common"] == 2 and b["n_common"] == 2
    # Both are ranked (they share the common set); A ran the whole co-ranked
    # universe, B skipped t3 — reported via n_missing, NOT a demotion.
    assert a["n_ran_total"] == 3 and a["n_missing"] == 0 and a["bucket"] == "ranked"
    assert b["n_ran_total"] == 2 and b["n_missing"] == 1 and b["bucket"] == "ranked"


def test_aggregate_disjoint_player_never_averaged_head_to_head():
    # C ran a disjoint set {t9}; it must not collapse the common set nor be
    # averaged against A/B.
    records = [
        _record("A", 0.8, "t1"), _record("A", 0.6, "t2"), _record("A", 0.4, "t3"),
        _record("B", 0.2, "t1"), _record("B", 0.4, "t2"),
        _record("C", 0.99, "t9"),
    ]
    rows = {r["player_id"]: r for r in aggregate(records)}
    # A/B still ranked over {t1,t2} (C did not poison the intersection).
    assert rows["A"]["n_common"] == 2 and rows["A"]["n_ranked"] == 2
    assert rows["B"]["n_common"] == 2
    c = rows["C"]
    assert c["disjoint"] is True
    assert c["n_ranked"] == 0          # nothing in the common set -> not in the mean
    assert c["bucket"] == "partial"


def test_aggregate_explicit_rank_task_ids():
    # Pin the comparison set explicitly: only t1 counts toward the mean.
    records = [
        _record("A", 0.8, "t1"), _record("A", 0.0, "t2"),
        _record("B", 0.2, "t1"), _record("B", 1.0, "t2"),
    ]
    rows = {r["player_id"]: r for r in aggregate(records, rank_task_ids={"t1"})}
    assert abs(rows["A"]["primary_mean"] - 0.8) < 1e-9
    assert abs(rows["B"]["primary_mean"] - 0.2) < 1e-9
    assert rows["A"]["n_common"] == 1


def test_aggregate_does_not_average_a_crash_as_zero():
    # §F / Finding 7: a crashing adapter records score.breakdown.error and must
    # be EXCLUDED from the mean — never silently counted as a 0.0 "skill" score
    # that floors the player. Here the good task scores 1.0; the crash on t2 must
    # not drag the mean to 0.5.
    records = [
        _record("p", 1.0, "t1"),
        _record("p", None, "t2", error="RuntimeError: boom"),
    ]
    row = aggregate(records)[0]
    assert row["n_errored"] == 1
    assert row["n_ranked"] == 1
    assert abs(row["primary_mean"] - 1.0) < 1e-9   # NOT 0.5


def test_aggregate_symmetric_all_ranked():
    # Everyone ran the same full split -> all ranked, none partial.
    records = [
        _record("A", 0.9, "t1"), _record("A", 0.7, "t2"),
        _record("B", 0.5, "t1"), _record("B", 0.3, "t2"),
        _record("C", 0.6, "t1"), _record("C", 0.6, "t2"),
    ]
    rows = aggregate(records)
    assert all(r["bucket"] == "ranked" for r in rows)
    assert all(r["n_missing"] == 0 for r in rows)
    assert all(r["n_common"] == 2 for r in rows)
