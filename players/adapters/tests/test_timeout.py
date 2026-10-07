"""Tests for the in-process adapter timeout wrapper (players/adapters/_timeout.py)."""
import time
from pathlib import Path

import pytest

from players.adapters._timeout import run_with_timeout


def test_returns_value_when_fast():
    assert run_with_timeout(lambda: 1 + 1, timeout_s=5) == 2


def test_no_timeout_when_zero_or_none():
    assert run_with_timeout(lambda: "ok", timeout_s=0) == "ok"
    assert run_with_timeout(lambda: "ok", timeout_s=None) == "ok"


def test_raises_timeout_error_on_slow_work():
    def slow():
        time.sleep(5)
        return "done"

    t0 = time.time()
    with pytest.raises(TimeoutError):
        run_with_timeout(slow, timeout_s=1, label="slow")
    # Returns control promptly (does NOT block on the orphan worker via shutdown).
    assert time.time() - t0 < 3


def test_propagates_callable_exception():
    def boom():
        raise ValueError("kaboom")

    with pytest.raises(ValueError, match="kaboom"):
        run_with_timeout(boom, timeout_s=5)


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[3] / "arenas" / "pdf-table-extraction-v1" / "scorer.py").is_file(),
    reason="pdf-table-extraction-v1 is a PRIVATE arena; its scorer is not in the public mirror",
)
def test_a_timeout_is_recorded_as_an_error_not_a_clean_zero():
    """docpluck asked (M-docpluck-arena-20260927T102719Z, A4) that a timeout stop
    reading as an ordinary 0.0. Measured 2026-09-28: the runner already records
    `breakdown.error = "TimeoutError: ..."` and classifies the record `timeout`, and
    the report's unscoreable policy keeps the reason under `unscoreable_reason`.
    This pins it, through the real `_play_one`, so it cannot regress silently."""
    import importlib.util
    from pathlib import Path

    from framework import runner

    repo = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        "tables_scorer_t", repo / "arenas" / "pdf-table-extraction-v1" / "scorer.py")
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)

    class Slow:
        def play_task(self, env, *, timeout_s):
            return run_with_timeout(lambda: time.sleep(5) or {"tables": []},
                                    timeout_s, label="docpluck tables")

        def resolved_tool_version(self):
            return None

        def last_usage(self):
            return None

    class NoErrors:
        def iter_errors(self, _output):
            return []

    envelope = {"task_id": "t", "visibility": "public", "input": {},
                "task_set_version": "v1", "arena_id": "pdf-table-extraction-v1"}
    entry = {"player_id": "docpluck-tables", "player_version": "x", "player_type": "platform",
             "instrument": "x", "adapter_class": "DocpluckTablesAdapter"}
    rec = runner._play_one(envelope, {"gold_tables": [{"label": "Table 1"}]}, entry, Slow(),
                           scorer, NoErrors(), 1, "pdf-table-extraction-v1",
                           trials=1, seed=0, split="revealed")
    assert rec["score"]["breakdown"]["error"].startswith("TimeoutError")
    assert runner._record_outcome(rec) == "timeout"
