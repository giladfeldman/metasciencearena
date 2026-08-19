"""Tests for the in-process adapter timeout wrapper (players/adapters/_timeout.py)."""
import time

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
