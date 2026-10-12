"""A declared label the running tool contradicts is refused before any task (d-19134a).

The publish gate (fairness-gates.mjs assertVersionConsistency) already rejects such
records; this makes the run fail first, so no run is spent on records that cannot land.
"""
from __future__ import annotations

import pytest

from framework.runner import _refuse_version_drift


class _Fake:
    def __init__(self, v):
        self.v = v

    def resolved_tool_version(self):
        if isinstance(self.v, Exception):
            raise self.v
        return self.v


def test_contradicting_version_is_refused():
    with pytest.raises(RuntimeError, match="version drift"):
        _refuse_version_drift({"player_id": "docpluck", "player_version": "docpluck-2.4.150"},
                              _Fake("2.4.151"))


def test_matching_version_passes():
    # Two-sided control.
    _refuse_version_drift({"player_id": "docpluck", "player_version": "docpluck-2.4.151"},
                          _Fake("docpluck 2.4.151"))


def test_unknown_or_failing_probe_is_left_to_the_publish_gate():
    entry = {"player_id": "x", "player_version": "x-1.0"}
    _refuse_version_drift(entry, _Fake(None))
    _refuse_version_drift(entry, _Fake(OSError("service down")))
