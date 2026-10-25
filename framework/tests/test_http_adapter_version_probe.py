"""The audit must be able to SEE an ESCImate version drift without playing a task.

WHY THIS FILE EXISTS (2026-09-21). ``HttpAdapter.resolved_tool_version()`` read
the version only off a ``/process-text`` RESPONSE, so it returned ``None`` until
a task had been played. ``framework audit --versions`` builds every adapter and
probes it WITHOUT running a task, so for the one player in this registry whose
entire purpose is to self-report its version it printed::

    [UNCHECKED:undeclared] escimate  declared=0.7.6

while the live service was running 0.7.10. The audit documented as the guard
against exactly that drift was structurally blind to it, and its UNCHECKED row
read identically whether the service was DOWN or merely idle -- two different
facts in one sentence, which is the ambiguity the VersionAbsence vocabulary in
``framework/player_adapter.py`` exists to remove.

MEASURED AGAINST THE LIVE SERVICE, both sides, before these tests were written:

* service up   -> ``escimate`` DRIFT declared=0.7.6 resolved=0.7.10;
  ``escimate-0-7-10`` OK declared=0.7.10 resolved=0.7.10
* service down -> UNCHECKED:probe_failed naming the unanswered health url

The tests below are hermetic (no socket is opened) so they run on a CI runner
that has no ESCImate; the live numbers above are the evidence that the stubbed
response shape is the real one. ``/health`` really answers
``{"status":["healthy"],"version":["0.7.10"],"engine_versions":{...}}`` -- jsonlite
boxes R length-1 vectors, which is why the unwrapping is tested.
"""
from __future__ import annotations

from framework.player_adapter import (
    VERSION_ABSENT_PROBE_FAILED,
    VERSION_ABSENT_UNDECLARED,
    HttpAdapter,
)

ENDPOINT = "http://127.0.0.1:9422/api/v1/process-text"
HEALTH = "http://127.0.0.1:9422/health"


def _adapter() -> HttpAdapter:
    return HttpAdapter(
        player_id="escimate-test", player_version="0.7.6", player_type="platform",
        confidence_strategy="implicit-1.0", deterministic=True, endpoint=ENDPOINT,
    )


class _Resp:
    def __init__(self, body, status=200):
        self._body = body
        self.status_code = status

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


def _stub_get(monkeypatch, body, status=200):
    """Replace requests.get and return the list of urls it was asked for."""
    calls: list[str] = []

    def fake_get(url, timeout=None, **kwargs):
        calls.append(url)
        if isinstance(body, Exception):
            raise body
        return _Resp(body, status)

    monkeypatch.setattr("framework.player_adapter.requests.get", fake_get)
    return calls


def test_health_supplies_the_version_when_no_task_has_been_played(monkeypatch):
    """The whole point: the audit never plays a task, and must still see 0.7.10."""
    calls = _stub_get(monkeypatch, {"status": ["healthy"], "version": ["0.7.10"]})
    assert _adapter().resolved_tool_version() == "0.7.10"
    assert calls == [HEALTH]


def test_a_bare_string_version_is_read_too(monkeypatch):
    """jsonlite boxing is a service detail, not a contract. Both shapes work."""
    _stub_get(monkeypatch, {"version": "0.7.10"})
    assert _adapter().resolved_tool_version() == "0.7.10"


def test_engine_versions_is_the_fallback_when_version_is_absent(monkeypatch):
    _stub_get(monkeypatch, {"engine_versions": {"effectcheck": "0.7.10", "r": "4.4.0"}})
    assert _adapter().resolved_tool_version() == "0.7.10"


def test_a_played_task_outranks_the_health_probe(monkeypatch):
    """A task that ANSWERED is stronger evidence, and is what lands on a record.

    The negative control is the call list: /health must not be reached at all,
    because a record has to carry the version that produced THAT score.
    """
    calls = _stub_get(monkeypatch, {"version": ["9.9.9"]})
    adapter = _adapter()
    adapter._capture_provenance({"effectcheck_version": ["0.7.6"]})
    assert adapter.resolved_tool_version() == "0.7.6"
    assert calls == []


def test_the_probe_runs_at_most_once_per_adapter(monkeypatch):
    """The runner asks after EVERY task. One socket per adapter, not per task."""
    calls = _stub_get(monkeypatch, {"version": ["0.7.10"]})
    adapter = _adapter()
    for _ in range(5):
        adapter.resolved_tool_version()
    assert calls == [HEALTH]


def test_a_failed_probe_is_not_retried_on_every_task(monkeypatch):
    """A DOWN service must not cost one timeout per task either."""
    calls = _stub_get(monkeypatch, ConnectionError("refused"))
    adapter = _adapter()
    for _ in range(5):
        assert adapter.resolved_tool_version() is None
    assert calls == [HEALTH]


def test_a_service_that_does_not_answer_is_probe_failed_and_says_so(monkeypatch):
    """An outage, named as one -- never the inherited `undeclared` default."""
    _stub_get(monkeypatch, ConnectionError("refused"))
    adapter = _adapter()
    assert adapter.resolved_tool_version() is None
    absence = adapter.version_absence_reason()
    assert absence.kind == VERSION_ABSENT_PROBE_FAILED
    assert absence.kind != VERSION_ABSENT_UNDECLARED
    assert HEALTH in absence.detail
    assert "not running" in absence.detail


def test_a_service_that_answers_but_names_no_version_is_a_different_failure(monkeypatch):
    """Corroboration loss. Same kind, different detail -- the two must not merge.

    This is the branch that matters most: a service that silently stops
    reporting its version looks exactly like a clean pass to any check keyed on
    "is the field present".
    """
    _stub_get(monkeypatch, {"status": ["healthy"]})
    adapter = _adapter()
    assert adapter.resolved_tool_version() is None
    absence = adapter.version_absence_reason()
    assert absence.kind == VERSION_ABSENT_PROBE_FAILED
    assert "named no version" in absence.detail
    assert "not running" not in absence.detail


def test_a_5xx_health_is_an_outage_not_an_answer(monkeypatch):
    _stub_get(monkeypatch, {"version": ["0.7.10"]}, status=503)
    adapter = _adapter()
    assert adapter.resolved_tool_version() is None
    assert "not running" in adapter.version_absence_reason().detail


def test_the_probe_never_raises_whatever_the_service_returns(monkeypatch):
    """Provenance is best-effort: it must never fail a task or an audit."""
    for body in (ValueError("not json"), ["a", "list"], None, {"version": []},
                 {"version": 7}, {"engine_versions": "not a dict"}):
        _stub_get(monkeypatch, body)
        assert _adapter().resolved_tool_version() is None


def test_the_stub_is_not_vacuous(monkeypatch):
    """If the monkeypatch missed, every assertion above would test nothing.

    A real requests.get here would hit a socket and (on a machine with ESCImate
    running) return a REAL 0.7.10 -- passing the first test for the wrong reason.
    Planting a version no service ever reported decides it.
    """
    _stub_get(monkeypatch, {"version": ["0.0.0-planted"]})
    assert _adapter().resolved_tool_version() == "0.0.0-planted"


def test_the_registry_entries_that_use_this_adapter_are_still_two():
    """Blast radius, pinned. HttpAdapter serves the ESCImate entries only.

    If another player adopts it, this file's live-measured claims no longer
    cover every user of the code path and the new one needs its own evidence.
    escimate-0-7-19 (2026-10-14) has it: all 72 of its stats-extraction-v1 v5
    records carry resolved_tool_version "0.7.19", read off the running service.
    (The name says "two"; it is kept so the test's history stays searchable.)
    """
    from framework.paths import registry_path
    from framework.registry import load_registry

    users = sorted(e["player_id"] for e in load_registry(registry_path())
                   if e.get("adapter_class") == "HttpAdapter")
    assert users == ["escimate", "escimate-0-7-10", "escimate-0-7-19"], users
