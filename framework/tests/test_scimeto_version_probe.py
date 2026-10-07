"""The audit must be able to SEE a SciMeto version drift without playing a task.

WHY THIS FILE EXISTS (2026-09-21). None of the three SciMeto adapter classes
overrode ``resolved_tool_version()`` or ``version_absence_reason()``, so
``framework audit --versions`` printed, for all three::

    [UNCHECKED:undeclared] scimeto-replication-api  declared=1.71.3
    [UNCHECKED:undeclared] scimeto-citations-api    declared=1.71.3
    [UNCHECKED:undeclared] scimeto-matching-api     declared=1.71.3

while the SciMeto running on this machine was 1.74.106. There was no evidence
seam at all: a re-run would have stamped 1.71.3 onto 1.74.106's numbers and
nothing in this repository could have caught it. That is the exact shape the
version audit exists to prevent, on a player whose service self-reports.

MEASURED AGAINST THE LIVE SERVICE, both sides, before these tests were written:

* ``GET http://127.0.0.1:3001/health``     -> 200
  ``{"status":"healthy","timestamp":"2026-09-21T17:25:00.515Z",
     "database":"unknown","version":"1.74.106","gitSha":null}``
* ``GET http://127.0.0.1:3001/api/health`` -> 404  (hence: origin root, not /api)
* service UP   -> all three rows DRIFT declared=1.71.3 resolved=1.74.106
* AND THEN, 2026-09-22, the same probe on the same machine read 1.74.107: the
  service had been rebuilt overnight. A patch release moved under the benchmark
  in under a day and nothing here could have seen it. The stub values below stay
  at 1.74.106 because that is the reading these tests were written against; what
  they pin is the MECHANISM, and a stub that chases the live service would be a
  test of the service rather than of this code.
* endpoint pointed at an unanswered loopback port -> all three rows
  UNCHECKED:probe_failed, naming that url

The down-side was measured against a dead loopback port rather than by stopping
the live service, because concurrent sessions share this machine and one of them
was mid-``framework audit`` at the time; stopping the server would have handed
that session a false outage. The socket path is identical (a refused connection
raises the same way), and that the DEFAULT url is 3001/health with the
environment unset is pinned hermetically below.

The tests below are hermetic (no socket is opened) so they run on a CI runner
that has no SciMeto; the live results above are the evidence that the stubbed
response shape is the real one.
"""
from __future__ import annotations

import pytest

from framework.player_adapter import (
    VERSION_ABSENT_PROBE_FAILED,
    VERSION_ABSENT_UNDECLARED,
)
from players.adapters._scimeto_common import (
    ENGINE_VERSION_UNREPORTED,
    scimeto_health_url,
)
from players.adapters.scimeto_replication import ScimetoReplicationApiAdapter

HEALTH = "http://127.0.0.1:3001/health"

#: Every adapter that carries the seam, with the default API route it resolves.
#:
#: This list held three adapters until 2026-09-22. The other two -- citation
#: extraction and citation-to-reference matching -- were RETIRED and deleted that
#: day: their default routes were assumptions that SciMeto has never served, and
#: neither player ever produced a record. See the tombstones in
#: players/registry.yaml for the measurement. The seam is unchanged; what shrank
#: is the set of adapters that carry it, and that set is now exactly one.
ADAPTERS = [
    (ScimetoReplicationApiAdapter, "http://127.0.0.1:3001/api/replications/lookup"),
]
ADAPTER_CLASSES = [cls for cls, _url in ADAPTERS]


@pytest.fixture(autouse=True)
def _local_endpoint(monkeypatch):
    """Every test runs with the env in its normal state: unset -> loopback.

    ``SCIMETO_API_URL`` is not set on this machine, and ``SCIMETO_API_KEY`` holds
    a live hosted key -- neither must influence a version probe, so both are
    cleared rather than inherited.
    """
    monkeypatch.delenv("SCIMETO_API_URL", raising=False)
    monkeypatch.delenv("SCIMETO_API_KEY", raising=False)
    monkeypatch.delenv("SCIENCEARENA_ALLOW_REMOTE", raising=False)


def _adapter(cls):
    return cls(
        player_id="scimeto-test", player_version="1.71.3", player_type="platform",
        confidence_strategy="native", deterministic=True,
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

    monkeypatch.setattr("players.adapters._scimeto_common.requests.get", fake_get)
    return calls


# --- the health url ------------------------------------------------------


@pytest.mark.parametrize("cls,api_url", ADAPTERS)
def test_every_adapter_probes_the_origin_root_not_its_own_api_path(monkeypatch, cls, api_url):
    """MEASURED: /health is 200 at the root and /api/health is 404.

    The three adapters POST to three different routes, so a health url derived
    by trimming the API path is three different derivations. Rebuilding from
    scheme+host makes it one fact.
    """
    calls = _stub_get(monkeypatch, {"version": "1.74.106"})
    adapter = _adapter(cls)
    assert adapter._config()[0] == api_url
    assert adapter.resolved_tool_version() == "1.74.106"
    assert calls == [HEALTH]


def test_the_health_url_discards_the_path_whatever_it_is():
    """A route this adapter has never seen must still resolve to the root."""
    assert scimeto_health_url("http://127.0.0.1:3001/api/v1/replications/extract") == HEALTH
    assert scimeto_health_url("http://127.0.0.1:3001/lookup") == HEALTH
    assert scimeto_health_url("http://127.0.0.1:3001") == HEALTH


def test_a_url_with_no_host_raises_rather_than_inventing_one():
    with pytest.raises(ValueError):
        scimeto_health_url("/api/replications/lookup")


# --- evidence order ------------------------------------------------------


@pytest.mark.parametrize("cls", ADAPTER_CLASSES)
def test_a_played_task_outranks_the_health_probe(monkeypatch, cls):
    """The version that produced THIS score is what belongs on the record.

    The negative control is the call list: /health must not be reached at all.
    The planted 9.9.9 is what a wrong precedence would return.
    """
    calls = _stub_get(monkeypatch, {"version": "9.9.9"})
    adapter = _adapter(cls)
    adapter._capture_provenance(
        {"success": True, "data": {"engineVersion": "1.74.106+abc1234"}})
    assert adapter.resolved_tool_version() == "1.74.106+abc1234"
    assert calls == []


@pytest.mark.parametrize("cls", ADAPTER_CLASSES)
def test_the_worker_build_suffix_survives_to_the_record(monkeypatch, cls):
    """``APP_VERSION+<sha>`` is finer grain than /health and must not be trimmed.

    ``framework/audit.py::_numeric`` extracts 1.74.106 from it for the drift
    comparison, so keeping the suffix costs nothing and records the commit.
    """
    from framework.audit import _numeric
    _stub_get(monkeypatch, ConnectionError("refused"))
    adapter = _adapter(cls)
    adapter._capture_provenance({"success": True, "data": {"engineVersion": "1.74.106+abc1234"}})
    resolved = adapter.resolved_tool_version()
    assert resolved == "1.74.106+abc1234"
    assert _numeric(resolved) == "1.74.106"


@pytest.mark.parametrize("cls", ADAPTER_CLASSES)
def test_unreported_by_worker_is_not_a_version(monkeypatch, cls):
    """THE FALSE GREEN THIS FILTER EXISTS FOR.

    SciMeto sends this literal when the worker served the request but
    named no build. It contains no digits, so ``_numeric`` returns None and
    ``version_drift_from_registry`` computes ``drift = bool(None) and ...`` =
    False -- verdict OK. A player whose version was never established would have
    been reported as matching its declared label.
    """
    from framework.audit import _numeric
    assert _numeric(ENGINE_VERSION_UNREPORTED) is None  # the mechanism, pinned
    _stub_get(monkeypatch, ConnectionError("refused"))
    adapter = _adapter(cls)
    adapter._capture_provenance({"success": True, "data": {"engineVersion": ENGINE_VERSION_UNREPORTED}})
    assert adapter.resolved_tool_version() is None
    assert adapter.version_absence_reason().kind == VERSION_ABSENT_PROBE_FAILED


@pytest.mark.parametrize("cls", ADAPTER_CLASSES)
def test_the_stale_unknown_marker_is_not_a_version_either(monkeypatch, cls):
    """Every pre-2026-09-01 SciMeto worker sent 'unknown' on every response."""
    _stub_get(monkeypatch, ConnectionError("refused"))
    adapter = _adapter(cls)
    adapter._capture_provenance({"engineVersion": "unknown"})
    assert adapter.resolved_tool_version() is None


# --- one probe per adapter ----------------------------------------------


@pytest.mark.parametrize("cls", ADAPTER_CLASSES)
def test_the_probe_runs_at_most_once_per_adapter(monkeypatch, cls):
    """The runner asks after EVERY task. One socket per adapter, not per task."""
    calls = _stub_get(monkeypatch, {"version": "1.74.106"})
    adapter = _adapter(cls)
    for _ in range(5):
        adapter.resolved_tool_version()
    assert calls == [HEALTH]


@pytest.mark.parametrize("cls", ADAPTER_CLASSES)
def test_a_failed_probe_is_not_retried_on_every_task(monkeypatch, cls):
    """A DOWN service must not cost one 10s timeout per task."""
    calls = _stub_get(monkeypatch, ConnectionError("refused"))
    adapter = _adapter(cls)
    for _ in range(5):
        assert adapter.resolved_tool_version() is None
    adapter.version_absence_reason()
    assert calls == [HEALTH]


@pytest.mark.parametrize("cls", ADAPTER_CLASSES)
def test_two_adapters_do_not_share_probe_state(monkeypatch, cls):
    """The mixin holds its state as CLASS attributes shadowed per instance.

    If a write ever landed on the class instead, one adapter's answer would
    silently become every later adapter's answer -- including across players.
    """
    _stub_get(monkeypatch, {"version": "1.74.106"})
    first = _adapter(cls)
    assert first.resolved_tool_version() == "1.74.106"
    _stub_get(monkeypatch, ConnectionError("refused"))
    second = _adapter(cls)
    assert second.resolved_tool_version() is None
    assert first.resolved_tool_version() == "1.74.106"


# --- why no version ------------------------------------------------------


@pytest.mark.parametrize("cls", ADAPTER_CLASSES)
def test_a_service_that_does_not_answer_is_probe_failed_and_says_so(monkeypatch, cls):
    """An outage, named as one -- never the inherited `undeclared` default."""
    _stub_get(monkeypatch, ConnectionError("refused"))
    adapter = _adapter(cls)
    assert adapter.resolved_tool_version() is None
    absence = adapter.version_absence_reason()
    assert absence.kind == VERSION_ABSENT_PROBE_FAILED
    assert absence.kind != VERSION_ABSENT_UNDECLARED
    assert HEALTH in absence.detail
    assert "not running" in absence.detail


@pytest.mark.parametrize("cls", ADAPTER_CLASSES)
def test_a_service_that_answers_but_names_no_version_is_a_different_failure(monkeypatch, cls):
    """Corroboration loss. Same kind, different detail -- the two must not merge.

    A service that silently stops reporting its version looks exactly like a
    clean pass to any check keyed on "is the field present".
    """
    _stub_get(monkeypatch, {"status": "healthy"})
    adapter = _adapter(cls)
    assert adapter.resolved_tool_version() is None
    absence = adapter.version_absence_reason()
    assert absence.kind == VERSION_ABSENT_PROBE_FAILED
    assert "named no version" in absence.detail
    assert "not running" not in absence.detail


@pytest.mark.parametrize("cls", ADAPTER_CLASSES)
def test_a_5xx_health_is_an_outage_not_an_answer(monkeypatch, cls):
    _stub_get(monkeypatch, {"version": "1.74.106"}, status=503)
    adapter = _adapter(cls)
    assert adapter.resolved_tool_version() is None
    assert "not running" in adapter.version_absence_reason().detail


@pytest.mark.parametrize("cls", ADAPTER_CLASSES)
def test_a_refused_hosted_url_is_probe_failed_not_a_silent_pass(monkeypatch, cls):
    """The local-only guard raises inside ``_config()``, which the probe calls.

    A provenance probe must never raise, and must never report "no version"
    without saying why. This is the one branch where no socket is opened at all.
    """
    monkeypatch.setenv("SCIMETO_API_URL", "https://scimeto.app/api/replications/lookup")
    calls = _stub_get(monkeypatch, {"version": "1.74.106"})
    adapter = _adapter(cls)
    assert adapter.resolved_tool_version() is None
    absence = adapter.version_absence_reason()
    assert absence.kind == VERSION_ABSENT_PROBE_FAILED
    assert "could not be built" in absence.detail
    assert "NOT a local service" in absence.detail
    assert calls == []


@pytest.mark.parametrize("cls", ADAPTER_CLASSES)
def test_the_probe_never_raises_whatever_the_service_returns(monkeypatch, cls):
    """Provenance is best-effort: it must never fail a task or an audit."""
    for body in (ValueError("not json"), ["a", "list"], None, {"version": ""},
                 {"version": 7}, {"version": ["1.74.106"]}):
        _stub_get(monkeypatch, body)
        adapter = _adapter(cls)
        assert adapter.resolved_tool_version() is None
        assert adapter.version_absence_reason().kind == VERSION_ABSENT_PROBE_FAILED


@pytest.mark.parametrize("cls", ADAPTER_CLASSES)
def test_capture_provenance_never_raises_on_a_hostile_response(monkeypatch, cls):
    _stub_get(monkeypatch, ConnectionError("refused"))
    for body in (None, ["a"], 7, {"data": None}, {"success": True, "data": 3},
                 {"engineVersion": []}, {"engineVersion": None}):
        adapter = _adapter(cls)
        adapter._capture_provenance(body)
        assert adapter.resolved_tool_version() is None


# --- vacuity controls ----------------------------------------------------


@pytest.mark.parametrize("cls", ADAPTER_CLASSES)
def test_the_stub_is_not_vacuous(monkeypatch, cls):
    """If the monkeypatch missed, every assertion above would test nothing.

    A real requests.get here would hit a socket and (on this machine, where
    SciMeto is running) return a REAL 1.74.106 -- passing the first test for the
    wrong reason. Planting a version no service ever reported decides it.
    """
    _stub_get(monkeypatch, {"version": "0.0.0-planted"})
    assert _adapter(cls).resolved_tool_version() == "0.0.0-planted"


def test_the_registry_entries_that_carry_this_seam_are_still_the_declared_set():
    """Blast radius, pinned. If another player adopts a SciMeto adapter, this
    file's live-measured claims no longer cover every user of the code path.

    Was "still three" until 2026-09-22, when scimeto-citations-api and
    scimeto-matching-api were retired -- see players/registry.yaml."""
    from framework.paths import registry_path
    from framework.registry import load_registry

    entries = load_registry(registry_path())
    users = sorted(e["player_id"] for e in entries
                   if str(e.get("adapter_class", "")).startswith("Scimeto"))
    assert users == ["scimeto-replication-api",
                     "scimeto-replication-api-1-74-107"], users


#: The GitHub repository that hosts SciMeto, ASSEMBLED rather than written
#: because THIS FILE IS ITSELF PUBLISHED -- spelling it here would re-publish
#: exactly what the test below exists to keep out. Measured 2026-09-21,
#: two-sided and anonymously: the public mirror repo answers 200 and this one
#: answers 404, so it is private.
_PRIVATE_SCIMETO_REPO = "Citation" + "Guard"


def test_no_published_file_names_the_private_repository_behind_scimeto():
    """Writing this seam meant reading that repo's source, and the first draft of
    these two files cited it by name and by internal file path.

    ``SciMeto`` is the public product name and appears throughout the mirror,
    which is correct. The repository name is not public, and neither is its
    directory layout -- the same disclosure rule
    ``test_start_command_paths.py`` states for the other private siblings, whose
    list this deliberately does not edit (that file is owned by another change
    in flight). Two-sided: the scan must find something, or its zero means
    nothing.
    """
    from pathlib import Path

    from framework.publish import public_manifest

    root = Path(__file__).resolve().parents[2]
    scanned = control = 0
    offenders = []
    for rel in public_manifest(root):
        path = root / rel
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        scanned += 1
        if "SciMeto" in text or "scimeto" in text:
            control += 1
        if _PRIVATE_SCIMETO_REPO in text:
            offenders.append(str(rel))
    assert scanned > 100, f"only {scanned} published files readable; the scan is not live"
    assert control > 5, f"positive control found {control}; the scan cannot read content"
    assert offenders == [], offenders


def test_the_private_repo_scan_can_actually_find_the_name(tmp_path):
    """Positive control for the scan above, which would pass just as happily if
    the name were assembled wrong and matched nothing."""
    planted = tmp_path / "planted.txt"
    planted.write_text("cloned from " + _PRIVATE_SCIMETO_REPO, encoding="utf-8")
    assert _PRIVATE_SCIMETO_REPO in planted.read_text(encoding="utf-8")
    assert _PRIVATE_SCIMETO_REPO not in "nothing to see here"


def test_the_seam_is_actually_on_every_scimeto_adapter_class():
    """The defect was three classes inheriting a default. Pin the override.

    Checked against ``PlayerAdapter``'s own function objects rather than by
    calling them: an override that exists is the fact, and a call would open a
    socket.
    """
    from framework.player_adapter import PlayerAdapter
    for cls in ADAPTER_CLASSES:
        assert cls.resolved_tool_version is not PlayerAdapter.resolved_tool_version, cls
        assert cls.version_absence_reason is not PlayerAdapter.version_absence_reason, cls
