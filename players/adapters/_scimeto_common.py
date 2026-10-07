"""Shared version-provenance seam for the SciMeto players.

WHY THIS MODULE EXISTS (measured 2026-09-21). ``framework audit --versions``
printed, verbatim, for all three SciMeto entries then in the registry::

    [UNCHECKED:undeclared] scimeto-replication-api  declared=1.71.3
    [UNCHECKED:undeclared] scimeto-citations-api    declared=1.71.3
    [UNCHECKED:undeclared] scimeto-matching-api     declared=1.71.3

while the SciMeto running on this machine answered ``GET /health`` with
``{"status":"healthy","version":"1.74.106","gitSha":null}``. None of the three
adapter classes overrode ``resolved_tool_version()`` or
``version_absence_reason()``, so they inherited ``PlayerAdapter``'s "no version,
and no reason given" default. (Two of those three -- citation extraction and
citation-to-reference matching -- were retired on 2026-09-22 for pointing at
routes SciMeto has never served. The seam is unchanged; it now carries the
replication adapter, which serves both the 1.71.3 and 1.74.107 entries.) There was NO EVIDENCE SEAM AT ALL: a re-run today
would have stamped the declared label ``1.71.3`` onto numbers produced by
1.74.106 and not one check in this repository could have noticed -- the audit
documented as the guard against exactly that drift was structurally blind to it.

This is the same defect ``HttpAdapter._probe_health_version`` was written for on
the same day (ESCImate, declared 0.7.6 / installed 0.7.10), and the treatment is
deliberately identical so the two read the same way:

* evidence order -- a PLAYED TASK outranks the health probe, because the version
  on a task response is the build that produced THAT score and is what the
  runner stamps onto the record;
* ONE probe per adapter instance, never one per task, whether it succeeds or
  fails (a down service must not cost a timeout per task);
* best-effort, never raises -- provenance must not fail a task or an audit;
* ``probe_failed``, never the inherited ``undeclared``, because a version
  genuinely exists on the other side of this endpoint and we could not read it.

MEASURED, BOTH SIDES, before this module was written:

* 2026-09-21 ``GET http://127.0.0.1:3001/health``     -> 200
  ``{"status":"healthy",...,"version":"1.74.106","gitSha":null}``
* 2026-09-21 ``GET http://127.0.0.1:3001/api/health`` -> 404
* 2026-09-22 ``GET http://127.0.0.1:3001/health``     -> ``"version":"1.74.107"``
  after an overnight rebuild of the service, on the same machine.

That last line is the argument for this whole module in one observation: the
installed version moved a patch release in under a day, with nothing in this
repository able to see it. The declared label would have kept saying 1.71.3.

The health route is at the ORIGIN ROOT and NOT under ``/api``. That is why
the URL here is rebuilt from the scheme and host with ``urlsplit`` instead of
being derived by trimming the API path: ``SCIMETO_API_URL`` may be pointed at any
route, and a string trim that happened to miss ``/api/`` would silently probe
``<api-path>/health`` and report a live service as an outage.

TWO VERSION LINES, ONE OF WHICH IS COMPARABLE. SciMeto's ``/health`` reports
the product version, generated at build time from the service's own package
metadata -- 1.74.106 on 2026-09-21 and 1.74.107 on 2026-09-22, which is the line
``players/registry.yaml``'s ``1.71.3`` is declared on. The replications-lookup
response additionally carries ``data.engineVersion``: the build identity of the
WORKER that served it, ``<product version>+<deploy-commit>``, and deliberately
never the API tier's own constant, because the two are separately deployed and
do run different builds. Same numeric line, finer grain, and it is attached to
the response that produced the score, which is why it is preferred.

``unreported-by-worker`` is SciMeto's explicit marker for "the worker served
this request but named no build". It must never be returned as a version: it
carries no digits, ``framework/audit.py::_numeric`` would return None for it, and
``version_drift_from_registry`` would then record verdict OK -- a clean green for
a player whose version was never established. It is filtered here so the absence
travels as an absence.
"""
from __future__ import annotations

from urllib.parse import urlsplit

import requests

from framework.player_adapter import (
    VERSION_ABSENT_PROBE_FAILED,
    VersionAbsence,
)

#: How long the one-shot ``/health`` version probe waits. Longer than a liveness
#: ping on purpose, and for the reason measured on ESCImate the same day: this
#: probe runs ONCE and its answer is reported to a human as "the service is not
#: running", so a short timeout publishes a transient stall as an outage.
SCIMETO_HEALTH_TIMEOUT_S = 10

#: SciMeto's marker for "the worker answered but named no build". A string that
#: is not a version and must never be treated as one -- see the module docstring
#: for why returning it would manufacture a false OK.
ENGINE_VERSION_UNREPORTED = "unreported-by-worker"

#: Values that look like a version but assert nothing. ``unknown`` is what every
#: pre-2026-09-01 SciMeto worker sent on every response, so it is exactly the
#: string a stale service still emits.
_NON_VERSIONS = frozenset({ENGINE_VERSION_UNREPORTED, "unknown", "none", "null"})


def scimeto_health_url(api_url: str) -> str:
    """The ``/health`` URL for a SciMeto API url, rebuilt from scheme + host.

    Measured 2026-09-21: ``/health`` answers at the origin root and
    ``/api/health`` is a 404, so the path is discarded rather than trimmed.
    """
    parts = urlsplit(api_url)
    if not parts.scheme or not parts.netloc:
        raise ValueError(f"not an absolute http url: {api_url!r}")
    return f"{parts.scheme}://{parts.netloc}/health"


def _clean_version(value) -> str | None:
    """A non-empty version string, or None for anything that asserts nothing."""
    if not isinstance(value, str):
        return None
    v = value.strip()
    if not v or v.lower() in _NON_VERSIONS:
        return None
    return v


class ScimetoHealthVersionMixin:
    """``resolved_tool_version`` / ``version_absence_reason`` for a SciMeto adapter.

    Mixed in BEFORE ``PlayerAdapter`` so these two override the base defaults.
    It carries no ``__init__``: the state below is declared as class attributes
    (no annotations, so ``PlayerAdapter``'s dataclass machinery never sees them
    as fields) and shadowed per instance on first write. That also keeps the
    mixin correct for an instance built with ``cls.__new__(cls)``, which
    ``framework/tests/test_scimeto_local_only.py`` does.

    The host class must supply ``_config() -> (url, key)``.
    """

    # Set on the instance the first time each is established. See the class
    # docstring for why these are class attributes rather than __init__ state.
    _last_tool_version = None      # from a played task -- strongest evidence
    _health_probed = False
    _health_reachable = False
    _health_tool_version = None
    _health_probe_detail = None    # why the probe could not be made at all

    def _health_url(self) -> str:
        url, _key = self._config()
        return scimeto_health_url(url)

    def _capture_provenance(self, data) -> None:
        """Read the build the SERVICE says served this response.

        Best-effort and never raises: provenance must not fail a task.

        MEASURED 2026-09-21 against the local service: the replications-lookup
        route returns
        ``{"success":true,"data":{...,"engineVersion":"<version>+<sha>",...}}``.
It is written here rather than on the adapter because a
        per-adapter copy is how the three ``_require_local_scimeto`` copies
        already drifted apart in whitespace, and because any future SciMeto
        adapter should inherit the seam rather than re-derive it.
        """
        try:
            if not isinstance(data, dict):
                return
            if data.get("success") and isinstance(data.get("data"), dict):
                data = data["data"]
            v = _clean_version(data.get("engineVersion"))
            if v is None:
                v = _clean_version(data.get("version"))
            if v is not None:
                self._last_tool_version = v
        except Exception:
            pass

    def _probe_health_version(self) -> None:
        """Read the service's version from ``/health``, WITHOUT playing a task.

        This is the whole point of the module: ``framework audit --versions``
        builds every adapter and probes it without running a single task, so an
        adapter whose only version source is a task response is invisible to it.

        Records BOTH outcomes -- whether the service answered at all
        (``_health_reachable``) and what version it named -- because
        ``version_absence_reason`` has to tell an outage apart from a service
        that answered and named nothing. Those are two different facts and this
        project has a standing rule against rendering them as one sentence.
        """
        self._health_probed = True
        try:
            url = self._health_url()
        except Exception as exc:
            # Includes the local-only guard refusing a hosted SCIMETO_API_URL.
            # An honest "we did not probe", never a silent pass.
            self._health_probe_detail = f"{type(exc).__name__}: {exc}"
            return
        try:
            r = requests.get(url, timeout=SCIMETO_HEALTH_TIMEOUT_S)
            if r.status_code >= 500:
                return
            body = r.json()
        except Exception:
            return
        self._health_reachable = True
        if not isinstance(body, dict):
            return
        self._health_tool_version = _clean_version(body.get("version"))

    def resolved_tool_version(self) -> str | None:
        """The version the SERVICE reports, not the one registry.yaml declares.

        Two sources, in order of evidential strength:

        1. the build carried on the response to a task we actually played --
           that is the version that produced THIS score, and the runner stamps it
           onto the record;
        2. failing that, ``/health``, probed at most once per adapter instance.

        The order matters for what lands on a RECORD: source 2 can only ever
        reach a record of a task that produced no answer at all, where the
        alternative was None.
        """
        if self._last_tool_version:
            return self._last_tool_version
        if not self._health_probed:
            self._probe_health_version()
        return self._health_tool_version

    def version_absence_reason(self) -> VersionAbsence:
        """Why no version: an unprobed endpoint, an OUTAGE, or a silent service.

        All three are ``probe_failed`` -- a version genuinely exists on the other
        side of this endpoint in every one of them and we could not read it --
        but the detail says which, so a reader is never left guessing whether the
        server is down. Never the inherited ``undeclared``, which is what these
        three adapters reported for months while a live 1.71.3-declared /
        1.74.106-installed drift sat in front of them.
        """
        if not self._health_probed:
            self._probe_health_version()
        if self._health_probe_detail is not None:
            return VersionAbsence(
                VERSION_ABSENT_PROBE_FAILED,
                f"the SciMeto health url could not be built, so no version probe "
                f"was made: {self._health_probe_detail}",
            )
        if not self._health_reachable:
            try:
                url = self._health_url()
            except Exception:
                url = "<unresolvable>"
            return VersionAbsence(
                VERSION_ABSENT_PROBE_FAILED,
                f"{url} did not answer within {SCIMETO_HEALTH_TIMEOUT_S}s -- "
                "SciMeto is not running, or is not responding. Start its local "
                "server (start_app.bat --headless) and re-run the probe.",
            )
        return VersionAbsence(
            VERSION_ABSENT_PROBE_FAILED,
            f"{self._health_url()} answered but named no version. This service "
            "reports one on `version`; losing it turns every drift check into a "
            "green that tests nothing.",
        )
