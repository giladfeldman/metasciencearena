"""ScimetoReplicationApiAdapter — HTTP adapter for SciMeto's replications API.

Reads SCIMETO_API_URL and SCIMETO_API_KEY from env. Sends one DOI at a time
with the requested direction; normalizes the response into the arena's output
schema. If no API key is configured, it falls back to the local browser-compatible
route and fetches a CSRF token.
"""
from __future__ import annotations

import os
from urllib.parse import urljoin

import requests

from framework.player_adapter import PlayerAdapter, register_adapter_class
from players.adapters._scimeto_common import ScimetoHealthVersionMixin


def _require_local_scimeto(url: str, env_name: str) -> str:
    """Refuse a non-loopback Scimeto endpoint unless explicitly permitted.

    These adapters read the endpoint from the environment at CALL time, so the
    value never reached ``is_cloud_player()`` -- which classifies from the
    endpoint DECLARED in registry.yaml, and these three players declare none.
    ``SCIMETO_API_URL=https://scimeto.app`` would therefore have billed the
    hosted service on every task with no gate firing at all.

    Standing user directive: cross-testing is ALWAYS local unless the user
    approves otherwise; hosted calls are metered and the user pays personally.
    Override with SCIENCEARENA_ALLOW_REMOTE=1, which is deliberately loud.
    """
    from framework.runner import _is_loopback_endpoint
    if _is_loopback_endpoint(url):
        return url
    if os.environ.get("SCIENCEARENA_ALLOW_REMOTE") == "1":
        print(f"[local-only] WARNING: {env_name} is REMOTE ({url}). "
              f"This is metered and costs money.", flush=True)
        return url
    raise RuntimeError(
        f"{env_name} is {url!r}, which is NOT a local service. "
        f"Cross-testing must run against a LOCAL Scimeto (user directive); "
        f"hosted calls are metered and the user pays personally. "
        f"Start Scimeto locally (start_app.bat -> 127.0.0.1:3001), "
        f"or set SCIENCEARENA_ALLOW_REMOTE=1 to override."
    )



class ScimetoReplicationApiAdapter(ScimetoHealthVersionMixin, PlayerAdapter):
    api_url_env: str
    api_key_env: str

    def __init__(self, *args, api_url_env: str = "SCIMETO_API_URL",
                 api_key_env: str = "SCIMETO_API_KEY", **kwargs):
        super().__init__(*args, **kwargs)
        self.api_url_env = api_url_env
        self.api_key_env = api_key_env

    def _config(self) -> tuple[str, str | None]:
        url = os.environ.get(self.api_url_env)
        if not url:
            url = "http://127.0.0.1:3001/api/replications/lookup"
        # WIRED 2026-08-28. This guard was written 2026-08-22 and had ZERO call
        # sites for six days -- the whole point of it is this one line.
        url = _require_local_scimeto(url, self.api_url_env)
        key = os.environ.get(self.api_key_env) or None
        return url, key

    def _csrf_headers(self, session: requests.Session, url: str) -> dict:
        root = url.split("/api/", 1)[0].rstrip("/") + "/"
        token_url = urljoin(root, "api/csrf-token")
        r = session.get(token_url, timeout=30)
        r.raise_for_status()
        token = r.json().get("csrfToken")
        return {"CSRF-Token": token} if token else {}

    def play_task(self, envelope: dict, *, timeout_s: int) -> dict:
        url, key = self._config()
        direction = envelope["input"]["direction"]
        doi = envelope["input"]["doi"]
        body = {"direction": direction, "dois": [doi], "includeProvenance": True, "useLlmVerifier": False}
        headers = {"Content-Type": "application/json"}
        if key:
            headers["x-internal-api-key"] = key
        if key:
            r = requests.post(url, json=body, headers=headers, timeout=timeout_s)
        else:
            session = requests.Session()
            if "/api/v1/" not in url:
                headers.update(self._csrf_headers(session, url))
            r = session.post(url, json=body, headers=headers, timeout=timeout_s)
        r.raise_for_status()
        data = r.json()
        # The build that produced THIS answer, read off the response. Stronger
        # evidence than the /health probe, and what the runner stamps onto the
        # record -- see players/adapters/_scimeto_common.py.
        self._capture_provenance(data)
        return self._normalize(data, direction, doi)

    def _normalize(self, response: dict, direction: str, input_doi: str) -> dict:
        if response.get("success") and isinstance(response.get("data"), dict):
            response = response["data"]
        matches = []
        for rec in response.get("records", []) or []:
            if rec.get("status") and rec.get("status") not in ("accepted", "verified", "ok"):
                continue
            method = rec.get("matchMethod") or rec.get("matchmethod") or ""
            provenance = []
            if method:
                provenance.append(method)
            provenance.append("scimeto")
            confidence = rec.get("confidence", 1.0)
            if isinstance(confidence, str):
                confidence = {"high": 0.95, "medium": 0.7, "low": 0.4}.get(confidence.lower(), 1.0)
            matches.append({
                "replication_doi": rec.get("replicationDoi"),
                "original_doi": rec.get("originalDoi"),
                "outcome": rec.get("outcome", "") or "",
                "confidence": float(confidence or 1.0),
                "evidence": rec.get("justificationPhrase", "") or "",
                "provenance": provenance,
                "abstained": False,
            })
        return {
            "direction": direction,
            "input_doi": input_doi,
            "matches": matches,
            "player_strategy_notes": "scimeto-replication-api",
        }


register_adapter_class("ScimetoReplicationApiAdapter", ScimetoReplicationApiAdapter)
