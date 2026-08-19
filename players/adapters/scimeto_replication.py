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


class ScimetoReplicationApiAdapter(PlayerAdapter):
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
