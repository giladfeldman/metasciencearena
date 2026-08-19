"""ScimetoMatchingAdapter — HTTP adapter for SciMeto's citation-to-reference
matching service.

Mirrors scimeto_citations.py: reads SCIMETO_API_URL and SCIMETO_API_KEY from
env (via the api_url_env / api_key_env kwargs), POSTs the PDF, and normalizes
the response into the pdf-citation-matching-v1 {markers, consistency} shape.

ASSUMPTION — endpoint path: SciMeto's exact citation-matching endpoint is not
documented in this repo. Where scimeto_citations.py defaults to
.../api/citations/extract, this matching adapter defaults to
.../api/citations/match — the matching counterpart of the extraction route.
If the real route differs, only _config()'s default URL needs changing.

ASSUMPTION — request shape: POSTs the PDF as a base64 JSON payload
{"document_bytes_b64": "..."}, matching how the arena's PDF task envelopes
carry the document (see grobid_references.py / llm_pdf.py / scimeto_citations.py).
If the real endpoint expects multipart, only play_task's request line needs
changing; the mapping function _scimeto_to_linkage is transport-independent.

ASSUMPTION — response shape: _scimeto_to_linkage accepts a dict with an
in-text-citations list (`in_text_citations`, `markers`, or `citations`) and an
optional `consistency` (or `report`) object. Every field read is defensive:
missing marker fields normalize to None / -1, missing consistency lists to [].
"""
from __future__ import annotations

import base64
import os
from urllib.parse import urljoin

import requests

from framework.player_adapter import PlayerAdapter, register_adapter_class

# Documented sentinel: char offset not provided by the upstream service.
_OFFSET_UNRESOLVED = -1


def _str_or_none(value):
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def _int_or_sentinel(value):
    """Coerce an offset to int; missing / unparseable -> the -1 sentinel."""
    if value is None:
        return _OFFSET_UNRESOLVED
    try:
        return int(value)
    except (TypeError, ValueError):
        return _OFFSET_UNRESOLVED


def _str_list(raw) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [str(x) for x in raw if x is not None]


def _group_list(raw) -> list[list[str]]:
    if not isinstance(raw, list):
        return []
    out: list[list[str]] = []
    for grp in raw:
        if isinstance(grp, list):
            out.append([str(x) for x in grp if x is not None])
    return out


def _normalize_marker(rec: dict) -> dict:
    """Coerce one SciMeto in-text-citation record to the arena marker schema."""
    return {
        "marker_text": str(rec.get("marker_text")
                            or rec.get("text")
                            or rec.get("marker") or ""),
        "char_start": _int_or_sentinel(rec.get("char_start")
                                       if rec.get("char_start") is not None
                                       else rec.get("start")),
        "char_end": _int_or_sentinel(rec.get("char_end")
                                     if rec.get("char_end") is not None
                                     else rec.get("end")),
        "reference_id": _str_or_none(rec.get("reference_id")
                                     or rec.get("ref_id")
                                     or rec.get("target")),
    }


def _scimeto_to_linkage(response: dict) -> dict:
    """Map a SciMeto citation-matching response to {markers, consistency}.

    Defensive: non-dict input or missing sections normalize to empty markers
    and an all-empty consistency report.
    """
    empty = {
        "markers": [],
        "consistency": {
            "orphan_markers": [],
            "uncited_reference_ids": [],
            "duplicate_reference_groups": [],
        },
    }
    if not isinstance(response, dict):
        return empty
    # Unwrap a {"success": true, "data": {...}} envelope if present.
    if response.get("success") and isinstance(response.get("data"), dict):
        response = response["data"]

    items = response.get("in_text_citations")
    if not isinstance(items, list):
        items = response.get("markers")
    if not isinstance(items, list):
        items = response.get("citations")
    if not isinstance(items, list):
        items = []
    markers = [_normalize_marker(r) for r in items if isinstance(r, dict)]

    cons_src = response.get("consistency")
    if not isinstance(cons_src, dict):
        cons_src = response.get("report")
    if not isinstance(cons_src, dict):
        cons_src = {}
    consistency = {
        "orphan_markers": _str_list(cons_src.get("orphan_markers")),
        "uncited_reference_ids": _str_list(cons_src.get("uncited_reference_ids")),
        "duplicate_reference_groups": _group_list(
            cons_src.get("duplicate_reference_groups")),
    }
    return {"markers": markers, "consistency": consistency}


class ScimetoMatchingAdapter(PlayerAdapter):
    """Adapter for SciMeto's citation-matching API. platform / deterministic."""

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
            url = "http://127.0.0.1:3001/api/citations/match"
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
        pdf_b64 = envelope["input"]["document_bytes_b64"]
        # Re-encode to canonical base64 (defensive — guarantees a clean payload).
        pdf_b64 = base64.b64encode(base64.b64decode(pdf_b64)).decode("ascii")
        body = {"document_bytes_b64": pdf_b64}
        headers = {"Content-Type": "application/json"}
        if key:
            headers["x-internal-api-key"] = key
            r = requests.post(url, json=body, headers=headers, timeout=timeout_s)
        else:
            session = requests.Session()
            if "/api/v1/" not in url:
                headers.update(self._csrf_headers(session, url))
            r = session.post(url, json=body, headers=headers, timeout=timeout_s)
        r.raise_for_status()
        data = r.json()
        return {
            **_scimeto_to_linkage(data),
            "player_strategy_notes": "scimeto-citation-matching-api",
        }


register_adapter_class("ScimetoMatchingAdapter", ScimetoMatchingAdapter)
