"""ScimetoCitationsAdapter — HTTP adapter for SciMeto's citation/reference
extraction service.

Mirrors scimeto_replication.py: reads SCIMETO_API_URL and SCIMETO_API_KEY from
env (via the api_url_env / api_key_env kwargs). play_task POSTs the PDF and
normalizes the response into the arena reference schema.

ASSUMPTION — request shape: SciMeto's exact citation-extraction endpoint shape
is not documented in this repo. This adapter POSTs the PDF as a base64 JSON
payload: {"document_bytes_b64": "..."} — matching how the arena's PDF task
envelopes carry the document (see grobid_references.py / llm_pdf.py). If the
real endpoint expects multipart, only play_task's request line needs changing;
the mapping function _scimeto_to_references is independent of transport.

ASSUMPTION — response shape: _scimeto_to_references accepts a dict with a
`references` (or `citations`) list. Each item may carry author/year/title/
venue/volume/issue/page/doi/pmid fields under common aliases. All field reads
are defensive — missing fields normalize to null.
"""
from __future__ import annotations

import base64
import os
import re
from urllib.parse import urljoin

import requests

from framework.player_adapter import PlayerAdapter, register_adapter_class


def _str_or_none(value):
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def _year_of(value):
    s = _str_or_none(value)
    if not s:
        return None
    m = re.search(r"\d{4}", s)
    return m.group(0) if m else s


def _split_pages(value):
    s = _str_or_none(value)
    if not s:
        return None, None
    parts = re.split(r"[-–—]", s, maxsplit=1)
    return parts[0].strip() or None, (parts[1].strip() if len(parts) > 1 else None)


def _normalize_authors(raw) -> list[dict]:
    """Map SciMeto's author list (tolerating several key conventions)."""
    out: list[dict] = []
    if not isinstance(raw, list):
        return out
    for a in raw:
        if isinstance(a, dict):
            surname = _str_or_none(a.get("surname") or a.get("family")
                                   or a.get("last") or a.get("last_name"))
            given = _str_or_none(a.get("given_names") or a.get("given")
                                 or a.get("first") or a.get("first_name"))
        elif isinstance(a, str):
            surname, given = _str_or_none(a), None
        else:
            continue
        if surname or given:
            out.append({"surname": surname, "given_names": given})
    return out


def _scimeto_to_references(response: dict) -> list[dict]:
    """Map a SciMeto citation-extraction response to the arena reference schema.

    Accepts a dict with a `references` or `citations` list. Defensive: every
    field is optional and normalizes to null when absent.
    """
    if not isinstance(response, dict):
        return []
    # Unwrap a {"success": true, "data": {...}} envelope if present.
    if response.get("success") and isinstance(response.get("data"), dict):
        response = response["data"]
    items = response.get("references")
    if not isinstance(items, list):
        items = response.get("citations")
    if not isinstance(items, list):
        return []

    out: list[dict] = []
    for n, rec in enumerate(items, start=1):
        if not isinstance(rec, dict):
            continue
        fpage = _str_or_none(rec.get("fpage"))
        lpage = _str_or_none(rec.get("lpage"))
        if fpage is None and lpage is None:
            fpage, lpage = _split_pages(rec.get("pages") or rec.get("page_range"))
        doi = _str_or_none(rec.get("doi"))
        out.append({
            "reference_id": _str_or_none(rec.get("id")
                                         or rec.get("reference_id")) or f"ref-{n}",
            "authors": _normalize_authors(rec.get("authors") or rec.get("author")),
            "year": _year_of(rec.get("year") or rec.get("date") or rec.get("issued")),
            "title": _str_or_none(rec.get("title")),
            "venue": _str_or_none(rec.get("venue") or rec.get("journal")
                                  or rec.get("container-title") or rec.get("source")),
            "volume": _str_or_none(rec.get("volume")),
            "issue": _str_or_none(rec.get("issue") or rec.get("number")),
            "fpage": fpage,
            "lpage": lpage,
            "doi": doi.lower() if doi else None,
            "pmid": _str_or_none(rec.get("pmid")),
            "raw_text": _str_or_none(rec.get("raw_text") or rec.get("text")
                                     or rec.get("original")) or "",
        })
    return out


class ScimetoCitationsAdapter(PlayerAdapter):
    """Adapter for SciMeto's citation-extraction API. platform / deterministic."""

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
            url = "http://127.0.0.1:3001/api/citations/extract"
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
        # Re-encode to canonical base64 (defensive — envelope value is trusted
        # but this guarantees a clean payload).
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
            "references": _scimeto_to_references(data),
            "player_strategy_notes": "scimeto-citations-api",
        }


register_adapter_class("ScimetoCitationsAdapter", ScimetoCitationsAdapter)
