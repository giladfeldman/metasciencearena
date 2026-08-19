"""GrobidReferencesAdapter — POST a PDF to GROBID's processReferences API,
parse the returned TEI <listBibl>/<biblStruct> into the arena's reference list.

GROBID's /api/processReferences endpoint extracts only the bibliography of a
paper. It returns a TEI document whose <listBibl> holds one <biblStruct> per
cited reference. Each <biblStruct> splits the work into an <analytic> part
(article-level: title, authors) and a <monogr> part (container-level: journal
title, imprint with date / volume / issue / pages). Identifiers (DOI, PMID)
arrive as <idno type="..."> elements.
"""
from __future__ import annotations

import base64
import io
import re

import requests

from framework.player_adapter import PlayerAdapter, register_adapter_class

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}
TEI_NS_URI = "http://www.tei-c.org/ns/1.0"
_XML_ID = "{http://www.w3.org/XML/1998/namespace}id"


def _local(tag) -> str:
    """Local-name of an lxml element tag, namespace stripped."""
    if not isinstance(tag, str):
        return ""
    return tag.rsplit("}", 1)[-1]


def _text(el) -> str:
    """Whitespace-collapsed full text content of an element (or '' if None)."""
    if el is None:
        return ""
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip()


def _find_local(parent, *names):
    """First descendant whose local-name matches any of `names` (namespace-agnostic)."""
    if parent is None:
        return None
    wanted = set(names)
    for el in parent.iter():
        if _local(el.tag) in wanted:
            return el
    return None


def _findall_local(parent, name):
    """All descendants whose local-name == `name`."""
    if parent is None:
        return []
    return [el for el in parent.iter() if _local(el.tag) == name]


def _child_local(parent, *names):
    """First *direct child* whose local-name matches any of `names`."""
    if parent is None:
        return None
    wanted = set(names)
    for el in parent:
        if _local(el.tag) in wanted:
            return el
    return None


class GrobidReferencesAdapter(PlayerAdapter):
    endpoint: str

    def __init__(self, *args, endpoint: str = "http://localhost:8070", **kwargs):
        super().__init__(*args, **kwargs)
        self.endpoint = endpoint.rstrip("/")

    def resolved_tool_version(self) -> str | None:
        from players.adapters._tool_version import grobid_version
        return grobid_version(self.endpoint)

    def prepare(self) -> None:
        try:
            r = requests.get(f"{self.endpoint}/api/isalive", timeout=3)
            if r.status_code != 200:
                raise RuntimeError(f"GROBID isalive returned {r.status_code}")
        except Exception as exc:
            raise RuntimeError(
                f"GROBID server not reachable at {self.endpoint}: {exc}. "
                "Start it via: docker run -p 8070:8070 lfoppiano/grobid:0.8.1"
            )

    def play_task(self, envelope: dict, *, timeout_s: int) -> dict:
        pdf_bytes = base64.b64decode(envelope["input"]["document_bytes_b64"])
        files = {"input": ("paper.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
        data = {"consolidateCitations": "0"}
        r = requests.post(
            f"{self.endpoint}/api/processReferences",
            files=files, data=data, timeout=timeout_s,
        )
        r.raise_for_status()
        references = self._tei_to_references(r.text)
        return {
            "references": references,
            "player_strategy_notes": "grobid processReferences",
        }

    def _tei_to_references(self, tei_str: str) -> list[dict]:
        try:
            from lxml import etree  # type: ignore
        except ImportError:
            return []
        root = etree.fromstring(tei_str.encode("utf-8"))
        out: list[dict] = []
        for n, bibl in enumerate(_findall_local(root, "biblStruct"), start=1):
            out.append(self._parse_biblstruct(bibl, n))
        return out

    def _parse_biblstruct(self, bibl, n: int) -> dict:
        analytic = _child_local(bibl, "analytic")
        monogr = _child_local(bibl, "monogr")

        # Title: analytic title wins; otherwise the monogr title.
        analytic_title = _text(_find_local(analytic, "title")) if analytic is not None else ""
        monogr_title = _text(_find_local(monogr, "title")) if monogr is not None else ""
        if analytic_title:
            title = analytic_title
            venue = monogr_title or None
        else:
            title = monogr_title or None
            venue = None

        # Authors: prefer analytic <author>s, fall back to monogr <author>s.
        author_scope = analytic if (analytic is not None and _findall_local(analytic, "author")) else monogr
        authors: list[dict] = []
        for author in _findall_local(author_scope, "author"):
            persname = _find_local(author, "persName")
            if persname is None:
                continue
            surname = _text(_find_local(persname, "surname")) or None
            givens = [_text(f) for f in _findall_local(persname, "forename")]
            given_names = " ".join(g for g in givens if g) or None
            if surname or given_names:
                authors.append({"surname": surname, "given_names": given_names})

        # Year: from <date when="..."> (or its text), year portion only.
        year = None
        date_el = _find_local(monogr if monogr is not None else bibl, "date")
        if date_el is not None:
            raw_date = date_el.get("when") or _text(date_el)
            m = re.search(r"\d{4}", raw_date)
            if m:
                year = m.group(0)

        # biblScope: volume / issue / page range.
        volume = issue = fpage = lpage = None
        for scope in _findall_local(monogr if monogr is not None else bibl, "biblScope"):
            unit = (scope.get("unit") or "").lower()
            if unit in ("volume", "vol"):
                volume = _text(scope) or None
            elif unit in ("issue", "number"):
                issue = _text(scope) or None
            elif unit == "page":
                frm = scope.get("from")
                to = scope.get("to")
                if frm or to:
                    fpage = frm or None
                    lpage = to or None
                else:
                    pages = _text(scope)
                    if pages:
                        parts = re.split(r"[-–]", pages, maxsplit=1)
                        fpage = parts[0].strip() or None
                        lpage = parts[1].strip() if len(parts) > 1 else None

        # Identifiers.
        doi = pmid = None
        for idno in _findall_local(bibl, "idno"):
            idtype = (idno.get("type") or "").upper()
            value = _text(idno)
            if not value:
                continue
            if idtype == "DOI":
                doi = value.lower()
            elif idtype == "PMID":
                pmid = value

        # raw_text: a GROBID-emitted raw_reference note if present, else the
        # full text content of the biblStruct.
        raw_note = None
        for note in _findall_local(bibl, "note"):
            if (note.get("type") or "") == "raw_reference":
                raw_note = _text(note)
                break
        raw_text = raw_note if raw_note else _text(bibl)

        reference_id = bibl.get(_XML_ID) or f"ref-{n}"

        return {
            "reference_id": reference_id,
            "authors": authors,
            "year": year,
            "title": title,
            "venue": venue,
            "volume": volume,
            "issue": issue,
            "fpage": fpage,
            "lpage": lpage,
            "doi": doi,
            "pmid": pmid,
            "raw_text": raw_text,
        }


register_adapter_class("GrobidReferencesAdapter", GrobidReferencesAdapter)
