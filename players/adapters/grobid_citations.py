"""GrobidCitationsAdapter — POST a PDF to GROBID's processFulltextDocument API,
parse the returned TEI into the pdf-citation-matching-v1 linkage shape.

Unlike processReferences (which extracts only the bibliography — see
grobid_references.py), processFulltextDocument returns the WHOLE TEI: the
<body> with its in-text <ref type="bibr"> citation callouts AND the <listBibl>
of <biblStruct> references. Each in-text <ref> carries a `target` attribute
(e.g. "#b12") pointing at a <biblStruct xml:id="b12"> in the bibliography.

This adapter builds the arena's {markers, consistency} linkage:
- markers: one per <ref type="bibr"> — its text is marker_text, its `target`
  (stripped of the leading '#') is reference_id when it resolves against a
  <biblStruct> id, else None.
- consistency: orphan_markers (markers whose target resolves to nothing),
  uncited_reference_ids (<biblStruct> ids never targeted), and
  duplicate_reference_groups (<biblStruct>s sharing a lowercased DOI, else a
  normalized title).

char offsets: GROBID's TEI <body> is a reflowed paragraph tree; reconstructing
the exact body string that the arena gold offsets index into is not reliable
across GROBID versions. char_start / char_end are therefore set to the
documented sentinel -1 ("offset unresolved"). The marker_text + reference_id
linkage — which the scorer weights 0.8 of primary — is exact.

The namespace-agnostic TEI helpers are reused from grobid_references.py.
"""
from __future__ import annotations

import base64
import io
import re

import requests

from framework.player_adapter import PlayerAdapter, register_adapter_class
from players.adapters.grobid_references import (
    _XML_ID,
    _child_local,
    _find_local,
    _findall_local,
    _local,
    _text,
)

# Documented sentinel: char offset could not be reconstructed from GROBID TEI.
_OFFSET_UNRESOLVED = -1


def _empty_linkage() -> dict:
    return {
        "markers": [],
        "consistency": {
            "orphan_markers": [],
            "uncited_reference_ids": [],
            "duplicate_reference_groups": [],
        },
    }


def _norm_title(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def _biblstruct_dup_key(bibl) -> str | None:
    """Duplicate-grouping key for a <biblStruct>: lowercased DOI, else title."""
    for idno in _findall_local(bibl, "idno"):
        if (idno.get("type") or "").upper() == "DOI":
            doi = _text(idno).lower()
            if doi:
                return "doi::" + doi
    # No DOI — fall back to the (analytic, else monogr) title.
    analytic = _child_local(bibl, "analytic")
    monogr = _child_local(bibl, "monogr")
    title = (_text(_find_local(analytic, "title")) if analytic is not None else "")
    if not title and monogr is not None:
        title = _text(_find_local(monogr, "title"))
    title = _norm_title(title)
    return ("title::" + title) if title else None


def _tei_to_linkage(tei_str: str) -> dict:
    """Map a GROBID processFulltextDocument TEI to {markers, consistency}.

    Defensive: unparseable input -> empty linkage. char offsets are the
    documented -1 sentinel (GROBID body text is not reliably reconstructable).
    """
    try:
        from lxml import etree  # type: ignore
    except ImportError:
        return _empty_linkage()
    try:
        root = etree.fromstring(tei_str.encode("utf-8"))
    except Exception:
        return _empty_linkage()

    # Bibliography: every <biblStruct> with an xml:id, in document order.
    ref_ids: list[str] = []
    bibl_by_id: dict[str, object] = {}
    for bibl in _findall_local(root, "biblStruct"):
        bid = bibl.get(_XML_ID)
        if not bid:
            continue
        if bid not in bibl_by_id:
            bibl_by_id[bid] = bibl
            ref_ids.append(bid)
    ref_id_set = set(ref_ids)

    # In-text markers: every <ref type="bibr"> in the TEI. GROBID usually
    # puts them inside <text>/<body>, but for short documents misclassified
    # as abstract-only it leaves <body/> empty and sticks the bibr refs in
    # <teiHeader>/<profileDesc>/<abstract>/<div>/<p>. Scanning the whole
    # root covers both placements; bibr refs in the abstract are still
    # legitimate in-text citations.
    markers: list[dict] = []
    cited: set[str] = set()
    orphans: list[str] = []
    for el in root.iter():
        if _local(el.tag) != "ref":
            continue
        if (el.get("type") or "") != "bibr":
            continue
        marker_text = _text(el)
        target = (el.get("target") or "").lstrip("#").strip()
        resolved = target if target in ref_id_set else None
        if resolved is not None:
            cited.add(resolved)
        else:
            orphans.append(marker_text)
        markers.append({
            "marker_text": marker_text,
            "char_start": _OFFSET_UNRESOLVED,
            "char_end": _OFFSET_UNRESOLVED,
            "reference_id": resolved,
        })

    uncited = [rid for rid in ref_ids if rid not in cited]

    # Duplicate detection: group biblStructs by DOI, else normalized title.
    groups: dict[str, list[str]] = {}
    for rid in ref_ids:
        key = _biblstruct_dup_key(bibl_by_id[rid])
        if key:
            groups.setdefault(key, []).append(rid)
    duplicate_groups = [ids for ids in groups.values() if len(ids) > 1]

    return {
        "markers": markers,
        "consistency": {
            "orphan_markers": orphans,
            "uncited_reference_ids": uncited,
            "duplicate_reference_groups": duplicate_groups,
        },
    }


class GrobidCitationsAdapter(PlayerAdapter):
    """Adapter for GROBID's full-text API as a citation matcher.

    tool / deterministic. Uses the `endpoint` config kwarg (same name as
    GrobidReferencesAdapter — do NOT introduce grobid_url).
    """

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
        data = {"consolidateCitations": "0", "consolidateHeader": "0"}
        r = requests.post(
            f"{self.endpoint}/api/processFulltextDocument",
            files=files, data=data, timeout=timeout_s,
        )
        r.raise_for_status()
        linkage = _tei_to_linkage(r.text)
        return {
            **linkage,
            "player_strategy_notes": "grobid processFulltextDocument",
        }


register_adapter_class("GrobidCitationsAdapter", GrobidCitationsAdapter)
