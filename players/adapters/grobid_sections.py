"""GrobidSectionsAdapter — POST a PDF to GROBID, parse TEI <div type="..."> sections.

GROBID returns one <div> per logical section, optionally with @type and a
nested <head>. We use the same `label_map.yaml` (grobid: section, owned by
the arena) to collapse @type onto the canonical 11.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

import requests
import yaml

from framework.player_adapter import PlayerAdapter, register_adapter_class

REPO_ROOT = Path(__file__).resolve().parents[2]
ARENA_LABEL_MAP_PATH = REPO_ROOT / "arenas" / "pdf-section-structure-v1" / "label_map.yaml"

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}


def _load_grobid_label_map() -> dict[str, str]:
    with ARENA_LABEL_MAP_PATH.open("r", encoding="utf-8") as f:
        full = yaml.safe_load(f) or {}
    return (full.get("grobid") or {})


class GrobidSectionsAdapter(PlayerAdapter):
    endpoint: str

    def __init__(self, *args, endpoint: str = "http://localhost:8070", **kwargs):
        super().__init__(*args, **kwargs)
        self.endpoint = endpoint.rstrip("/")

    def resolved_tool_version(self) -> str | None:
        from players.adapters._tool_version import grobid_version
        return grobid_version(self.endpoint)

    def prepare(self) -> None:
        self._label_map = _load_grobid_label_map()
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
        if not hasattr(self, "_label_map"):
            self.prepare()
        pdf_bytes = base64.b64decode(envelope["input"]["document_bytes_b64"])
        files = {"input": ("paper.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
        data = {"consolidateHeader": "0", "consolidateCitations": "0"}
        r = requests.post(
            f"{self.endpoint}/api/processFulltextDocument",
            files=files, data=data, timeout=timeout_s,
        )
        r.raise_for_status()
        tei_xml = r.text
        sections, full_text = self._extract_sections(tei_xml)
        return {
            "sections": sections,
            "full_text": full_text,
            "player_strategy_notes": "grobid /api/processFulltextDocument tei -> sections",
        }

    def _extract_sections(self, tei_xml: str) -> tuple[list[dict], str]:
        try:
            from lxml import etree  # type: ignore
        except ImportError:
            return [], ""
        root = etree.fromstring(tei_xml.encode("utf-8"))
        body = root.find("tei:text/tei:body", TEI_NS)
        out: list[dict] = []
        chunks: list[str] = []
        idx = 0
        running = 0

        # Synthesize an abstract section from teiHeader/profileDesc.
        abstract = root.find("tei:teiHeader/tei:profileDesc/tei:abstract", TEI_NS)
        if abstract is not None:
            paras = []
            for p in abstract.iter("{http://www.tei-c.org/ns/1.0}p"):
                t = "".join(p.itertext()).strip()
                if t:
                    paras.append(t)
            if paras:
                joined = "\n\n".join(paras)
                out.append({
                    "label": "abstract",
                    "heading_text": "Abstract",
                    "section_index": idx,
                    "char_start": running,
                    "char_end": running + len(joined),
                    "page": None,
                })
                chunks.append(joined)
                running += len(joined) + 2
                idx += 1

        if body is not None:
            for div in body.findall("tei:div", TEI_NS):
                div_type = (div.get("type") or "").strip().lower()
                head_el = div.find("tei:head", TEI_NS)
                heading_text = "".join(head_el.itertext()).strip() if head_el is not None else None
                label = self._label_map.get(div_type, "other")
                # Body paragraphs (no nested div recursion in v1).
                paras = []
                for p in div.findall("tei:p", TEI_NS):
                    t = "".join(p.itertext()).strip()
                    if t:
                        paras.append(t)
                joined = "\n\n".join(paras)
                if not joined and not heading_text:
                    continue
                out.append({
                    "label": label,
                    "heading_text": heading_text or None,
                    "section_index": idx,
                    "char_start": running,
                    "char_end": running + len(joined),
                    "page": None,
                })
                chunks.append(joined)
                running += len(joined) + 2
                idx += 1
        return out, "\n\n".join(chunks)


register_adapter_class("GrobidSectionsAdapter", GrobidSectionsAdapter)
