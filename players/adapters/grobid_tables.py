"""GrobidTablesAdapter — POST a PDF to GROBID, parse TEI <figure type="table">.

GROBID emits each table as <figure type="table"> with a nested <table>
of <row>/<cell>. <cell @cols="N"> indicates a colspan (rowspans are rare
in GROBID output but supported via <cell @rows>). Captions live in
<figDesc>.
"""
from __future__ import annotations

import base64
import io

import requests

from framework.player_adapter import PlayerAdapter, register_adapter_class

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}


def _text(el) -> str:
    return "".join(el.itertext()).strip() if el is not None else ""


class GrobidTablesAdapter(PlayerAdapter):
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
        data = {"consolidateHeader": "0", "consolidateCitations": "0"}
        r = requests.post(
            f"{self.endpoint}/api/processFulltextDocument",
            files=files, data=data, timeout=timeout_s,
        )
        r.raise_for_status()
        return {
            "tables": self._extract_tables(r.text),
            "player_strategy_notes": "grobid /api/processFulltextDocument tei -> tables",
        }

    @staticmethod
    def _extract_tables(tei_xml: str) -> list[dict]:
        try:
            from lxml import etree  # type: ignore
        except ImportError:
            return []
        # A malformed GROBID response must degrade to "no tables", not crash the
        # task with an uncaught XMLSyntaxError (DR-0014; matches grobid_citations).
        try:
            root = etree.fromstring(tei_xml.encode("utf-8"))
        except Exception:
            return []
        out: list[dict] = []
        for fig in root.iter("{http://www.tei-c.org/ns/1.0}figure"):
            if (fig.get("type") or "").lower() != "table":
                continue
            head_el = fig.find("tei:head", TEI_NS)
            desc_el = fig.find("tei:figDesc", TEI_NS)
            label = _text(head_el) or None
            caption = _text(desc_el) or None
            tbl = fig.find("tei:table", TEI_NS)
            cells_flat: list[dict] = []
            n_rows = 0
            n_cols = 0
            header_rows = 0
            if tbl is not None:
                rows = list(tbl.findall("tei:row", TEI_NS))
                # GROBID doesn't expose <thead>; treat rows where every cell has @role="head" or all cells are bold-tagged as headers.
                occupied: set[tuple[int, int]] = set()
                for r_idx, row in enumerate(rows):
                    c_cursor = 0
                    row_all_header = True
                    cells = list(row.findall("tei:cell", TEI_NS))
                    for cell in cells:
                        while (r_idx, c_cursor) in occupied:
                            c_cursor += 1
                        rs = int(cell.get("rows", "1") or "1")
                        cs = int(cell.get("cols", "1") or "1")
                        role = (cell.get("role") or "").lower()
                        is_h = role == "head"
                        if not is_h:
                            row_all_header = False
                        cells_flat.append({
                            "r": r_idx,
                            "c": c_cursor,
                            "rowspan": rs,
                            "colspan": cs,
                            "text": _text(cell),
                            "is_header": is_h,
                        })
                        for dr in range(rs):
                            for dc in range(cs):
                                occupied.add((r_idx + dr, c_cursor + dc))
                        c_cursor += cs
                    if cells and row_all_header and header_rows == r_idx:
                        header_rows = r_idx + 1
                n_rows = (max((c["r"] + c["rowspan"] for c in cells_flat), default=0))
                n_cols = (max((c["c"] + c["colspan"] for c in cells_flat), default=0))
            out.append({
                "label": label,
                "page": None,
                "caption": caption,
                "n_rows": n_rows,
                "n_cols": n_cols,
                "header_rows": header_rows,
                "cells": cells_flat,
            })
        return out


register_adapter_class("GrobidTablesAdapter", GrobidTablesAdapter)
