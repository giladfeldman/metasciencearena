"""GrobidTextAdapter — POST a PDF to a GROBID server, return body text.

Calls GROBID's `processFulltextDocument` endpoint, parses TEI XML, and
concatenates `<text><body>` paragraph content. Requires a running GROBID
server (e.g. via Docker: `docker run -p 8070:8070 lfoppiano/grobid:0.8.1`).

Adapter degrades gracefully: if the server isn't reachable, `prepare()`
fails fast with a clear message. The framework treats this as a player-
level error and continues with the other players.
"""
from __future__ import annotations

import base64
import io

import requests

from framework.player_adapter import PlayerAdapter, register_adapter_class


class GrobidTextAdapter(PlayerAdapter):
    endpoint: str  # e.g. http://localhost:8070

    def __init__(self, *args, endpoint: str = "http://localhost:8070", **kwargs):
        super().__init__(*args, **kwargs)
        self.endpoint = endpoint.rstrip("/")

    def prepare(self) -> None:
        try:
            r = requests.get(f"{self.endpoint}/api/isalive", timeout=3)
            if r.status_code != 200:
                raise RuntimeError(f"GROBID isalive returned {r.status_code}")
        except Exception as exc:
            raise RuntimeError(
                f"GROBID server not reachable at {self.endpoint}: {exc}. "
                f"Start it via: docker run -p 8070:8070 lfoppiano/grobid:0.8.1"
            )

    def resolved_tool_version(self) -> str | None:
        from players.adapters._tool_version import grobid_version
        return grobid_version(self.endpoint)

    def play_task(self, envelope: dict, *, timeout_s: int) -> dict:
        pdf_bytes = base64.b64decode(envelope["input"]["document_bytes_b64"])
        files = {"input": ("paper.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
        data = {"consolidateHeader": "0", "consolidateCitations": "0"}
        r = requests.post(
            f"{self.endpoint}/api/processFulltextDocument",
            files=files, data=data, timeout=timeout_s,
        )
        r.raise_for_status()
        tei_xml = r.text

        body_text = self._extract_body_text(tei_xml)
        return {
            "full_text": body_text,
            "pages": [body_text],
            "footnotes": [],
            "player_strategy_notes": "grobid /api/processFulltextDocument body text",
        }

    @staticmethod
    def _extract_body_text(tei_xml: str) -> str:
        """Concatenate paragraph text from <text><body>, in document order."""
        try:
            from lxml import etree  # type: ignore
        except ImportError:
            # Fallback: dumb regex extraction
            import re
            paragraphs = re.findall(r"<p[^>]*>(.*?)</p>", tei_xml, flags=re.DOTALL)
            stripped = [re.sub(r"<[^>]+>", "", p).strip() for p in paragraphs]
            return "\n\n".join(s for s in stripped if s)

        ns = {"tei": "http://www.tei-c.org/ns/1.0"}
        root = etree.fromstring(tei_xml.encode("utf-8"))
        body = root.find("tei:text/tei:body", ns)
        if body is None:
            return ""
        chunks: list[str] = []
        # Abstract is in teiHeader/profileDesc/abstract; include its paragraphs.
        abstract = root.find("tei:teiHeader/tei:profileDesc/tei:abstract", ns)
        if abstract is not None:
            for p in abstract.iter("{http://www.tei-c.org/ns/1.0}p"):
                t = "".join(p.itertext()).strip()
                if t:
                    chunks.append(t)
        for p in body.iter("{http://www.tei-c.org/ns/1.0}p"):
            t = "".join(p.itertext()).strip()
            if t:
                chunks.append(t)
        return "\n\n".join(chunks)


register_adapter_class("GrobidTextAdapter", GrobidTextAdapter)
