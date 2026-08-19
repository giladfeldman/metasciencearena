"""CermineReferencesAdapter — run CERMINE (a Java JAR) over a PDF, parse the
NLM-JATS XML it emits into the arena reference schema.

CERMINE (https://github.com/CeON/CERMINE) extracts structured metadata and a
reference list from scientific PDFs. Its `ContentExtractor` command processes a
directory of PDFs and writes one NLM-JATS XML file (`<name>.cermxml`) next to
each input PDF:

    java -cp <cermine-jar> pl.edu.icm.cermine.ContentExtractor -path <dir>

The produced NLM XML carries the bibliography under <back>/<ref-list>, with one
<ref> per cited work. CERMINE wraps each reference in a <mixed-citation> whose
child elements follow JATS conventions:

    <string-name><surname>..</surname><given-names>..</given-names></string-name>
    <article-title>..</article-title>
    <source>..</source>            (journal / container title)
    <year>..</year>
    <volume>.. </volume>
    <issue>..</issue>
    <fpage>..</fpage> <lpage>..</lpage>
    <pub-id pub-id-type="doi">..</pub-id>

The parser below is namespace-agnostic (matching grobid_references.py) so it
tolerates both bare and JATS-namespaced documents.
"""
from __future__ import annotations

import base64
import os
import re
import subprocess
import tempfile
from pathlib import Path

from framework.player_adapter import PlayerAdapter, register_adapter_class


def _local(tag) -> str:
    """Local-name of an lxml/ElementTree tag, namespace stripped."""
    if not isinstance(tag, str):
        return ""
    return tag.rsplit("}", 1)[-1]


def _text(el) -> str:
    """Whitespace-collapsed full text content of an element (or '' if None)."""
    if el is None:
        return ""
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip()


def _findall_local(parent, name):
    """All descendants whose local-name == `name`."""
    if parent is None:
        return []
    return [el for el in parent.iter() if _local(el.tag) == name]


def _find_local(parent, *names):
    """First descendant whose local-name matches any of `names`."""
    if parent is None:
        return None
    wanted = set(names)
    for el in parent.iter():
        if _local(el.tag) in wanted:
            return el
    return None


def _first_text(parent, *names) -> str | None:
    el = _find_local(parent, *names)
    t = _text(el)
    return t or None


def _parse_ref(ref, n: int) -> dict:
    """Map one NLM <ref> element to the arena reference schema."""
    authors: list[dict] = []
    # JATS: <string-name> (CERMINE) or <name> (canonical JATS).
    for name in _findall_local(ref, "string-name") + _findall_local(ref, "name"):
        surname = _first_text(name, "surname")
        given = _first_text(name, "given-names")
        if surname or given:
            authors.append({"surname": surname, "given_names": given})

    year = None
    raw_year = _first_text(ref, "year")
    if raw_year:
        m = re.search(r"\d{4}", raw_year)
        year = m.group(0) if m else raw_year

    fpage = _first_text(ref, "fpage")
    lpage = _first_text(ref, "lpage")
    # Some emitters fold the range into a single <page-range>.
    if fpage is None and lpage is None:
        page_range = _first_text(ref, "page-range", "pages")
        if page_range:
            parts = re.split(r"[-–—]", page_range, maxsplit=1)
            fpage = parts[0].strip() or None
            lpage = parts[1].strip() if len(parts) > 1 else None

    doi = pmid = None
    for pub_id in _findall_local(ref, "pub-id"):
        idtype = (pub_id.get("pub-id-type") or "").lower()
        value = _text(pub_id)
        if not value:
            continue
        if idtype == "doi":
            doi = value.lower()
        elif idtype == "pmid":
            pmid = value

    # raw_text: the literal <mixed-citation> text if present, else the <ref> text.
    mixed = _find_local(ref, "mixed-citation")
    raw_text = _text(mixed) if mixed is not None else _text(ref)

    ref_id = ref.get("id") or f"ref-{n}"

    return {
        "reference_id": ref_id,
        "authors": authors,
        "year": year,
        "title": _first_text(ref, "article-title", "chapter-title"),
        "venue": _first_text(ref, "source"),
        "volume": _first_text(ref, "volume"),
        "issue": _first_text(ref, "issue"),
        "fpage": fpage,
        "lpage": lpage,
        "doi": doi,
        "pmid": pmid,
        "raw_text": raw_text,
    }


def _nlm_to_references(nlm_xml: str | bytes) -> list[dict]:
    """Parse a CERMINE NLM-JATS document (or <ref-list> fragment) into the
    arena reference schema. Namespace-agnostic."""
    try:
        from lxml import etree  # type: ignore
        if isinstance(nlm_xml, str):
            nlm_xml = nlm_xml.encode("utf-8")
        root = etree.fromstring(nlm_xml)
    except ImportError:
        import xml.etree.ElementTree as etree  # type: ignore
        if isinstance(nlm_xml, bytes):
            nlm_xml = nlm_xml.decode("utf-8", errors="replace")
        root = etree.fromstring(nlm_xml)
    out: list[dict] = []
    for n, ref in enumerate(_findall_local(root, "ref"), start=1):
        out.append(_parse_ref(ref, n))
    return out


class CermineReferencesAdapter(PlayerAdapter):
    """Adapter wrapping the CERMINE Java JAR. deterministic / tool."""

    cermine_jar: str
    java_binary: str

    def __init__(self, *args, cermine_jar: str, java_binary: str = "java", **kwargs):
        super().__init__(*args, **kwargs)
        self.cermine_jar = cermine_jar
        self.java_binary = java_binary

    def prepare(self) -> None:
        import shutil
        if shutil.which(self.java_binary) is None:
            raise RuntimeError(f"java binary not found on PATH: {self.java_binary}")
        if not Path(self.cermine_jar).is_file():
            raise RuntimeError(f"CERMINE jar not found: {self.cermine_jar}")

    def play_task(self, envelope: dict, *, timeout_s: int) -> dict:
        pdf_bytes = base64.b64decode(envelope["input"]["document_bytes_b64"])
        # CERMINE's ContentExtractor processes a directory: write the PDF into
        # an isolated temp dir, run it, then read the sibling .cermxml output.
        with tempfile.TemporaryDirectory(prefix="cermine_") as work_dir:
            pdf_path = os.path.join(work_dir, "paper.pdf")
            with open(pdf_path, "wb") as f:
                f.write(pdf_bytes)
            proc = subprocess.run(
                [self.java_binary, "-cp", self.cermine_jar,
                 "pl.edu.icm.cermine.ContentExtractor", "-path", work_dir],
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=timeout_s, check=False,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"CERMINE exited {proc.returncode}: {proc.stderr.strip()[:300]}")
            xml_path = os.path.join(work_dir, "paper.cermxml")
            if not os.path.exists(xml_path):
                raise RuntimeError(f"CERMINE produced no output at {xml_path}")
            nlm_xml = Path(xml_path).read_text(encoding="utf-8", errors="replace")
        references = _nlm_to_references(nlm_xml)
        return {
            "references": references,
            "player_strategy_notes": "cermine ContentExtractor",
        }


register_adapter_class("CermineReferencesAdapter", CermineReferencesAdapter)
