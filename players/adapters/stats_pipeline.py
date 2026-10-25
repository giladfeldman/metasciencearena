"""StatsPipelineAdapter — a PDF extractor feeding a statistics checker (stats-from-pdf-v1).

Each registry entry fixes one (extractor, checker) pair, so the same checker on two
extractors is two instruments. Extractors run here; checkers run in R
(players/adapters/stats_pipeline.R). Every tool's version is recorded on the result.

  checker "statcheck"   + extractor pypdf | pdftotext | grobid | docpluck | bibr
  checker "statcheck"   + extractor "statcheck_pdf"   (statcheck's own checkPDF reader)
  checker "effectcheck" + extractor docpluck | grobid | docling | pdftotext  (check_document)
  checker "metacheck"   + extractor grobid | bibr     (metacheck convert -> stat_check module)

Extraction choices follow the 2026-10-08 Eindhoven workshop script so results are
comparable with it: pypdf joins page.extract_text() with newlines and keeps the text
exactly; GROBID TEI becomes text from title, abstract, heads, paragraphs, formulas,
figure/table captions and table rows (" | "), bibliography skipped; docpluck is
extract_pdf + normalize_text(academic).

Verdict mapping (documented, applied identically to every pipeline):
  statcheck/metacheck: decision_error TRUE -> "decision_error"; error TRUE -> "inconsistent";
                       both NA -> "unverifiable"; else "consistent". A t or r df that
                       statcheck keeps in df2 moves to df1 (the answer key's layout).
  effectcheck:         p_check_status DECISION_ERROR -> "decision_error"; INCONSISTENT ->
                       "inconsistent"; CONSISTENT -> "consistent"; anything else (MISSING,
                       UNVERIFIABLE) -> "unverifiable"; the decision_error flag only where no
                       p check was recorded. Never the overall `status` (it folds in ES/CI).
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import requests

from framework.player_adapter import PlayerAdapter, register_adapter_class, resolve_rscript_binary

R_SCRIPT = Path(__file__).resolve().parent / "stats_pipeline.R"
REPO = Path(__file__).resolve().parents[2]

_FAM = {"t": "t", "F": "F", "Chi2": "chi2", "chisq": "chi2", "chi2": "chi2", "r": "r", "Z": "z", "z": "z",
        "H": "H", "U": "U", "W": "W", "b": "t", "beta": "t", "OR": "z",
        # no "Q": statcheck's heterogeneity Q / Qb / Qw is no family of this arena, and a
        # row the answer key cannot hold only counts against precision (consult 2026-10-18).
        # effectcheck labels a coefficient test "regression"; its stat_value is the t
        # (b = 0.21, SE = 0.08, t(436) = 2.52). Unmapped, every one was dropped (2026-10-18).
        "regression": "t"}


def _num(x):
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def _rel(x) -> str | None:
    """A checker's p relation as the output schema's enum. Anything else (statcheck's "ns",
    a "<=") carries no comparable relation and becomes None, rather than voiding the whole
    task's output with a schema violation."""
    return x if x in ("=", "<", ">") else None


def _docpluck_token() -> str:
    """The LOCAL docpluck service's internal token: DOCPLUCK_INTERNAL_TOKEN (the variable
    effectcheck reads), else tmp/docpluck_local_token (a throwaway token the local service
    was started with). Never a hosted-service credential."""
    t = os.environ.get("DOCPLUCK_INTERNAL_TOKEN")
    if t:
        return t
    f = REPO / "tmp" / "docpluck_local_token"
    if f.is_file():
        return f.read_text().strip()
    raise RuntimeError("no local docpluck token: set DOCPLUCK_INTERNAL_TOKEN or start the service with tmp/docpluck_local_token")


def tei_to_text(tei: str) -> str:
    """GROBID TEI -> plain text (the Eindhoven workshop's conversion, abstract included)."""
    from lxml import etree
    root = etree.fromstring(tei.encode("utf-8"))
    ns = {"t": "http://www.tei-c.org/ns/1.0"}
    out = []
    for x in root.xpath("//t:titleStmt/t:title", namespaces=ns):
        out.append("".join(x.itertext()).strip())
    for p in root.xpath("//t:profileDesc/t:abstract//t:p", namespaces=ns):
        out.append("".join(p.itertext()).strip())
    for el in root.xpath("//t:text/t:body//*[self::t:head or self::t:p or self::t:formula]", namespaces=ns):
        out.append("".join(el.itertext()).strip())
    for fig in root.xpath("//t:text//t:figure", namespaces=ns):
        for d in fig.xpath("./t:head|./t:label|./t:figDesc", namespaces=ns):
            out.append("".join(d.itertext()).strip())
        for row in fig.xpath(".//t:row", namespaces=ns):
            out.append(" | ".join("".join(c.itertext()).strip() for c in row.xpath("./t:cell", namespaces=ns)))
    for el in root.xpath("//t:text/t:back//*[self::t:head or self::t:p]", namespaces=ns):
        if not el.xpath("ancestor::t:listBibl", namespaces=ns):
            out.append("".join(el.itertext()).strip())
    return "\n\n".join(s for s in out if s)


class StatsPipelineAdapter(PlayerAdapter):

    def __init__(self, *args, extractor: str, checker: str = "statcheck", grobid_endpoint: str = "http://localhost:8072",
                 docpluck_url: str = "http://127.0.0.1:6117",
                 docling_python: str | None = None, metacheck_lib: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.extractor, self.checker = extractor, checker
        self.grobid_endpoint = grobid_endpoint.rstrip("/")
        self.docpluck_url = docpluck_url.rstrip("/")
        from framework.runner import _is_loopback_endpoint  # host-parsed, never a prefix test
        for u in (self.grobid_endpoint, self.docpluck_url):
            if not _is_loopback_endpoint(u):
                raise ValueError(f"stats pipelines use LOCAL services only (held-out PDFs never leave the machine): {u}")
        self.docling_python = docling_python
        self.metacheck_lib = (metacheck_lib or os.environ.get(
            "METACHECK_DEV_RLIB", str(Path.home() / "AppData/Local/R/sciencearena-libs/metacheck-0.3.3.12"))
        ).replace("\\", "/")  # R string literal: forward slashes
        self.rscript = resolve_rscript_binary(None)
        self._versions: dict = {}

    # ---------------------------------------------------------------- lifecycle
    def prepare(self) -> None:
        if self.extractor == "grobid" or (self.checker == "metacheck" and self.extractor == "grobid"):
            r = requests.get(f"{self.grobid_endpoint}/api/version", timeout=5)
            r.raise_for_status()
            self._versions["grobid"] = r.json().get("version")
        if self.extractor == "pdftotext" and not shutil.which("pdftotext"):
            raise RuntimeError("pdftotext not on PATH")
        if self.extractor == "pypdf":
            import pypdf
            self._versions["pypdf"] = pypdf.__version__
        if self.extractor == "docpluck":
            # ONE docpluck for the whole arena: the local service effectcheck's docpluck
            # engine also calls, so statcheck-on-docpluck and escimate-on-docpluck read the
            # same extraction (never the hosted app; loopback only).
            h = requests.get(f"{self.docpluck_url}/health", timeout=5).json()
            self._versions["docpluck"] = h.get("docpluck_identity") or h.get("docpluck_version")

    def _r_pkg_version(self, pkg: str, lib: str | None = None) -> str | None:
        expr = (f'cat(as.character(packageVersion("{pkg}", lib.loc="{lib}")))' if lib
                else f'cat(as.character(packageVersion("{pkg}")))')
        r = subprocess.run([self.rscript, "-e", expr], capture_output=True, text=True, timeout=60)
        return r.stdout.strip() or None if r.returncode == 0 else None

    def resolved_tool_version(self) -> str | None:
        """'<checker>-<version>+<extractor>-<version>', e.g. statcheck-1.5.0+grobid-0.9.1.
        The registry's player_version must equal this exactly (provenance rule)."""
        try:
            self.prepare()
        except Exception:
            return None
        if self.checker == "metacheck":
            cv = self._r_pkg_version("metacheck", self.metacheck_lib)
        else:
            cv = self._r_pkg_version(self.checker)
        if self.extractor == "statcheck_pdf":
            ev = "pdftools-" + (self._r_pkg_version("pdftools") or "?")
            return f"{self.checker}-{cv}+{ev}"
        if self.extractor == "pdftotext":
            r = subprocess.run(["pdftotext", "-v"], capture_output=True, text=True)
            m = re.search(r"version ([\d.]+)", (r.stderr or "") + (r.stdout or ""))
            ev = m.group(1) if m else None
        elif self.extractor == "docling":
            py = self.docling_python or str(REPO / "players/tool_envs/docling/.venv/Scripts/python.exe")
            r = subprocess.run([py, "-c", "import importlib.metadata as m;print(m.version('docling'))"],
                               capture_output=True, text=True, timeout=120)
            ev = r.stdout.strip() or None
        else:
            ev = self._versions.get(self.extractor)
        if cv is None or ev is None:
            return None
        return f"{self.checker}-{cv}+{self.extractor}-{ev}"

    # -------------------------------------------------------------- extraction
    def _text(self, pdf: bytes, path: Path) -> str:
        if self.extractor == "pypdf":
            import pypdf
            import logging
            logging.getLogger("pypdf").setLevel(logging.ERROR)
            reader = pypdf.PdfReader(io.BytesIO(pdf))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        if self.extractor == "pdftotext":
            r = subprocess.run(["pdftotext", "-enc", "UTF-8", str(path), "-"], capture_output=True, timeout=300)
            if r.returncode != 0:
                raise RuntimeError(f"pdftotext failed: {r.stderr[:200]!r}")
            return r.stdout.decode("utf-8", "replace")
        if self.extractor == "grobid":
            r = requests.post(f"{self.grobid_endpoint}/api/processFulltextDocument",
                              files={"input": (path.name, pdf, "application/pdf")},
                              data={"consolidateHeader": "0", "consolidateCitations": "0"}, timeout=600)
            r.raise_for_status()
            return tei_to_text(r.text)
        if self.extractor == "docpluck":
            r = requests.post(f"{self.docpluck_url}/extract?normalize=academic",
                              files={"file": (path.name, pdf, "application/pdf")},
                              headers={"X-Internal-Service-Token": _docpluck_token()}, timeout=600)
            r.raise_for_status()
            return r.json()["text"]
        raise ValueError(f"no text extractor {self.extractor!r}")

    def _r(self, payload: dict, timeout_s: int) -> dict:
        env = dict(os.environ)
        if self.extractor == "docpluck":
            env["DOCPLUCK_INTERNAL_TOKEN"] = _docpluck_token()
        r = subprocess.run([self.rscript, str(R_SCRIPT)], input=json.dumps(payload), capture_output=True,
                           text=True, encoding="utf-8", timeout=timeout_s, env=env)
        if r.returncode != 0:
            raise RuntimeError(f"R pipeline ({payload['mode']}) failed: {(r.stderr or '')[-600:]}")
        return json.loads(r.stdout)

    # ----------------------------------------------------------------- playing
    def play_task(self, envelope: dict, *, timeout_s: int) -> dict:
        pdf = base64.b64decode(envelope["input"]["document_bytes_b64"])
        with tempfile.TemporaryDirectory(prefix="sfp_") as d:
            path = Path(d) / "paper.pdf"
            path.write_bytes(pdf)
            if self.checker == "statcheck" and self.extractor == "statcheck_pdf":
                res = self._r({"mode": "statcheck_pdf", "pdf_path": str(path)}, timeout_s)
                results = [self._from_statcheck(row) for row in res["rows"]]
            elif self.checker == "statcheck":
                text = self._text(pdf, path)
                res = self._r({"mode": "statcheck_text", "text": text}, timeout_s)
                results = [self._from_statcheck(row) for row in res["rows"]]
            elif self.checker == "effectcheck":
                py = sys.executable
                payload = {"mode": "effectcheck", "pdf_path": str(path), "engine": self.extractor,
                           "table_typer_python": py}
                if self.extractor == "grobid":
                    payload["url"] = self.grobid_endpoint
                if self.extractor == "docpluck":
                    payload["url"] = self.docpluck_url
                    # Explicit, not via the environment: R applies the user's .Renviron AFTER
                    # the process environment, and a DOCPLUCK_INTERNAL_TOKEN / DOCPLUCK_API_KEY
                    # there silently replaced the local token (every task refused, 2026-10-17).
                    payload["token"] = _docpluck_token()
                if self.extractor == "docling":
                    payload["python"] = self.docling_python or str(REPO / "players/tool_envs/docling/.venv/Scripts/python.exe")
                res = self._r(payload, timeout_s)
                results = [self._from_effectcheck(row) for row in res["rows"]]
            elif self.checker == "metacheck":
                res = self._r({"mode": "metacheck", "pdf_path": str(path), "method": self.extractor,
                               "grobid_url": self.grobid_endpoint, "lib": self.metacheck_lib}, timeout_s)
                results = [self._from_statcheck(row) for row in res["rows"]]
            else:
                raise ValueError(f"unknown checker {self.checker!r}")
        self._versions.update({k: v for k, v in (res.get("versions") or {}).items() if v})
        return {"results": [r for r in results if r is not None],
                "extraction_engine": self.extractor, "checker": self.checker,
                "tool_versions": dict(self._versions),
                "player_strategy_notes": f"{self.checker} on {self.extractor} text; verdict mapping in stats_pipeline.py"}

    @staticmethod
    def _from_statcheck(row: dict) -> dict | None:
        fam = _FAM.get(str(row.get("test_type")), None)
        if fam is None:
            return None
        de, err = row.get("decision_error"), row.get("error")
        if de is None and err is None:
            verdict = "unverifiable"  # NA in both: no check ran, never a "consistent"
        else:
            verdict = "decision_error" if de else "inconsistent" if err else "consistent"
        df1, df2 = _num(row.get("df1")), _num(row.get("df2"))
        if fam in ("t", "r") and df1 is None and df2 is not None:
            # statcheck keeps a single df in df2; the answer key (and effectcheck) in df1
            df1, df2 = df2, None
        return {"test_type": fam, "df1": df1, "df2": df2,
                "statistic": _num(row.get("test_value")), "p_reported": _num(row.get("reported_p")),
                "p_relation": _rel(row.get("p_comp")), "verdict": verdict, "raw": row.get("raw")}

    @staticmethod
    def _from_effectcheck(row: dict) -> dict | None:
        fam = _FAM.get(str(row.get("test_type")), None)
        if fam is None:
            return None
        # The arena's verdict is p-value consistency, so it comes from effectcheck's own
        # p check. Its overall `status` also folds in the effect-size and CI checks: reading
        # it passed F(2, 138) = 0.37, p = .360 (p check INCONSISTENT, status NOTE) and flagged
        # p-consistent results whose effect size was off (2026-10-17 run).
        # p_check_status is the authority; its DECISION_ERROR already carries the flag. The
        # bare decision_error flag decides only where no p check was recorded (an UNVERIFIABLE
        # resampling p beside a flag is not graded as an error; consult 2026-10-18).
        pcs = str(row.get("p_check_status") or "").upper()
        if pcs == "DECISION_ERROR" or (not pcs and row.get("decision_error") is True):
            v = "decision_error"
        elif pcs == "INCONSISTENT":
            v = "inconsistent"
        elif pcs == "CONSISTENT":
            v = "consistent"
        else:
            v = "unverifiable"  # MISSING / UNVERIFIABLE / no p check: no verdict, never a guess
        return {"test_type": fam, "df1": _num(row.get("df1")), "df2": _num(row.get("df2")),
                "statistic": _num(row.get("stat_value")), "p_reported": _num(row.get("p_reported")),
                "p_relation": _rel(row.get("p_symbol")), "effect_size_type": row.get("effect_reported_name"),
                "effect_size": _num(row.get("effect_reported")), "verdict": v, "raw": row.get("raw_text")}


register_adapter_class("StatsPipelineAdapter", StatsPipelineAdapter)
