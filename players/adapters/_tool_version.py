"""Runtime version detection for adapters' ``resolved_tool_version()``.

Every helper here is best-effort and returns ``None`` on any failure (missing
package, binary not on PATH, server down) so version detection NEVER fails a
task. The runner stamps the result into each run record; the leaderboard ranks
on the version a score was actually produced with rather than the static label
in registry.yaml (Finding 1 of the 2026-06-12 accuracy handoff).
"""
from __future__ import annotations

import importlib
import importlib.metadata
import os
import re
import shutil
import subprocess


def module_version(module_name: str) -> str | None:
    """``<module>-<version>`` for an importable Python package, else None.

    Prefers the INSTALLED DISTRIBUTION version over the module's ``__version__``
    attribute. They disagree more often than you would like: liteparse 2.0.8
    ships ``__version__ == "2.0.0"``, so reading the attribute stamped a release
    we were not running into every run record's ``resolved_tool_version`` — the
    provenance the leaderboard ranks on and the outreach letters point at.
    Distribution metadata is what pip installed and what the registry
    declaration is compared against; ``__version__`` is an author convention
    upstream can forget to bump.

    Falls back to ``__version__`` when there is no installed distribution (a
    vendored or namespace module), keeping detection best-effort per this
    module's contract. Pinned by
    ``players/adapters/tests/test_tool_version_provenance.py``.
    """
    try:
        mod = importlib.import_module(module_name)
    except Exception:
        return None
    try:
        return f"{module_name}-{importlib.metadata.version(module_name)}"
    except Exception:
        pass
    v = getattr(mod, "__version__", None)
    return f"{module_name}-{v}" if v else None


def pdftotext_version(binary: str = "pdftotext") -> str | None:
    """Parse ``pdftotext -v`` (poppler/xpdf write the banner to stderr)."""
    try:
        resolved = shutil.which(binary) or binary
        proc = subprocess.run([resolved, "-v"], capture_output=True, timeout=10)
        text = (proc.stderr or b"").decode("utf-8", "replace") + \
               (proc.stdout or b"").decode("utf-8", "replace")
        m = re.search(r"pdftotext version (\S+)", text)
        return f"poppler-pdftotext-{m.group(1)}" if m else None
    except Exception:
        return None


# The first `library(<pkg>)` in an adapter script names the R package that does the
# actual work (jsonlite is I/O plumbing every script loads, never the tool itself).
# NOT anchored to line start: adapters legitimately nest the call, e.g. metacheck.R
# does `capture.output(suppressMessages(library(metacheck)))` to silence startup
# chatter that would otherwise corrupt the JSON on stdout.
_R_LIBRARY_RE = re.compile(r"\blibrary\(\s*([A-Za-z][A-Za-z0-9._]*)\s*\)")
_R_PLUMBING_PACKAGES = {"jsonlite"}


def r_adapter_package(r_script: str | "os.PathLike[str]") -> str | None:
    """Name the R package an adapter script depends on, read from the script itself.

    Derived rather than hardcoded so a new R adapter gets version detection for free
    and a renamed dependency cannot silently desync from a lookup table. Returns
    None for pure base-R adapters (e.g. pcurve.R), which have no package version.
    """
    try:
        with open(r_script, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception:
        return None
    for name in _R_LIBRARY_RE.findall(text):
        if name not in _R_PLUMBING_PACKAGES:
            return name
    return None


def r_package_version(package: str, rscript_binary: str | None = None) -> str | None:
    """``<package>-<version>`` for an installed R package, else None.

    Closes F7 (2026-08-04): R-backed reference tools previously resolved to None, so
    `framework audit --versions` could not detect drift for scrutiny / metacheck /
    oddpub / rtransparent / statcheck / metafor / effectsize / rsprite2 / zcurve —
    exactly the deterministic tools that cross-validate arena gold. Their labels had
    to be reconciled by hand (cycle 6), which is precisely the kind of provenance
    that silently re-ranks history when it goes stale.
    """
    if not package:
        return None
    try:
        from framework.player_adapter import resolve_rscript_binary
        binary = resolve_rscript_binary(rscript_binary)
        resolved = shutil.which(binary) or binary
        proc = subprocess.run(
            [resolved, "--vanilla", "-e",
             f'cat(as.character(utils::packageVersion("{package}")))'],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=60, check=False,
        )
        if proc.returncode != 0:
            return None
        # packageVersion() prints bare "0.6.1"; guard against R warnings on stdout.
        m = re.search(r"\b(\d+(?:[.-]\d+)*)\b", (proc.stdout or "").strip())
        return f"{package}-{m.group(1)}" if m else None
    except Exception:
        return None


def grobid_version(endpoint: str) -> str | None:
    """Query a running GROBID server's ``/api/version`` endpoint."""
    try:
        import requests
        r = requests.get(f"{endpoint.rstrip('/')}/api/version", timeout=3)
        if r.status_code == 200 and r.text.strip():
            return f"grobid-{r.text.strip()}"
    except Exception:
        pass
    return None
