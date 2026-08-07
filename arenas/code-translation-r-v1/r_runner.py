"""Execute an R script and recover the JSON statistics it prints.

This is the heart of the arena: scoring is EXECUTABLE EQUIVALENCE, so a player's
emitted R has to actually run against the fixed dataset and produce numbers we
can compare to gold. Textual similarity to a reference translation is explicitly
not the standard — a script that looks right but computes a different quantity
must score near zero.

Safety: the code being run is model-generated and therefore untrusted. Every
execution is bounded by a timeout, runs in a scratch working directory, and is
launched with --vanilla so no user profile, site file, or saved workspace can
influence it (that also keeps results reproducible across machines).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

DEFAULT_TIMEOUT_S = 60

# Candidate Rscript locations. R is frequently installed on Windows without
# being added to PATH, so fall back to the standard install root before giving
# up — an arena that reports "does not execute" for every player because R is
# merely unfindable would be silently, uniformly wrong.
_WINDOWS_R_ROOTS = (
    Path(r"C:\Program Files\R"),
    Path(r"C:\Program Files (x86)\R"),
)


def find_rscript() -> str | None:
    """Locate an Rscript binary, or None when R is unavailable."""
    env = os.environ.get("ARENA_RSCRIPT")
    if env and Path(env).exists():
        return env
    which = shutil.which("Rscript")
    if which:
        return which
    for root in _WINDOWS_R_ROOTS:
        if not root.is_dir():
            continue
        # Newest R version first (R-4.10 must sort above R-4.9).
        def _ver_key(p: Path) -> list[int]:
            return [int(x) for x in re.findall(r"\d+", p.name)] or [0]

        for d in sorted((p for p in root.iterdir() if p.is_dir()), key=_ver_key, reverse=True):
            cand = d / "bin" / "Rscript.exe"
            if cand.exists():
                return str(cand)
    return None


@dataclass
class RResult:
    """Outcome of one R execution.

    `ok` means the script ran AND printed parseable JSON. Anything else carries
    an `error` explaining which of those two failed, so the scorer can
    distinguish "did not execute" from "ran but emitted nothing usable".
    """

    ok: bool
    stats: dict
    stdout: str
    stderr: str
    error: str | None = None


_JSON_RE = re.compile(r"\{.*\}", re.S)


def _extract_json(text: str) -> dict | None:
    """Recover the JSON object from R stdout.

    Players routinely print extra output (a stray `print()`, a package startup
    message that escaped stderr). Rather than demand pristine stdout, take the
    last balanced {...} block — the contract says the JSON is printed last.
    """
    text = text.strip()
    if not text:
        return None
    try:
        v = json.loads(text)
        return v if isinstance(v, dict) else None
    except json.JSONDecodeError:
        pass
    for m in reversed(list(_JSON_RE.finditer(text))):
        try:
            v = json.loads(m.group(0))
            if isinstance(v, dict):
                return v
        except json.JSONDecodeError:
            continue
    return None


HARVESTER = Path(__file__).resolve().parent / "harvest.R"


def run_r(r_code: str, data_csv: Path, *, required: list[str] | None = None,
          timeout_s: int = DEFAULT_TIMEOUT_S, rscript: str | None = None) -> RResult:
    """Run `r_code` against `data_csv` and HARVEST the statistics it produced.

    The player's code is executed verbatim by ``harvest.R``, which then walks the
    resulting environment and recovers the requested statistics from whatever it
    finds — a jmv results object, an ``lm``/``htest``, an ANOVA table, plain
    named variables, or an explicit JSON block.

    This is deliberately permissive about FORM and strict about VALUE. The arena
    asks whether a translation is accurate, not whether the translator happened
    to adopt our print contract; requiring a JSON block scored real converters
    0.00 for emitting perfectly correct analyses in their own idiom.

    The dataset path travels in ARENA_DATA so no script hard-codes a path.
    """
    binary = rscript or find_rscript()
    if binary is None:
        return RResult(False, {}, "", "", error="rscript_not_found")

    with tempfile.TemporaryDirectory(prefix="arena-r-") as tmp:
        script = Path(tmp) / "player.R"
        script.write_text(r_code, encoding="utf-8")
        env = {
            **os.environ,
            "ARENA_DATA": str(data_csv.resolve()),
            "ARENA_PLAYER_FILE": str(script),
            "ARENA_REQUIRED": ",".join(required or []),
        }
        try:
            proc = subprocess.run(
                [binary, "--vanilla", str(HARVESTER)],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=timeout_s, cwd=tmp, env=env,
            )
        except subprocess.TimeoutExpired:
            return RResult(False, {}, "", "", error=f"timeout_after_{timeout_s}s")
        except OSError as e:  # binary vanished, permission denied, ...
            return RResult(False, {}, "", "", error=f"spawn_failed: {e}")

    if proc.returncode != 0:
        return RResult(False, {}, proc.stdout, proc.stderr,
                       error=f"nonzero_exit_{proc.returncode}")

    stats = _extract_json(proc.stdout)
    if stats is None:
        # The harvester always prints a JSON object, so this means IT failed —
        # a harness fault, not a translation fault.
        return RResult(False, {}, proc.stdout, proc.stderr, error="harvester_produced_no_output")

    # The harvester reports the player's own failure this way, so a broken
    # translation is distinguishable from a broken harness.
    if "__arena_error__" in stats:
        return RResult(False, {}, proc.stdout, proc.stderr,
                       error=str(stats["__arena_error__"])[:300])

    return RResult(True, stats, proc.stdout, proc.stderr)
