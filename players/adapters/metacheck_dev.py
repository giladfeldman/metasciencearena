"""MetacheckDevAdapter — the same metacheck.R player, run against metacheck's DEVELOPMENT version.

Owner decision 2026-10-17: re-check metacheck at its development version (it moved
PDF conversion from GROBID to bibr). The development build is installed in its own R
library (players/local_paths.yaml placeholder METACHECK_DEV_RLIB, default
%LOCALAPPDATA%/R/sciencearena-libs/metacheck-0.3.3.12), pinned to one commit, so the
published metacheck-0.1.0 player keeps running unchanged next to it.

Both the run AND the version probe use that library: a probe against the default
library would label a 0.3.3.12 run "metacheck-0.1.0", the exact provenance defect
framework/tests/test_provenance_labels.py exists to catch.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from framework.player_adapter import RCliAdapter, register_adapter_class


def _dev_lib() -> str:
    lib = os.environ.get("METACHECK_DEV_RLIB") or str(
        Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData/Local"))) / "R/sciencearena-libs/metacheck-0.3.3.12")
    return lib.replace("\\", "/")


class MetacheckDevAdapter(RCliAdapter):

    def _env(self) -> dict:
        env = dict(os.environ)
        lib = _dev_lib()
        if not Path(lib).is_dir():
            raise RuntimeError(f"metacheck development library not found at {lib} (set METACHECK_DEV_RLIB)")
        env["R_LIBS"] = lib + os.pathsep + env.get("R_LIBS", "")
        return env

    def resolved_tool_version(self) -> str | None:
        try:
            p = subprocess.run([self.rscript_binary, "-e",
                                f'cat(as.character(packageVersion("metacheck", lib.loc="{_dev_lib()}")))'],
                               capture_output=True, text=True, timeout=60)
        except Exception:
            return None
        v = p.stdout.strip()
        return f"metacheck-{v}" if p.returncode == 0 and v else None

    def play_task(self, envelope: dict, *, timeout_s: int) -> dict:
        proc = subprocess.run([self.rscript_binary, str(self.r_script)], input=json.dumps(envelope),
                              capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=timeout_s, check=False, env=self._env())
        if proc.returncode != 0:
            raise RuntimeError(f"{self.r_script.name} (metacheck dev) exited {proc.returncode}: {proc.stderr.strip()[-800:]}")
        try:
            return json.loads(proc.stdout)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"{self.r_script.name} produced invalid JSON: {e}; raw: {proc.stdout[:300]}")


register_adapter_class("MetacheckDevAdapter", MetacheckDevAdapter)
