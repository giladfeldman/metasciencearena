"""Vendored-fixture adapter for code-translation-r-v1.

The SPSS/Stata -> R converters in this arena (spss2rmarkdown, SPSStoR,
skranz/stata2r) are R packages, two of which are GitHub-only and one of which
has been dormant since 2021. Invoking them live would make the arena unrunnable
on any machine without R plus that exact set of packages installed from source.

So each converter is run ONCE by hand against the arena's source scripts and its
emitted R is committed as a version-pinned fixture (user decision 2026-08-03).
This adapter simply serves those fixtures. The result is still an honest
measurement — the fixture is that tool's real output, and it is scored by the
same executable-equivalence path as every other player — but the version is
pinned at capture time rather than detected at run time.

A missing fixture is reported as an explicit refusal (empty `r_code`, which the
scorer records as `does_not_execute`) rather than an exception: a converter that
cannot handle a construct is a real, interesting result, not a harness bug.
"""
from __future__ import annotations

import json
from pathlib import Path

from framework.player_adapter import PlayerAdapter, register_adapter_class


class XlatFixtureAdapter(PlayerAdapter):
    """Serve a converter's pre-captured R output for a task.

    Fixtures live at ``<fixture_dir>/<analysis_id>.json`` and hold::

        {"r_code": "...", "note": "optional capture note"}

    ``analysis_id`` is recovered from the task_id (``xlat-<analysis>-<lang>-s<seed>``)
    so a fixture is shared by both language variants only when the tool genuinely
    produced the same translation; normally each tool covers one language and the
    other resolves to a refusal.
    """

    def __init__(self, *args, fixture_dir: str | Path,
                 fixture_tool_version: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fixture_dir = Path(fixture_dir)
        self.fixture_tool_version = fixture_tool_version

    @staticmethod
    def _analysis_id(task_id: str) -> str:
        # xlat-<analysis_id>-<language>-s<seed>; analysis ids contain '_' not '-',
        # so strip the fixed prefix and the two trailing segments.
        parts = task_id.split("-")
        return "-".join(parts[1:-2]) if len(parts) >= 4 else task_id

    def play_task(self, envelope: dict, *, timeout_s: int) -> dict:
        analysis = self._analysis_id(envelope["task_id"])
        language = envelope.get("input", {}).get("source_language", "")

        # Prefer a language-specific fixture, fall back to the analysis-level one.
        for name in (f"{analysis}.{language}.json", f"{analysis}.json"):
            path = self.fixture_dir / name
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                return {"r_code": data.get("r_code", ""), "confidence": 1.0}

        # No fixture: this converter does not handle this task. Scored as a
        # non-execution, which is exactly what would happen with the real tool.
        return {"r_code": "", "confidence": 1.0}

    def resolved_tool_version(self) -> str | None:
        """The version captured at fixture time.

        Reported from the registry rather than probed, because the tool is not
        installed here — the fixtures ARE the tool for this arena, and pinning
        the version at capture is what keeps the record honest.
        """
        return self.fixture_tool_version


register_adapter_class("XlatFixtureAdapter", XlatFixtureAdapter)
