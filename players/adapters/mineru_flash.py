"""MinerU 4.0.10, `flash` tier, for the three PDF arenas.

Runs out of process in players/tool_envs/mineru/ against a private MinerU server
(own home outside the repository, own port, telemetry disabled and read back
before the first parse; see that worker). AGPL-3.0 code, run locally, unmodified.

ONLY `flash`: it reads the PDF's own text layer. MinerU's basic / standard /
advanced tiers run a vision-language model and are a later, separately-classified
phase; a `mineru-...-flash` row must never be read as MinerU's best tier.

  MineruTextAdapter      markdown -> plain text (_markdown_projection.md_to_plain)
  MineruSectionsAdapter  markdown `#` headings -> flat sections; labels from the
                         arena label map (`tool_native: false`)
  MineruTablesAdapter    MinerU's own structured result: each `table` block's
                         HTML body and the caption MinerU placed inside it

Resolved version: `mineru-<v>+flash` (+`+arena-label-map` for sections), read
back from both the installed client and the server that answered.
"""
from __future__ import annotations

import base64

from framework.player_adapter import PlayerAdapter, register_adapter_class

from players.adapters._markdown_projection import md_outline, md_to_plain, mineru_middle_tables
from players.adapters._outline_sections import sections_from_outline
from players.adapters._tool_worker import ToolWorker, ToolWorkerCrashed, mineru_env, pdf_on_disk

LABEL_STEP = "arena-label-map"


class _MineruBase(PlayerAdapter):
    version_suffix = ""

    def __init__(self, *args, tool_env: str | None = "mineru", **kwargs):
        super().__init__(*args, **kwargs)
        if tool_env != "mineru":
            raise ValueError(f"MinerU players run only in tool_env 'mineru', got {tool_env!r}")
        self.tool_env = tool_env
        self._worker = ToolWorker("mineru", env=mineru_env())

    def prepare(self) -> None:
        self._worker.start()

    def cleanup(self) -> None:
        self._worker.stop()

    def resolved_tool_version(self) -> str | None:
        try:
            v = self._worker.versions()
        except Exception:  # noqa: BLE001 - version detection never fails a task
            return None
        if not v.get("mineru") or v.get("tier") != "flash":
            return None
        return f"mineru-{v['mineru']}+flash{self.version_suffix}"

    def version_absence_reason(self):
        from framework.player_adapter import VERSION_ABSENT_PROBE_FAILED, VersionAbsence

        return VersionAbsence(VERSION_ABSENT_PROBE_FAILED,
                              "the MinerU client/server did not report one agreed version")

    def _convert(self, envelope: dict, timeout_s: int) -> dict:
        pdf_bytes = base64.b64decode(envelope["input"]["document_bytes_b64"])
        with pdf_on_disk(pdf_bytes, envelope["input"].get("filename") or "task.pdf") as path:
            try:
                return self._worker.call({"pdf_path": path, "wait_s": max(60, timeout_s - 60),
                                          "task_id": envelope.get("task_id")}, timeout_s)
            except (TimeoutError, ToolWorkerCrashed):
                self._recover(pdf_bytes)
                raise

    @staticmethod
    def _recover(pdf_bytes: bytes) -> None:
        """After a timeout or crash: stop the server and delete this paper's parse.

        Killing the worker does not stop the MinerU SERVER, which keeps parsing
        the paper with its single parse worker; every later task then queues
        behind it and times out too, and the worker's own cleanup never ran, so
        the paper's stored text would stay behind (Sonnet review 2026-10-07).
        The next task's worker starts a fresh server.
        """
        import hashlib
        import json
        import os
        import shutil
        import subprocess
        from pathlib import Path

        home = Path(mineru_env()["MINERU_HOME"])
        try:
            pid = json.loads((home / "doclib.endpoint.json").read_text(encoding="utf-8")).get("pid")
            if pid:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
                else:
                    os.kill(int(pid), 9)
        except (OSError, ValueError):
            pass
        sha = hashlib.sha256(pdf_bytes).hexdigest()
        shutil.rmtree(home / "doclib" / "parsed" / sha[:2] / sha, ignore_errors=True)

    def _notes(self) -> str:
        return (f"{self.resolved_tool_version()}; mineru parse --tier flash --pages all --force "
                f"--no-marker; local server, telemetry disabled; no VLM")


class MineruTextAdapter(_MineruBase):
    def play_task(self, envelope: dict, *, timeout_s: int) -> dict:
        text = md_to_plain(self._convert(envelope, timeout_s)["markdown"]).strip()
        return {"full_text": text, "pages": [text], "footnotes": [],
                "player_strategy_notes": self._notes() + "; markdown -> plain text, nothing removed for scope"}


class MineruSectionsAdapter(_MineruBase):
    version_suffix = f"+{LABEL_STEP}"

    def play_task(self, envelope: dict, *, timeout_s: int) -> dict:
        sections, full_text = sections_from_outline(md_outline(self._convert(envelope, timeout_s)["markdown"]))
        return {"sections": sections, "full_text": full_text,
                "player_strategy_notes": self._notes() + f"; headings = markdown '#' lines; labels = {LABEL_STEP} (arena code)"}


class MineruTablesAdapter(_MineruBase):
    def play_task(self, envelope: dict, *, timeout_s: int) -> dict:
        tables = mineru_middle_tables(self._convert(envelope, timeout_s)["pages"])
        return {"tables": tables,
                "player_strategy_notes": self._notes() + "; tables from MinerU's structured result; caption only where MinerU placed it in the table block"}


register_adapter_class("MineruTextAdapter", MineruTextAdapter)
register_adapter_class("MineruSectionsAdapter", MineruSectionsAdapter)
register_adapter_class("MineruTablesAdapter", MineruTablesAdapter)
