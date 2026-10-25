"""AgenticContainerAdapter -- a Claude model that may run code, one locked container per task.

The arena prompt (the same template the single-shot players get, with the same extracted text)
is written to /task/task_prompt.txt beside the task's PDFs (`envelope.input.documents`); the
agent is told, by players/prompts/ma_agentic_wrapper_v1.txt, to read it, work with Bash / R /
Python / pdftotext, and write its JSON answer to /out/answer.json. Containment, egress lock and
token handling are framework/agentic_container.py's; this adapter adds nothing to them and never
falls back to the host when the container is unavailable.

The answer is taken from /out/answer.json when the agent wrote one, else from the last JSON
object in the CLI's final message. Which of the two was used is recorded in
`player_strategy_notes`, so a run can be audited for agents that never wrote the file.
"""
from __future__ import annotations

import base64
import json
import re
from pathlib import Path

from framework import agentic_container as ac
from framework.player_adapter import (VERSION_ABSENT_BY_DESIGN, PlayerAdapter, VersionAbsence,
                                      register_adapter_class)

WRAPPER = Path(__file__).resolve().parents[1] / "prompts" / "ma_agentic_wrapper_v1.txt"


def _last_json_object(text: str) -> dict | None:
    s = (text or "").strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else None
    except ValueError:
        pass
    for m in reversed(list(re.finditer(r"\{", s))):
        try:
            v, _end = json.JSONDecoder().raw_decode(s[m.start():])
            if isinstance(v, dict):
                return v
        except ValueError:
            continue
    return None


class AgenticContainerAdapter(PlayerAdapter):

    def __init__(self, *args, cli_command: list[str], prompt_template_path: str | Path,
                 docker_container: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        # cli_command carries the model the same way the single-shot entries do:
        # ["claude", "--print", "--model", "<model>"]; only the model is used here.
        if "--model" not in cli_command:
            raise ValueError("cli_command must name --model")
        self.model = cli_command[cli_command.index("--model") + 1]
        self.image = docker_container or ac.DEFAULT_IMAGE
        self.prompt_template_path = Path(prompt_template_path)
        self._template: str | None = None
        self._wrapper: str | None = None
        self._last_usage: dict | None = None

    def version_absence_reason(self) -> VersionAbsence:
        return VersionAbsence(VERSION_ABSENT_BY_DESIGN,
                              "the model id is the version; the image id is in containment_argv")

    def prepare(self) -> None:
        self._template = self.prompt_template_path.read_text(encoding="utf-8")
        self._wrapper = WRAPPER.read_text(encoding="utf-8")
        ac.image_id(self.image)  # raises ContainerUnavailable now, not mid-run
        self.containment_argv = ac.build_argv(self.model, Path("task"), Path("out"), image=self.image)

    def containment_block(self) -> dict:
        """provenance.containment for this player: the container, not the CLI flags.
        `verified` is False on purpose: the proof is the live leak test, which the runner
        cannot see; run_pretest.ps1 refuses agentic players without a green one."""
        # Fits run-record schema v11 as it is: the container is named by profile_version
        # (the classifier keys on it), the image tag in cli_version (the CLI is pinned in the
        # image), and the image digest enters `fingerprint` via config_fingerprint.
        return {"profile_version": ac.AGENTIC_PROFILE_VERSION, "cli": "claude",
                "cli_version": f"image {self.image}",
                "allow_tools": ac.AGENT_TOOLS.split(","), "verified": False, "flags_insufficient": False,
                "verified_note": "live leak test framework/tests/test_agentic_container_leak.py",
                "fingerprint": ac.config_fingerprint(self.model, self.image)}

    def last_usage(self) -> dict | None:
        return self._last_usage

    def play_task(self, envelope: dict, *, timeout_s: int) -> dict:
        if self._template is None:
            self.prepare()
        self._last_usage = None
        inp = envelope["input"]
        prompt = self._template.replace("{{INPUT_TEXT}}", inp["text"])
        files = {"task_prompt.txt": prompt.encode("utf-8")}
        for name, b64 in (inp.get("documents") or {}).items():
            files[name] = base64.b64decode(b64)
        res = ac.run_task(self.model, self._wrapper, files, image=self.image, timeout_s=timeout_s)
        if not res.egress_locked:
            raise RuntimeError("container did not report its egress lock; refusing the answer")
        envelope_json = _last_json_object(res.stdout) or {}
        if isinstance(envelope_json.get("usage"), dict):
            self._last_usage = envelope_json["usage"]
        saved = _last_json_object(res.outputs["answer.json"]) if "answer.json" in res.outputs else None
        if res.returncode != 0 and saved is None:
            raise RuntimeError(f"agent exited {res.returncode} "
                               f"({envelope_json.get('subtype') or 'no result envelope'}): {res.stderr.strip()[-300:]}")
        answer, source = None, None
        if saved is not None:
            # also after a non-zero exit (e.g. error_max_turns): the agent was told to save its
            # answer early and refine it, and the file is what it last committed to
            answer = saved
            source = "/out/answer.json" + ("" if res.returncode == 0 else
                                          f" (agent then exited {res.returncode}: {envelope_json.get('subtype')})")
        if answer is None:
            answer, source = _last_json_object(str(envelope_json.get("result", ""))), "final message"
        if answer is None:
            # ValueError, as SubprocessCliAdapter's "no parseable JSON": a missing answer is a
            # PARSE failure. Held-out redaction keeps only the exception type, and the pretest
            # gate counts RuntimeError as infrastructure (2026-10-17).
            raise ValueError("agent produced no JSON answer (no /out/answer.json, none in its final message)")
        answer.setdefault("notes", "")
        answer["player_strategy_notes"] = (f"agentic container {self.image} ({res.image_id[:19]}), "
                                           f"answer from {source}, {res.seconds}s")
        return answer


register_adapter_class("AgenticContainerAdapter", AgenticContainerAdapter)
