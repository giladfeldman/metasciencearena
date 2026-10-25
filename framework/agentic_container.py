"""Agentic CLI players inside a locked-down container.

A *single-shot hermetic* player (framework/hermetic.py) gets no tools at all. An *agentic*
player may run code, which is what reproducing an effect size actually involves, so its
containment cannot rest on CLI flags: it rests on the filesystem and the network.

    * one fresh container per task (no state carries between tasks);
    * only the task directory is mounted, read-only; the repo, the article repository and
      the gold are simply absent from the container's filesystem;
    * egress is locked to the inference host by the image's entrypoint (iptables default
      DROP, DNS closed), and the entrypoint refuses to start the CLI if its self-check fails;
    * the CLI runs unprivileged with no capabilities, so the model cannot lift the lock;
    * web tools are not offered (WebSearch runs on the vendor's servers and would bypass
      any local firewall), and user settings, CLAUDE.md and MCP servers are not loaded.

Image: players/docker/agentic-claude (pinned CLI). Auth: CLAUDE_CODE_OAUTH_TOKEN from the
caller's environment (Claude Max), passed by NAME so the value never appears in argv or in
a run record. The leak test is framework/tests/test_agentic_container_leak.py; no score from
this mode counts until it is green for the exact image tag in use.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

CONTAINMENT_AGENTIC_CONTAINER = "agentic-container"
#: provenance.containment.profile_version of an agentic-container record (framework/hermetic.py
#: classifies on the "agentic-container" prefix).
AGENTIC_PROFILE_VERSION = "agentic-container-1"

DEFAULT_IMAGE = "sciencearena/agentic-claude:2.1.291-r1"
#: Built-in tools offered to the agent. No WebFetch/WebSearch, no MCP.
AGENT_TOOLS = "Bash,Read,Write,Edit,Glob,Grep"
TOKEN_ENV = "CLAUDE_CODE_OAUTH_TOKEN"
FORBIDDEN_ENV = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL")


class ContainerUnavailable(RuntimeError):
    """Docker, the image or the subscription token is missing. Never fall back to the host."""


@dataclass
class ContainerResult:
    returncode: int
    stdout: str
    stderr: str
    seconds: float
    image: str
    image_id: str
    argv_redacted: list[str]
    outputs: dict[str, str] = field(default_factory=dict)

    @property
    def egress_locked(self) -> bool:
        return "entrypoint: egress locked to:" in self.stderr


def _docker() -> str:
    exe = shutil.which("docker")
    if not exe:
        raise ContainerUnavailable("docker not on PATH")
    return exe


def image_id(image: str = DEFAULT_IMAGE) -> str:
    """Content digest of the local image; part of the player's config fingerprint."""
    out = subprocess.run([_docker(), "image", "inspect", "--format", "{{.Id}}", image],
                         capture_output=True, text=True)
    if out.returncode != 0 or not out.stdout.strip():
        raise ContainerUnavailable(f"image {image} not built: {out.stderr.strip()[:200]}")
    return out.stdout.strip()


def config_fingerprint(model: str, image: str = DEFAULT_IMAGE) -> str:
    """A containerised agent is a different instrument from the same model single-shot."""
    blob = json.dumps({"mode": CONTAINMENT_AGENTIC_CONTAINER, "image_id": image_id(image),
                       "tools": AGENT_TOOLS, "model": model}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def container_prefix(task_dir: Path, out_dir: Path, *, image: str = DEFAULT_IMAGE,
                     memory: str = "4g", cpus: str = "2", name: str | None = None) -> list[str]:
    """Everything up to and including the image: mounts, limits, lock capability, token."""
    return [
        _docker(), "run", "--rm", "-i", *(["--name", name] if name else []),
        "--cap-add", "NET_ADMIN",          # entrypoint needs it for iptables; dropped before the CLI
        "--memory", memory, "--cpus", cpus, "--pids-limit", "512",
        "-e", TOKEN_ENV,                   # by NAME: value is read from our env, never in argv
        "-v", f"{task_dir.resolve()}:/task:ro",
        "-v", f"{out_dir.resolve()}:/out",
        "--tmpfs", "/work:exec,size=1g,uid=1000,gid=1000",
        "-w", "/work",
        image,
    ]


#: Pretest round 1 (2026-10-17): on a 50 MB meta-analysis PDF all three models used up 40 turns
#: before answering (error_max_turns). Raised for round 2; the answer saved so far is kept.
MAX_TURNS = 80


def agent_command(model: str, *, max_turns: int = MAX_TURNS) -> list[str]:
    return [
        "claude", "--print", "--model", model,
        "--tools", AGENT_TOOLS,
        "--setting-sources", "",
        "--strict-mcp-config",
        "--dangerously-skip-permissions",
        "--max-turns", str(max_turns),
        "--output-format", "json",
    ]


def build_argv(model: str, task_dir: Path, out_dir: Path, *, image: str = DEFAULT_IMAGE,
               max_turns: int = MAX_TURNS) -> list[str]:
    return container_prefix(task_dir, out_dir, image=image) + agent_command(model, max_turns=max_turns)


def run_task(model: str, prompt: str, files: dict[str, bytes], *, image: str = DEFAULT_IMAGE,
             timeout_s: int = 1800, env: dict[str, str] | None = None,
             command: list[str] | None = None) -> ContainerResult:
    """Run one task in a fresh container. ``files`` land read-only under /task.

    ``command`` replaces the agent CLI with an arbitrary command run as the same
    unprivileged player with the same mounts and environment: the leak test uses it to
    probe exactly the surface the agent's Bash tool would have."""
    base_env = dict(os.environ if env is None else env)
    if not base_env.get(TOKEN_ENV):
        raise ContainerUnavailable(f"{TOKEN_ENV} not set (Claude Max token; never an API key)")
    for k in FORBIDDEN_ENV:
        base_env.pop(k, None)
    iid = image_id(image)
    with tempfile.TemporaryDirectory(prefix="sa-agentic-") as tmp:
        task_dir, out_dir = Path(tmp, "task"), Path(tmp, "out")
        task_dir.mkdir(); out_dir.mkdir()
        for name, data in files.items():
            if "/" in name or "\\" in name or name.startswith("."):
                raise ValueError(f"task file names must be flat: {name!r}")
            (task_dir / name).write_bytes(data)
        # A named container, so a timeout can remove it: killing the docker CLIENT leaves the
        # container (and the agent, holding the token) running. Measured 2026-10-17: five
        # orphaned agentic containers after 120 s client timeouts.
        name = "sa-agentic-" + uuid.uuid4().hex[:12]
        prefix = container_prefix(task_dir, out_dir, image=image, name=name)
        argv = prefix + (command if command else agent_command(model))
        t0 = time.monotonic()
        try:
            proc = subprocess.run(argv, input=prompt, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", env=base_env, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            subprocess.run([_docker(), "rm", "-f", name], capture_output=True, timeout=120)
            raise
        outputs = {p.name: p.read_text(encoding="utf-8", errors="replace")
                   for p in out_dir.iterdir() if p.is_file()}
        return ContainerResult(proc.returncode, proc.stdout, proc.stderr,
                               round(time.monotonic() - t0, 1), image, iid,
                               [a if a != base_env.get(TOKEN_ENV) else "<redacted>" for a in argv],
                               outputs)
