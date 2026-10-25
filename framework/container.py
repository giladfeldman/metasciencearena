"""Run each player's adapter inside the pinned arena runner image (CC8, schema v12).

DIRECTIVE (owner, 2026-10-12/14; ScienceArena CLAUDE.md): no arena run outside
the runner image, and a record without ``env_sha`` is rejected by the runner.

How it works. The host runner keeps the parts that need the answer key: it
generates tasks, scores answers and writes records. For every model call it
launches ``docker run --rm`` on the runner image with ONLY the task envelope
mounted read-only. The container has no repository, no gold, no seed and no
home directory. Its network is an ``--internal`` Docker network with no route
out; the only exit is an egress proxy (framework/docker/egress) that admits
CONNECT to the player's provider host and nothing else. The image is built from
a staged copy of the code only (framework/, players/ code and prompts, arena
prompts), so its tag is a content hash of exactly what ran.

What is not yet containerised is an explicit, named list (EXEMPT_ADAPTER_CLASSES)
and every record from it says so: its env is ``isolation: host-exempt`` with the
exemption named. Each exemption has a TODO item; none is silent.

Measured basis (Stage B spike, docs/design/2026-10-runner-harness-decision.md):
per-call container start adds ~1-2 s; the egress proxy blocks a non-allowlisted
host ("403 Filtered") and the internal network has no DNS for a direct call.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse

from framework import run_env

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCKER_DIR = REPO_ROOT / "framework" / "docker"
IMAGE_REPO = "sciencearena/runner"
EGRESS_IMAGE = "sciencearena/egress:1"
INTERNAL_NETWORK = "sa-int"
RESOURCES = {"memory": "1g", "cpus": "1", "read_only_rootfs": True, "user": "10001", "pids_limit": 256}

#: Adapter classes that run in the runner image.
CONTAINER_ADAPTER_CLASSES = frozenset({"OpenAIChatCompletionsAdapter", "GeminiGenerateContentAdapter"})

#: Adapter classes that still run on the host, each with the reason. A record
#: from one of these carries an env with isolation "host-exempt" naming it.
#: Removing an entry means containerising that class; adding one needs a reason.
EXEMPT_ADAPTER_CLASSES: dict[str, str] = {
    "SubprocessCliAdapter": "subscription CLI players need OAuth tokens and per-CLI images (CC8 design doc s.2)",
    "AntigravityCliAdapter": "subscription CLI player; needs its own image and token path",
    "LlmCliPdfAdapter": "subscription CLI player with PDF input",
    "LlmCliPdfSectionsAdapter": "subscription CLI player with PDF input",
    "LlmCliPdfTablesAdapter": "subscription CLI player with PDF input",
    "LlmPdfReferencesAdapter": "subscription CLI player with PDF input",
    "LlmPdfCitationsAdapter": "subscription CLI player with PDF input",
    "LocalGgufAdapter": "local llama.cpp server on the host GPU/CPU; container must reach host.docker.internal",
    "RCliAdapter": "R tools need an R layer in the runner image",
    "StatsPipelineAdapter": "host PDF extractor + R checker (statcheck/effectcheck/metacheck); needs an R layer in the runner image (d-98e5b5)",
    "MetacheckDevAdapter": "R tool (metacheck dev build); needs an R layer in the runner image (d-98e5b5)",
    "AgenticContainerAdapter": "subscription CLI agent in its own locked per-task container (framework/agentic_container.py); needs OAuth tokens",
    "HttpAdapter": "local platform servers (ESCImate etc.) run on the host",
    "ScimetoReplicationApiAdapter": "local Scimeto server runs on the host",
    "DocpluckLibraryAdapter": "PDF tool players: image needs poppler and the tool's pinned version",
    "DocpluckSectionsAdapter": "PDF tool players: image needs poppler and the tool's pinned version",
    "DocpluckTablesAdapter": "PDF tool players: image needs poppler and the tool's pinned version",
    "DoclingTextAdapter": "ML extractor with its own venv and model weights",
    "DoclingTablesAdapter": "ML extractor with its own venv and model weights",
    "DoclingSectionsAdapter": "ML extractor with its own venv and model weights",
    "MarkerTextAdapter": "ML extractor with its own venv and model weights",
    "MarkerSectionsAdapter": "ML extractor with its own venv and model weights",
    "MarkerTablesAdapter": "ML extractor with its own venv and model weights",
    "MineruTextAdapter": "ML extractor with its own venv and model weights",
    "MineruSectionsAdapter": "ML extractor with its own venv and model weights",
    "MineruTablesAdapter": "ML extractor with its own venv and model weights",
    "LiteparseTextAdapter": "Node tool; needs a Node layer",
    "LiteparseSectionsHeuristicAdapter": "Node tool; needs a Node layer",
    "LiteparseTablesHeuristicAdapter": "Node tool; needs a Node layer",
    "PdftotextSubprocessAdapter": "poppler binary on the host",
    "GrobidTextAdapter": "talks to the local GROBID container",
    "GrobidSectionsAdapter": "talks to the local GROBID container",
    "GrobidTablesAdapter": "talks to the local GROBID container",
    "GrobidReferencesAdapter": "talks to the local GROBID container",
    "GrobidCitationsAdapter": "talks to the local GROBID container",
    "AnystyleReferencesAdapter": "already runs in its own tool container (players/docker)",
    "CermineReferencesAdapter": "already runs in its own tool container (players/docker)",
    "RegcheckShimAdapter": "third-party shim with its own client stack",
    "ModelProxyAdapter": "Kaggle model proxy client",
    "XlatFixtureAdapter": "no I/O fixture",
    "FredFridaOnlyAdapter": "no I/O fixture",
    "HumanReplayAdapter": "replays human answers from a file; no model call",
    "AnthropicApiAdapter": "published runs replay Anthropic Batch API answers from local result files (no model call during the run); live mode needs a key path and api.anthropic.com egress in the image (d-7aca62)",
    "StubPassAdapter": "test stub, no I/O",
    "StubFailAdapter": "test stub, no I/O",
    "StubSchemaViolationAdapter": "test stub, no I/O",
}


class RunnerUnavailable(RuntimeError):
    """Docker or the runner image cannot be used; the run must not fall back to the host."""


def adapter_class_of(entry: dict) -> str:
    return (entry.get("adapter_class") or "").rsplit(".", 1)[-1]


def placement(entry: dict) -> str:
    """'container' or 'host-exempt'. Raises for an adapter class nobody classified."""
    cls = adapter_class_of(entry)
    if cls in CONTAINER_ADAPTER_CLASSES and not entry.get("document_to_text"):
        return "container"
    if cls in EXEMPT_ADAPTER_CLASSES or cls in CONTAINER_ADAPTER_CLASSES:
        return "host-exempt"
    raise RunnerUnavailable(
        f"adapter class {cls!r} is neither containerised nor a named exemption "
        f"(framework/container.py). Classify it before it can play.")


def exemption_reason(entry: dict) -> str:
    cls = adapter_class_of(entry)
    if cls in CONTAINER_ADAPTER_CLASSES and entry.get("document_to_text"):
        return f"{cls} with document_to_text={entry['document_to_text']}: the image has no poppler"
    return f"{cls}: {EXEMPT_ADAPTER_CLASSES[cls]}"


# ------------------------------------------------------------------ image
def _staged_files() -> list[tuple[Path, str]]:
    """(source, path-in-image) for exactly the code that goes into the image."""
    out: list[tuple[Path, str]] = []

    def add_tree(rel: str, suffixes: tuple[str, ...], skip_parts=("tests", "__pycache__", "_experiments", "_archive")):
        base = REPO_ROOT / rel
        for p in sorted(base.rglob("*")):
            if p.is_file() and p.suffix in suffixes and not (set(p.relative_to(REPO_ROOT).parts) & set(skip_parts)):
                out.append((p, p.relative_to(REPO_ROOT).as_posix()))

    add_tree("framework", (".py", ".json"))
    add_tree("players/adapters", (".py",))
    add_tree("players/prompts", (".txt", ".md"))
    out.append((REPO_ROOT / "players" / "__init__.py", "players/__init__.py"))
    for p in sorted((REPO_ROOT / "arenas").glob("*/prompts/*")):
        if p.is_file():
            out.append((p, p.relative_to(REPO_ROOT).as_posix()))
    return out


def code_hash() -> str:
    h = hashlib.sha256()
    for src, dst in _staged_files():
        h.update(dst.encode() + b"\0" + src.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    for f in ("Dockerfile", "requirements.txt"):
        p = DOCKER_DIR / "runner" / f
        h.update(f.encode() + b"\0" + p.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def _docker(*args: str, check: bool = True, timeout: int | None = None, input: str | None = None):
    try:
        cp = subprocess.run(["docker", *args], capture_output=True, text=True, encoding="utf-8",
                            errors="replace", timeout=timeout, input=input)
    except FileNotFoundError as exc:
        raise RunnerUnavailable("docker is not installed or not on PATH") from exc
    if check and cp.returncode != 0:
        raise RunnerUnavailable(f"docker {' '.join(args[:2])} failed: {(cp.stderr or cp.stdout)[-600:]}")
    return cp


def ensure_image() -> dict:
    """Build the runner image for the current code if missing; return its identity."""
    ch = code_hash()
    tag = f"{IMAGE_REPO}:env-{ch[:16]}"
    cp = _docker("image", "inspect", "--format", "{{.Id}}", tag, check=False)
    if cp.returncode != 0:
        with tempfile.TemporaryDirectory(prefix="sa-runner-ctx-") as ctx:
            ctxp = Path(ctx)
            for src, dst in _staged_files():
                target = ctxp / "code" / dst
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(src.read_bytes().replace(b"\r\n", b"\n"))
            shutil.copy(DOCKER_DIR / "runner" / "requirements.txt", ctxp / "requirements.txt")
            shutil.copy(DOCKER_DIR / "runner" / "Dockerfile", ctxp / "Dockerfile")
            _docker("build", "-t", tag, "--label", f"sciencearena.code_hash={ch}", ctx, timeout=3600)
        cp = _docker("image", "inspect", "--format", "{{.Id}}", tag)
    image_id = cp.stdout.strip()
    py = _docker("run", "--rm", "--network", "none", "--entrypoint", "python", tag, "-c",
                 "import sys,platform;print(sys.version.split()[0])").stdout.strip()
    freeze = _docker("run", "--rm", "--network", "none", "--entrypoint", "cat", tag,
                     "/opt/sa-pip-freeze.txt").stdout.replace("\r\n", "\n")
    packages = sorted(l.strip() for l in freeze.splitlines() if l.strip())
    return {"image": tag, "image_id": image_id, "code_hash": ch, "python": py,
            "pip_freeze": packages,
            "pip_freeze_sha256": hashlib.sha256("\n".join(packages).encode("utf-8")).hexdigest(),
            "dockerfile_sha256": run_env.file_sha256(DOCKER_DIR / "runner" / "Dockerfile"),
            "requirements_sha256": run_env.file_sha256(DOCKER_DIR / "runner" / "requirements.txt")}


def ensure_egress_image() -> dict:
    cp = _docker("image", "inspect", "--format", "{{.Id}}", EGRESS_IMAGE, check=False)
    if cp.returncode != 0:
        _docker("build", "-t", EGRESS_IMAGE, str(DOCKER_DIR / "egress"), timeout=1800)
        cp = _docker("image", "inspect", "--format", "{{.Id}}", EGRESS_IMAGE)
    return {"image": EGRESS_IMAGE, "image_id": cp.stdout.strip(),
            "dockerfile_sha256": run_env.file_sha256(DOCKER_DIR / "egress" / "Dockerfile")}


# ------------------------------------------------------------------ network
def provider_host(entry: dict) -> str:
    cls = adapter_class_of(entry)
    if cls == "GeminiGenerateContentAdapter":
        return "generativelanguage.googleapis.com"
    env = entry.get("openai_base_url_env")
    base = os.environ.get(env, "").strip() if env else ""
    host = urlparse(base).hostname if base else None
    if not host:
        raise RunnerUnavailable(f"{entry['player_id']}: cannot derive the provider host "
                                f"({env} is unset or not a URL)")
    if host in ("localhost", "127.0.0.1", "::1") or host.endswith(".localhost"):
        raise RunnerUnavailable(f"{entry['player_id']}: loopback endpoint {host}; local servers are host-exempt")
    return host


def ensure_proxy(host: str) -> str:
    """One egress proxy container per provider host on the internal network."""
    name = "sa-egress-" + "".join(c if c.isalnum() else "-" for c in host)[:50]
    _docker("network", "create", "--internal", INTERNAL_NETWORK, check=False)
    st = _docker("inspect", "--format", "{{.State.Running}}|{{range .Config.Env}}{{.}};{{end}}", name, check=False)
    if st.returncode == 0 and st.stdout.startswith("true|") and f"ALLOW_HOSTS={host};" in st.stdout:
        return name
    _docker("rm", "-f", name, check=False)
    _docker("run", "-d", "--name", name, "--restart", "unless-stopped", "-e", f"ALLOW_HOSTS={host}", EGRESS_IMAGE)
    _docker("network", "connect", INTERNAL_NETWORK, name)
    for _ in range(20):
        if "Listening" in (_docker("logs", name, check=False).stderr or "") + (_docker("logs", name, check=False).stdout or ""):
            break
        time.sleep(0.25)
    return name


def api_key_env(entry: dict) -> str | None:
    return entry.get("gemini_api_key_env") or entry.get("openai_api_key_env") or (
        "GEMINI_API_KEY" if adapter_class_of(entry) == "GeminiGenerateContentAdapter" else None)


# ------------------------------------------------------------------ env
def container_env(entry: dict, runner: dict, egress: dict, generator_fingerprint: str | None) -> dict:
    return run_env.build_env(
        isolation="container", runner=runner, egress=egress,
        network={"mode": "internal-network+egress-proxy", "allow_hosts": [provider_host(entry)]},
        resources=RESOURCES, generator_fingerprint=generator_fingerprint)


def host_env(entry: dict, generator_fingerprint: str | None) -> dict:
    return run_env.build_env(
        isolation="host-exempt", runner=None, egress=None,
        network={"mode": "host", "allow_hosts": "unrestricted"}, resources=None,
        generator_fingerprint=generator_fingerprint, exemption=exemption_reason(entry))


# ------------------------------------------------------------------ the call
def run_in_container(entry: dict, envelope: dict, timeout_s: int, runner: dict, host: str) -> dict:
    """Play one task in a fresh container; return the container's result dict.

    The job is written to the SYSTEM temp directory (outside every project:
    held-out envelopes carry real article text) and removed afterwards.
    """
    proxy = ensure_proxy(host)
    key_env = api_key_env(entry)
    work = Path(tempfile.mkdtemp(prefix="sa-job-"))
    try:
        (work / "in").mkdir()
        (work / "out").mkdir()
        (work / "in" / "job.json").write_text(json.dumps(
            {"entry": entry, "envelope": envelope, "timeout_s": timeout_s}, ensure_ascii=False), encoding="utf-8")
        args = ["run", "--rm", "--network", INTERNAL_NETWORK,
                "--memory", RESOURCES["memory"], "--cpus", RESOURCES["cpus"], "--pids-limit", str(RESOURCES["pids_limit"]),
                "--read-only", "--tmpfs", "/tmp", "--user", RESOURCES["user"], "--cap-drop", "ALL",
                "--security-opt", "no-new-privileges",
                "-e", f"HTTPS_PROXY=http://{proxy}:8888", "-e", f"HTTP_PROXY=http://{proxy}:8888",
                "-v", f"{work / 'in'}:/job:ro", "-v", f"{work / 'out'}:/out"]
        if key_env:
            if not os.environ.get(key_env):
                raise RunnerUnavailable(f"{entry['player_id']}: {key_env} is not set")
            args += ["-e", key_env]
        # The base URL, and the provider's model allowlist: without the latter the
        # adapter's "refuse a model not on the list" check would silently pass
        # inside the container, because the variable would simply be absent.
        for var_key in ("openai_base_url_env", "openai_available_models_env"):
            var = entry.get(var_key)
            if var and os.environ.get(var):
                args += ["-e", var]
        args += [runner["image"], "/job/job.json", "/out/result.json"]
        cp = _docker(*args, check=False, timeout=timeout_s + 120)
        res_path = work / "out" / "result.json"
        if not res_path.exists():
            return {"output": None, "error": f"RunnerContainerFailed: exit {cp.returncode}: "
                                             f"{(cp.stderr or cp.stdout)[-400:]}"}
        return json.loads(res_path.read_text(encoding="utf-8"))
    finally:
        shutil.rmtree(work, ignore_errors=True)


class ContainerizedAdapter:
    """Host-side stand-in with the PlayerAdapter telemetry surface the runner reads."""

    def __init__(self, entry: dict, runner: dict):
        self.entry = entry
        self.runner = runner
        self.player_id = entry["player_id"]
        self.host = provider_host(entry)
        self._res: dict = {}
        self.containment_argv = None

    def prepare(self) -> None:
        ensure_proxy(self.host)

    def play_task(self, envelope: dict, *, timeout_s: int):
        # Cleared FIRST: if the call raises, the runner must read no telemetry rather
        # than the previous task's, which it would store under this task's run_id
        # (Sol, consult 2026-10-19; test_stale_raw_response.py).
        self._res = {}
        self._res = run_in_container(self.entry, envelope, timeout_s, self.runner, self.host)
        if self._res.get("error"):
            raise RuntimeError(self._res["error"])
        return self._res.get("output")

    def last_usage(self):
        return self._res.get("usage")

    def last_response_meta(self):
        return self._res.get("response_meta")

    def last_request_max_tokens(self):
        v = self._res.get("request_max_tokens")
        return v if isinstance(v, int) and not isinstance(v, bool) else None

    def last_request_body(self):
        return self._res.get("request_body")

    def last_raw_response(self):
        return self._res.get("raw_response")

    def resolved_tool_version(self):
        return self._res.get("resolved_tool_version") or self.entry.get("openai_model") or self.entry.get("gemini_model")

    def cleanup(self) -> None:
        pass
