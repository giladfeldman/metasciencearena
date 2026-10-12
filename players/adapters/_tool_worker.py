"""Client for the out-of-process tool workers in players/tool_envs/<tool>/.

See players/tool_envs/_worker_common.py for the protocol and for why these
extractors run in their own pinned venvs. This side owns three guarantees:

* A CRASH IS A FAILURE, NEVER AN EMPTY SUCCESS. If the worker process dies
  mid-task (native crash, heap corruption, OOM kill) `call` raises
  `ToolWorkerCrashed` carrying the exit code and the tail of the worker's stderr;
  the runner records that as an errored task scored 0.0 (framework/runner.py
  `_play_one`). The next task starts a fresh worker.
* A HANG IS A TIMEOUT. The per-task ceiling the runner passes is meant to be
  enforced by killing the worker; the record says `TimeoutError`, which the run
  summary counts as a timeout rather than as incapability. NOT YET RELIABLE: on
  2026-10-08 a Docling task recorded 8541 s against an 1800 s ceiling before the
  TimeoutError (debt d-2d727b) -- do not rely on the ceiling until that is fixed.
* THE VERSION IS READ BACK, NOT ASSUMED. The worker reports its versions from
  its own installed distribution metadata in the ready line, and that is what
  `versions()` returns -- never the pin in requirements.lock.

The venv is resolved from `players/tool_envs/<tool>/.venv` unless the env var
`SCIENCEARENA_TOOL_ENV_<TOOL>` names another interpreter (a second machine, CI).
A missing venv fails `start()` loudly with the command that creates it.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path

import tempfile

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_ENVS = REPO_ROOT / "players" / "tool_envs"
# OUTSIDE the repository: a tool's stderr can quote the paper it was parsing,
# and held-out text must never be stored in a project directory, gitignored or
# not (Sonnet review 2026-10-07).
LOG_DIR = Path(tempfile.gettempdir()) / "sciencearena-tool-worker-logs"


class ToolWorkerCrashed(RuntimeError):
    """The worker process exited while a task was in flight."""


def python_for(tool: str) -> Path:
    override = os.environ.get(f"SCIENCEARENA_TOOL_ENV_{tool.upper()}")
    if override:
        return Path(override)
    sub = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    return TOOL_ENVS / tool / ".venv" / sub


def _exit_code_text(rc: int | None) -> str:
    if rc is None:
        return "unknown"
    u = rc & 0xFFFFFFFF
    return f"{rc} (0x{u:08X})" if rc < 0 or rc > 255 else str(rc)


class ToolWorker:
    def __init__(self, tool: str, *, env: dict[str, str] | None = None, start_timeout_s: int = 1800):
        self.tool = tool
        self.env_extra = dict(env or {})
        self.start_timeout_s = start_timeout_s
        self._proc: subprocess.Popen | None = None
        self._lines: queue.Queue = queue.Queue()
        self._versions: dict | None = None
        self._log_path: Path | None = None
        self.load_seconds: float | None = None
        self.restarts = 0

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        py = python_for(self.tool)
        script = TOOL_ENVS / self.tool / "worker.py"
        if not py.is_file():
            raise RuntimeError(
                f"{self.tool} venv missing at {py}. Create it with: "
                f"uv venv --python 3.12 {py.parents[1]} && "
                f"uv pip sync {TOOL_ENVS / self.tool / 'requirements.lock'} --python {py}"
            )
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self._log_path = LOG_DIR / f"{self.tool}-{os.getpid()}-{int(time.time())}.stderr.log"
        env = {**os.environ, **self.env_extra, "PYTHONIOENCODING": "utf-8"}
        self._lines = queue.Queue()
        self._proc = subprocess.Popen(
            [str(py), str(script)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self._log_path.open("wb"),
            env=env, cwd=str(REPO_ROOT),
        )
        proc = self._proc
        threading.Thread(target=self._pump, args=(proc, self._lines), daemon=True).start()
        ready = self._read(self.start_timeout_s, what="start-up")
        if not ready.get("ready"):
            raise RuntimeError(f"{self.tool} worker sent {ready!r} instead of a ready line")
        self._versions = ready.get("versions") or {}
        self.load_seconds = ready.get("load_seconds")

    @staticmethod
    def _pump(proc: subprocess.Popen, q: queue.Queue) -> None:
        assert proc.stdout is not None
        for raw in proc.stdout:
            q.put(raw)
        q.put(None)  # EOF

    def _stderr_tail(self, n: int = 800) -> str:
        try:
            data = self._log_path.read_bytes() if self._log_path else b""
        except OSError:
            return ""
        return data[-n:].decode("utf-8", "replace")

    def _read(self, timeout_s: float, *, what: str) -> dict:
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.stop()
                raise TimeoutError(f"{self.tool} worker exceeded {int(timeout_s)}s during {what}")
            try:
                raw = self._lines.get(timeout=min(remaining, 5))
            except queue.Empty:
                if self._proc is not None and self._proc.poll() is not None and self._lines.empty():
                    raw = None
                else:
                    continue
            if raw is None:
                rc = self._proc.wait() if self._proc else None
                self._proc = None
                raise ToolWorkerCrashed(
                    f"{self.tool} worker exited with code {_exit_code_text(rc)} during {what}; "
                    f"stderr tail: {self._stderr_tail()!r}"
                )
            line = raw.decode("utf-8", "replace").strip()
            if line.startswith("{"):
                return json.loads(line)

    def stop(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()
        except OSError:
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            # /T: a tool that forked helpers (Marker's layout server) leaves
            # orphans holding RAM and CPU under a plain kill.
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                               capture_output=True)
            else:
                proc.kill()
            proc.wait(timeout=30)

    # -- per task ----------------------------------------------------------
    def call(self, request: dict, timeout_s: int) -> dict:
        if self._proc is None or self._proc.poll() is not None:
            if self._versions is not None:
                self.restarts += 1
            self.start()
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write((json.dumps(request) + "\n").encode("utf-8"))
        self._proc.stdin.flush()
        resp = self._read(timeout_s, what=f"task {request.get('task_id', '?')}")
        if not resp.get("ok"):
            raise RuntimeError(f"{self.tool}: {resp.get('error')}")
        return resp["result"]

    def versions(self) -> dict:
        """Versions from the running worker's ready line; else a model-free probe.

        The probe (`worker.py --versions`) exists so a worker that cannot START
        -- the very case a crash record needs a version for -- still yields one.
        """
        if self._versions is not None:
            return dict(self._versions)
        py = python_for(self.tool)
        env = {**os.environ, **self.env_extra, "PYTHONIOENCODING": "utf-8"}
        r = subprocess.run([str(py), str(TOOL_ENVS / self.tool / "worker.py"), "--versions"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace",
                           env=env, cwd=str(REPO_ROOT), timeout=600)
        for line in reversed((r.stdout or "").splitlines()):
            if line.startswith('{"versions"'):
                self._versions = json.loads(line)["versions"]
                return dict(self._versions)
        raise RuntimeError(f"{self.tool} --versions probe failed (exit {r.returncode}): {r.stderr[-400:]!r}")


class pdf_on_disk:
    """Write a task's PDF bytes to a temp file OUTSIDE the repository.

    Held-out papers are licensed material in article-finder custody; they must
    never land inside a project directory, gitignored or not. The OS temp dir is
    outside it, and the file is removed when the task ends.
    """

    def __init__(self, pdf_bytes: bytes, filename: str = "task.pdf"):
        self.pdf_bytes = pdf_bytes
        self.suffix = Path(filename).suffix or ".pdf"
        self.path: str | None = None

    def __enter__(self) -> str:
        import tempfile

        fd, self.path = tempfile.mkstemp(prefix="sa-task-", suffix=self.suffix)
        with os.fdopen(fd, "wb") as f:
            f.write(self.pdf_bytes)
        return self.path

    def __exit__(self, *exc) -> None:
        if self.path:
            try:
                os.unlink(self.path)
            except OSError:
                pass


def mineru_env() -> dict[str, str]:
    """MinerU's state lives OUTSIDE the repo and outside the shared ~/.mineru.

    Its library caches parsed document text, so it must not sit in the project
    (article custody), and the shared default would put this arena's parses in
    another session's server. See players/tool_envs/mineru/worker.py.
    """
    import tempfile

    return {"MINERU_HOME": os.environ.get("SCIENCEARENA_MINERU_HOME")
            or str(Path(tempfile.gettempdir()) / "sciencearena-mineru-home")}
