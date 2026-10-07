"""Local GGUF model player, served by llama.cpp's ``llama-server`` over loopback.

WHY A SEPARATE CLASS RATHER THAN A REGISTRY ENTRY ON ``OpenAIChatCompletionsAdapter``
------------------------------------------------------------------------------------
``llama-server`` speaks the OpenAI chat-completions API, so the parent class can
already talk to it. Two things make that insufficient, and both are failure modes
this project has already shipped once in another guise:

1. **The held-out egress gate reads ``entry["endpoint"]``.** The parent resolves
   its URL from ``openai_base_url_env`` -- an environment variable *name* -- so a
   registry entry for it declares no endpoint at all. ``is_cloud_player``
   (framework/runner.py) therefore sees no evidence, falls through to the
   ``OpenAIChatCompletions`` cloud prefix, and refuses every held-out run against
   a player that never leaves the machine. Declaring an ``endpoint`` alongside an
   env var would be worse: the gate would licence held-out runs from a key the
   request path ignores, which is exactly the GROBID hole documented at
   framework/runner.py:409. So here the declared ``endpoint`` is the ONLY source
   of the URL, and no environment variable can redirect it.

2. **The quantization is part of the player identity.** ``resolved_tool_version``
   on the parent returns the bare model id. A GGUF at UD-IQ3_S and the same model
   at Q8_0 produce different numbers; publishing either under the plain model name
   is the liteparse-2.0.8 defect (the module's ``__version__`` said 2.0.0 while
   2.0.8 ran) with the stakes raised, because here the mislabel changes the SCORE
   and not merely the label. Provenance therefore carries model stem, quant label,
   engine build, and the sha256 of the weights actually loaded.

WHY THIS PLAYER IS WORTH HAVING
-------------------------------
A pinned GGUF + a pinned llama.cpp build + a fixed seed is the only LLM player in
this benchmark that can be re-run identically later. Every API player drifts:
the same model, prompt, task set and scorer six weeks apart scored 0.10 apart, and
14 of 22 arenas rank players run more than 30 days apart. A local player is the
control that spans the same axis as the comparison.

CONTAINMENT: ``llama-server`` is a bare weights server. It has no tools, no shell
and no filesystem access, so it cannot read the answer key the way an agentic CLI
player could (framework/hermetic.py). ``containment_state`` classifies it ``n-a``,
which is correct and needs no special case.
"""
from __future__ import annotations

import hashlib
import logging
import os
import platform
import re
import subprocess
from pathlib import Path
from urllib.parse import urlsplit

import requests

from framework import host
from framework.player_adapter import register_adapter_class
from players.adapters.openai_compatible import OpenAIChatCompletionsAdapter

logger = logging.getLogger(__name__)

#: Env var names the parent class reads for its URL and key. We set them ourselves
#: from the declared `endpoint` and restore them immediately, so this adapter
#: inherits every validation `OpenAIChatCompletionsAdapter.prepare` performs while
#: never actually consulting the ambient environment for a destination. The names
#: are deliberately private and unlikely to be set by anyone: if one leaks, it is
#: overwritten before use rather than read.
_BASE_URL_ENV = "SCIENCEARENA_LOCAL_GGUF_BASE_URL"
_API_KEY_ENV = "SCIENCEARENA_LOCAL_GGUF_API_KEY"

#: llama-server ignores Authorization unless started with --api-key. A placeholder
#: keeps the parent's "both must be set" check meaningful without inventing a
#: secret. It is not a credential and must never be treated as one.
_NO_AUTH_PLACEHOLDER = "local-no-auth"

#: Output ceiling for local runs, replacing the parent's 32000. That number is
#: sized for hosted providers where an over-long completion costs money and
#: seconds; here it costs MINUTES. The incident that motivated this whole player
#: was a 19,712-token budget on a CPU-bound machine: at the ~4 tok/s a dense 27B
#: sustains on LPDDR5x, one runaway answer is ~80 minutes. Measured arena
#: completions across the 16 locally-feasible arenas run 26-629 tokens at the
#: median, so 4096 truncates nothing real while capping a repetition loop at a few
#: minutes. Any task that does hit it is visible as
#: `response_meta.finish_reason == "length"` rather than scored as incapability.
#: Override per player with `openai_extra_body: {max_tokens: N}`.
_LOCAL_MAX_TOKENS = 4096

#: `<stem>-<quant>.gguf`. The quant families llama.cpp emits all start with one of
#: these tokens followed by a digit, which is what separates `...-A3B-UD-Q3_K_XL`
#: (stem `...-A3B`, quant `UD-Q3_K_XL`) from a model name that merely contains
#: digits and hyphens. Non-greedy stem so the FIRST position that yields a valid
#: quant wins.
_GGUF_NAME_RE = re.compile(
    r"^(?P<stem>.+?)-(?P<quant>(?:UD-)?(?:IQ|Q|BF|F|MXFP)\d[A-Za-z0-9_]*)\.gguf$"
)


def parse_gguf_name(filename: str) -> tuple[str, str] | None:
    """``("gpt-oss-20b", "Q8_0")`` from ``gpt-oss-20b-Q8_0.gguf``, else ``None``.

    Returns None rather than guessing: an unrecognised name means we cannot say
    which artifact produced a score, and a wrong quant label is worse than a
    missing one. The caller turns None into a refusal that names the escape hatch.
    """
    m = _GGUF_NAME_RE.match(Path(filename).name)
    return (m.group("stem"), m.group("quant")) if m else None


def gguf_sha256(path: Path) -> str:
    """sha256 of the weights, cached in a ``<file>.sha256`` sidecar.

    The digest is the only identifier a re-quantized file cannot fake -- filenames
    and model ids are both just labels. Cached because these files are 12-23 GB and
    re-hashing on every `prepare()` would add minutes of wall-clock to every run
    for a value that cannot change without the file changing.

    The sidecar is trusted only when it is NEWER than the weights, so replacing a
    GGUF in place invalidates it rather than silently publishing the old digest.
    """
    sidecar = path.with_suffix(path.suffix + ".sha256")
    try:
        if sidecar.exists() and sidecar.stat().st_mtime >= path.stat().st_mtime:
            cached = sidecar.read_text(encoding="utf-8").strip()
            if re.fullmatch(r"[0-9a-f]{64}", cached):
                return cached
    except OSError:
        pass  # unreadable sidecar: recompute rather than fail

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    digest = h.hexdigest()
    try:
        sidecar.write_text(digest + "\n", encoding="utf-8")
    except OSError as exc:
        logger.warning("could not cache digest beside %s: %s", path, exc)
    return digest


def _server_command_line(port: int) -> str | None:
    """The llama-server process's own argv, or ``None`` if it can't be read.

    THE ONLY PLACE THE OFFLOAD CONFIG IS OBSERVABLE. Measured 2026-08-21 against
    llama.cpp b10472: ``GET /props`` returns ``build_info``, ``model_path``,
    ``model_ftype``, ``model_alias``, ``total_slots`` and a
    ``default_generation_settings`` holding only ``n_ctx`` and sampling
    ``params``. It says nothing about ``-ngl``, threads, or the compute backend.

    That silence matters more here than it would on a hosted provider. Measured
    on this laptop the same week, gpt-oss-20b Q8_0 on real arena prompts:
    ``-ngl 99`` (Vulkan iGPU) sustains 8.1-8.5 tok/s decode against 4.7-5.5 for
    ``-ngl 0``. Those are different kernels, so they are different instruments —
    and every record published before today was silent about which one ran.

    Best-effort by contract: the caller labels the absence rather than dropping
    the field, and never lets a failure here fail a run.
    """
    system = platform.system()
    try:
        if system == "Windows":
            # Emits ONE line per listening process, so Python can apply the same
            # "ambiguous means none" rule as the POSIX branch. The earlier form
            # used `Select-Object -First 1`, which silently picked a winner when
            # several processes held the port (Sonnet cross-review, 2026-08-21:
            # the POSIX fix had not been mirrored here).
            script = (
                f"Get-NetTCPConnection -LocalPort {int(port)} -State Listen "
                "-ErrorAction SilentlyContinue | ForEach-Object { "
                "(Get-CimInstance Win32_Process -Filter "
                "\"ProcessId=$($_.OwningProcess)\").CommandLine } | "
                "Sort-Object -Unique"
            )
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command", script],
                capture_output=True, text=True, timeout=20, check=False)
            lines = [ln.strip() for ln in (out.stdout or "").splitlines() if ln.strip()]
            if len(lines) != 1:
                return None
            out = type("R", (), {"stdout": lines[0]})()
        else:
            # lsof gives the PID(s) bound to the port; ps turns one into an argv.
            pid = subprocess.run(["lsof", "-t", f"-i:{int(port)}", "-sTCP:LISTEN"],
                                 capture_output=True, text=True, timeout=20, check=False)
            pids = [p.strip() for p in (pid.stdout or "").strip().splitlines() if p.strip()]
            # AMBIGUOUS MEANS NONE (Codex cross-review, 2026-08-21). Under
            # SO_REUSEPORT two processes can listen on one port, and picking the
            # first would publish some other process's argv as this player's
            # configuration. A missing field is recoverable; a confidently wrong
            # one is the failure this whole change exists to prevent.
            if len(pids) != 1:
                return None
            out = subprocess.run(["ps", "-p", pids[0], "-o", "args="],
                                 capture_output=True, text=True, timeout=20, check=False)
        line = (out.stdout or "").strip()
        if not line:
            return None
        # The argv must actually name the server we just probed. Closes the
        # port-reuse race Codex raised: if llama-server exited between the /props
        # call and this one and something else grabbed the port, we would
        # otherwise publish that process's command line as our provenance.
        if "llama-server" not in line.lower():
            return None
        return line
    except Exception:
        return None


def _is_loopback(endpoint: str) -> bool:
    """Delegates to the egress gate's own parser so the two cannot disagree.

    Imported lazily: `framework.runner` autoloads every adapter module at import
    time, so a module-level import here would be circular. By the time an adapter
    is CONSTRUCTED, runner is fully initialised.
    """
    from framework.runner import _is_loopback_endpoint

    return _is_loopback_endpoint(endpoint)


def _same_file(a: Path, b: Path) -> bool:
    """Are these two paths the same file on disk?

    `os.path.samefile` is the real answer (it follows symlinks and hardlinks, and
    the HF cache uses both). It raises when either path is missing, so fall back
    to a normalised-path comparison -- case-insensitive on Windows, where a drive
    letter and directory name may differ only in case and still be one directory.
    """
    try:
        return os.path.samefile(a, b)
    except OSError:
        return (os.path.normcase(os.path.abspath(a))
                == os.path.normcase(os.path.abspath(b)))


from framework.player_adapter import (  # noqa: E402
    VersionAbsence, VERSION_ABSENT_BY_DESIGN, VERSION_ABSENT_PROBE_FAILED,
    VERSION_ABSENT_NEEDS_PREPARE)

class LocalGgufAdapter(OpenAIChatCompletionsAdapter):
    """A GGUF served by a local ``llama-server`` on loopback."""

    def version_absence_reason(self) -> VersionAbsence:
        """NOT absent -- UNBUILT. `prepare()` computes it and the audit never calls it.

        `__init__` sets `_tool_version = None`; only `prepare()` fills it with
        `<stem>/<quant>@<engine>@host-<fingerprint>`, the richest provenance
        string this project produces. `framework/runner.py` calls `prepare()`
        before any task, so every RUN RECORD carries it; `framework audit` builds
        the adapter and asks immediately, because preparing a local player loads
        a multi-gigabyte model off disk.

        So the audit's "no installed version" was never true for these 19 players
        -- it was measuring a lifecycle stage, not a capability. Verified by AST
        2026-08-27: the assignment is inside `prepare()`, not `__init__`.
        """
        return VersionAbsence(
            VERSION_ABSENT_NEEDS_PREPARE,
            "version is computed by prepare(), which the audit does not call "
            "because it would load the model",
        )

    def __init__(
        self,
        *args,
        endpoint: str,
        gguf_path: str | Path,
        gguf_quant: str | None = None,
        **kwargs,
    ):
        # The parent requires these two; we supply our own private names and fill
        # them from `endpoint` at prepare() time. A registry entry that tries to
        # set them is refused rather than silently ignored -- a key the operator
        # believes is in effect but is not is how the wrong config gets published.
        for forbidden in ("openai_base_url_env", "openai_api_key_env"):
            if forbidden in kwargs:
                raise RuntimeError(
                    f"LocalGgufAdapter does not read {forbidden}: the destination "
                    "comes from `endpoint` alone, so the egress gate and the "
                    "request path can never disagree. Remove it from registry.yaml."
                )
        kwargs["openai_base_url_env"] = _BASE_URL_ENV
        kwargs["openai_api_key_env"] = _API_KEY_ENV
        super().__init__(*args, **kwargs)

        endpoint = str(endpoint).strip().rstrip("/")
        if not _is_loopback(endpoint):
            raise RuntimeError(
                f"{self.player_id}: endpoint {endpoint!r} is not loopback. "
                "LocalGgufAdapter is local by construction -- a class named Local* "
                "must not be pointable off-machine, and the held-out egress gate "
                "would classify this entry as cloud anyway. Use an OpenAI-compatible "
                "player for a remote endpoint."
            )
        self.endpoint = endpoint
        # Expanded, never hardcoded: a GGUF lives wherever the operator keeps
        # models, and this repo's rule is that no absolute user path is written
        # into config. `${SCIENCEARENA_GGUF_DIR}/foo.gguf` resolves here; an unset
        # variable stays literal and fails the exists() check below by name,
        # rather than silently resolving to a wrong file.
        self.gguf_path = Path(
            os.path.expanduser(os.path.expandvars(str(gguf_path)))
        )
        self.gguf_quant = gguf_quant
        self._local_meta: dict = {}
        self._tool_version: str | None = None
        self._served_identity: str = ""

    # -- provenance ----------------------------------------------------------

    def _probe_server_props(self) -> dict | None:
        """Ask the running server what it actually loaded. Measured, not declared.

        llama-server's ``/props`` reports ``model_path`` (the GGUF it opened),
        ``model_ftype`` (the quant read from that file's own metadata) and
        ``build_info``. Those three make the served artifact CHECKABLE, which is
        the whole difference between provenance and a label.
        """
        try:
            parts = urlsplit(self.endpoint)
            root = f"{parts.scheme}://{parts.netloc}"
            r = requests.get(f"{root}/props", timeout=10)
            if r.status_code != 200:
                return None
            return r.json() or None
        except Exception:
            return None

    @staticmethod
    def _engine_label(build_info: str | None) -> str:
        """``b10472-7a556b8f9`` -> ``llama.cpp-b10472-7a556b8f9``.

        The WHOLE build string is kept, commit hash and all. An earlier version
        parsed out just the bNNNN release number; keeping everything is both
        simpler and better provenance, because two llama.cpp builds can share a
        release tag and differ by commit -- and this player's entire purpose is
        to be re-runnable to the same numbers later.

        An unknown build is LABELLED rather than omitted: a missing engine reads
        as "no engine", which is a different and false claim.
        """
        info = str(build_info or "").strip()
        return f"llama.cpp-{info}" if info else "llama.cpp-unknown"

    def resolved_tool_version(self) -> str | None:
        """``<stem>/<quant>@llama.cpp-b<build>`` -- the artifact, not the model name."""
        return self._tool_version

    def last_response_meta(self) -> dict | None:
        meta = dict(super().last_response_meta() or {})
        meta.update(self._local_meta)
        return meta or None

    # -- lifecycle -----------------------------------------------------------

    def prepare(self) -> None:
        parsed = parse_gguf_name(self.gguf_path.name)
        quant = self.gguf_quant or (parsed[1] if parsed else None)
        stem = (parsed[0] if parsed else None) or self.openai_model
        if not quant:
            raise RuntimeError(
                f"{self.player_id}: cannot read a quantization label from "
                f"{self.gguf_path.name!r}. Set `gguf_quant` explicitly in "
                "registry.yaml -- publishing a score without naming the quant "
                "would attribute it to an artifact nobody can identify."
            )
        if not self.gguf_path.exists():
            raise RuntimeError(
                f"{self.player_id}: weights not found at {self.gguf_path}. "
                "The provenance digest is computed from the file this player "
                "claims to be running, so a missing file is a hard error rather "
                "than an unlabelled run."
            )

        # Hand the parent its URL through our private env names, then put the
        # environment back. This inherits EVERY validation the parent performs
        # (reserved body keys, rotating-model refusal, template load) instead of
        # duplicating them here, where they would drift.
        saved = {k: os.environ.get(k) for k in (_BASE_URL_ENV, _API_KEY_ENV)}
        os.environ[_BASE_URL_ENV] = self.endpoint
        os.environ[_API_KEY_ENV] = _NO_AUTH_PLACEHOLDER
        try:
            super().prepare()
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

        # WHAT DID THE SERVER ACTUALLY LOAD? Everything below hangs on this.
        # Until 2026-08-20 the adapter hashed the file the REGISTRY named and
        # asked the server nothing, so a server left running on a different GGUF
        # -- routine between benchmark configs -- published one file's digest for
        # another file's outputs, with every test green. (Codex cross-review.)
        props = self._probe_server_props()
        if not props:
            raise RuntimeError(
                f"{self.player_id}: could not read {self.endpoint} /props, so the "
                "served model cannot be verified. Refusing rather than publishing "
                "a provenance label nobody can check. Is llama-server running?"
            )
        served_raw = str(props.get("model_path") or "").strip()
        if not served_raw:
            raise RuntimeError(
                f"{self.player_id}: /props reported no `model_path`, so the served "
                "artifact is unverifiable. Refusing."
            )
        served = Path(served_raw)
        if not _same_file(served, self.gguf_path):
            raise RuntimeError(
                f"{self.player_id}: the server is serving {served.name!r} but "
                f"registry.yaml names {self.gguf_path.name!r}. The digest would "
                "describe a file that did not produce the answers. Point the "
                "player at the served file, or restart llama-server on the "
                f"declared one.\n  served:   {served}\n  declared: {self.gguf_path}"
            )

        # The LABEL comes from the served filename, never the registry: an
        # override would let a config edit relabel a score, which is the failure
        # this class exists to prevent. `gguf_quant` survives only as a fallback
        # for a file whose name carries no recognisable quant token.
        parsed = parse_gguf_name(served.name)
        quant = (parsed[1] if parsed else None) or self.gguf_quant
        stem = (parsed[0] if parsed else None) or self.openai_model
        if not quant:
            raise RuntimeError(
                f"{self.player_id}: cannot read a quantization label from "
                f"{served.name!r}. Set `gguf_quant` in registry.yaml as a fallback "
                "-- publishing a score without naming the quant would attribute it "
                "to an artifact nobody can identify."
            )

        # The string llama-server echoes in every completion's `model` field.
        # Measured 2026-08-20: it ignores the requested model name and reports what
        # it actually loaded, so this is a free per-task fingerprint (see play_task).
        self._served_identity = str(
            props.get("model_alias") or props.get("model_path") or "").strip()

        engine = self._engine_label(props.get("build_info"))
        digest = gguf_sha256(self.gguf_path)

        # THE MACHINE IS EVIDENCE OF WHAT RAN — it is not the instrument.
        #
        # RECONCILED 2026-09-17. This comment used to open "THE MACHINE IS PART
        # OF THE IDENTITY, for the same reason the quant is", and that sentence
        # overstated what this line achieves and contradicted the identity design
        # all three consult seats agreed on: host is PROVENANCE, and two machines
        # running the same instrument are a REPLICATION, not two models. The
        # facts underneath it were right and are kept below; the conclusion was
        # not, and the conclusion was the part a reader would act on.
        #
        # WHAT IS TRUE. registry.yaml records that this model reproduces 16/16
        # byte-identical completions up to ~2400 tokens across days, sessions and
        # server processes — the drift anchor this whole player class exists to
        # provide. That reproduction is a property of ONE CPU and ONE llama.cpp
        # backend; a different SIMD path or a different (or absent) iGPU reorders
        # the reductions, one logit flips, and the continuation diverges. So two
        # machines must never be silently pooled into one mean.
        #
        # WHAT THIS LINE ACTUALLY DOES ABOUT THAT: nothing, on its own. It writes
        # the host digest into `resolved_tool_version`, which is the EVIDENCE
        # field ("what actually ran") and not a grouping key — and
        # `framework/leaderboard.py` keeps only the FIRST `resolved_tool_version`
        # it sees per player_id, so a differing digest does not split a row.
        # Keeping it here is still right: it is the finest-grained record of what
        # produced the number, and `framework audit --versions` reads it.
        #
        # WHAT ENFORCES THE RULE is `framework.audit.host_split_players`, which
        # flags any player_id whose records come from more than one machine, and
        # which treats a MISSING fingerprint as a third answer rather than a
        # match. That is where "do not pool two machines" lives.
        #
        # And what a reader is shown is the DECLARED instrument
        # (framework/identity.py): the model at one inference configuration,
        # carrying no host. Same instrument on two boxes is one row plus a
        # replication, which is the honest shape. See framework/host.py.
        host_ident = host.host_identity()
        self._tool_version = f"{stem}/{quant}@{engine}@host-{host_ident['fingerprint']}"

        # The server's runtime configuration. `/props` cannot report the offload
        # config (measured — see `_server_command_line`), so it is read from the
        # OS and redacted. Best-effort: a failure is LABELLED, never fatal and
        # never silently omitted.
        try:
            raw_argv = _server_command_line(urlsplit(self.endpoint).port or 80)
        except Exception:
            raw_argv = None
        server_argv = host.redact_paths(raw_argv) if raw_argv else None

        self._local_meta = {
            "gguf_file": served.name,
            "gguf_sha256": digest,
            "quant": quant,
            "engine_build": engine,
            "endpoint": self.endpoint,
            # Independent of the filename: `model_ftype` is read out of the GGUF's
            # own header by the server, so a renamed file is still described
            # correctly here and the two can be compared after the fact.
            "model_ftype": props.get("model_ftype"),
            # NO ABSOLUTE PATH HERE. `served` is something like
            # C:/Users/<name>/.sciencearena/models/..., and every field in this
            # block is copied into the PUBLIC data bundle. `gguf_file` above is
            # the basename of that same served file, which is all a reader needs;
            # the full path only ever appears in the local mismatch error.
            # The sampling config is part of what produced the number. Recorded
            # because a local player is meant to be a REPRODUCIBLE anchor, and an
            # anchor whose sampling nobody wrote down is not one. `server_defaults`
            # captures what llama-server would apply for anything we do not send --
            # its out-of-the-box temperature is 0.8 with a random seed, which is
            # why two runs of one prompt can differ by hundreds of tokens.
            "sampling": {
                "temperature": self.openai_temperature,
                **{k: v for k, v in (self.openai_extra_body or {}).items()
                   if k != "max_tokens"},
            },
            "server_defaults": (props.get("default_generation_settings") or {}).get("params"),
            "max_tokens": (self.openai_extra_body or {}).get(
                "max_tokens", _LOCAL_MAX_TOKENS),
            # Slot geometry. registry.yaml names this as the MECHANISM that ends
            # byte-for-byte reproduction past ~2400 tokens: llama.cpp picks a slot
            # by prefix similarity and reuses part of a previous request's KV, so
            # batch composition varies with how many slots exist and how large
            # each one's context is. Recording the sampling config while leaving
            # the slot geometry out documented half the cause.
            "server_n_ctx": (props.get("default_generation_settings") or {}).get("n_ctx"),
            "server_total_slots": props.get("total_slots"),
            # The offload/thread configuration, redacted. `None` + an explicit
            # source label, never a bare omission: absent must not read the same
            # as "we looked and there was nothing".
            "server_argv": server_argv,
            "server_argv_source": "os-process" if server_argv else "unavailable",
        }

    def play_task(self, envelope: dict, *, timeout_s: int) -> dict:
        # Cap output locally. The parent sends its hosted-provider default of
        # 32000, which on this hardware is a multi-hour runaway rather than an
        # expensive one. Set only when the registry has not chosen a ceiling.
        if not (self.openai_extra_body or {}).get("max_tokens"):
            body = dict(self.openai_extra_body or {})
            body["max_tokens"] = _LOCAL_MAX_TOKENS
            self.openai_extra_body = body
        out = super().play_task(envelope, timeout_s=timeout_s)

        # THE MODEL CAN CHANGE UNDER US MID-RUN. `prepare()` verifies the served
        # artifact once; a tournament then runs for hours. If llama-server is
        # restarted onto a different GGUF partway through -- a crash-restart, or
        # exactly the config-swapping done while benchmarking -- every later task
        # is answered by the new weights and published under the first probe's
        # stem, quant and sha256. Nothing throws and no score looks wrong.
        # (Sonnet 5 cross-review, 2026-08-20.)
        #
        # Free to check: llama-server echoes the model it actually loaded in every
        # response, which the parent already captures as `served_model`. Raising
        # fails THIS task loudly instead of silently mislabelling it and every one
        # after it.
        served = (self._last_meta or {}).get("served_model")
        if served and self._served_identity and served != self._served_identity:
            raise RuntimeError(
                f"{self.player_id}: the server changed model mid-run. Provenance "
                f"was verified against {self._served_identity!r} at startup but this "
                f"task was answered by {served!r}. Every record since the swap would "
                "carry the wrong quant and digest. Restart the run against one "
                "server, or split it into two runs."
            )
        return out


register_adapter_class("LocalGgufAdapter", LocalGgufAdapter)
