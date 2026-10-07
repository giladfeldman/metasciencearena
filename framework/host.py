"""Which machine produced a score.

WHY THIS MODULE EXISTS (2026-08-21)
-----------------------------------
``provenance.host`` used to be::

    os.environ.get("SCIENCEARENA_HOST", "win11-local")

``SCIENCEARENA_HOST`` is set nowhere — not in the repo, not in any run script,
not in the environment of the machine that produced the records. Measured the
day this module was written: **all 3,924 tracked run records carry the
byte-identical string** ``"win11-local"``. The field looked like provenance and
encoded nothing.

Harmless while one laptop produced every score. Not harmless the moment a second
machine runs the same player:

* ``players/registry.yaml`` records that gpt-oss-20b reproduced **16/16
  byte-identical** completions up to ~2400 tokens, across days, sessions and
  server processes. That bounded determinism is this project's only genuine
  drift anchor — no hosted player has it.
* It is a property of ONE MACHINE. A different CPU is a different SIMD path; a
  different (or absent) iGPU is a different llama.cpp backend entirely. Reduction
  order changes, one logit flips, the continuation diverges. Same weights, same
  seed, same temperature — different text.
* Both leaderboard engines group strictly by ``player_id``
  (``leaderboard-app/scripts/build-data.mjs``, ``framework/leaderboard.py``), and
  both keep the FIRST ``player_version`` / ``resolved_tool_version`` they see. So
  two machines under one ``player_id`` publish one machine's label over both
  machines' scores, and nothing fails.

WHAT IS AND IS NOT IN HERE
--------------------------
Every field this module returns is copied verbatim into the PUBLIC data bundle,
so the same bar applies that ``players/adapters/local_gguf.py`` already holds for
the served model path ("NO ABSOLUTE PATH HERE ... C:/Users/<name>/").

Recorded — hardware facts that can change a completion, none of them identifying:
``cpu`` (the CPUID family/model/stepping string, i.e. a part number),
``arch``, ``os``, ``os_release``, ``cpu_logical``, ``ram_gb``.

Deliberately NOT recorded: hostname, username, MAC address, serial numbers,
absolute paths. None of them affect a logit, and all of them identify a person or
a machine on a network.

``label`` vs ``fingerprint``: the label is a NAME an operator chooses; the
fingerprint is EVIDENCE derived from the hardware. They are kept apart on
purpose. If renaming a machine moved its fingerprint, an operator could split one
machine into two instruments — or merge two into one — by editing an environment
variable.

STDLIB ONLY. ``psutil`` is present in this venv transitively but is not a
declared dependency, and ``scripts/check_clean_install.py`` asserts the package
imports with core dependencies alone.
"""
from __future__ import annotations

import ctypes
import functools
import hashlib
import json
import os
import platform
import re
import subprocess
from collections.abc import Mapping

#: Bumped when the RECIPE changes — which facts are hashed, or how they are
#: canonicalised. Without it, a reader diffing two fingerprints cannot tell
#: "different machine" from "we changed how we hash".
FINGERPRINT_VERSION = 1

#: Facts REPORTED in the published block. A superset of what the digest covers.
_FACT_KEYS = ("cpu", "arch", "os", "os_release", "cpu_logical", "ram_gb",
              "accelerator")

#: Facts the digest actually covers. Adding or removing one is a
#: FINGERPRINT_VERSION bump: every previously published digest becomes
#: non-comparable, and saying so is the point of the version.
#:
#: `os_release` is REPORTED but deliberately NOT hashed (Codex cross-review,
#: 2026-08-21, reproduced). On Linux `platform.release()` is the kernel build
#: (`6.8.0-40`), which moves on every routine patch. Hashing it reported one
#: machine as two after an apt upgrade and failed the audit gate for a box nobody
#: touched — a false positive, and a gate that cries wolf is a gate the operator
#: learns to skip. A kernel patch does not reorder a SIMD reduction.
_DIGEST_KEYS = ("cpu", "arch", "os", "cpu_logical", "ram_gb", "accelerator")

#: The string this module replaces. Named so the test that forbids it can point
#: at the reason rather than at a magic literal.
LEGACY_CONSTANT_HOST = "win11-local"

#: HOME DIRECTORIES, matched before anything else and tolerant of spaces.
#: This is where usernames live, so it gets its own rule rather than relying on
#: the generic path regexes below, which stop at whitespace and therefore leaked
#: `C:\Users\Jane Doe\...` verbatim (Sonnet cross-review, 2026-08-21, reproduced).
_HOME_DIR_RE = re.compile(
    r'(?:[A-Za-z]:)?[\\/](?:Users|home)[\\/][^\\/"]+[\\/]', re.IGNORECASE)

#: A quoted argument holding a path. The only form in which a path containing
#: SPACES is unambiguously delimited — see `redact_paths`.
_QUOTED_PATH_RE = re.compile(r'"((?:[A-Za-z]:[\\/]|[\\/])[^"]*)"')
_WIN_PATH_RE = re.compile(r"[A-Za-z]:[\\/](?:[^\s\\/]+[\\/])+")
_POSIX_PATH_RE = re.compile(r"/(?:[^\s/]+/)+")
_PATH_PLACEHOLDER = "<path>/"


# --- measuring the machine ---------------------------------------------------

@functools.lru_cache(maxsize=1)
def _cpu_brand() -> str:
    """A CPU identifier, best available on this platform.

    CACHED because ``host_identity()`` is called once per run record and the
    Darwin branch shells out to ``sysctl``. The CPU does not change mid-process.
    The cache sits on the probe, not on ``collect_facts()``, so a test can still
    substitute the whole fact set.

    ``platform.processor()`` is the portable answer and on Windows returns the
    CPUID string (``Intel64 Family 6 Model 170 Stepping 4, GenuineIntel``), which
    is *better* provenance than a marketing name: family/model/stepping is what
    actually selects the SIMD path. On Linux it is frequently empty, so fall
    back to the kernel's own answer.
    """
    proc = (platform.processor() or "").strip()
    if proc:
        return proc
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        elif platform.system() == "Darwin":
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5, check=False)
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
    except Exception:
        pass
    return platform.machine() or "unknown"


@functools.lru_cache(maxsize=1)
def _total_ram_gb() -> int | None:
    """Total physical memory in whole GB, or ``None`` if it cannot be read.

    WHOLE GB on purpose. Raw bytes move when firmware reserves a different
    amount, so a BIOS update would mint a new "machine" for the same box.

    ``None`` rather than 0: absent is not zero (the same distinction the run
    record's ``schema_version`` exists to preserve).
    """
    try:
        system = platform.system()
        if system == "Windows":
            class _MemStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            status = _MemStatus()
            status.dwLength = ctypes.sizeof(_MemStatus)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return None
            total = int(status.ullTotalPhys)
        elif hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names \
                and "SC_PHYS_PAGES" in os.sysconf_names:
            total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        else:
            return None
        return int(round(total / (1024 ** 3))) or None
    except Exception:
        return None


#: The probe ran and found no accelerator. A real fact about the machine.
ACCELERATOR_NONE = "<none>"
#: The probe could NOT run (no lspci, no permission, unsupported platform). NOT
#: the same fact — Sonnet cross-review 2026-08-21, reproduced: collapsing the two
#: let a box with a discrete GPU whose probe failed share a fingerprint with a
#: genuinely CPU-only box, which is the exact collision this fact was added to
#: prevent. Absent is not zero.
ACCELERATOR_UNKNOWN = "<unknown>"


@functools.lru_cache(maxsize=1)
def _accelerator() -> str:
    """Display adapter name(s), or an explicit none/unknown sentinel.

    Closes most of Codex #1 (2026-08-21). Without it, two boxes with the same
    CPU, core count and RAM collided onto one fingerprint even when one had a
    discrete GPU and the other did not — which is exactly the difference that
    changes llama.cpp's backend and therefore the completion. Measured on the
    reference laptop the same week: ``-ngl 99`` (Vulkan iGPU) sustains 8.1-8.5
    tok/s decode against 4.7-5.5 for ``-ngl 0``, running different kernels.

    It does NOT make collisions impossible — two genuinely identical machines
    still collide, which is documented in the module docstring rather than
    papered over. Cached: this shells out on Windows and the GPU does not change
    mid-process.
    """
    try:
        system = platform.system()
        if system == "Windows":
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_VideoController).Name -join '|'"],
                capture_output=True, text=True, timeout=20, check=False)
        elif system == "Linux":
            out = subprocess.run(
                ["sh", "-c", "lspci 2>/dev/null | grep -iE 'vga|3d|display' "
                             "| sed 's/.*: //' | paste -sd'|' -"],
                capture_output=True, text=True, timeout=20, check=False)
        elif system == "Darwin":
            out = subprocess.run(
                ["sh", "-c", "system_profiler SPDisplaysDataType 2>/dev/null "
                             "| awk -F': ' '/Chipset Model/{print $2}' | paste -sd'|' -"],
                capture_output=True, text=True, timeout=20, check=False)
        else:
            return ACCELERATOR_UNKNOWN
        # A NON-ZERO exit means the probe itself failed (binary absent, denied),
        # which is "unknown". A clean exit with empty output means the machine
        # really has no display adapter, which is "none". Two different facts.
        if out.returncode != 0:
            return ACCELERATOR_UNKNOWN
        name = (out.stdout or "").strip()
        return name or ACCELERATOR_NONE
    except Exception:
        return ACCELERATOR_UNKNOWN


def collect_facts() -> dict:
    """Measure this machine. Never raises — a telemetry probe must not fail a run."""
    return {
        "cpu": _cpu_brand(),
        "arch": platform.machine() or "unknown",
        "os": platform.system() or "unknown",
        "os_release": platform.release() or "unknown",
        "cpu_logical": os.cpu_count() or 0,
        "ram_gb": _total_ram_gb(),
        "accelerator": _accelerator(),
    }


# --- the digest --------------------------------------------------------------

def fingerprint(facts: Mapping) -> str:
    """Stable short digest over ``_FACT_KEYS`` only.

    Extra keys are ignored, so ``fingerprint(host_identity())`` reproduces the
    identity's own digest — which is what makes the published block checkable by
    a reader rather than a number they have to trust.
    """
    canonical = json.dumps(
        {k: facts.get(k) for k in _DIGEST_KEYS},
        sort_keys=True, separators=(",", ":"), default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def _derive_label(facts: Mapping, digest: str) -> str:
    """A readable, non-identifying name: ``windows-11-amd64-14c-32g-a1b2c3d4e5f6``.

    Never the hostname. The digest suffix is what makes it unique; the rest is
    there so a human reading a leaderboard row knows what kind of box it was
    without looking anything up.
    """
    parts = [
        str(facts.get("os") or "unknown").lower(),
        str(facts.get("os_release") or "").lower(),
        str(facts.get("arch") or "").lower(),
    ]
    if facts.get("cpu_logical"):
        parts.append(f"{facts['cpu_logical']}c")
    if facts.get("ram_gb"):
        parts.append(f"{facts['ram_gb']}g")
    parts.append(digest)
    return "-".join(p for p in parts if p)


def host_identity(facts: Mapping | None = None, *,
                  env: Mapping | None = None) -> dict:
    """The ``provenance.host_fingerprint`` block for one run record.

    ``facts`` and ``env`` are injectable so the tests can pin behaviour for a
    machine that is not the one running them.
    """
    facts = dict(facts) if facts is not None else collect_facts()
    env = os.environ if env is None else env
    digest = fingerprint(facts)
    label = str(env.get("SCIENCEARENA_HOST") or "").strip() or _derive_label(facts, digest)
    return {
        **{k: facts.get(k) for k in _FACT_KEYS},
        "fingerprint": digest,
        "fingerprint_version": FINGERPRINT_VERSION,
        "label": label,
    }


# --- redaction for captured command lines ------------------------------------

def redact_paths(text: str) -> str:
    """Strip directory components, keeping basenames and flags.

    Used on the llama-server command line, which is the only place the GPU
    offload configuration is observable (``/props`` does not report it —
    measured 2026-08-21 against llama.cpp b10472: its keys are ``build_info``,
    ``model_path``, ``model_ftype``, ``model_alias``, ``total_slots`` and
    ``default_generation_settings``, and the last holds only ``n_ctx`` and
    sampling ``params``).

    That command line carries ``C:/Users/<name>/...``, so it cannot be published
    raw. Directories go; ``--n-gpu-layers 99`` and the GGUF basename stay,
    because they are the provenance we came for and the basename is already
    published as ``response_meta.gguf_file``.
    """
    if not text:
        return text
    # QUOTED paths first. Codex #4 (High, reproduced 2026-08-21): the bare
    # regexes below stop at whitespace, so a Windows profile name with a space in
    # it survived redaction and the username was published. A quoted argument is
    # the only case where a path with spaces is unambiguously delimited, so it is
    # handled as a whole and reduced to its basename.
    def _quoted(m: re.Match) -> str:
        inner = m.group(1)
        base = re.split(r"[\\/]", inner)[-1]
        return f'"{_PATH_PLACEHOLDER}{base}"'

    out = _QUOTED_PATH_RE.sub(_quoted, text)
    # Home directories next, spaces and all — see `_HOME_DIR_RE`.
    out = _HOME_DIR_RE.sub(_PATH_PLACEHOLDER, out)
    out = _WIN_PATH_RE.sub(_PATH_PLACEHOLDER, out)
    out = _POSIX_PATH_RE.sub(_PATH_PLACEHOLDER, out)
    # Belt and braces for a username that survived as a path component. It must
    # be DELIMITED BY A PATH SEPARATOR — Codex #5 (reproduced): a bare substring
    # replace turned `-ngl 99` into `-<user> 99` for an operator whose account is
    # named `ngl`, destroying the single most numerically-significant flag on
    # this hardware. Short names are skipped as well; scrubbing a 1-2 character
    # string mangles unrelated text for no privacy gain.
    for name in (os.environ.get("USERNAME"), os.environ.get("USER")):
        if name and len(name) >= 3:
            out = re.sub(rf"(?<=[\\/]){re.escape(name)}(?=[\\/\"\s]|$)",
                         "<user>", out, flags=re.IGNORECASE)
    # Two rules can each fire on one path (`/home/x/` then `/models/`), leaving
    # `<path><path>/`. Purely cosmetic, but a reader should not have to wonder
    # whether the doubling means something.
    return re.sub(r"(?:<path>/?)+", _PATH_PLACEHOLDER, out)
