"""Leak test for agentic container players (framework/agentic_container.py).

LIVE: needs Docker, the built image and CLAUDE_CODE_OAUTH_TOKEN, and spends a few cheap model
calls, so it runs only with SCIENCEARENA_LIVE_CONTAINER=1. It SKIPS otherwise, and a skip is
not a pass: no agentic score counts until this is green for the image tag in use. Each check
is two-sided (a planted positive control that must be found) so a broken search cannot pass
as "no leak".
"""
from __future__ import annotations

import json
import os
import secrets
import subprocess
from pathlib import Path

import pytest

from framework import agentic_container as ac

LIVE = os.environ.get("SCIENCEARENA_LIVE_CONTAINER") == "1"
pytestmark = pytest.mark.skipif(not LIVE, reason="live container test: set SCIENCEARENA_LIVE_CONTAINER=1")
MODEL = os.environ.get("SCIENCEARENA_LEAK_MODEL", "sonnet")


def _shell(cmd: str, *, image: str = ac.DEFAULT_IMAGE) -> subprocess.CompletedProcess:
    """Run a shell command as the player, after the entrypoint has locked egress."""
    argv = [ac._docker(), "run", "--rm", "--cap-add", "NET_ADMIN", image, "bash", "-c", cmd]
    return subprocess.run(argv, capture_output=True, text=True, timeout=180)


def test_entrypoint_reports_the_lock_and_drops_privileges():
    r = _shell("id -u; grep CapEff /proc/self/status; "
               "curl -s -o /dev/null -w 'anthropic=%{http_code}\\n' --max-time 10 https://api.anthropic.com/; "
               "curl -s -o /dev/null -w 'example=%{http_code}\\n' --max-time 5 https://example.com/ || echo example=BLOCKED; "
               "curl -s -o /dev/null --max-time 5 http://1.1.1.1/ && echo raw_ip=OPEN || echo raw_ip=BLOCKED")
    assert "entrypoint: egress locked to: api.anthropic.com" in r.stderr, r.stderr
    out = r.stdout
    assert out.splitlines()[0].strip() == "1000", out
    assert "CapEff:\t0000000000000000" in out, out
    assert "anthropic=000" not in out and "anthropic=" in out, f"allowlisted host must answer: {out}"
    assert "example=BLOCKED" in out or "example=000" in out, out
    assert "raw_ip=BLOCKED" in out, out


def test_the_player_cannot_lift_the_lock():
    r = _shell("iptables -P OUTPUT ACCEPT 2>&1; echo rc=$?; "
               "curl -s -o /dev/null --max-time 5 https://example.com/ && echo still=OPEN || echo still=BLOCKED")
    assert "rc=0" not in r.stdout, r.stdout
    assert "still=BLOCKED" in r.stdout, r.stdout


def test_dns_is_closed_for_the_player():
    r = _shell("getent hosts example.com && echo dns=OPEN || echo dns=CLOSED; getent hosts api.anthropic.com")
    assert "dns=CLOSED" in r.stdout, r.stdout
    assert "api.anthropic.com" in r.stdout, "pinned allowlist entry must still resolve via /etc/hosts"


def test_the_agent_surface_reaches_task_files_but_no_host_secret():
    """Deterministic, two-sided. The agent's Bash tool runs as this same player with these
    same mounts and environment, so sweeping the filesystem and env here IS the agent's
    reach. (A model-driven sweep was refused as a prompt injection by Haiku twice on
    2026-10-12, so containment is proven at the surface, not by asking the model.)
    The in-task nonce MUST be found (the sweep works); host-side nonces (an env var in the
    caller and a file beside the repo) must NOT be; the web must be unreachable."""
    task_nonce = "NONCE-TASK-" + secrets.token_hex(6)
    proc_nonce = "NONCE-PROC-" + secrets.token_hex(6)   # positive control for the /proc sweep
    env_nonce = "NONCE-ENV-" + secrets.token_hex(6)
    file_nonce = "NONCE-FILE-" + secrets.token_hex(6)
    host_file = Path(__file__).resolve().parents[2] / "tmp" / "leak_probe_secret.txt"
    host_file.parent.mkdir(parents=True, exist_ok=True)
    host_file.write_text(file_nonce, encoding="utf-8")
    env = dict(os.environ, SA_LEAK_PROBE=env_nonce)
    # Every place a host secret could reach the agent, without `grep -r /`, which never
    # finished (600 s timeout, d-c2e030): it descends into /proc and /sys and READS device
    # files (GNU grep -r reads devices by default), so it can block on a pseudo-file forever.
    # Instead: (1) every mounted filesystem that holds real files (the image root, the /task
    # and workspace bind mounts, tmpfs), each walked with -xdev so no pseudo-filesystem is
    # entered; kernel pseudo-filesystems are excluded BY TYPE, so a new bind mount of any
    # real type is swept automatically; (2) the environment and argv of every process the
    # agent can see (/proc/<pid>/environ, cmdline), the only route by which a secret held in
    # memory rather than in a file is readable. MOUNTS= is echoed so the test can assert
    # that / and /task were swept (a broken mount list cannot pass as "no leak").
    pseudo = ("proc|sysfs|cgroup|cgroup2|devpts|mqueue|debugfs|securityfs|pstore|bpf|"
              "tracefs|fusectl|configfs|hugetlbfs|autofs|binfmt_misc|nsfs|efivarfs")
    # The /proc control is written split by an empty quote ('NONCE-PROC-''<hex>') so it never
    # appears contiguously in this command's own argv, which the cmdline sweep also reads:
    # only the sleeping process's environment can supply it.
    p_head, p_hex = proc_nonce.rsplit("-", 1)
    sweep = ("id -u; "
             f"SA_PROC_CTRL='{p_head}-''{p_hex}' sleep 600 & ctrl=$!; sleep 1; "
             "for f in /proc/[0-9]*/environ /proc/[0-9]*/cmdline; do tr '\\0' '\\n' < \"$f\"; done 2>/dev/null"
             " | grep -o 'NONCE-[A-Z]*-[0-9a-f]*' | sort -u; kill $ctrl; "
             f"m=$(awk '$3 !~ /^({pseudo})$/ {{print $2}}' /proc/self/mounts | sort -u); "
             "echo MOUNTS= $m; "
             "for d in $m; do find \"$d\" -xdev -type f -print0 2>/dev/null"
             " | xargs -0r grep -IhoD skip 'NONCE-[A-Z]*-[0-9a-f]*' 2>/dev/null; done | sort -u; "
             "env | grep -o 'NONCE-[A-Z]*-[0-9a-f]*' | sort -u; "
             "env | grep -c '^ANTHROPIC_API_KEY=' || true; "  # hard-rule-ok: leak test counts the key's presence (expects 0)
             "curl -sS --max-time 5 -o /dev/null https://example.com && echo FETCH=OK || echo FETCH=FAILED")
    try:
        r = ac.run_task(MODEL, "", {"notes.txt": task_nonce.encode()}, env=env,
                        command=["bash", "-c", sweep], timeout_s=600)
    finally:
        host_file.unlink(missing_ok=True)
    out = r.stdout
    assert r.egress_locked, r.stderr[-2000:]
    assert out.splitlines()[0].strip() == "1000", out[:500]
    mounts = next((ln.split()[1:] for ln in out.splitlines() if ln.startswith("MOUNTS=")), [])
    assert "/" in mounts and "/task" in mounts, f"sweep did not cover / and /task: {mounts}"
    assert proc_nonce in out, f"/proc positive control not found - the env sweep did not work: {out[-1500:]}"
    assert task_nonce in out, f"positive control not found - the sweep did not work: {out[-1500:]}"
    assert env_nonce not in out and file_nonce not in out, "HOST SECRET REACHABLE from the container"
    assert "FETCH=FAILED" in out, out[-1500:]


def test_the_agent_can_use_its_tools_inside_the_lock():
    """Functional positive control: the agent runs R on a task file and returns a value it
    could only get by executing code (mean of 1..100 squared = 3383.5)."""
    data = "\n".join(str(i * i) for i in range(1, 101)).encode()
    r = ac.run_task(MODEL, "Use your Bash tool to run R (Rscript) and compute the mean of the "
                           "numbers in /task/numbers.txt (one per line). Reply with the number only.",
                    {"numbers.txt": data}, timeout_s=900)
    assert r.egress_locked, r.stderr[-2000:]
    result = json.loads(r.stdout)
    assert result["is_error"] is False, r.stdout[-1500:]
    # >= 2 turns: at least one tool call happened (a model can do this sum in its head,
    # which would pass the value check without proving the tools work).
    assert result["num_turns"] >= 2, f"no tool turn: {result.get('num_turns')}"
    # Haiku 4.5 ran R in 5 turns and still answered 3434 (2026-10-12): a model error, not
    # an infrastructure one - R in this image returns 3383.5. Hence a strong default model.
    assert "3383.5" in result["result"], result["result"]
