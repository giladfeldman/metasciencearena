"""Isolation of the real runner image, against real Docker (skipped without it).

The in-process fake (framework/tests/_container_fakes.py) proves the data path;
only a real container can prove the containment the env record claims:

  * no direct route out (the internal network has no DNS),
  * the egress proxy refuses a host outside the allowlist,
  * the image holds no task data, gold or seed,
  * the planted-nonce probe: a file on the host is not readable inside.
"""
from __future__ import annotations

import secrets
import subprocess
import tempfile
from pathlib import Path

import pytest

from framework import container


def _docker_state() -> tuple[bool, str]:
    """(usable, skip reason). A slow engine and an absent one are different findings."""
    try:
        # 60 s, not 20: under load `docker info` took 13.8 s and a 20 s probe timed out,
        # skipping all five tests on a healthy engine (2026-10-25) -- a false skip.
        ok = subprocess.run(["docker", "info"], capture_output=True, timeout=60).returncode == 0
        return ok, "Docker engine not reachable"
    except subprocess.TimeoutExpired:
        return False, "docker info did not answer within 60 s (engine overloaded?) -- NOT verified"
    except Exception:
        return False, "Docker engine not reachable"


_HAS_BUILD_FILES = (container.DOCKER_DIR / "runner" / "Dockerfile").exists()
# Probe only where the tests could run: in the public mirror the build files are absent, so a
# `docker info` (up to 60 s) at collection would be paid for nothing.
_DOCKER_OK, _DOCKER_WHY = _docker_state() if _HAS_BUILD_FILES else (False, "skipped: no build files")

pytestmark = [
    # The runner image's build files are not in the public mirror (contract/public_manifest.txt),
    # so these run only where the image can be built: here. Found 2026-10-25, when Docker was up
    # during a mirror --verify-tests run and all five errored on the missing Dockerfile.
    pytest.mark.skipif(not _HAS_BUILD_FILES,
                       reason="runner image build files are private, not in the public mirror"),
    pytest.mark.skipif(not _DOCKER_OK, reason=_DOCKER_WHY),
]


@pytest.fixture(scope="module")
def runner():
    return container.ensure_image()


def _py(runner: dict, code: str, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", "run", "--rm", *extra, "--entrypoint", "python", runner["image"], "-c", code],
                          capture_output=True, text=True, timeout=120)


def test_no_direct_egress_from_the_internal_network(runner):
    container.ensure_egress_image()
    container.ensure_proxy("api.groq.com")
    cp = _py(runner, "import urllib.request;urllib.request.urlopen('https://api.groq.com',timeout=8)",
             "--network", container.INTERNAL_NETWORK)
    assert cp.returncode != 0 and ("resolution" in cp.stderr or "URLError" in cp.stderr)


def test_proxy_refuses_a_host_outside_the_allowlist(runner):
    proxy = container.ensure_proxy("api.groq.com")
    cp = _py(runner, "import urllib.request;urllib.request.urlopen('https://example.com',timeout=8)",
             "--network", container.INTERNAL_NETWORK, "-e", f"HTTPS_PROXY=http://{proxy}:8888")
    assert cp.returncode != 0 and "403" in cp.stderr


def test_proxy_admits_the_allowlisted_host(runner):
    """Control for the refusal above: the same path reaches the allowed host
    (any HTTP status means the TLS tunnel was opened)."""
    proxy = container.ensure_proxy("api.groq.com")
    code = ("import urllib.request,urllib.error\n"
            "try: urllib.request.urlopen('https://api.groq.com/openai/v1/models',timeout=15)\n"
            "except urllib.error.HTTPError as e: print('HTTP', e.code)\n")
    cp = _py(runner, code, "--network", container.INTERNAL_NETWORK, "-e", f"HTTPS_PROXY=http://{proxy}:8888")
    assert cp.returncode == 0 and cp.stdout.startswith("HTTP"), cp.stderr[-400:]


def test_planted_nonce_on_the_host_is_not_readable_inside(runner):
    nonce = "NONCE-" + secrets.token_hex(8)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "nonce.txt"
        p.write_text(nonce, encoding="utf-8")
        code = ("import os\nhits=[]\n"
                "for root,_,files in os.walk('/'):\n"
                "  if root.startswith(('/proc','/sys','/dev')): continue\n"
                "  for f in files:\n"
                "    if f=='nonce.txt': hits.append(os.path.join(root,f))\n"
                "print(len(hits))\n")
        cp = _py(runner, code, "--network", "none")
    assert cp.returncode == 0 and cp.stdout.strip() == "0"
    assert nonce not in cp.stdout


def test_image_contains_no_task_sets_gold_or_seeds(runner):
    code = ("import os\nbad=[]\n"
            "for root,dirs,files in os.walk('/opt/sa'):\n"
            "  for f in files:\n"
            "    p=os.path.join(root,f)\n"
            "    if '/task_sets/' in p or '_held_out' in p or f.endswith('.jsonl') or '.private_seed' in f: bad.append(p)\n"
            "print(len(bad))\n")
    cp = _py(runner, code, "--network", "none")
    assert cp.returncode == 0 and cp.stdout.strip() == "0", cp.stdout
