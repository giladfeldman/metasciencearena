"""Model-written code must not inherit our environment (2026-09-27).

`code-translation-r-v1` executed the player's R with `{**os.environ}`; a planted
nonce showed it could read SCIENCEARENA_PRIVATE_SEED and the real profile.
`hermetic.untrusted_code_env` is an allowlist -- these tests plant secrets and
assert they do not cross, and assert what MUST cross still does.
"""
from __future__ import annotations

from pathlib import Path

from framework import hermetic

BASE = {
    "SCIENCEARENA_PRIVATE_SEED": "nonce-seed",
    "CLAUDE_CODE_OAUTH_TOKEN": "nonce-token",
    "ANTHROPIC_API_KEY": "nonce-key",
    "USERPROFILE": r"C:\Users\someone",
    "HOME": "/home/someone",
    "LOCALAPPDATA": r"C:\Users\someone\AppData\Local",
    "PATH": r"C:\Windows\system32",
    "SystemRoot": r"C:\Windows",
    "R_HOME": r"C:\Program Files\R\R-4.4.0",
    "ARENA_RSCRIPT": "Rscript",
}


def test_secrets_and_profile_do_not_cross(tmp_path: Path) -> None:
    env = hermetic.untrusted_code_env(tmp_path, base=BASE)
    joined = "\n".join(f"{k}={v}" for k, v in env.items())
    for secret in ("nonce-seed", "nonce-token", "nonce-key", "someone"):
        assert secret not in joined, f"{secret!r} reached model-written code"


def test_home_points_at_scratch_and_the_minimum_survives(tmp_path: Path) -> None:
    env = hermetic.untrusted_code_env(tmp_path, {"ARENA_DATA": "d.csv"}, base=BASE)
    for key in ("HOME", "USERPROFILE", "R_USER", "TEMP", "TMP"):
        assert env[key] == str(tmp_path)
    # Two-sided: an allowlist that kept nothing would also pass the test above.
    assert env["PATH"] == BASE["PATH"]
    assert env["SystemRoot"] == BASE["SystemRoot"]
    assert env["R_HOME"] == BASE["R_HOME"]
    assert env["ARENA_RSCRIPT"] == "Rscript"
    assert env["ARENA_DATA"] == "d.csv"


def test_extra_is_applied_last(tmp_path: Path) -> None:
    env = hermetic.untrusted_code_env(tmp_path, {"HOME": "override"}, base=BASE)
    assert env["HOME"] == "override"
