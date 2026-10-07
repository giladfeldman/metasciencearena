"""``provenance.host`` must identify the machine, not a constant.

REGRESSION (2026-08-21). Every run record this project has ever published —
3,924 of them, measured — carries the byte-identical ``provenance.host`` value
``"win11-local"``, because the field was
``os.environ.get("SCIENCEARENA_HOST", "win11-local")`` and ``SCIENCEARENA_HOST``
was never set anywhere.

The tests below were written against that code and watched fail before the fix.
They pin the property that matters: run the same player on two machines and the
records must say so.

See ``framework/host.py`` for why the machine is part of a local player's
identity at all (short version: llama.cpp's bounded byte-for-byte determinism is
a property of one CPU and one backend, and it is this project's only drift
anchor).
"""
from pathlib import Path

import pytest

from framework import host, runner


class _FakeAdapter:
    def play_task(self, envelope, *, timeout_s):
        return {"answer": "ok"}

    def resolved_tool_version(self):
        return "fake-tool/1.0"

    def last_usage(self):
        return None

    def last_response_meta(self):
        return None


class _FakeValidator:
    def iter_errors(self, output):
        return []


class _FakeScorer:
    @staticmethod
    def score(output, ground_truth):
        return {"primary": 1.0}


ENTRY = {
    "player_id": "fake-player",
    "player_version": "fake/1.0",
    "player_type": "tool",
    "adapter_class": "FakeAdapter",
    "openai_temperature": 0.0,
}

ENVELOPE = {
    "task_id": "t1",
    "task_set_version": "v1",
    "visibility": "public",
    "split": "revealed",
}


def _build_record():
    return runner._play_one(
        ENVELOPE, {"gold": 1}, ENTRY, _FakeAdapter(), _FakeScorer(),
        _FakeValidator(), 60, "fake-arena", trials=1, seed=0, split="revealed",
    )


def test_host_is_no_longer_the_hardcoded_constant():
    rec = _build_record()
    assert rec["provenance"]["host"] != host.LEGACY_CONSTANT_HOST


def test_record_carries_a_host_fingerprint_block():
    block = _build_record()["provenance"]["host_fingerprint"]
    assert block["fingerprint"]
    assert block["fingerprint_version"] == host.FINGERPRINT_VERSION
    for key in ("cpu", "arch", "os", "os_release", "cpu_logical", "ram_gb"):
        assert key in block


def test_two_machines_produce_two_different_host_values(monkeypatch):
    """THE regression. Under the replaced code both of these were "win11-local"."""
    def _laptop():
        return {"cpu": "Intel64 Family 6 Model 170 Stepping 4, GenuineIntel",
                "arch": "AMD64", "os": "Windows", "os_release": "11",
                "cpu_logical": 14, "ram_gb": 32}

    def _desktop():
        return {"cpu": "Intel64 Family 6 Model 154 Stepping 3, GenuineIntel",
                "arch": "AMD64", "os": "Windows", "os_release": "11",
                "cpu_logical": 12, "ram_gb": 32}

    monkeypatch.delenv("SCIENCEARENA_HOST", raising=False)

    monkeypatch.setattr(host, "collect_facts", _laptop)
    a = _build_record()["provenance"]

    monkeypatch.setattr(host, "collect_facts", _desktop)
    b = _build_record()["provenance"]

    assert a["host"] != b["host"]
    assert a["host_fingerprint"]["fingerprint"] != b["host_fingerprint"]["fingerprint"]


def test_operator_label_still_wins(monkeypatch):
    """``SCIENCEARENA_HOST`` was the documented escape hatch; keep it working."""
    monkeypatch.setenv("SCIENCEARENA_HOST", "desk")
    assert _build_record()["provenance"]["host"] == "desk"


def test_operator_label_cannot_forge_the_fingerprint(monkeypatch):
    """Renaming a machine must not let it masquerade as a different instrument."""
    monkeypatch.delenv("SCIENCEARENA_HOST", raising=False)
    bare = _build_record()["provenance"]["host_fingerprint"]["fingerprint"]
    monkeypatch.setenv("SCIENCEARENA_HOST", "something-else-entirely")
    named = _build_record()["provenance"]["host_fingerprint"]["fingerprint"]
    assert bare == named


def test_the_published_host_block_carries_no_pii():
    """Every provenance field is copied into the public data bundle."""
    import getpass
    import json
    import platform

    blob = json.dumps(_build_record()["provenance"]).lower()
    for secret in (platform.node(), getpass.getuser()):
        if secret and len(secret) >= 3:
            assert secret.lower() not in blob


def test_the_legacy_constant_is_gone_from_the_framework():
    """Fix the class, not the symptom: no other module may reintroduce it.

    ``framework/host.py`` is exempt — it names the constant so the guard above
    can point at a reason instead of a magic literal.
    """
    root = Path(__file__).resolve().parents[1]
    offenders = [
        p.relative_to(root).as_posix()
        for p in root.rglob("*.py")
        if p.name not in {"host.py"}
        and "__pycache__" not in p.parts
        and "tests" not in p.parts
        and host.LEGACY_CONSTANT_HOST in p.read_text(encoding="utf-8", errors="replace")
    ]
    assert not offenders, f"hardcoded host constant reintroduced in: {offenders}"
