"""``framework/host.py`` — which machine produced a score.

WHY THIS MODULE EXISTS (2026-08-21, written before any second machine ran a task).

``provenance.host`` was ``os.environ.get("SCIENCEARENA_HOST", "win11-local")``,
and ``SCIENCEARENA_HOST`` is set nowhere — not in the repo, not in any run
script, not in the environment of the machine that produced the records.
Measured the day this module was written: **all 3,924 tracked run records carry
the byte-identical string ``"win11-local"``**. A field that returns one constant
on every machine is not provenance; it is a label that looks like provenance.

That was harmless while one laptop produced every score. It stops being harmless
the moment a second machine runs the same player, because:

* ``registry.yaml`` records that gpt-oss-20b reproduces **16/16 byte-identical**
  completions up to ~2400 tokens across days, sessions and server processes.
  That bounded determinism is this project's only genuine drift anchor.
* It is a property of ONE MACHINE. A different CPU is a different SIMD path; a
  different (or absent) iGPU is a different llama.cpp backend. Reduction order
  changes, one logit flips, the continuation diverges. Same weights, same seed,
  same temperature — different text.
* Both leaderboard engines group strictly by ``player_id``
  (``build-data.mjs``, ``framework/leaderboard.py``), and both keep the FIRST
  ``player_version`` / ``resolved_tool_version`` they see. Two machines under one
  ``player_id`` therefore publish one machine's label over both machines' scores.

So the tests below pin two things a constant cannot give: the value must DIFFER
between machines, and it must carry no personally-identifying material, because
every field here is copied verbatim into the public data bundle.
"""
import getpass
import json
import platform

import pytest

from framework import host


# --- the regression the module exists to prevent -----------------------------

def test_two_different_machines_get_different_fingerprints():
    """THE point of the module.

    Under the code this replaces, both of these returned ``"win11-local"``.
    """
    laptop = host.fingerprint({
        "cpu": "Intel64 Family 6 Model 170 Stepping 4, GenuineIntel",
        "arch": "AMD64", "os": "Windows", "os_release": "11",
        "cpu_logical": 14, "ram_gb": 32,
    })
    desktop = host.fingerprint({
        "cpu": "Intel64 Family 6 Model 154 Stepping 3, GenuineIntel",
        "arch": "AMD64", "os": "Windows", "os_release": "11",
        "cpu_logical": 12, "ram_gb": 32,
    })
    assert laptop != desktop


def test_the_derived_label_is_not_the_old_constant():
    """``win11-local`` described every machine, which is why it described none."""
    ident = host.host_identity(env={})
    assert ident["label"] != "win11-local"


@pytest.mark.parametrize("changed", [
    {"cpu": "AMD Ryzen 9 7950X"},
    {"arch": "arm64"},
    {"os": "Linux"},
    {"cpu_logical": 8},
    {"ram_gb": 64},
    {"accelerator": "NVIDIA GeForce RTX 3090"},
])
def test_every_hashed_fact_moves_the_fingerprint(changed):
    """A fact that cannot change the digest is a fact the digest does not cover.

    Parametrised over ``_DIGEST_KEYS``, not ``_FACT_KEYS``. ``os_release`` is
    reported but deliberately unhashed — see
    ``test_a_kernel_patch_does_not_split_one_machine_in_two``.
    """
    base = {
        "cpu": "Intel64 Family 6 Model 170 Stepping 4, GenuineIntel",
        "arch": "AMD64", "os": "Windows", "os_release": "11",
        "cpu_logical": 14, "ram_gb": 32, "accelerator": "Intel Arc",
    }
    assert host.fingerprint(base) != host.fingerprint({**base, **changed})


def test_the_digest_key_set_matches_what_is_parametrised_above():
    """Otherwise a new hashed fact silently escapes the test above."""
    assert set(host._DIGEST_KEYS) == {
        "cpu", "arch", "os", "cpu_logical", "ram_gb", "accelerator"}
    assert set(host._DIGEST_KEYS) <= set(host._FACT_KEYS), \
        "every hashed fact must also be reported, or the digest is unreproducible"


def test_fingerprint_is_stable_across_calls():
    """A provenance value that changes per call cannot group anything."""
    assert host.host_identity()["fingerprint"] == host.host_identity()["fingerprint"]


def test_fingerprint_ignores_key_order():
    a = {"cpu": "x", "arch": "AMD64", "os": "Windows", "os_release": "11",
         "cpu_logical": 4, "ram_gb": 8}
    b = {"ram_gb": 8, "os_release": "11", "os": "Windows", "arch": "AMD64",
         "cpu_logical": 4, "cpu": "x"}
    assert host.fingerprint(a) == host.fingerprint(b)


# --- this block is published, so it must carry no PII ------------------------

def test_identity_leaks_neither_hostname_nor_username():
    """Every field here is copied into the PUBLIC data bundle.

    ``local_gguf.py`` already refuses to publish the served model's absolute path
    for exactly this reason ("NO ABSOLUTE PATH HERE ... C:/Users/<name>/"). The
    same bar applies to the machine block.
    """
    blob = json.dumps(host.host_identity()).lower()
    node = (platform.node() or "").lower()
    user = (getpass.getuser() or "").lower()
    if node:
        assert node not in blob, f"hostname {node!r} leaked into the published host block"
    if user:
        assert user not in blob, f"username {user!r} leaked into the published host block"


def test_identity_contains_no_absolute_paths():
    blob = json.dumps(host.host_identity())
    assert "C:\\Users" not in blob and "C:/Users" not in blob
    assert "/home/" not in blob and "/Users/" not in blob


# --- the operator-facing label ------------------------------------------------

def test_env_label_wins_when_set():
    ident = host.host_identity(env={"SCIENCEARENA_HOST": "desk"})
    assert ident["label"] == "desk"


def test_env_label_does_not_change_the_fingerprint():
    """The label is a NAME. The fingerprint is EVIDENCE.

    If renaming a machine changed its fingerprint, an operator could split one
    machine into two instruments — or merge two into one — by editing an
    environment variable. The digest must come from the hardware alone.
    """
    named = host.host_identity(env={"SCIENCEARENA_HOST": "desk"})
    unnamed = host.host_identity(env={})
    assert named["fingerprint"] == unnamed["fingerprint"]


def test_identity_reports_the_facts_it_hashed():
    """A digest nobody can reproduce is a number, not provenance."""
    ident = host.host_identity()
    for key in ("cpu", "arch", "os", "os_release", "cpu_logical", "ram_gb"):
        assert key in ident, f"{key} is hashed but not reported"
    assert ident["fingerprint"] == host.fingerprint(ident)


def test_identity_carries_a_fingerprint_version():
    """When the recipe changes, old and new digests are not comparable.

    Without a version, a reader diffing two fingerprints cannot tell 'different
    machine' from 'we changed how we hash'.
    """
    assert isinstance(host.host_identity()["fingerprint_version"], int)


def test_collect_facts_never_raises_on_this_machine():
    facts = host.collect_facts()
    assert facts["cpu"] and facts["arch"] and facts["os"]
    assert isinstance(facts["cpu_logical"], int) and facts["cpu_logical"] > 0


def test_ram_is_coarse_enough_to_be_stable():
    """Reported in whole GB.

    Total RAM as raw bytes moves when firmware reserves a different amount, which
    would mint a new 'machine' for the same box after a BIOS update.
    """
    ram = host.collect_facts()["ram_gb"]
    assert ram is None or (isinstance(ram, int) and ram > 0)


# --- path redaction, used for the captured server command line ---------------

def test_redact_paths_removes_the_user_directory():
    line = (r"C:\Users\testuser\.unsloth\llama.cpp\build\bin\Release\llama-server.exe "
            r"--model C:/Users/testuser/.sciencearena/models/gpt-oss-20b-Q8_0.gguf "
            r"--n-gpu-layers 99 --threads 8")
    out = host.redact_paths(line)
    assert "testuser" not in out
    assert "C:\\Users" not in out and "C:/Users" not in out
    # The FLAGS are the whole point — redaction must not eat them.
    assert "--n-gpu-layers 99" in out
    assert "--threads 8" in out
    # The basename identifies the artifact and is already published as `gguf_file`.
    assert "gpt-oss-20b-Q8_0.gguf" in out
    assert "llama-server.exe" in out


def test_redact_paths_handles_posix_home():
    out = host.redact_paths("/home/jdoe/models/x.gguf --threads 4")
    assert "jdoe" not in out
    assert "--threads 4" in out
    assert "x.gguf" in out


def test_redact_paths_is_idempotent():
    once = host.redact_paths(r"C:\Users\testuser\a\b.gguf --ngl 99")
    assert host.redact_paths(once) == once


# --- Codex cross-review, 2026-08-21 ------------------------------------------
#
# All three reproduced locally before being fixed. Findings are hypotheses until
# they run; these ran.

def test_redact_paths_handles_a_user_directory_containing_spaces():
    """Codex #4 (High), reproduced: PII survived redaction.

    The path regex stopped at whitespace, so a Windows profile name containing a
    space came through verbatim -- the directory prefix was replaced but the
    profile name and everything after it survived. Every field in this block is
    published, so this was a live privacy defect.
    """
    out = host.redact_paths(r'llama-server -m "C:\Users\Jane Doe\Models\gpt.gguf" --threads 8')
    assert "Jane Doe" not in out
    assert "Users" not in out
    assert "--threads 8" in out
    assert "gpt.gguf" in out


def test_redact_paths_does_not_eat_a_flag_that_contains_the_username(monkeypatch):
    """Codex #5 (Medium), reproduced: the username scrub destroyed provenance.

    The scrub was a bare case-insensitive substring replace, so a user account
    named `ngl` turned `-ngl 99` — the single most numerically-significant flag
    on this hardware — into `-<user> 99`.
    """
    monkeypatch.setenv("USERNAME", "ngl")
    out = host.redact_paths(r"llama-server -m C:\models\a.gguf -ngl 99 --threads 8")
    assert "-ngl 99" in out


def test_redact_paths_still_scrubs_a_username_sitting_in_a_path(monkeypatch):
    """The narrower rule must not lose the case it was there for."""
    monkeypatch.setenv("USERNAME", "testuser")
    out = host.redact_paths(r'llama-server --data "D:\shares\testuser\models\a.gguf"')
    assert "testuser" not in out


def test_a_kernel_patch_does_not_split_one_machine_in_two(monkeypatch):
    """Codex #2 (High), reproduced: `os_release` was in the digest.

    On Linux `platform.release()` is the kernel build (`6.8.0-40`), which moves on
    every routine patch. A fingerprint that changes with it reports one machine as
    two and fails the audit gate for a machine nobody touched — a false positive
    that would train the operator to ignore the gate.

    `os_release` is still REPORTED, because it is real context. It is simply not
    part of the identity, since a kernel patch does not reorder a SIMD reduction.
    """
    base = {"cpu": "x", "arch": "x86_64", "os": "Linux", "os_release": "6.8.0-40",
            "cpu_logical": 8, "ram_gb": 32, "accelerator": None}
    after = {**base, "os_release": "6.8.0-41"}
    assert host.fingerprint(base) == host.fingerprint(after)
    assert "os_release" in host.host_identity(base)


def test_the_accelerator_is_part_of_the_identity():
    """Codex #1 (High), partially closed.

    Two boxes with the same CPU/RAM/core-count collided, even when one had a
    discrete GPU and the other did not — which is precisely the case that changes
    the llama.cpp backend and therefore the completion.

    This does not make collisions impossible (two genuinely identical machines
    still collide, and that is documented rather than papered over). It closes the
    case that actually changes the numbers.
    """
    cpu_only = {"cpu": "x", "arch": "AMD64", "os": "Windows", "os_release": "11",
                "cpu_logical": 14, "ram_gb": 32, "accelerator": "Intel Arc"}
    with_gpu = {**cpu_only, "accelerator": "NVIDIA GeForce RTX 3090"}
    assert host.fingerprint(cpu_only) != host.fingerprint(with_gpu)


# --- Sonnet cross-review, 2026-08-21 (after the Codex round) ------------------
#
# All reproduced locally before being fixed.

def test_redact_paths_handles_an_UNQUOTED_user_directory_with_spaces():
    """Sonnet #2 (High), reproduced.

    The Codex round fixed the QUOTED case. The unquoted one still leaked, because
    the generic path regex stops at whitespace, so everything from the profile
    name onward survived. A home directory is where usernames live, so it gets a
    rule of its own that tolerates spaces rather than relying on that regex.
    """
    out = host.redact_paths(r"llama-server --model C:\Users\Jane Doe\models\x.gguf --threads 8")
    assert "Jane Doe" not in out
    assert "--threads 8" in out


def test_redact_paths_scrubs_posix_home_with_spaces():
    out = host.redact_paths("llama-server -m /home/jane doe/models/x.gguf -t 4")
    assert "jane doe" not in out
    assert "-t 4" in out


def test_an_unprobeable_accelerator_is_not_the_same_as_having_none():
    """Sonnet #1 (High), reproduced: absent read as zero.

    A machine whose GPU probe FAILED hashed identically to a machine that
    genuinely has no accelerator — so a box with a discrete card could silently
    share an identity with a CPU-only box, which is exactly the collision the
    accelerator fact was added to prevent. Same 'absent is not zero' rule the
    record schema's own `schema_version` exists to preserve.
    """
    unknown = {"cpu": "x", "arch": "AMD64", "os": "Windows", "cpu_logical": 14,
               "ram_gb": 32, "accelerator": host.ACCELERATOR_UNKNOWN}
    none_at_all = {**unknown, "accelerator": host.ACCELERATOR_NONE}
    assert host.fingerprint(unknown) != host.fingerprint(none_at_all)
