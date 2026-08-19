"""A `--split private` run must REFUSE the dev-fallback seed.

`parity.resolve_seed` falls back to `revealed_seed + _DEV_PRIVATE_OFFSET` when
`task_sets/<version>/.private_seed` is missing. That fallback is correct for local
parity checks — it keeps the two splits distinct without needing the secret — but
it is *publicly computable*: `benchmark_splits.revealed.seed` is committed in
`arena.yaml`, and the offset is a constant in the source.

Before this gate the only signal was `print("WARNING: ...")` from
`framework/cli.py::_resolve_seed`. A warning on stdout does not survive a
120-call tournament: the run completes, writes run records that look exactly like
official ones, and the leaderboard publishes a "held-out" score for a split that
anyone with repo access can regenerate. That is the same failure class as
`5885189` ("never certify a scan that checked nothing") and `f878f39` ("a missing
corpus must RAISE, not silently shrink the benchmark") — a fallback that is
labelled somewhere but not *enforced* anywhere.

Written 2026-08-09, before creating stats-extraction-v1's task-set v2, because
that migration's whole point is a private split that is genuinely private.
"""
import shutil

import pytest
import yaml

from framework import cli


class _Args:
    def __init__(self, **kw):
        defaults = dict(
            arena="fake_split_arena", task_set="v1", split="private", seed=None,
            players=["stub-pass"], trials=1, timeout=60, tag="gate-test",
            public_only=False, include_held_out=False, held_out_only=False,
            max_tasks=1, overwrite=True,
        )
        defaults.update(kw)
        self.__dict__.update(defaults)


@pytest.fixture()
def arena_root(tmp_path, fixtures_dir, monkeypatch):
    """A copy of the real split fixture, switched to `seed_source: secret_file`."""
    root = tmp_path / "arenas"
    root.mkdir()
    arena = root / "fake_split_arena"
    shutil.copytree(fixtures_dir / "fake_split_arena", arena)

    manifest = yaml.safe_load((arena / "arena.yaml").read_text(encoding="utf-8"))
    manifest["benchmark_splits"]["private"] = {"seed_source": "secret_file"}
    (arena / "arena.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    (arena / "task_sets" / "v1").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(cli, "arenas_root", lambda: root)
    monkeypatch.setattr(cli, "registry_path", lambda: fixtures_dir / "fake_registry.yaml")
    return arena


def _write_seed(arena):
    (arena / "task_sets" / "v1" / ".private_seed").write_text("987654321", encoding="utf-8")


def test_private_run_refuses_the_dev_fallback_seed(arena_root):
    """The defect: a private run with no secret silently used a derivable seed."""
    with pytest.raises(SystemExit) as excinfo:
        cli._cmd_run(_Args())
    msg = str(excinfo.value)
    assert ".private_seed" in msg, msg
    assert "private" in msg.lower(), msg


def test_private_run_succeeds_when_the_secret_is_present(arena_root):
    _write_seed(arena_root)
    assert cli._cmd_run(_Args()) is None or True  # completes without SystemExit


def test_explicit_seed_override_is_still_allowed(arena_root):
    """`--seed` is a deliberate, recorded choice, so it bypasses the gate."""
    cli._cmd_run(_Args(seed=42))


def test_revealed_run_is_unaffected_by_a_missing_private_seed(arena_root):
    """The revealed seed is committed; a missing .private_seed is irrelevant to it."""
    cli._cmd_run(_Args(split="revealed"))


# --- the seed must never be printed for a private run (2026-08-09) -----------


def test_private_run_does_not_print_the_seed(arena_root, capsys):
    """A private run announced its own secret on stdout.

    `framework run --split private` ended with
    `wrote 36 run records to ... (split=private, seed=<9 digits>)`, and
    `runner.run_tournament` logged the same value at INFO. That is the entire
    secret, written into terminal scrollback, CI logs, and any transcript of the
    session — for a value whose only protection is that it is not written down.
    Caught on 2026-08-09 during the stats-extraction-v1 v2 tournament, which
    printed the freshly-minted v2 seed on its first private run.

    The revealed seed is committed in `arena.yaml`, so printing THAT is fine and
    is still useful for reproducing a public run.
    """
    _write_seed(arena_root)
    secret = (arena_root / "task_sets" / "v1" / ".private_seed").read_text(encoding="utf-8").strip()
    cli._cmd_run(_Args(split="private"))
    out = capsys.readouterr()
    assert secret not in out.out, "the private seed was printed on stdout"
    assert secret not in out.err, "the private seed was printed on stderr"


def test_revealed_run_still_reports_its_seed(arena_root, capsys):
    """The public seed is not a secret; keep it visible for reproducibility."""
    cli._cmd_run(_Args(split="revealed"))
    out = capsys.readouterr()
    assert "seed=0" in out.out


# --- retry-failed must mirror run's revealed/public-only default (2026-08-09) --


def test_retry_failed_on_revealed_does_not_trip_the_heldout_egress_gate(arena_root, monkeypatch):
    """`retry-failed --split revealed` was unusable for every cloud player.

    `_cmd_run` applies a safety default — `--split revealed` implies
    `--public-only` — added 2026-08-04 after a revealed run shipped 9 held-out PMC
    papers to a provider. `_cmd_retry_failed` calls `run_tournament` directly and
    never got that default, so it passed `public_only=False`, the egress gate saw
    `will_play_held_out=True`, and it refused:

        RuntimeError: refusing to send HELD-OUT tasks to third-party player(s)

    …for a REVEALED backfill, where there are no held-out tasks to send. Hit for
    real on 2026-08-09 topping up a partial stats-extraction-v1 v2 run. The two
    commands must agree: same split, same visibility filter.
    """
    _write_seed(arena_root)

    class _RetryArgs(_Args):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.player = self.players[0]
            self.max_rounds = 1
            self.cooldown = 0

    # A cloud-classified player is what makes the gate fire at all.
    monkeypatch.setattr(
        "framework.runner.is_cloud_player", lambda entry: True, raising=True
    )
    # No prior records -> everything is "to retry", which is the path that ran.
    cli._cmd_retry_failed(_RetryArgs(split="revealed", tag="retry-gate-test"))
