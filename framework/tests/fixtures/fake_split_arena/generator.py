"""Split-aware fixture generator for runner tests.

Mirrors the dual-benchmark contract: accepts `split`, tags each envelope with
`split` + the derived `visibility`. Gold deliberately differs from
StubPassAdapter's output so the scorer emits a content-bearing finding (lets the
runner-redaction path be tested).
"""


def generate(task_set_version, seed, split="revealed"):
    visibility = "public" if split == "revealed" else "held_out"
    yield {
        "task_id": "t1",
        "arena_id": "fake-split-arena",
        "task_set_version": "v1",
        "difficulty": {"tier": 1},
        "input": {"text": "hello"},
        "split": split,
        "visibility": visibility,
    }


def ground_truth(task_id):
    return {"label": "gold"}
