def generate(task_set_version, seed):
    yield {"task_id": "t1", "arena_id": "fake-arena", "task_set_version": "v1", "difficulty": {"tier": 1}, "input": {"text": "hello"}}
    yield {"task_id": "t2", "arena_id": "fake-arena", "task_set_version": "v1", "difficulty": {"tier": 1}, "input": {"text": "world"}}


def ground_truth(task_id):
    return {"label": "ok"}
