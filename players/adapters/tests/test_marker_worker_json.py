"""Marker worker's JSON path must survive Marker's own output model.

Measured 2026-10-08: 22 of 24 held-out pdf-table-extraction-v1 tasks errored with
`TypeError: unhashable type: 'dict'`. Marker had finished every conversion; our
worker then called `rendered.model_dump()`, and any page with a Picture/Figure/
Diagram block carries `images: {BlockId: str}` -- a dict keyed by a pydantic
model, which python-mode dumping turns into an (unhashable) dict key. Marker's
own JSON export (`marker.output.text_from_rendered`) uses `model_dump_json`,
which stringifies the key; the worker now does the same. On the 6 tasks probed,
both paths agreed wherever `model_dump` did not crash.

Runs in the main venv with a stand-in for Marker's models (no marker import).
"""
import importlib.util
from pathlib import Path

import pytest

from pydantic import BaseModel

WORKER = Path(__file__).resolve().parents[2] / "tool_envs" / "marker" / "worker.py"


def _load_worker():
    spec = importlib.util.spec_from_file_location("marker_worker_under_test", WORKER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _BlockId(BaseModel):  # stands in for marker.schema.blocks.base.BlockId
    page_id: int
    block_id: int

    def __str__(self):
        return f"/page/{self.page_id}/Picture/{self.block_id}"

    def __hash__(self):
        return hash(str(self))


class _Block(BaseModel):
    id: str
    block_type: str
    html: str
    children: list["_Block"] | None = None
    images: dict | None = None


class _Rendered(BaseModel):
    children: list[_Block]
    block_type: str = "Document"
    metadata: dict


def _rendered_with_picture() -> _Rendered:
    pic = _Block(id="/page/0/Picture/3", block_type="Picture", html="",
                 images={_BlockId(page_id=0, block_id=3): "base64..."})
    table = _Block(id="/page/0/Table/4", block_type="Table", html="<table><tr><td>1</td></tr></table>")
    page = _Block(id="/page/0/Page/0", block_type="Page", html="", children=[pic, table])
    return _Rendered(children=[page], metadata={"table_of_contents": []})


def test_stand_in_reproduces_the_python_mode_crash():
    # Guards the stand-in: if this stops raising, the test below proves nothing.
    try:
        _rendered_with_picture().model_dump(exclude={"metadata"})
    except TypeError as e:
        assert "unhashable" in str(e)
    else:
        raise AssertionError("stand-in no longer reproduces Marker's BlockId-keyed images dict")


@pytest.mark.skipif(not WORKER.is_file(),
                    reason="players/tool_envs/ (the extractor workers) is not in the public mirror")
def test_json_output_survives_picture_blocks_and_drops_metadata():
    worker = _load_worker()
    state = {"models": None, "converters": {"json": lambda _path: _rendered_with_picture()}}
    out = worker.handle(state, {"output_format": "json", "pdf_path": "unused.pdf"})
    doc = out["json"]
    assert "metadata" not in doc
    page = doc["children"][0]
    assert [c["block_type"] for c in page["children"]] == ["Picture", "Table"]
    assert page["children"][0]["images"] == {"/page/0/Picture/3": "base64..."}
    assert page["children"][1]["html"].startswith("<table>")
