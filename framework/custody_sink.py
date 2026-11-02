"""Opt-in raw run evidence staged outside every project for article-finder custody.

The normal run record is still redacted. This sink is never a publication output.
"""
from __future__ import annotations
import json
import tempfile
from pathlib import Path


class CustodyRunSink:
    def __init__(self, path: Path, *, maximum_bytes: int = 512 * 1024 * 1024):
        self.path = path.resolve()
        temporary = Path(tempfile.gettempdir()).resolve()
        project = Path(__file__).resolve().parents[1]
        if not self.path.is_relative_to(temporary) or self.path.is_relative_to(project) or self.path.suffix != '.jsonl':
            raise ValueError('Raw custody evidence must be a JSONL file in OS temporary staging outside the project')
        if maximum_bytes <= 0:
            raise ValueError('Positive custody byte budget required')
        self.maximum_bytes = maximum_bytes
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise ValueError('Use a new custody file; original attempts are never overwritten or mixed')
        self.path.touch(exist_ok=False)

    def write(self, record: dict) -> None:
        data = (json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')
        if self.path.stat().st_size + len(data) > self.maximum_bytes:
            raise RuntimeError('Raw custody evidence budget exhausted; retained attempts must be registered')
        with self.path.open('ab') as stream:
            stream.write(data)
            stream.flush()
