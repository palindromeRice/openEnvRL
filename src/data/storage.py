from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any


def write_jsonl_atomic(path: str | Path, rows: list[dict[str, Any]]) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=file_path.parent,
            prefix=f".{file_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            for row in rows:
                handle.write(json.dumps(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

        with temp_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    json.loads(line)

        temp_path.replace(file_path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Compatibility wrapper for callers that require JSONL persistence."""
    write_jsonl_atomic(path, rows)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
