import json
from pathlib import Path
from typing import Any


def ensure_parent(path: str | Path) -> Path:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    file_path = ensure_parent(path)
    file_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
