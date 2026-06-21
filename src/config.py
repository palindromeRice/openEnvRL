from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    return dict(loaded)


def ensure_runtime_dirs(config: dict[str, Any]) -> None:
    for key in ("results_dir", "data_dir", "models_dir"):
        if key in config:
            Path(config[key]).mkdir(parents=True, exist_ok=True)
