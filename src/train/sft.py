from __future__ import annotations

from pathlib import Path
from typing import Any


def build_sft_job_config(dataset_path: str, output_dir: str) -> dict[str, Any]:
    return {
        "model_id": "Qwen/Qwen3.5-0.8B",
        "dataset_path": dataset_path,
        "output_dir": output_dir,
        "trainer": "unsloth_sft",
        "lora": True,
    }


def ensure_sft_artifact_dirs(config: dict[str, Any]) -> None:
    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)
