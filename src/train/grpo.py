from __future__ import annotations

from pathlib import Path
from typing import Any


def build_grpo_job_config(init_model_path: str, output_dir: str) -> dict[str, Any]:
    return {
        "init_model_path": init_model_path,
        "output_dir": output_dir,
        "trainer": "unsloth_grpo",
        "online_env_interaction": True,
        "reward_source": "native_env_reward",
    }


def ensure_grpo_artifact_dirs(config: dict[str, Any]) -> None:
    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)
