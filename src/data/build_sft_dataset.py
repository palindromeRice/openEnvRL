from __future__ import annotations

from pathlib import Path
from typing import Any

from src.data.storage import write_jsonl


def build_step_dataset(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        for step in episode.get("steps", []):
            rows.append(
                {
                    "instruction": step["instruction"],
                    "observation": step["observation"],
                    "target_action": step["action"],
                }
            )
    return rows


def write_step_dataset(episodes: list[dict[str, Any]], path: str | Path = "data/sft_dataset.jsonl") -> list[dict[str, Any]]:
    rows = build_step_dataset(episodes)
    write_jsonl(path, rows)
    return rows
