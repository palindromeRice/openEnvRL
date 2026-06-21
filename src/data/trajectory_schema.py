from __future__ import annotations

from typing import Any


def build_step_record(
    instruction: str,
    observation: str,
    action: str,
    reward: float,
    done: bool,
    info: dict[str, Any],
) -> dict[str, Any]:
    return {
        "instruction": instruction,
        "observation": observation,
        "action": action,
        "reward": reward,
        "done": done,
        "info": info,
    }
