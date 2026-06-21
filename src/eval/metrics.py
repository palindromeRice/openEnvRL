from __future__ import annotations

from typing import Any


def aggregate_metrics(episodes: list[dict[str, Any]]) -> dict[str, float]:
    count = len(episodes)
    if count == 0:
        return {"success_rate": 0.0, "average_reward": 0.0, "average_steps": 0.0}

    successes = sum(1 for episode in episodes if episode.get("success") is True)
    total_reward = sum(float(episode.get("total_reward", 0.0)) for episode in episodes)
    total_steps = sum(int(episode.get("num_steps", 0)) for episode in episodes)
    return {
        "success_rate": successes / count,
        "average_reward": total_reward / count,
        "average_steps": total_steps / count,
    }
