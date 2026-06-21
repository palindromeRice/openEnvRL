from __future__ import annotations

from typing import Any

from src.agents.rollout import run_episode
from src.eval.metrics import aggregate_metrics
from src.utils.logging import write_json


def evaluate_policy(
    env_factory: Any,
    policy: Any,
    num_episodes: int,
    max_steps: int,
    output_path: str | None = None,
) -> dict[str, float]:
    episodes = []
    for _ in range(num_episodes):
        env = env_factory()
        episodes.append(run_episode(env, policy, max_steps=max_steps))
    metrics = aggregate_metrics(episodes)
    if output_path is not None:
        write_json(output_path, metrics)
    return metrics
