from __future__ import annotations

from typing import Any, Iterable

from src.agents.rollout import run_episode
from src.data.storage import write_jsonl


def collect_successful_trajectories(
    episodes: Iterable[dict[str, Any]],
    target_successes: int,
    output_path: str | None = None,
) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    for episode in episodes:
        if episode.get("success") is True:
            collected.append(episode)
        if len(collected) >= target_successes:
            break

    if output_path is not None:
        write_jsonl(output_path, collected)
    return collected


def run_collection_loop(
    env_factory: Any,
    policy: Any,
    target_successes: int,
    max_steps: int,
    output_path: str = "data/successful_trajectories.jsonl",
) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    while len(collected) < target_successes:
        env = env_factory()
        episode = run_episode(env, policy, max_steps=max_steps)
        if episode["success"] is True:
            collected.append(episode)
    write_jsonl(output_path, collected)
    return collected
