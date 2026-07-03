from __future__ import annotations

from typing import Any, Iterable

from src.agents.rollout import run_episode
from src.data.storage import write_jsonl
from src.env.openenv_task import OpenEnvTransientError


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
    max_retries_per_episode: int = 2,
    output_path: str = "data/successful_trajectories.jsonl",
) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    while len(collected) < target_successes:
        episode: dict[str, Any] | None = None
        for attempt in range(max_retries_per_episode + 1):
            env = env_factory()
            try:
                episode = run_episode(env, policy, max_steps=max_steps)
                break
            except OpenEnvTransientError as exc:
                if attempt >= max_retries_per_episode:
                    raise RuntimeError(
                        "OpenEnv session kept disconnecting during trajectory collection. "
                        "Reduce target successes for smoke tests or rerun when the remote service is stable."
                    ) from exc
            finally:
                close_fn = getattr(env, "close", None)
                if callable(close_fn):
                    close_fn()
        if episode is None:
            continue
        if episode["success"] is True:
            collected.append(episode)
    write_jsonl(output_path, collected)
    return collected
