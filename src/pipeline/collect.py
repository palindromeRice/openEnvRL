from __future__ import annotations

from numbers import Real
from pathlib import Path
from typing import Any, Iterable
from uuid import UUID

from src.agents.rollout import run_episode
from src.data.storage import read_jsonl, write_jsonl_atomic
from src.env.openenv_task import OpenEnvTransientError


def validate_successful_episodes(
    rows: list[dict[str, Any]], target_successes: int
) -> None:
    if len(rows) > target_successes:
        raise ValueError("successful episode count exceeds target_successes")

    episode_ids: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("successful trajectory row must be an object")
        episode_id = row.get("episode_id")
        if not isinstance(episode_id, str):
            raise ValueError("episode_id must be a UUID4 string")
        try:
            if UUID(episode_id).version != 4:
                raise ValueError
        except (ValueError, AttributeError) as exc:
            raise ValueError("episode_id must be a UUID4 string") from exc
        if episode_id in episode_ids:
            raise ValueError(f"duplicate episode_id: {episode_id}")
        episode_ids.add(episode_id)

        if row.get("success") is not True:
            raise ValueError("successful trajectory must have success=True")
        reward = row.get("total_reward")
        if not isinstance(reward, Real) or isinstance(reward, bool):
            raise ValueError("successful trajectory must have a numeric total_reward")
        steps = row.get("steps")
        if not isinstance(steps, list) or not steps:
            raise ValueError("successful trajectory must have nonempty steps")


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
        validate_successful_episodes(collected, target_successes)
        write_jsonl_atomic(output_path, collected)
    return collected


def run_collection_loop(
    env_factory: Any,
    policy: Any,
    target_successes: int,
    max_steps: int,
    max_retries_per_episode: int = 2,
    output_path: str = "data/successful_trajectories.jsonl",
) -> list[dict[str, Any]]:
    output = Path(output_path)
    collected = read_jsonl(output) if output.exists() else []
    validate_successful_episodes(collected, target_successes)

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
            proposed = [*collected, episode]
            validate_successful_episodes(proposed, target_successes)
            write_jsonl_atomic(output_path, proposed)
            collected = proposed
    return collected
