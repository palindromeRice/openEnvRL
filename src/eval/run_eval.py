from __future__ import annotations

from typing import Any

from src.agents.rollout import run_episode
from src.data.storage import write_jsonl
from src.env.openenv_task import OpenEnvTransientError
from src.eval.metrics import aggregate_metrics
from src.utils.logging import write_json


def evaluate_policy(
    env_factory: Any,
    policy: Any,
    num_episodes: int,
    max_steps: int,
    max_retries_per_episode: int = 2,
    output_path: str | None = None,
    episodes_output_path: str | None = None,
) -> dict[str, float]:
    episodes = []
    for _ in range(num_episodes):
        last_error: Exception | None = None
        for attempt in range(max_retries_per_episode + 1):
            env = env_factory()
            try:
                episodes.append(run_episode(env, policy, max_steps=max_steps))
                last_error = None
                break
            except OpenEnvTransientError as exc:
                last_error = exc
                if attempt >= max_retries_per_episode:
                    raise RuntimeError(
                        "OpenEnv session kept disconnecting during evaluation. "
                        "Try fewer episodes first or rerun later if the remote service is unstable."
                    ) from exc
            finally:
                close_fn = getattr(env, "close", None)
                if callable(close_fn):
                    close_fn()
        if last_error is not None:
            raise last_error
    metrics = aggregate_metrics(episodes)
    if episodes_output_path is not None:
        write_jsonl(episodes_output_path, episodes)
    if output_path is not None:
        write_json(output_path, metrics)
    return metrics
