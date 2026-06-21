from __future__ import annotations

from typing import Any, Iterable

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
