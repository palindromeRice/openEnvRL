from __future__ import annotations

from typing import Any
from uuid import uuid4

from src.data.trajectory_schema import build_step_record


def run_episode(env: Any, policy: Any, max_steps: int) -> dict[str, Any]:
    state = env.reset()
    instruction = state["instruction"]
    observation = state["observation"]
    steps: list[dict[str, Any]] = []
    total_reward = 0.0
    done = False
    final_info: dict[str, Any] = {"success": False}

    for _ in range(max_steps):
        action = policy.act(instruction, observation)
        next_state, reward, done, info = env.step(action)
        steps.append(
            build_step_record(
                instruction=instruction,
                observation=observation,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )
        )
        total_reward += reward
        final_info = info
        instruction = next_state["instruction"]
        observation = next_state["observation"]
        if done:
            break

    return {
        "episode_id": str(uuid4()),
        "steps": steps,
        "success": bool(done and final_info.get("success") is True),
        "total_reward": total_reward,
        "num_steps": len(steps),
        "final_info": final_info,
    }
