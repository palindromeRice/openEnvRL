from __future__ import annotations

from dataclasses import dataclass

from src.agents.prompting import build_action_prompt


@dataclass
class PolicyConfig:
    model_id: str
    checkpoint_path: str | None = None


class BrowserAgentPolicy:
    def __init__(self, config: PolicyConfig) -> None:
        self.config = config

    def build_prompt(self, instruction: str, observation: str) -> str:
        return build_action_prompt(instruction=instruction, observation=observation)

    def act(self, instruction: str, observation: str) -> str:
        # Placeholder until real model integration is added in Colab runtime.
        _ = self.build_prompt(instruction=instruction, observation=observation)
        return "CLICK ref=1"
