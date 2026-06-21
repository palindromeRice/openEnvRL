from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.agents.prompting import build_action_prompt
from src.env.action_schema import normalize_action_text


@dataclass
class PolicyConfig:
    model_id: str
    checkpoint_path: str | None = None
    max_new_tokens: int = 32
    temperature: float = 0.0
    top_p: float = 1.0
    device_map: str = "auto"


class BrowserAgentPolicy:
    def __init__(self, config: PolicyConfig) -> None:
        self.config = config
        self.model: Any | None = None
        self.tokenizer: Any | None = None

    def build_prompt(self, instruction: str, observation: str) -> str:
        if self.tokenizer is not None and hasattr(self.tokenizer, "apply_chat_template"):
            messages = [
                {
                    "role": "user",
                    "content": build_action_prompt(instruction=instruction, observation=observation),
                }
            ]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return build_action_prompt(instruction=instruction, observation=observation)

    def load(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = self.config.checkpoint_path or self.config.model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map=self.config.device_map,
        )

    def act(self, instruction: str, observation: str) -> str:
        self.load()
        assert self.model is not None
        assert self.tokenizer is not None

        prompt = self.build_prompt(instruction=instruction, observation=observation)
        inputs = self.tokenizer(prompt, return_tensors="pt")

        if hasattr(self.model, "device"):
            inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

        generated = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=self.config.temperature > 0.0,
            temperature=max(self.config.temperature, 1e-5),
            top_p=self.config.top_p,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        prompt_length = inputs["input_ids"].shape[1]
        output_text = self.tokenizer.decode(
            generated[0][prompt_length:],
            skip_special_tokens=True,
        ).strip()
        return normalize_action_text(output_text)
