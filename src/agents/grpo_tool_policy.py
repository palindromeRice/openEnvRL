from __future__ import annotations

import re
from typing import Any


_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=click>\s*<parameter=target>\s*([^<\s]+)\s*"
    r"</parameter>\s*</function>\s*</tool_call>",
    re.DOTALL,
)


class GRPOToolPolicy:
    """Evaluate GRPO checkpoints through the same click tool contract used in training."""

    def __init__(self, checkpoint_path: str) -> None:
        self.checkpoint_path = checkpoint_path
        self.model: Any | None = None
        self.tokenizer: Any | None = None

    @staticmethod
    def parse_completion(text: str) -> str:
        match = _TOOL_CALL_RE.search(text)
        if match is None:
            return "NOOP"
        target = match.group(1)
        return f"click('{target}')"

    def load(self) -> None:
        if self.model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint_path, trust_remote_code=True, device_map="auto"
        )

    def act(self, instruction: str, observation: str) -> str:
        self.load()
        assert self.model is not None and self.tokenizer is not None
        messages = [
            {"role": "system", "content": "Use available browser tool click(target) to solve task."},
            {"role": "user", "content": f"Task: {instruction}\nObservation: {observation}"},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if hasattr(self.model, "device"):
            inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        generated = self.model.generate(**inputs, max_new_tokens=512, do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
        completion = self.tokenizer.decode(generated[0][inputs["input_ids"].shape[1] :], skip_special_tokens=False)
        return self.parse_completion(completion)

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
