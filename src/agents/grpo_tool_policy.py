from __future__ import annotations

import re
from typing import Any


_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=click>\s*<parameter=target>\s*([^<\s]+)\s*"
    r"</parameter>\s*</function>\s*</tool_call>",
    re.DOTALL,
)
_VISIBLE_CONTROL_RE = re.compile(r"\[(\d+)\]\s+(?:checkbox|button)\s+'([^']+)'")


def click(target: str) -> str:
    """Click a browser element.

    Args:
        target: BrowserGym element ID, such as "12" or "a3".

    Returns:
        The browser observation after the click.
    """
    raise RuntimeError("Schema-only evaluator tool; BrowserGym executes parsed actions.")


class GRPOToolPolicy:
    """Evaluate GRPO checkpoints through the same click tool contract used in training."""

    def __init__(self, checkpoint_path: str) -> None:
        self.checkpoint_path = checkpoint_path
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self.last_completion = ""
        self.last_action = "NOOP"
        self._pending_actions: list[str] = []

    @staticmethod
    def parse_completion(text: str) -> str:
        actions = GRPOToolPolicy.parse_completion_actions(text, "")
        return actions[0] if actions else "NOOP"

    @staticmethod
    def parse_completion_actions(text: str, observation: str) -> list[str]:
        visible_controls = {
            name: ref for ref, name in _VISIBLE_CONTROL_RE.findall(observation)
        }
        actions: list[str] = []
        for target_text in _TOOL_CALL_RE.findall(text):
            target = target_text.strip().strip("'\"")
            ref = target if target.isdigit() else visible_controls.get(target)
            if ref is not None:
                actions.append(f"click('{ref}')")
        return actions

    def _next_pending_action(self) -> str | None:
        if not self._pending_actions:
            return None
        action = self._pending_actions.pop(0)
        self.last_action = action
        return action

    def load(self) -> None:
        if self.model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint_path, trust_remote_code=True, device_map="auto"
        )

    def build_prompt(self, instruction: str, observation: str) -> str:
        assert self.tokenizer is not None
        messages = [
            {"role": "system", "content": "Use available browser tool click(target) to solve task."},
            {"role": "user", "content": f"Task: {instruction}\nObservation: {observation}"},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tools=[click],
            tokenize=False,
            add_generation_prompt=True,
        )

    def act(self, instruction: str, observation: str) -> str:
        queued_action = self._next_pending_action()
        if queued_action is not None:
            return queued_action
        self.load()
        assert self.model is not None and self.tokenizer is not None
        prompt = self.build_prompt(instruction, observation)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if hasattr(self.model, "device"):
            inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        # Match training's max_completion_length; 512 made every eval episode crawl.
        generated = self.model.generate(**inputs, max_new_tokens=128, do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
        completion = self.tokenizer.decode(generated[0][inputs["input_ids"].shape[1] :], skip_special_tokens=False)
        actions = self.parse_completion_actions(completion, observation)
        action = actions[0] if actions else "NOOP"
        self._pending_actions = actions[1:]
        self.last_completion = completion
        self.last_action = action
        return action

    def get_last_diagnostics(self) -> dict[str, str]:
        return {
            "raw_completion": self.last_completion,
            "parsed_action": self.last_action,
        }

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
