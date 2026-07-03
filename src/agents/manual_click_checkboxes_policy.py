from __future__ import annotations

import re
from dataclasses import dataclass


CHECKBOX_RE = re.compile(r"\[(\d+)\]\s+checkbox\s+'([^']+)'(\s+checked)?", re.IGNORECASE)
BUTTON_RE = re.compile(r"\[(\d+)\]\s+button\s+'([^']+)'", re.IGNORECASE)
WORD_RE = re.compile(r"[A-Za-z0-9]+")


@dataclass
class CheckboxEntry:
    ref: int
    label: str
    checked: bool


class ManualClickCheckboxesPolicy:
    """Deterministic smoke-test policy for the real MiniWoB click-checkboxes task."""

    def act(self, instruction: str, observation: str) -> str:
        targets = self._extract_targets(instruction)
        checkboxes = self._extract_checkboxes(observation)

        for target in targets:
            match = self._find_checkbox(target, checkboxes)
            if match is None:
                raise ValueError(f"Could not find target checkbox for '{target}' in observation.")
            if not match.checked:
                return f"CLICK ref={match.ref}"

        submit_ref = self._find_submit_button(observation)
        if submit_ref is None:
            raise ValueError("Could not find submit button after selecting all targets.")
        return f"CLICK ref={submit_ref}"

    def _extract_targets(self, instruction: str) -> list[str]:
        if not instruction.strip():
            raise ValueError("Instruction is empty.")

        cleaned_instruction = instruction.strip()
        lowered = cleaned_instruction.lower()
        if lowered.startswith("select "):
            cleaned_instruction = cleaned_instruction[7:]
            lowered = cleaned_instruction.lower()

        cleaned_instruction = re.sub(
            r"\s+and\s+click\s+submit\.?\s*$",
            "",
            cleaned_instruction,
            flags=re.IGNORECASE,
        ).strip()
        lowered = cleaned_instruction.lower()

        if lowered == "nothing":
            return []
        if instruction.lower().startswith("select "):
            instruction = cleaned_instruction
        else:
            instruction = cleaned_instruction

        parts = re.split(r",|\band\b", instruction, flags=re.IGNORECASE)
        targets = []
        for part in parts:
            words = WORD_RE.findall(part)
            if words:
                targets.append(" ".join(words).strip().lower())
        if not targets:
            raise ValueError(f"Could not parse checkbox targets from instruction: {instruction}")
        return targets

    def _extract_checkboxes(self, observation: str) -> list[CheckboxEntry]:
        checkboxes = []
        for match in CHECKBOX_RE.finditer(observation):
            checkboxes.append(
                CheckboxEntry(
                    ref=int(match.group(1)),
                    label=match.group(2).strip().lower(),
                    checked=bool(match.group(3)),
                )
            )
        return checkboxes

    def _find_checkbox(self, target: str, checkboxes: list[CheckboxEntry]) -> CheckboxEntry | None:
        for checkbox in checkboxes:
            if checkbox.label == target:
                return checkbox
        return None

    def _find_submit_button(self, observation: str) -> int | None:
        for match in BUTTON_RE.finditer(observation):
            label = match.group(2).strip().lower()
            if label in {"submit", "done"}:
                return int(match.group(1))
        return None
