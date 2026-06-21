from __future__ import annotations

from typing import Any


def parse_model_action(text: str) -> dict[str, Any]:
    parts = text.strip().split()
    if len(parts) != 2 or parts[0] != "CLICK" or not parts[1].startswith("ref="):
        raise ValueError(f"Unsupported action format: {text}")
    return {"type": "CLICK", "ref": int(parts[1].split("=", 1)[1])}


def format_action(action: dict[str, Any]) -> dict[str, Any]:
    if action.get("type") != "CLICK":
        raise ValueError(f"Unsupported action type: {action}")
    return {"action_type": "click", "ref": action["ref"]}
