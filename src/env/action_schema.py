from __future__ import annotations

import re
from typing import Any


CANONICAL_CLICK_RE = re.compile(r"^CLICK\s+ref=(\d+)$")
BROWSERGYM_CLICK_RE = re.compile(r"""click\(['"]([^'"]+)['"]\)""")


def strip_action_prefix(text: str) -> str:
    cleaned = text.strip().strip("`")
    if cleaned.lower().startswith("action:"):
        cleaned = cleaned.split(":", 1)[1].strip()
    return cleaned


def parse_model_action(text: str) -> dict[str, Any]:
    cleaned = strip_action_prefix(text)
    canonical_match = CANONICAL_CLICK_RE.match(cleaned)
    if canonical_match:
        return {"type": "CLICK", "ref": int(canonical_match.group(1))}

    browsergym_match = BROWSERGYM_CLICK_RE.search(cleaned)
    if browsergym_match:
        ref_text = browsergym_match.group(1)
        if ref_text.isdigit():
            return {"type": "CLICK", "ref": int(ref_text)}
        return {"type": "CLICK_TEXT", "target": ref_text}

    raise ValueError(f"Unsupported action format: {text}")


def format_action(action: dict[str, Any]) -> dict[str, Any]:
    action_type = action.get("type")
    if action_type == "CLICK":
        ref = action["ref"]
        return {
            "action_type": "click",
            "ref": ref,
            "action_str": f"click('{ref}')",
            "canonical": f"CLICK ref={ref}",
        }
    if action_type == "CLICK_TEXT":
        target = action["target"]
        return {
            "action_type": "click",
            "target": target,
            "action_str": f"click('{target}')",
            "canonical": f"click('{target}')",
        }
    raise ValueError(f"Unsupported action type: {action}")


def normalize_action_text(text: str) -> str:
    action = parse_model_action(text)
    return format_action(action)["canonical"]
