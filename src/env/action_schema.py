from __future__ import annotations

import re
from typing import Any


CANONICAL_CLICK_RE = re.compile(r"^CLICK\s+ref=(\d+)$")
BRACKETED_CLICK_RE = re.compile(r"^CLICK\s+\[(\d+)\]$", re.IGNORECASE)
SIMPLE_CLICK_RE = re.compile(r"^CLICK\s+(\d+)$", re.IGNORECASE)
BROWSERGYM_CLICK_RE = re.compile(r"""click\(['"]([^'"]+)['"]\)""")
NATURAL_LANGUAGE_CLICK_RE = re.compile(
    r"""^click(?:\s+the\s+\w+)?\s+['"]([^'"]+)['"]\.?$""",
    re.IGNORECASE,
)
NOOP_RE = re.compile(r"^NOOP$", re.IGNORECASE)


def strip_action_prefix(text: str) -> str:
    cleaned = text.strip().strip("`")
    if cleaned.lower().startswith("action:"):
        cleaned = cleaned.split(":", 1)[1].strip()
    return cleaned


def parse_model_action(text: str) -> dict[str, Any]:
    cleaned = strip_action_prefix(text)
    if NOOP_RE.match(cleaned):
        return {"type": "NOOP"}
    canonical_match = CANONICAL_CLICK_RE.match(cleaned)
    if canonical_match:
        return {"type": "CLICK", "ref": int(canonical_match.group(1))}

    bracketed_match = BRACKETED_CLICK_RE.match(cleaned)
    if bracketed_match:
        return {"type": "CLICK", "ref": int(bracketed_match.group(1))}

    simple_match = SIMPLE_CLICK_RE.match(cleaned)
    if simple_match:
        return {"type": "CLICK", "ref": int(simple_match.group(1))}

    browsergym_match = BROWSERGYM_CLICK_RE.search(cleaned)
    if browsergym_match:
        ref_text = browsergym_match.group(1)
        if ref_text.isdigit():
            return {"type": "CLICK", "ref": int(ref_text)}
        return {"type": "CLICK_TEXT", "target": ref_text}

    natural_language_match = NATURAL_LANGUAGE_CLICK_RE.match(cleaned)
    if natural_language_match:
        target = natural_language_match.group(1)
        if target.isdigit():
            return {"type": "CLICK", "ref": int(target)}
        return {"type": "CLICK_TEXT", "target": target}

    raise ValueError(f"Unsupported action format: {text}")


def format_action(action: dict[str, Any]) -> dict[str, Any]:
    action_type = action.get("type")
    if action_type == "NOOP":
        return {"action_type": "noop", "action_str": "NOOP", "canonical": "NOOP"}
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
