from __future__ import annotations

from typing import Any


def summarize_probe_result(raw_obs: dict[str, Any]) -> dict[str, Any]:
    dom_elements = raw_obs.get("dom_elements", [])
    return {
        "instruction": raw_obs.get("utterance", ""),
        "fields": raw_obs.get("fields", {}),
        "dom_refs": [element.get("ref") for element in dom_elements if "ref" in element],
        "raw_keys": sorted(raw_obs.keys()),
    }


def sample_probe_payload() -> dict[str, Any]:
    raw_obs = {
        "utterance": "Select Apple",
        "fields": {"target 0": "Apple"},
        "dom_elements": [{"ref": 1, "text": "Apple"}],
    }
    return {
        "summary": summarize_probe_result(raw_obs),
        "sample_action_attempt": {"action_type": "CLICK", "ref": 1},
    }
