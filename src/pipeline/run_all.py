from __future__ import annotations

from typing import Any


def list_pipeline_phases() -> list[str]:
    return [
        "native_probe",
        "openenv_probe",
        "baseline_eval",
        "collect_successes",
        "build_sft_dataset",
        "sft_train",
        "sft_eval",
        "grpo_train",
        "grpo_eval",
        "analysis",
    ]


def run_named_phase(name: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
    if name not in list_pipeline_phases():
        raise ValueError(f"Unknown phase: {name}")
    return {"phase": name, "status": "stub", "context": context or {}}


def run_full_pipeline(context: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    return [run_named_phase(name, context=context) for name in list_pipeline_phases()]
