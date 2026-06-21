from __future__ import annotations

from pathlib import Path
from typing import Any


def ensure_figure_dir(path: str = "results/figures") -> Path:
    figure_dir = Path(path)
    figure_dir.mkdir(parents=True, exist_ok=True)
    return figure_dir


def describe_plot_series(metrics: dict[str, dict[str, Any]]) -> dict[str, list[float]]:
    return {
        "success_rate": [float(metrics[name]["success_rate"]) for name in metrics],
        "average_reward": [float(metrics[name]["average_reward"]) for name in metrics],
        "average_steps": [float(metrics[name]["average_steps"]) for name in metrics],
    }
