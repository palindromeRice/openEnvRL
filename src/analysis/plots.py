from __future__ import annotations

import math
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.analysis.artifact_analysis import StageMetric, TraceView


@dataclass(frozen=True)
class FigureMetadata:
    figure_id: str
    figure_path: Path
    caption_path: Path
    evidence_type: str
    source_paths: tuple[Path, ...]


STAGE_COLORS = {"base": "#777777", "sft": "#3973ac", "grpo": "#4d8b57"}


def ensure_figure_dir(path: str | Path = "results/figures") -> Path:
    figure_dir = Path(path)
    figure_dir.mkdir(parents=True, exist_ok=True)
    return figure_dir


def _metric_value(metric: StageMetric | Mapping[str, Any], field: str) -> float:
    value = getattr(metric, field, None)
    if value is None:
        value = metric[field]
    return float(value)


def describe_plot_series(
    metrics: Mapping[str, StageMetric | Mapping[str, Any]],
) -> dict[str, list[float]]:
    return {
        "success_rate": [_metric_value(metrics[name], "success_rate") for name in metrics],
        "average_reward": [_metric_value(metrics[name], "average_reward") for name in metrics],
        "average_steps": [_metric_value(metrics[name], "average_steps") for name in metrics],
    }


def _source_path(metric: StageMetric | Mapping[str, Any], fallback: str) -> Path:
    path = getattr(metric, "source_path", None)
    if path is None and isinstance(metric, Mapping):
        path = metric.get("source_path")
    return Path(path or fallback)


def _save_figure(
    figure: plt.Figure,
    output_path: Path,
    figure_id: str,
    caption: str,
    evidence_type: str,
    source_paths: Sequence[Path],
) -> FigureMetadata:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    caption_path = output_path.with_suffix(".caption.md")
    figure.savefig(output_path, dpi=180, bbox_inches="tight", metadata={"Description": caption})
    caption_path.write_text(caption + "\n", encoding="utf-8")
    plt.close(figure)
    return FigureMetadata(
        figure_id=figure_id,
        figure_path=output_path,
        caption_path=caption_path,
        evidence_type=evidence_type,
        source_paths=tuple(Path(path) for path in source_paths),
    )


def plot_stage_progression(
    metrics: Mapping[str, StageMetric | Mapping[str, Any]],
    output_path: Path,
) -> FigureMetadata:
    stages = [name for name in ("base", "sft", "grpo") if name in metrics]
    values = [_metric_value(metrics[name], "success_rate") * 100 for name in stages]
    figure, axis = plt.subplots(figsize=(7, 4.2))
    bars = axis.bar(stages, values, color=[STAGE_COLORS[name] for name in stages])
    axis.set_ylim(0, 100)
    axis.set_ylabel("Success rate (%)")
    axis.set_title("Browser-agent success progression")
    axis.text(
        0.5,
        -0.22,
        "Separate randomized evaluations; descriptive comparison, not paired test.",
        ha="center",
        transform=axis.transAxes,
        fontsize=8,
    )
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.0f}%", ha="center")
    return _save_figure(
        figure,
        output_path,
        "figure1_stage_progression",
        "Base, SFT, and corrected final GRPO success rates. Evaluations use separate randomized samples.",
        "stage_aggregate_metrics",
        [_source_path(metrics[name], f"{name}_metrics.json") for name in stages],
    )


def plot_tool_use_timeline(trace: TraceView, output_path: Path) -> FigureMetadata:
    figure, axis = plt.subplots(figsize=(15, 4.8))
    axis.axis("off")
    labels = ["Instruction", "Observation", "Raw completion", "Executed actions", "Reward"]
    values = [
        trace.instruction,
        trace.initial_observation,
        trace.raw_completions[0] if trace.raw_completions else "",
        " → ".join(trace.executed_actions),
        str(trace.rewards[-1] if trace.rewards else 0.0),
    ]
    widths = [28, 30, 28, 30, 10]
    xs = np.linspace(0.1, 0.9, len(labels))
    for index, (x, label, value, width) in enumerate(zip(xs, labels, values, widths)):
        color = "#fff4df" if label == "Executed actions" else "#e8f1ff"
        wrapped_value = textwrap.fill(str(value), width=width)
        axis.text(
            x,
            0.58,
            f"{label}\n\n{wrapped_value}",
            ha="center",
            va="center",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.6", "facecolor": color, "edgecolor": "#777"},
            transform=axis.transAxes,
        )
        if index < len(labels) - 1:
            axis.annotate("", xy=(xs[index + 1] - 0.08, 0.58), xytext=(x + 0.08, 0.58),
                          xycoords=axis.transAxes, textcoords=axis.transAxes,
                          arrowprops={"arrowstyle": "->", "color": "#777"})
    caption = (
        "Representative final GRPO trace. Raw multi-call completion and executed actions are distinct; "
        "queue interpretation is derived from saved step records, not persisted as its own field."
    )
    return _save_figure(
        figure,
        output_path,
        "figure2_tool_use_timeline",
        caption,
        "final_grpo_evaluation_trace",
        [Path("final_grpo_eval_episodes.jsonl")],
    )


def plot_demonstration_evaluation_cards(
    sft_demonstrations: Sequence[Mapping[str, Any]],
    grpo_traces: Sequence[TraceView],
    output_path: Path,
) -> FigureMetadata:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    cards = [
        ("SFT training demonstration", sft_demonstrations[0] if sft_demonstrations else {}),
        ("GRPO final evaluation trace", grpo_traces[0] if grpo_traces else None),
    ]
    for axis, (title, value) in zip(axes, cards):
        axis.axis("off")
        if isinstance(value, TraceView):
            body = "\n".join(value.executed_actions)
        else:
            steps = value.get("steps", []) if isinstance(value, Mapping) else []
            body = "\n".join(str(step.get("action", "")) for step in steps)
        axis.text(0.5, 0.5, f"{title}\n\n{body or 'No trace available'}", ha="center", va="center",
                  bbox={"boxstyle": "round,pad=0.8", "facecolor": "#eef7ee" if "GRPO" in title else "#e8f1ff"},
                  transform=axis.transAxes)
    caption = "Qualitative cards: SFT card is a training demonstration; GRPO card is a final evaluation trace. Neither is a cross-stage distribution estimate."
    return _save_figure(
        figure,
        output_path,
        "figure3_demonstration_evaluation_cards",
        caption,
        "qualitative_training_and_evaluation_traces",
        [Path("sft_successful_trajectories.jsonl"), Path("final_grpo_eval_episodes.jsonl")],
    )


def plot_evaluator_flow(output_path: Path) -> FigureMetadata:
    figure, axis = plt.subplots(figsize=(11, 2.6))
    axis.axis("off")
    labels = ["Model\ncompletion", "Parse every\ntool call", "Derived\nqueue", "Resolve\nvisible IDs", "Execute\none action", "Reward +\nterminal state"]
    xs = np.linspace(0.08, 0.92, len(labels))
    for index, (x, label) in enumerate(zip(xs, labels)):
        axis.text(x, 0.5, label, ha="center", va="center", transform=axis.transAxes,
                  bbox={"boxstyle": "round,pad=0.6", "facecolor": "#fff4df" if label == "Derived\nqueue" else "#f5f5f5"})
        if index < len(labels) - 1:
            axis.annotate("", xy=(xs[index + 1] - 0.06, 0.5), xytext=(x + 0.06, 0.5),
                          xycoords=axis.transAxes, textcoords=axis.transAxes,
                          arrowprops={"arrowstyle": "->", "color": "#777"})
    caption = "Runtime evaluator flow. Queue is derived runtime behavior reconstructed from saved traces, not a standalone persisted artifact field."
    return _save_figure(
        figure,
        output_path,
        "figure4_evaluator_flow",
        caption,
        "evaluator_procedure",
        [Path("src/agents/grpo_tool_policy.py")],
    )


def plot_episode_summary(summary: Mapping[str, Any], output_path: Path) -> FigureMetadata:
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    axes[0].bar(["Success", "Failure"], [summary.get("success_count", 0), summary.get("failure_count", 0)],
                color=["#4d8b57", "#b85c5c"])
    axes[0].set_title("Final GRPO episodes")
    axes[0].set_ylabel("Episodes")
    axes[1].hist(summary.get("episode_lengths", []), bins=range(1, max(summary.get("episode_lengths", [1])) + 2),
                  color="#3973ac", rwidth=0.8)
    axes[1].set_title("Episode length")
    axes[1].set_xlabel("Executed steps")
    caption = "Descriptive summary of all final GRPO evaluation episodes. Source: corrected final per-episode JSONL."
    return _save_figure(
        figure,
        output_path,
        "figure5_final_episode_summary",
        caption,
        "final_grpo_episode_summary",
        [Path("final_grpo_eval_episodes.jsonl")],
    )


def plot_episode_reliability_audit(summary: Mapping[str, Any], output_path: Path) -> FigureMetadata:
    """Render human-readable final-policy outcome and failure behavior."""
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.0, 1.45, 1.2]})
    success_count = int(summary.get("success_count", 0))
    failure_count = int(summary.get("failure_count", 0))
    total = int(summary.get("episode_count", success_count + failure_count))

    axes[0].bar(["Success", "Failure"], [success_count, failure_count], color=["#4d8b57", "#b85c5c"])
    axes[0].set_ylim(0, max(total, 1) * 1.12)
    axes[0].set_ylabel("Episodes")
    axes[0].set_title(f"Outcome\n{success_count}/{total} successful")
    for index, value in enumerate((success_count, failure_count)):
        axes[0].text(index, value + 0.8, str(value), ha="center", fontweight="bold")

    labels = ["Correct selection,\nno Submit", "Premature\nSubmit", "Wrong selection\nthen Submit"]
    categories = (
        "no_submit_after_correct_selection",
        "premature_submit",
        "wrong_selection_then_submit",
    )
    counts = (summary.get("failure_behavior") or {}).get("counts", {})
    values = [int(counts.get(category, 0)) for category in categories]
    axes[1].barh(labels[::-1], values[::-1], color=["#d9973f", "#c77745", "#b85c5c"])
    axes[1].set_xlim(0, max(values + [1]) + 1)
    axes[1].set_xlabel("Failed episodes")
    axes[1].set_title("Observed failure behavior")
    for index, value in enumerate(values[::-1]):
        axes[1].text(value + 0.05, index, str(value), va="center", fontweight="bold")

    lengths = list(summary.get("episode_lengths", []))
    success_lengths = [length for length, record_success in zip(lengths, summary.get("episode_successes", [])) if record_success]
    failure_lengths = [length for length, record_success in zip(lengths, summary.get("episode_successes", [])) if not record_success]
    if not summary.get("episode_successes"):
        success_lengths = lengths[:success_count]
        failure_lengths = lengths[success_count:]
    for x, values_for_group, color, label in ((0, success_lengths, "#4d8b57", "Success"), (1, failure_lengths, "#b85c5c", "Failure")):
        axes[2].scatter([x] * len(values_for_group), values_for_group, s=36, color=color, alpha=0.8, label=label)
    axes[2].set_xticks([0, 1], ["Success", "Failure"])
    axes[2].set_ylabel("Executed steps")
    axes[2].set_title("Steps by outcome")
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].legend(frameon=False, fontsize=8)
    figure.suptitle("Final GRPO reliability audit — 50 evaluation episodes", fontsize=14, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    caption = (
        f"Corrected final GRPO evaluation: {success_count}/{total} successful. "
        "Failure categories are deterministic classifications from requested labels, visible controls, "
        "executed actions, and terminal state: 2 correct selections with no Submit, 2 premature Submit, "
        "and 2 wrong selection then Submit. This is final-policy behavioral evidence, not internal reasoning."
    )
    return _save_figure(
        figure,
        output_path,
        "figure5_final_episode_summary",
        caption,
        "final_grpo_behavioral_reliability",
        [Path("final_grpo_eval_episodes.jsonl")],
    )


def plot_episode_reliability_summary(summary: Mapping[str, Any], output_path: Path) -> FigureMetadata:
    """Render final evaluation outcome as one readable 50-episode strip."""
    figure, axis = plt.subplots(figsize=(12, 3.4))
    successes = list(summary.get("episode_successes", []))
    total = int(summary.get("episode_count", len(successes)))
    success_count = int(summary.get("success_count", sum(successes)))
    failure_count = int(summary.get("failure_count", total - success_count))
    if len(successes) != total:
        successes = [True] * success_count + [False] * failure_count
    colors = ["#4d8b57" if success else "#b85c5c" for success in successes]
    axis.scatter(range(1, total + 1), [0] * total, c=colors, marker="s", s=220, edgecolors="white", linewidths=0.7)
    axis.set_xlim(0, total + 1)
    axis.set_ylim(-1, 1)
    axis.set_yticks([])
    axis.set_xlabel("Evaluation episode")
    axis.set_title(f"Final GRPO reliability — {success_count}/{total} successful", fontsize=15, fontweight="bold")
    axis.text(0.01, 0.18, f"SUCCESS  {success_count}", transform=axis.transAxes, color="#286b39", fontweight="bold")
    axis.text(0.16, 0.18, f"FAILURE  {failure_count}", transform=axis.transAxes, color="#934141", fontweight="bold")
    axis.text(0.99, 0.18, "Each square = one episode", transform=axis.transAxes, ha="right", color="#536579")
    axis.grid(axis="x", alpha=0.15)
    caption = (
        f"Corrected final GRPO evaluation: {success_count}/{total} successful episodes. "
        "Each square represents one saved episode; green indicates success and red indicates failure. "
        "Final-policy reliability evidence only; no cross-stage episode comparison is implied."
    )
    return _save_figure(
        figure,
        output_path,
        "figure5_final_episode_summary",
        caption,
        "final_grpo_episode_outcomes",
        [Path("final_grpo_eval_episodes.jsonl")],
    )


def plot_episode_failure_modes(summary: Mapping[str, Any], output_path: Path) -> FigureMetadata:
    """Render only the observable failure taxonomy for final GRPO."""
    figure, axis = plt.subplots(figsize=(8.6, 4.3))
    labels = ["Correct selections\nbut no Submit", "Premature\nSubmit", "Wrong selection\nthen Submit"]
    categories = (
        "no_submit_after_correct_selection",
        "premature_submit",
        "wrong_selection_then_submit",
    )
    counts = (summary.get("failure_behavior") or {}).get("counts", {})
    values = [int(counts.get(category, 0)) for category in categories]
    bars = axis.bar(labels, values, color=["#d9973f", "#c77745", "#b85c5c"], width=0.62)
    axis.set_ylim(0, max(values + [1]) + 1)
    axis.set_ylabel("Failed episodes")
    axis.set_title("Why the 6 final-GRPO episodes failed", fontsize=15, fontweight="bold")
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.08, str(value), ha="center", fontweight="bold")
    axis.text(0.5, -0.23, "Deterministic categories from requested labels, visible controls, executed actions, and terminal state.",
              transform=axis.transAxes, ha="center", fontsize=8, color="#536579")
    caption = (
        "Final GRPO failure modes: 2 correct selections with no Submit, 2 premature Submit, "
        "and 2 wrong selection then Submit. Categories describe observable behavior, not model intent."
    )
    return _save_figure(
        figure,
        output_path,
        "figure6_final_episode_failure_modes",
        caption,
        "final_grpo_failure_behavior",
        [Path("final_grpo_eval_episodes.jsonl")],
    )


def plot_artifact_lineage(source_paths: Sequence[Path], output_path: Path) -> FigureMetadata:
    figure, axis = plt.subplots(figsize=(11, 2.8))
    axis.axis("off")
    labels = ["Successful\ntrajectories", "SFT\ndataset", "SFT\nmodel", "GRPO\ncheckpoint-20", "Corrected\nevaluation", "Article +\nmetrics"]
    xs = np.linspace(0.08, 0.92, len(labels))
    for index, (x, label) in enumerate(zip(xs, labels)):
        axis.text(x, 0.5, label, ha="center", va="center", transform=axis.transAxes,
                  bbox={"boxstyle": "round,pad=0.6", "facecolor": "#e8f1ff"})
        if index < len(labels) - 1:
            axis.annotate("", xy=(xs[index + 1] - 0.06, 0.5), xytext=(x + 0.06, 0.5),
                          xycoords=axis.transAxes, textcoords=axis.transAxes,
                          arrowprops={"arrowstyle": "->", "color": "#777"})
    return _save_figure(
        figure,
        output_path,
        "figure6_artifact_lineage",
        "Artifact lineage from successful demonstrations through final corrected evaluation and report outputs.",
        "artifact_lineage",
        source_paths,
    )


def plot_capability_radar(
    metrics: Mapping[str, StageMetric | Mapping[str, Any]],
    output_path: Path,
) -> FigureMetadata:
    stages = [name for name in ("base", "sft", "grpo") if name in metrics]
    steps = [_metric_value(metrics[name], "average_steps") for name in stages]
    low, high = min(steps), max(steps)
    step_scores = [100.0 if high == low else 100.0 * (high - value) / (high - low) for value in steps]
    axes_labels = ["Task success", "Average reward", "Relative steps"]
    angles = np.linspace(0, 2 * math.pi, len(axes_labels), endpoint=False).tolist()
    angles += angles[:1]
    figure = plt.figure(figsize=(6.5, 5.6))
    axis = figure.add_subplot(111, polar=True)
    for stage, step_score in zip(stages, step_scores):
        values = [
            _metric_value(metrics[stage], "success_rate") * 100,
            _metric_value(metrics[stage], "average_reward") * 100,
            step_score,
        ]
        values += values[:1]
        axis.plot(angles, values, label=stage.upper(), color=STAGE_COLORS[stage])
        axis.fill(angles, values, color=STAGE_COLORS[stage], alpha=0.08)
    axis.set_xticks(angles[:-1])
    axis.set_xticklabels(axes_labels)
    axis.set_ylim(0, 100)
    axis.set_title("Evidence-backed capability profile")
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    caption = (
        "Radar uses shared measured metrics only. Average reward duplicates success for this binary-reward task. "
        "Relative steps is inverted for display, but failed episodes can terminate early; it is not an independent efficiency score."
    )
    return _save_figure(
        figure,
        output_path,
        "figure7_capability_radar",
        caption,
        "shared_stage_metrics_with_caveats",
        [_source_path(metrics[name], f"{name}_metrics.json") for name in stages],
    )
