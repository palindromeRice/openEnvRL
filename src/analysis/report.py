from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.analysis.artifact_analysis import StageMetric, TraceView
from src.analysis.plots import FigureMetadata


def _metric_value(metric: StageMetric | Mapping[str, Any], field: str) -> float:
    value = getattr(metric, field, None)
    if value is None:
        value = metric[field]
    return float(value)


def build_comparison_table(metrics: Mapping[str, Mapping[str, Any] | StageMetric]) -> str:
    lines = [
        "| Model | Success Rate | Avg Reward | Avg Steps |",
        "| --- | --- | --- | --- |",
    ]
    for model_name, values in metrics.items():
        lines.append(
            f"| {model_name} | {_metric_value(values, 'success_rate'):.3f} | "
            f"{_metric_value(values, 'average_reward'):.3f} | {_metric_value(values, 'average_steps'):.3f} |"
        )
    return "\n".join(lines)


def build_analysis_sections() -> list[str]:
    return [
        "What is a trajectory? A trajectory is one full episode record made of step-by-step observations, actions, rewards, and termination state.",
        "Why does SFT help RL? SFT gives the policy a better starting point so online RL spends less time exploring obviously bad actions.",
        "What is GRPO? GRPO improves the policy from reward-ranked rollout groups using the native environment reward.",
        "What are the limits? One task, small data budget, separate randomized evaluations, and no independent rerun.",
    ]


def build_article_report(
    metrics: Mapping[str, StageMetric | Mapping[str, Any]],
    episode_summary: Mapping[str, Any],
    trace: TraceView,
    figure_manifest: Sequence[FigureMetadata],
) -> str:
    figure_lines = []
    for figure in figure_manifest:
        sources = ", ".join(str(path) for path in figure.source_paths)
        figure_lines.append(f"- `{figure.figure_id}`: `{figure.figure_path}` — source: `{sources}`")
    trace_actions = " → ".join(trace.executed_actions)
    sections = [
        "# From Demonstrations to Reward Learning: An Inspectable Browser Agent",
        "## Summary",
        "This engineering report uses existing artifacts only. It follows a small browser agent from Base to SFT to corrected final GRPO, then opens one tool-use trace so success is tied to observable actions and rewards.",
        "## Stage comparison",
        build_comparison_table(metrics),
        "Headline success progression: "
        + ", ".join(
            f"{stage.upper()} {_metric_value(metrics[stage], 'success_rate'):.0%}"
            for stage in ("base", "sft", "grpo")
            if stage in metrics
        ),
        "The stages use separate randomized evaluations. Scores are descriptive, not paired per-puzzle comparisons.",
        "## What the agent does",
        f"Representative instruction: `{trace.instruction}`\n\nExecuted action sequence: `{trace_actions}`\n\nFinal reward: `{trace.rewards[-1] if trace.rewards else 0.0}`.",
        "Raw model completion, parsed action, executed browser action, and reward are separate evidence layers. The queue shown in figures is derived from saved trace records; it is not a standalone persisted field.",
        "## Final GRPO episode evidence",
        f"The corrected final evaluation contains `{episode_summary.get('success_count', 0)}/{episode_summary.get('episode_count', 0)}` successful episodes and `{episode_summary.get('failure_count', 0)}` failures.",
        "## Interpretability boundary",
        "This is behavioral trace interpretability and procedural transparency, not neuron-level mechanistic interpretability. Durable Base/SFT model-evaluation traces are unavailable, so no cross-stage action distributions are claimed.",
        "## Reproducibility artifacts",
        "\n".join(figure_lines),
        *build_analysis_sections(),
        "## Limitations",
        "One MiniWoB++ task family, small evaluation samples, separate randomized samples, and no independent rerun. The result is a strong single-task engineering baseline, not general web-agent capability.",
    ]
    return "\n\n".join(sections) + "\n"


def build_figure_manifest(figures: Sequence[FigureMetadata]) -> dict[str, Any]:
    return {
        "figures": [
            {
                "figure_id": figure.figure_id,
                "figure_path": str(figure.figure_path),
                "caption_path": str(figure.caption_path),
                "evidence_type": figure.evidence_type,
                "source_paths": [str(path) for path in figure.source_paths],
            }
            for figure in figures
        ]
    }


def build_schema_validation_markdown(validation_payload: Mapping[str, Any]) -> str:
    lines = [
        "# Schema Validation",
        f"- Sources: {', '.join(str(item) for item in validation_payload.get('sources', []))}",
        f"- Record count: {validation_payload.get('record_count', 'unknown')}",
        f"- Aggregate reconciled: {validation_payload.get('aggregate_reconciled', False)}",
        "",
        "## Required fields",
    ]
    lines.extend(f"- `{field}`" for field in validation_payload.get("required_fields", []))
    lines.extend(["", "## Invalid records"])
    invalid_records = validation_payload.get("invalid_records", [])
    if not invalid_records:
        lines.append("None.")
    else:
        for issue in invalid_records:
            lines.append(
                f"- line {issue.get('source_line')}: `{issue.get('field_path')}` — {issue.get('message')}"
            )
    return "\n".join(lines) + "\n"


def write_article_report(content: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return output_path


def write_schema_validation_report(
    validation_payload: Mapping[str, Any],
    json_path: Path,
    markdown_path: Path,
) -> None:
    json_path.write_text(json.dumps(validation_payload, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(build_schema_validation_markdown(validation_payload), encoding="utf-8")


def write_figure_manifest(figures: Sequence[FigureMetadata], output_path: Path) -> None:
    output_path.write_text(json.dumps(build_figure_manifest(figures), indent=2) + "\n", encoding="utf-8")


def write_report(metrics: Mapping[str, Mapping[str, Any]], path: str = "results/report.md") -> str:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n\n".join(["# Browser Agent RL Report", build_comparison_table(metrics), *build_analysis_sections()])
    report_path.write_text(content, encoding="utf-8")
    return content
