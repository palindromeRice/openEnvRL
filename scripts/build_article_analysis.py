from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.artifact_analysis import (
    ArtifactValidationError,
    _metric_from_json,
    extract_representative_trace,
    load_final_grpo_episodes,
    load_manifest,
    load_sft_demonstrations,
    load_stage_metrics,
    reconcile_grpo_metrics,
    summarize_failure_behavior,
    summarize_final_episodes,
    validate_episode_records,
)
from src.analysis.plots import (
    FigureMetadata,
    plot_artifact_lineage,
    plot_capability_radar,
    plot_episode_reliability_audit,
    plot_episode_failure_modes,
    plot_episode_reliability_summary,
    plot_stage_progression,
)
from src.analysis.report import (
    build_article_report,
    build_schema_validation_markdown,
    write_article_report,
    write_figure_manifest,
    write_schema_validation_report,
)


DEFAULT_SFT_ROOT = Path("artifacts/2026-07-12-sft-full")
DEFAULT_GRPO_ROOT = Path("artifacts/2026-07-26-grpo-gate-r5")
DEFAULT_BASE_REPORT = Path("docs/notes/2026-07-27-click-checkboxes-sft-grpo-technical-report.md")
# Base/SFT/GRPO scored by one evaluator (GRPOToolPolicy), 50 episodes each.
DEFAULT_CONTROLS_ROOT = Path("artifacts/2026-10-22-same-evaluator-controls")
FORBIDDEN_OUTPUT_MARKERS = ("grpo_eval_step_20.json", "invalid intermediate grpo")


def _ensure_output_dir(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory must be non-empty-free: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def _with_sources(metadata: FigureMetadata, sources: Sequence[Path]) -> FigureMetadata:
    return replace(metadata, source_paths=tuple(sources))


def _scan_forbidden_markers(output_dir: Path) -> None:
    for path in output_dir.rglob("*"):
        if not path.is_file():
            continue
        data = path.read_bytes().lower()
        for marker in FORBIDDEN_OUTPUT_MARKERS:
            if marker.encode("utf-8") in data:
                raise ArtifactValidationError(f"forbidden public-result marker in output: {path}")


def run_analysis(
    output_dir: Path,
    sft_root: Path,
    grpo_root: Path,
    base_report_path: Path,
    controls_root: Path | None = None,
) -> dict[str, Path]:
    _ensure_output_dir(output_dir)
    sft_metrics_path = sft_root / "results" / "sft_metrics.json"
    sft_trajectories_path = sft_root / "data" / "successful_trajectories.jsonl"
    sft_manifest_path = sft_root / "run_manifest.json"
    grpo_metrics_path = grpo_root / "results" / "grpo_eval_step_20_corrected_50.json"
    grpo_episodes_path = grpo_root / "results" / "grpo_eval_step_20_corrected_50_episodes.jsonl"
    grpo_manifest_path = grpo_root / "grpo_run_manifest.json"

    if controls_root is None:
        metrics = load_stage_metrics(base_report_path, sft_metrics_path, grpo_metrics_path)
    else:
        results = controls_root / "results"
        base_metrics_path = results / "grpo_eval_step_base_control_50_bracketfix.json"
        sft_metrics_path = results / "grpo_eval_step_sft_control_50.json"
        grpo_metrics_path = results / "grpo_eval_step_grpo_r5_step20_rescore_50.json"
        grpo_episodes_path = results / "grpo_eval_step_grpo_r5_step20_rescore_50_episodes.jsonl"
        base_report_path = base_metrics_path
        metrics = {
            "base": _metric_from_json(base_metrics_path, "base", "same_evaluator_control"),
            "sft": _metric_from_json(sft_metrics_path, "sft", "same_evaluator_control"),
            "grpo": _metric_from_json(grpo_metrics_path, "grpo", "same_evaluator_control"),
        }
    records = load_final_grpo_episodes(grpo_episodes_path)
    validation = validate_episode_records(records, grpo_episodes_path, expected_count=50)
    if not reconcile_grpo_metrics(records, metrics["grpo"]):
        raise ArtifactValidationError("final GRPO aggregate does not reconcile with per-episode records")
    load_sft_demonstrations(sft_trajectories_path, limit=1)
    load_manifest(sft_manifest_path)
    load_manifest(grpo_manifest_path)
    summary = summarize_final_episodes(records)
    summary["failure_behavior"] = summarize_failure_behavior(records)
    trace = extract_representative_trace(next(record for record in records if record.success))

    figure_paths = [
        output_dir / "figure1_stage_progression.png",
        output_dir / "figure5_final_episode_summary.png",
        output_dir / "figure6_final_episode_failure_modes.png",
        output_dir / "figure7_capability_radar.png",
    ]
    figures = [
        plot_stage_progression(metrics, figure_paths[0]),
        _with_sources(plot_episode_reliability_summary(summary, figure_paths[1]), [grpo_episodes_path]),
        _with_sources(plot_episode_failure_modes(summary, figure_paths[2]), [grpo_episodes_path]),
        plot_capability_radar(metrics, figure_paths[3]),
    ]
    figure_manifest_path = output_dir / "figure_manifest.json"
    write_figure_manifest(figures, figure_manifest_path)
    article_path = output_dir / "article.md"
    write_article_report(build_article_report(metrics, summary, trace, figures), article_path)

    validation_payload = {
        "sources": [str(grpo_episodes_path), str(grpo_metrics_path), str(sft_metrics_path), str(base_report_path)],
        "record_count": validation.record_count,
        "required_fields": list(validation.required_fields),
        "invalid_records": [issue.__dict__ for issue in validation.invalid_records],
        "aggregate_reconciled": True,
        "unavailable_evidence": (
            ["Base model evaluation episode traces", "SFT model evaluation episode traces"]
            if controls_root is None
            else []
        ),
    }
    validation_json_path = output_dir / "schema_validation.json"
    validation_md_path = output_dir / "schema_validation.md"
    write_schema_validation_report(validation_payload, validation_json_path, validation_md_path)
    (output_dir / "normalized_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _scan_forbidden_markers(output_dir)
    return {
        "article": article_path,
        "figure_manifest": figure_manifest_path,
        "schema_validation_json": validation_json_path,
        "schema_validation_markdown": validation_md_path,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build artifact-only browser-agent article analysis.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sft-root", type=Path, default=DEFAULT_SFT_ROOT)
    parser.add_argument("--grpo-root", type=Path, default=DEFAULT_GRPO_ROOT)
    parser.add_argument("--base-report", type=Path, default=DEFAULT_BASE_REPORT)
    parser.add_argument("--controls-root", type=Path, default=DEFAULT_CONTROLS_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_analysis(args.output_dir, args.sft_root, args.grpo_root, args.base_report, args.controls_root)
    print(f"article analysis written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
