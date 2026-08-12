from src.analysis.report import build_comparison_table
from pathlib import Path

from src.analysis.artifact_analysis import StageMetric, TraceView
from src.analysis.plots import FigureMetadata
from src.analysis.report import (
    build_article_report,
    build_figure_manifest,
    build_schema_validation_markdown,
)


def test_build_comparison_table_formats_three_models():
    table = build_comparison_table({
        "base": {"success_rate": 0.2, "average_reward": 0.3, "average_steps": 4.0},
        "sft": {"success_rate": 0.4, "average_reward": 0.5, "average_steps": 3.0},
        "grpo": {"success_rate": 0.6, "average_reward": 0.7, "average_steps": 2.0},
    })
    assert "base" in table.lower()
    assert "sft" in table.lower()
    assert "grpo" in table.lower()


def test_build_article_report_contains_provenance_and_limits(tmp_path):
    metrics = {
        "base": StageMetric("base", 0.30, 0.30, 4.36, Path("base.md"), "base"),
        "sft": StageMetric("sft", 0.58, 0.58, 1.42, Path("sft.json"), "sft"),
        "grpo": StageMetric("grpo", 0.88, 0.88, 3.32, Path("grpo.json"), "grpo"),
    }
    trace = TraceView(
        "episode-1", "Select A and click Submit.", "obs",
        ("raw",), ("click('33')",), ("click('33')",), (1.0,), (True,), True, True,
    )
    figure = FigureMetadata("figure1", tmp_path / "figure1.png", tmp_path / "figure1.caption.md", "metrics", (Path("grpo.json"),))

    report = build_article_report(metrics, {
        "success_count": 44,
        "failure_count": 6,
        "episode_count": 50,
        "failure_behavior": {"counts": {
            "no_submit_after_correct_selection": 2,
            "premature_submit": 2,
            "wrong_selection_then_submit": 2,
        }},
    }, trace, [figure])

    assert "30%" in report
    assert "58%" in report
    assert "88%" in report
    assert "behavioral" in report.lower()
    assert "separate randomized" in report.lower()
    assert "invalid intermediate" not in report.lower()
    assert "grpo.json" in report
    assert "no Submit" in report
    assert "premature Submit" in report


def test_schema_markdown_and_figure_manifest_are_source_explicit(tmp_path):
    figure = FigureMetadata("figure1", tmp_path / "figure1.png", tmp_path / "figure1.caption.md", "metrics", (Path("grpo.json"),))
    manifest = build_figure_manifest([figure])
    validation = {
        "sources": ["grpo.json"],
        "record_count": 50,
        "required_fields": ["steps[].policy_diagnostics.raw_completion"],
        "invalid_records": [],
        "aggregate_reconciled": True,
    }

    assert manifest["figures"][0]["figure_id"] == "figure1"
    markdown = build_schema_validation_markdown(validation)
    assert "50" in markdown
    assert "raw_completion" in markdown
