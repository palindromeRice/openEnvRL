import json
from pathlib import Path

import pytest

from src.analysis.artifact_analysis import (
    ArtifactValidationError,
    load_final_grpo_episodes,
    load_manifest,
    load_sft_demonstrations,
    load_stage_metrics,
    reconcile_grpo_metrics,
    extract_representative_trace,
    summarize_final_episodes,
    validate_episode_records,
)


FIXTURES = Path(__file__).parents[1] / "fixtures" / "analysis"


def test_load_stage_metrics_preserves_values_and_provenance():
    metrics = load_stage_metrics(
        base_report_path=FIXTURES / "base_report.md",
        sft_metrics_path=FIXTURES / "sft_metrics.json",
        grpo_metrics_path=FIXTURES / "final_grpo_metrics.json",
    )

    assert metrics["base"].success_rate == 0.30
    assert metrics["sft"].average_steps == 1.42
    assert metrics["grpo"].success_rate == 0.88
    assert metrics["grpo"].source_path.name == "final_grpo_metrics.json"
    assert metrics["grpo"].evidence_type == "corrected_final_grpo_aggregate"


def test_load_final_grpo_fixture_and_reconcile_metrics():
    records = load_final_grpo_episodes(FIXTURES / "final_grpo_episodes.jsonl")
    metrics = json.loads((FIXTURES / "final_grpo_metrics.json").read_text())

    assert len(records) == 50
    assert sum(record.success for record in records) == 44
    assert reconcile_grpo_metrics(records, metrics) is True


def test_load_manifest_preserves_provenance():
    manifest = load_manifest(FIXTURES / "final_grpo_manifest.json")

    assert manifest["run_id"] == "fixture-grpo"
    assert manifest["config"]["task_name"] == "click-checkboxes"


def test_validate_episode_records_reports_required_schema():
    records = load_final_grpo_episodes(FIXTURES / "final_grpo_episodes.jsonl")
    report = validate_episode_records(
        records,
        source_path=FIXTURES / "final_grpo_episodes.jsonl",
        expected_count=50,
    )

    assert report.valid is True
    assert report.record_count == 50
    assert "steps[].policy_diagnostics.raw_completion" in report.required_fields
    assert not report.invalid_records


def test_load_sft_demonstrations_reads_original_action_format():
    demonstrations = load_sft_demonstrations(FIXTURES / "sft_trajectories.jsonl")

    assert demonstrations[0]["evidence_type"] == "SFT training demonstration"
    assert demonstrations[0]["steps"][0]["action"] == "CLICK ref=36"


def test_missing_metric_file_fails_with_source_path():
    with pytest.raises(ArtifactValidationError, match="missing.json"):
        load_stage_metrics(
            base_report_path=FIXTURES / "base_report.md",
            sft_metrics_path=FIXTURES / "missing.json",
            grpo_metrics_path=FIXTURES / "final_grpo_metrics.json",
        )


def test_wrong_episode_count_fails():
    records = load_final_grpo_episodes(FIXTURES / "final_grpo_episodes.jsonl")

    with pytest.raises(ArtifactValidationError, match="50"):
        validate_episode_records(
            records[:2],
            source_path=FIXTURES / "final_grpo_episodes.jsonl",
            expected_count=50,
        )


def test_malformed_record_is_reported_in_diagnostic_mode():
    records = load_final_grpo_episodes(FIXTURES / "final_grpo_episodes.jsonl")
    records = list(records)
    records[0] = records[0].__class__(
        episode_id=records[0].episode_id,
        steps=(),
        success=records[0].success,
        total_reward=records[0].total_reward,
        source_line=records[0].source_line,
    )

    report = validate_episode_records(
        records,
        source_path=FIXTURES / "final_grpo_episodes.jsonl",
        expected_count=50,
        diagnostic_mode=True,
    )

    assert report.valid is False
    assert report.invalid_records[0].field_path == "steps"


def test_summary_reconciles_successes_lengths_and_diagnostics():
    records = load_final_grpo_episodes(FIXTURES / "final_grpo_episodes.jsonl")
    summary = summarize_final_episodes(records)

    assert summary["episode_count"] == 50
    assert summary["success_count"] == 44
    assert summary["failure_count"] == 6
    assert sum(summary["episode_lengths"]) == 166
    assert summary["diagnostic_step_count"] == 166


def test_trace_preserves_raw_completion_and_marks_queue_as_derived():
    record = load_final_grpo_episodes(FIXTURES / "final_grpo_episodes.jsonl")[0]
    trace = extract_representative_trace(record)

    assert trace.instruction == "Select A and click Submit."
    assert len(trace.raw_completions) == 3
    assert trace.executed_actions == ("click('21')", "click('24')", "click('33')")
    assert trace.queue_is_derived is True
