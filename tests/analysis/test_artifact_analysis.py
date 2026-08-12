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
    classify_failure,
    summarize_failure_behavior,
    EpisodeRecord,
    EpisodeStep,
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


def test_failure_summary_explains_all_final_failures():
    records = _classification_fixture()
    summary = summarize_failure_behavior(records)

    assert summary["counts"] == {
        "no_submit_after_correct_selection": 2,
        "premature_submit": 2,
        "wrong_selection_then_submit": 2,
    }
    assert sum(summary["counts"].values()) == 6


def test_failure_classifier_uses_observable_behavior_only():
    records = _classification_fixture()
    failed = [record for record in records if not record.success]

    assert {classify_failure(record).category for record in failed} == {
        "no_submit_after_correct_selection",
        "premature_submit",
        "wrong_selection_then_submit",
    }


def _classification_fixture():
    def record(number, actions):
        observation = "RootWebArea\n[18] checkbox 'X'\n[21] checkbox 'A'\n[24] checkbox 'B'\n[33] button 'Submit'"
        steps = tuple(
            EpisodeStep(
                instruction="Select A, B and click Submit.",
                observation=observation,
                action=action,
                reward=0.0,
                done=index == len(actions) - 1,
                raw_completion=None,
                parsed_action=action,
            )
            for index, action in enumerate(actions)
        )
        return EpisodeRecord(f"fixture-{number}", steps, False, 0.0, number)

    return (
        record(1, ("click('21')", "click('24')")),
        record(2, ("click('21')", "click('24')")),
        record(3, ("click('21')", "click('33')")),
        record(4, ("click('21')", "click('33')")),
        record(5, ("click('18')", "click('33')")),
        record(6, ("click('18')", "click('33')")),
    )
