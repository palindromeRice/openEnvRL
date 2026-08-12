from pathlib import Path

from src.analysis.artifact_analysis import StageMetric, TraceView
from src.analysis.artifact_analysis import load_final_grpo_episodes, summarize_failure_behavior, summarize_final_episodes
from src.analysis.plots import (
    plot_artifact_lineage,
    plot_capability_radar,
    plot_episode_failure_modes,
    plot_demonstration_evaluation_cards,
    plot_episode_summary,
    plot_episode_reliability_summary,
    plot_evaluator_flow,
    plot_stage_progression,
    plot_tool_use_timeline,
)


def _metrics() -> dict[str, StageMetric]:
    return {
        "base": StageMetric("base", 0.30, 0.30, 4.36, Path("base.md"), "base"),
        "sft": StageMetric("sft", 0.58, 0.58, 1.42, Path("sft.json"), "sft"),
        "grpo": StageMetric("grpo", 0.88, 0.88, 3.32, Path("grpo.json"), "grpo"),
    }


def _trace() -> TraceView:
    return TraceView(
        episode_id="episode-1",
        instruction="Select A and click Submit.",
        initial_observation="[21] checkbox 'A'\n[33] button 'Submit'",
        raw_completions=("raw call 1\nraw call 2", "raw call 2", "raw submit"),
        parsed_actions=("click('21')", "click('24')", "click('33')"),
        executed_actions=("click('21')", "click('24')", "click('33')"),
        rewards=(0.0, 0.0, 1.0),
        done_flags=(False, False, True),
        success=True,
        queue_is_derived=True,
    )


def _assert_output(metadata, output_path: Path):
    assert metadata.figure_path == output_path
    assert output_path.is_file()
    assert output_path.stat().st_size > 0
    assert metadata.caption_path.is_file()
    assert metadata.source_paths


def test_all_renderers_write_nonempty_outputs(tmp_path):
    metrics = _metrics()
    trace = _trace()
    outputs = [
        (plot_stage_progression(metrics, tmp_path / "figure1.png"), tmp_path / "figure1.png"),
        (plot_tool_use_timeline(trace, tmp_path / "figure2.png"), tmp_path / "figure2.png"),
        (
            plot_demonstration_evaluation_cards(
                [{"evidence_type": "SFT training demonstration", "steps": [{"action": "CLICK ref=33"}]}],
                [trace],
                tmp_path / "figure3.png",
            ),
            tmp_path / "figure3.png",
        ),
        (plot_evaluator_flow(tmp_path / "figure4.png"), tmp_path / "figure4.png"),
        (
            plot_episode_summary(
                {"success_count": 44, "failure_count": 6, "episode_lengths": [3, 3, 4]},
                tmp_path / "figure5.png",
            ),
            tmp_path / "figure5.png",
        ),
        (
            plot_artifact_lineage([Path("trajectories.jsonl"), Path("checkpoint-20")], tmp_path / "figure6.png"),
            tmp_path / "figure6.png",
        ),
        (plot_capability_radar(metrics, tmp_path / "figure7.png"), tmp_path / "figure7.png"),
    ]

    for metadata, output_path in outputs:
        _assert_output(metadata, output_path)


def test_radar_caption_discloses_binary_reward_and_step_caveat(tmp_path):
    metadata = plot_capability_radar(_metrics(), tmp_path / "radar.png")
    caption = metadata.caption_path.read_text(encoding="utf-8")

    assert "binary" in caption.lower()
    assert "early" in caption.lower()


def test_reliability_audit_has_failure_taxonomy_and_steps(tmp_path):
    from src.analysis.plots import plot_episode_reliability_audit

    records = load_final_grpo_episodes(Path("tests/fixtures/analysis/final_grpo_episodes.jsonl"))
    summary = summarize_final_episodes(records)
    summary["failure_behavior"] = {"counts": {
        "no_submit_after_correct_selection": 2,
        "premature_submit": 2,
        "wrong_selection_then_submit": 2,
    }}
    metadata = plot_episode_reliability_audit(summary, tmp_path / "figure5.png")

    _assert_output(metadata, tmp_path / "figure5.png")
    caption = metadata.caption_path.read_text(encoding="utf-8")
    assert "44/50" in caption
    assert "no Submit" in caption


def test_reliability_figures_split_outcome_and_failure_modes(tmp_path):
    from src.analysis.plots import plot_episode_failure_modes, plot_episode_reliability_summary

    summary = {
        "episode_count": 50,
        "success_count": 44,
        "failure_count": 6,
        "episode_successes": [True] * 44 + [False] * 6,
        "failure_behavior": {"counts": {
            "no_submit_after_correct_selection": 2,
            "premature_submit": 2,
            "wrong_selection_then_submit": 2,
        }},
    }
    outcome = plot_episode_reliability_summary(summary, tmp_path / "figure5.png")
    failure = plot_episode_failure_modes(summary, tmp_path / "figure6.png")

    _assert_output(outcome, tmp_path / "figure5.png")
    _assert_output(failure, tmp_path / "figure6.png")
    assert "44/50" in outcome.caption_path.read_text(encoding="utf-8")
    failure_caption = failure.caption_path.read_text(encoding="utf-8")
    assert failure_caption.count("2") == 3
