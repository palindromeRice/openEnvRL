from src.pipeline.run_all import list_pipeline_phases


def test_list_pipeline_phases_matches_spec_order():
    phases = list_pipeline_phases()
    assert phases == [
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
