from src.train.sft import build_sft_job_config


def test_build_sft_job_config_points_to_dataset_and_output():
    cfg = build_sft_job_config(
        dataset_path="data/sft_dataset.jsonl",
        output_dir="models/sft_model",
    )
    assert cfg["output_dir"] == "models/sft_model"
