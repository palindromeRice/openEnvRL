from src.train.grpo import build_grpo_job_config


def test_build_grpo_job_config_uses_sft_checkpoint():
    cfg = build_grpo_job_config(
        init_model_path="models/sft_model",
        output_dir="models/grpo_model",
    )
    assert cfg["init_model_path"] == "models/sft_model"
