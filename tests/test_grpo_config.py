from pathlib import Path

import pytest

from src.train.grpo import build_grpo_job_config, run_grpo_training


def test_build_grpo_job_config_uses_sft_checkpoint():
    cfg = build_grpo_job_config(
        init_model_path="models/sft_model",
        output_dir="models/grpo_model",
    )
    assert cfg["init_model_path"] == "models/sft_model"


def test_build_grpo_job_config_applies_run_overrides():
    cfg = build_grpo_job_config(
        "/drive/sft_model",
        "/drive/models/grpo_model",
        max_steps=100,
        save_steps=50,
        resume_from_checkpoint="/drive/models/grpo_model/checkpoint-50",
    )

    assert cfg["max_steps"] == 100
    assert cfg["save_steps"] == 50
    assert cfg["resume_from_checkpoint"].endswith("checkpoint-50")


def test_run_grpo_training_passes_resume_and_closes_environments(monkeypatch, tmp_path):
    checkpoint = tmp_path / "checkpoint-50"
    checkpoint.mkdir()
    calls = []

    class Env:
        closed = False

        def close(self):
            self.closed = True

    env = Env()

    class Trainer:
        _openenvrl_environments = [env]

        def train(self, *, resume_from_checkpoint):
            calls.append(("train", resume_from_checkpoint))
            return {"global_step": 100}

        def save_model(self, output_dir):
            calls.append(("save", output_dir))

    monkeypatch.setattr("src.train.grpo.build_grpo_trainer", lambda config: Trainer())
    output_dir = tmp_path / "model"
    result = run_grpo_training(
        {"output_dir": str(output_dir), "resume_from_checkpoint": str(checkpoint)}
    )

    assert result == {"global_step": 100}
    assert calls == [("train", str(checkpoint)), ("save", str(output_dir))]
    assert env.closed


def test_run_grpo_training_rejects_missing_resume_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        run_grpo_training(
            {
                "output_dir": str(tmp_path / "model"),
                "resume_from_checkpoint": str(tmp_path / "missing"),
            }
        )
