import inspect
import sys
import types
from pathlib import Path

import pytest

from src.train.grpo import (
    BrowserGymToolEnv,
    build_grpo_job_config,
    build_grpo_trainer,
    run_grpo_training,
)


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

        def _close(self):
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


def test_grpo_job_config_anchors_policy_to_sft():
    """Gate r2 showed 0.40 -> 0.20 regression: no KL anchor, entropy 0.387 -> 0.018."""
    cfg = build_grpo_job_config("models/sft_model", "models/grpo_model")

    assert cfg["beta"] > 0.0
    assert cfg["learning_rate"] <= 1e-6


def test_build_grpo_trainer_omits_unsupported_prompt_length_and_forwards_kl(monkeypatch):
    """Current TRL releases reject max_prompt_length but still need the KL anchor."""
    captured = {}

    class FakeGRPOConfig:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_trl = types.SimpleNamespace(
        GRPOConfig=FakeGRPOConfig,
        GRPOTrainer=lambda **kwargs: types.SimpleNamespace(),
    )
    monkeypatch.setitem(sys.modules, "trl", fake_trl)
    monkeypatch.setattr(
        "src.train.grpo.build_grpo_prompt_dataset", lambda num_rows=32: ["row"]
    )

    build_grpo_trainer(build_grpo_job_config("m", "o"))

    assert "max_prompt_length" not in captured
    assert captured["beta"] > 0.0


def test_browsergym_tool_env_reset_shows_task_instruction(monkeypatch):
    class FakeEnv:
        def __init__(self, config=None):
            pass

        def reset(self):
            return {
                "instruction": "Select Lx9, Qz and click Submit.",
                "observation": "RootWebArea\n[33] checkbox 'Lx9'",
            }

    monkeypatch.setattr("src.train.grpo.NativeMiniWoBClickCheckboxesEnv", FakeEnv)
    prompt = BrowserGymToolEnv().reset()

    assert "Select Lx9, Qz and click Submit." in prompt
    assert "[33] checkbox 'Lx9'" in prompt


def test_browsergym_tool_env_does_not_expose_cleanup_as_a_tool():
    public_methods = {
        name
        for name, method in inspect.getmembers(BrowserGymToolEnv, inspect.isfunction)
        if not name.startswith("_")
    }

    assert "close" not in public_methods
    assert {"click", "noop"}.issubset(public_methods)
