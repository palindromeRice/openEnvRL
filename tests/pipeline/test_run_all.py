import json
from pathlib import Path
from uuid import uuid4

import pytest

from src.data.storage import read_jsonl, write_jsonl
from src.pipeline.run_all import list_pipeline_phases, run_named_phase


def test_list_pipeline_phases_matches_spec_order():
    phases = list_pipeline_phases()
    assert phases == [
        "native_probe",
        "openenv_probe",
        "manual_smoke",
        "baseline_eval",
        "collect_successes",
        "build_sft_dataset",
        "sft_train",
        "sft_eval",
        "sft_manifest",
        "grpo_train",
        "grpo_eval",
        "analysis",
    ]


def test_build_sft_dataset_phase_loads_and_validates_persisted_successes(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    source_path = data_dir / "successful_trajectories.jsonl"
    output_path = data_dir / "sft_dataset.jsonl"
    episodes = [
        {
            "episode_id": str(uuid4()),
            "success": True,
            "total_reward": 1.0,
            "steps": [
                {"instruction": "Pick", "observation": "One", "action": "CLICK ref=1"},
                {"instruction": "Pick", "observation": "Two", "action": "CLICK ref=2"},
            ],
        },
        {
            "episode_id": str(uuid4()),
            "success": True,
            "total_reward": 1.0,
            "steps": [
                {"instruction": "Pick", "observation": "Three", "action": "CLICK ref=3"},
            ],
        },
    ]
    write_jsonl(source_path, episodes)
    validated = []
    monkeypatch.setattr(
        "src.pipeline.run_all.validate_successful_episodes",
        lambda rows, target: validated.append((rows, target)),
    )

    result = run_named_phase(
        "build_sft_dataset",
        context={
            "config": {
                "data_dir": str(data_dir),
                "sft_dataset_path": str(output_path),
                "target_successes": 2,
            }
        },
    )

    assert result == {"phase": "build_sft_dataset", "num_rows": 3}
    assert validated == [(episodes, 2)]
    assert len(read_jsonl(output_path)) == sum(len(episode["steps"]) for episode in episodes)


def test_sft_train_phase_builds_job_config_and_runs_training(monkeypatch):
    job_config = {"dataset_path": "data/sft_dataset.jsonl", "output_dir": "models/sft_model"}
    builder_calls = []
    trained_configs = []
    monkeypatch.setattr(
        "src.pipeline.run_all.build_sft_job_config",
        lambda dataset_path, output_dir, adapter_dir: builder_calls.append(
            (dataset_path, output_dir, adapter_dir)
        )
        or job_config,
    )
    monkeypatch.setattr(
        "src.pipeline.run_all.run_sft_training",
        lambda config: trained_configs.append(config) or {"global_step": 60},
    )

    result = run_named_phase(
        "sft_train",
        context={
            "config": {
                "sft_dataset_path": "data/sft_dataset.jsonl",
                "sft_output_dir": "models/sft_model",
                "sft_adapter_dir": "models/sft_model/adapter",
            }
        },
    )

    assert builder_calls == [
        ("data/sft_dataset.jsonl", "models/sft_model", "models/sft_model/adapter")
    ]
    assert trained_configs == [job_config]
    assert result == {"phase": "sft_train", "result": {"global_step": 60}}


def test_sft_eval_phase_validates_exports_and_writes_drive_metrics(tmp_path, monkeypatch):
    calls = []
    metrics_path = tmp_path / "results" / "sft_metrics.json"
    config = {
        "model_id": "Qwen/Qwen3.5-0.8B",
        "sft_output_dir": str(tmp_path / "models" / "sft_model"),
        "sft_adapter_dir": str(tmp_path / "models" / "sft_adapter"),
        "results_dir": str(metrics_path.parent),
        "default_eval_episodes": 2,
        "max_steps_per_episode": 4,
        "env_retry_attempts": 1,
    }
    monkeypatch.setattr(
        "src.pipeline.run_all.validate_sft_exports",
        lambda job_config: calls.append(("validate", job_config)) or {},
    )
    monkeypatch.setattr(
        "src.pipeline.run_all._build_env_factory",
        lambda job_config: calls.append(("env", job_config)) or "env-factory",
    )
    monkeypatch.setattr(
        "src.pipeline.run_all._build_policy",
        lambda job_config, checkpoint_path: calls.append(("policy", job_config, checkpoint_path))
        or "policy",
    )

    def fake_evaluate_policy(**kwargs):
        calls.append(("evaluate", kwargs))
        output_path = Path(kwargs["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({"success_rate": 0.5}), encoding="utf-8")
        return {"success_rate": 0.5}

    monkeypatch.setattr("src.pipeline.run_all.evaluate_policy", fake_evaluate_policy)

    result = run_named_phase("sft_eval", context={"config": config})

    assert result == {"phase": "sft_eval", "result": {"success_rate": 0.5}}
    assert calls[:3] == [
        ("validate", {**config, "output_dir": config["sft_output_dir"], "adapter_dir": config["sft_adapter_dir"]}),
        ("env", config),
        ("policy", config, config["sft_output_dir"]),
    ]
    assert calls[3] == (
        "evaluate",
        {
            "env_factory": "env-factory",
            "policy": "policy",
            "num_episodes": 2,
            "max_steps": 4,
            "max_retries_per_episode": 1,
            "output_path": str(metrics_path),
        },
    )
    assert json.loads(metrics_path.read_text(encoding="utf-8")) == {"success_rate": 0.5}


def test_sft_manifest_phase_requires_drive_artifact_root(monkeypatch):
    monkeypatch.setattr(
        "src.pipeline.run_all.write_run_manifest",
        lambda paths, config: (_ for _ in ()).throw(AssertionError("must not write")),
    )

    with pytest.raises(ValueError, match="Drive artifact root"):
        run_named_phase(
            "sft_manifest",
            context={"config": {"data_dir": "data", "artifact_root": "not-drive"}},
        )
