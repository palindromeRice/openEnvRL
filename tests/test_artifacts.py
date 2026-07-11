import json
from pathlib import Path
from uuid import uuid4

import pytest

from src.artifacts import prepare_drive_artifact_root, write_run_manifest
from src.data.storage import write_jsonl_atomic


def test_prepare_drive_artifact_root_creates_drive_run_paths_and_removes_sentinel(tmp_path):
    drive_prefix = tmp_path / "drive" / "MyDrive" / "openEnvRL" / "artifacts"
    root = drive_prefix / "sft-rerun-20260711"

    paths = prepare_drive_artifact_root(root, drive_prefix=drive_prefix)

    assert paths == {
        "data_dir": root / "data",
        "results_dir": root / "results",
        "models_dir": root / "models",
        "sft_dataset_path": root / "data" / "sft_dataset.jsonl",
        "sft_output_dir": root / "models" / "sft_model",
        "sft_adapter_dir": root / "models" / "sft_adapter",
    }
    assert all(path.parent.exists() for path in paths.values())
    assert not (root / ".drive-write-test").exists()


@pytest.mark.parametrize("root", ["outside-run", "artifacts"])
def test_prepare_drive_artifact_root_rejects_paths_outside_drive_prefix(tmp_path, root):
    drive_prefix = tmp_path / "drive" / "MyDrive" / "openEnvRL" / "artifacts"
    outside_root = tmp_path / root

    with pytest.raises(ValueError, match="Drive artifact root"):
        prepare_drive_artifact_root(outside_root, drive_prefix=drive_prefix)


def test_prepare_drive_artifact_root_requires_nonempty_run_id(tmp_path):
    drive_prefix = tmp_path / "drive" / "MyDrive" / "openEnvRL" / "artifacts"

    with pytest.raises(ValueError, match="run ID"):
        prepare_drive_artifact_root(drive_prefix, drive_prefix=drive_prefix)


def test_write_run_manifest_requires_complete_consistent_sft_artifacts(tmp_path):
    drive_prefix = tmp_path / "drive" / "MyDrive" / "openEnvRL" / "artifacts"
    root = drive_prefix / "sft-smoke"
    paths = prepare_drive_artifact_root(root, drive_prefix=drive_prefix)
    successes = [
        {
            "episode_id": str(uuid4()),
            "success": True,
            "total_reward": 1.0,
            "steps": [{"action": "CLICK ref=1"}, {"action": "CLICK ref=2"}],
        },
        {
            "episode_id": str(uuid4()),
            "success": True,
            "total_reward": 1.0,
            "steps": [{"action": "CLICK ref=3"}],
        },
    ]
    write_jsonl_atomic(paths["data_dir"] / "successful_trajectories.jsonl", successes)
    write_jsonl_atomic(paths["sft_dataset_path"], [{"row": index} for index in range(3)])
    (paths["sft_output_dir"] / "config.json").write_text("{}", encoding="utf-8")
    (paths["sft_output_dir"] / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (paths["sft_output_dir"] / "model.safetensors").write_bytes(b"merged")
    (paths["sft_adapter_dir"] / "adapter_config.json").write_text("{}", encoding="utf-8")
    (paths["sft_adapter_dir"] / "adapter_model.safetensors").write_bytes(b"adapter")
    metrics_path = paths["results_dir"] / "sft_metrics.json"
    metrics_path.write_text(
        json.dumps({"success_rate": 0.5, "average_reward": 0.5, "average_steps": 2.0}),
        encoding="utf-8",
    )

    manifest = write_run_manifest(
        paths,
        config={"target_successes": 2, "model_id": "Qwen/Qwen3.5-0.8B"},
    )

    assert manifest["run_id"] == "sft-smoke"
    assert manifest["artifact_paths"]["sft_model"] == str(paths["sft_output_dir"])
    assert manifest["collection"]["count"] == 2
    assert manifest["dataset"]["count"] == 3
    assert manifest["metrics"]["success_rate"] == 0.5
    assert manifest["model_directories"]["sft_model"]["total_bytes"] > 0
    assert manifest["model_directories"]["sft_adapter"]["total_bytes"] > 0
    assert len(manifest["json_artifacts"]["sft_metrics.json"]["sha256"]) == 64
    assert json.loads((root / "run_manifest.json").read_text(encoding="utf-8")) == manifest


@pytest.mark.parametrize("target_successes", [2, 50])
def test_write_run_manifest_requires_exact_configured_success_count(tmp_path, target_successes):
    paths = prepare_drive_artifact_root(
        tmp_path / "drive" / "MyDrive" / "openEnvRL" / "artifacts" / "run",
        drive_prefix=tmp_path / "drive" / "MyDrive" / "openEnvRL" / "artifacts",
    )
    write_jsonl_atomic(paths["data_dir"] / "successful_trajectories.jsonl", [])
    write_jsonl_atomic(paths["sft_dataset_path"], [{"row": 1}])
    (paths["sft_output_dir"] / "config.json").write_text("{}", encoding="utf-8")
    (paths["sft_output_dir"] / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (paths["sft_output_dir"] / "model.safetensors").write_bytes(b"merged")
    (paths["sft_adapter_dir"] / "adapter_config.json").write_text("{}", encoding="utf-8")
    (paths["sft_adapter_dir"] / "adapter_model.safetensors").write_bytes(b"adapter")
    (paths["results_dir"] / "sft_metrics.json").write_text(
        json.dumps({"success_rate": 0.0, "average_reward": 0.0, "average_steps": 0.0}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exactly"):
        write_run_manifest(paths, config={"target_successes": target_successes})
