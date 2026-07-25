import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.cli.run_phase import apply_profile, main


def test_main_runs_phase_with_smoke_profile_and_artifact_root(tmp_path, monkeypatch, capsys):
    artifact_root = tmp_path / "artifacts"
    bootstrap_calls = []
    runtime_dir_calls = []
    phase_calls = []

    def fake_load_config(path):
        return {
            "task_id": "click-checkboxes",
            "env_backend": "native_browsergym",
            "miniwob_root": "/content/browsergym-assets",
            "results_dir": "results",
            "data_dir": "data",
            "models_dir": "models",
            "default_eval_episodes": 50,
            "target_successes": 50,
            "max_steps_per_episode": 10,
            "manual_smoke_episodes": 3,
        }

    def fake_ensure_runtime_dirs(config):
        runtime_dir_calls.append(
            (config["results_dir"], config["data_dir"], config["models_dir"])
        )

    def fake_bootstrap(root_dir):
        bootstrap_calls.append(root_dir)
        return f"file://{root_dir}/miniwob/"

    def fake_run_named_phase(name, context):
        phase_calls.append((name, context["config"].copy()))
        return {"phase": name, "result": {"success_rate": 1.0}}

    monkeypatch.setattr("src.cli.run_phase.load_config", fake_load_config)
    monkeypatch.setattr(
        "src.cli.run_phase.prepare_drive_artifact_root",
        lambda root: {
            "results_dir": root / "results",
            "data_dir": root / "data",
            "models_dir": root / "models",
            "sft_dataset_path": root / "data" / "sft_dataset.jsonl",
                "sft_output_dir": root / "models" / "sft_model",
                "sft_adapter_dir": root / "models" / "sft_model" / "adapter",
                "grpo_output_dir": root / "models" / "grpo_model",
        },
    )
    monkeypatch.setattr("src.cli.run_phase.ensure_runtime_dirs", fake_ensure_runtime_dirs)
    monkeypatch.setattr("src.cli.run_phase.bootstrap_miniwob_assets", fake_bootstrap)
    monkeypatch.setattr("src.cli.run_phase.run_named_phase", fake_run_named_phase)

    exit_code = main(
        [
            "--phase",
            "manual_smoke",
            "--profile",
            "smoke",
            "--artifact-root",
            str(artifact_root),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["phase"] == "manual_smoke"
    assert runtime_dir_calls == [
        (
            str(artifact_root / "results"),
            str(artifact_root / "data"),
            str(artifact_root / "models"),
        )
    ]
    assert bootstrap_calls == [str(artifact_root / "browsergym-assets")]
    assert phase_calls[0][0] == "manual_smoke"
    config = phase_calls[0][1]
    assert config["default_eval_episodes"] == 5
    assert config["target_successes"] == 5
    assert config["max_steps_per_episode"] == 6
    assert config["manual_smoke_episodes"] == 2
    assert config["results_dir"] == str(artifact_root / "results")
    assert config["miniwob_url"] == f"file://{artifact_root / 'browsergym-assets'}/miniwob/"


def test_main_rejects_non_drive_artifact_root_before_bootstrap_or_phase_dispatch(
    tmp_path, monkeypatch
):
    calls = []

    monkeypatch.setattr("src.cli.run_phase.load_config", lambda path: {"env_backend": "native_browsergym"})
    monkeypatch.setattr(
        "src.cli.run_phase.bootstrap_miniwob_assets",
        lambda root_dir: calls.append("bootstrap"),
    )
    monkeypatch.setattr(
        "src.cli.run_phase.run_named_phase",
        lambda name, context: calls.append("phase"),
    )

    with pytest.raises(ValueError, match="Drive artifact root"):
        main(
            [
                "--phase",
                "manual_smoke",
                "--artifact-root",
                str(tmp_path / "not-drive"),
            ]
        )

    assert calls == []


def test_main_skips_bootstrap_for_openenv_backend(monkeypatch, capsys):
    phase_calls = []

    def fake_load_config(path):
        return {
            "task_id": "click-checkboxes",
            "env_backend": "openenv",
            "results_dir": "results",
            "data_dir": "data",
            "models_dir": "models",
            "default_eval_episodes": 50,
            "target_successes": 50,
            "max_steps_per_episode": 10,
        }

    def fake_run_named_phase(name, context):
        phase_calls.append((name, context["config"].copy()))
        return {"phase": name, "result": "ok"}

    monkeypatch.setattr("src.cli.run_phase.load_config", fake_load_config)
    monkeypatch.setattr("src.cli.run_phase.ensure_runtime_dirs", lambda config: None)
    monkeypatch.setattr(
        "src.cli.run_phase.bootstrap_miniwob_assets",
        lambda root_dir: (_ for _ in ()).throw(AssertionError("bootstrap should not run")),
    )
    monkeypatch.setattr("src.cli.run_phase.run_named_phase", fake_run_named_phase)

    exit_code = main(["--phase", "openenv_probe", "--profile", "full"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["phase"] == "openenv_probe"
    assert phase_calls == [("openenv_probe", fake_load_config(None))]


def test_sft_smoke_profile_uses_two_collection_and_evaluation_episodes():
    profile = apply_profile(
        {"target_successes": 50, "default_eval_episodes": 50, "max_steps_per_episode": 10},
        "sft_smoke",
    )

    assert profile["target_successes"] == 2
    assert profile["default_eval_episodes"] == 2
    assert profile["max_steps_per_episode"] == 6


def test_sft_smoke_profile_reaches_drive_rooted_phase(tmp_path, monkeypatch, capsys):
    artifact_root = tmp_path / "artifacts" / "sft-smoke"
    phase_configs = []
    monkeypatch.setattr(
        "src.cli.run_phase.load_config",
        lambda path: {"env_backend": "openenv", "target_successes": 50, "default_eval_episodes": 50},
    )
    monkeypatch.setattr(
        "src.cli.run_phase.prepare_drive_artifact_root",
        lambda root: {
            "data_dir": root / "data",
            "results_dir": root / "results",
            "models_dir": root / "models",
            "sft_dataset_path": root / "data" / "sft_dataset.jsonl",
                "sft_output_dir": root / "models" / "sft_model",
                "sft_adapter_dir": root / "models" / "sft_adapter",
                "grpo_output_dir": root / "models" / "grpo_model",
        },
    )
    monkeypatch.setattr("src.cli.run_phase.ensure_runtime_dirs", lambda config: None)
    monkeypatch.setattr(
        "src.cli.run_phase.run_named_phase",
        lambda name, context: phase_configs.append(context["config"]) or {"phase": name},
    )

    assert main(["--phase", "collect_successes", "--profile", "sft_smoke", "--artifact-root", str(artifact_root)]) == 0

    assert phase_configs[0]["target_successes"] == 2
    assert phase_configs[0]["default_eval_episodes"] == 2
    assert phase_configs[0]["artifact_root"] == str(artifact_root)
    assert phase_configs[0]["sft_adapter_dir"] == str(artifact_root / "models" / "sft_adapter")
    assert json.loads(capsys.readouterr().out) == {"phase": "collect_successes"}


def test_wrapper_script_runs_from_repo_root():
    result = subprocess.run(
        [sys.executable, "scripts/run_phase.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Run one browser-agent pipeline phase." in result.stdout
