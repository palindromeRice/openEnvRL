from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping

from src.data.storage import read_jsonl
from src.pipeline.collect import validate_successful_episodes


DEFAULT_DRIVE_ARTIFACT_PREFIX = Path("/content/drive/MyDrive/openEnvRL/artifacts")


def prepare_drive_artifact_root(
    root: Path, drive_prefix: Path = DEFAULT_DRIVE_ARTIFACT_PREFIX
) -> dict[str, Path]:
    resolved_root = Path(root).resolve()
    resolved_prefix = Path(drive_prefix).resolve()
    try:
        run_id = resolved_root.relative_to(resolved_prefix)
    except ValueError as error:
        raise ValueError(
            f"Drive artifact root must be under {resolved_prefix}: {resolved_root}"
        ) from error
    if not run_id.parts:
        raise ValueError("Drive artifact root requires a nonempty run ID")

    resolved_root.mkdir(parents=True, exist_ok=True)
    sentinel = resolved_root / ".drive-write-test"
    try:
        sentinel.write_text("ok", encoding="utf-8")
    finally:
        sentinel.unlink(missing_ok=True)

    data_dir = resolved_root / "data"
    results_dir = resolved_root / "results"
    models_dir = resolved_root / "models"
    sft_output_dir = models_dir / "sft_model"
    sft_adapter_dir = models_dir / "sft_adapter"
    for directory in (data_dir, results_dir, models_dir, sft_output_dir, sft_adapter_dir):
        directory.mkdir(parents=True, exist_ok=True)

    return {
        "data_dir": data_dir,
        "results_dir": results_dir,
        "models_dir": models_dir,
        "sft_dataset_path": data_dir / "sft_dataset.jsonl",
        "sft_output_dir": sft_output_dir,
        "sft_adapter_dir": sft_adapter_dir,
    }


def _sha256_and_size(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def _model_directory_details(directory: Path) -> dict[str, Any]:
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    return {
        "files": [
            {"path": str(path.relative_to(directory)), "bytes": path.stat().st_size}
            for path in files
        ],
        "total_bytes": sum(path.stat().st_size for path in files),
    }


def _require_file(directory: Path, candidates: tuple[str, ...], label: str) -> None:
    if not any((directory / candidate).is_file() for candidate in candidates):
        raise ValueError(f"Missing {label} in {directory}")


def _validate_model_exports(sft_model_dir: Path, sft_adapter_dir: Path) -> None:
    _require_file(sft_model_dir, ("config.json",), "merged model config")
    _require_file(sft_model_dir, ("tokenizer_config.json",), "merged tokenizer config")
    _require_file(
        sft_model_dir,
        ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json", "pytorch_model.bin.index.json"),
        "merged model weights",
    )
    _require_file(sft_adapter_dir, ("adapter_config.json",), "adapter config")
    _require_file(
        sft_adapter_dir,
        ("adapter_model.safetensors", "adapter_model.bin"),
        "adapter weights",
    )


def _sanitize_config(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): ("[redacted]" if any(token in str(key).lower() for token in ("token", "secret", "password", "api_key")) else _sanitize_config(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_config(item) for item in value]
    return value


def _git_metadata() -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    revision = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    dirty = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain"],
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "revision": revision.stdout.strip() if revision.returncode == 0 else "unavailable",
        "dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
    }


def _package_versions() -> dict[str, str]:
    versions = {"python": os.sys.version.split()[0]}
    for package in ("transformers", "peft", "trl", "unsloth"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    return versions


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary_path = Path(handle.name)
        try:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)


def write_run_manifest(paths: Mapping[str, str | Path], config: Mapping[str, Any]) -> dict[str, Any]:
    normalized = {key: Path(value) for key, value in paths.items()}
    data_dir = normalized["data_dir"]
    root = data_dir.parent
    successful_trajectories = data_dir / "successful_trajectories.jsonl"
    dataset_path = normalized["sft_dataset_path"]
    metrics_path = normalized["results_dir"] / "sft_metrics.json"
    sft_model_dir = normalized["sft_output_dir"]
    sft_adapter_dir = normalized["sft_adapter_dir"]
    required_paths = (successful_trajectories, dataset_path, metrics_path, sft_model_dir, sft_adapter_dir)
    if any(not path.exists() for path in required_paths):
        raise ValueError("Missing required SFT artifact")

    target_successes = config.get("target_successes")
    if not isinstance(target_successes, int) or target_successes <= 0:
        raise ValueError("target_successes must be a positive integer")
    episodes = read_jsonl(successful_trajectories)
    validate_successful_episodes(episodes, target_successes)
    if len(episodes) != target_successes:
        raise ValueError(f"Expected exactly {target_successes} successful trajectories, found {len(episodes)}")
    dataset_rows = read_jsonl(dataset_path)
    source_steps = sum(len(episode["steps"]) for episode in episodes)
    if not dataset_rows or len(dataset_rows) != source_steps:
        raise ValueError("SFT dataset row count must equal source trajectory steps")

    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("Invalid SFT metrics JSON") from error
    required_metrics = ("success_rate", "average_reward", "average_steps")
    if not isinstance(metrics, dict) or any(
        not isinstance(metrics.get(key), (int, float)) or isinstance(metrics.get(key), bool)
        for key in required_metrics
    ):
        raise ValueError("SFT metrics require numeric success_rate, average_reward, and average_steps")
    _validate_model_exports(sft_model_dir, sft_adapter_dir)

    artifact_paths = {
        "successful_trajectories": str(successful_trajectories),
        "sft_dataset": str(dataset_path),
        "sft_model": str(sft_model_dir),
        "sft_adapter": str(sft_adapter_dir),
        "sft_metrics": str(metrics_path),
    }
    manifest = {
        "run_id": root.name,
        "artifact_paths": artifact_paths,
        "collection": {"count": len(episodes), "target_successes": target_successes},
        "dataset": {"count": len(dataset_rows), "source_steps": source_steps},
        "metrics": metrics,
        "json_artifacts": {
            "successful_trajectories.jsonl": _sha256_and_size(successful_trajectories),
            "sft_dataset.jsonl": _sha256_and_size(dataset_path),
            "sft_metrics.json": _sha256_and_size(metrics_path),
        },
        "model_directories": {
            "sft_model": _model_directory_details(sft_model_dir),
            "sft_adapter": _model_directory_details(sft_adapter_dir),
        },
        "git": _git_metadata(),
        "package_versions": _package_versions(),
        "config": _sanitize_config(dict(config)),
        "model_id": config.get("model_id"),
        "sft_config": {
            key: config[key]
            for key in ("sft_output_dir", "sft_adapter_dir", "max_steps_per_episode")
            if key in config
        },
    }
    _write_json_atomic(root / "run_manifest.json", manifest)
    return manifest
