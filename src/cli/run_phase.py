from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.artifacts import prepare_drive_artifact_root


def list_pipeline_phases() -> list[str]:
    from src.pipeline.run_all import list_pipeline_phases as _list_pipeline_phases

    return _list_pipeline_phases()


def load_config(path: str) -> dict[str, Any]:
    from src.config import load_config as _load_config

    return _load_config(path)


def ensure_runtime_dirs(config: dict[str, Any]) -> None:
    from src.config import ensure_runtime_dirs as _ensure_runtime_dirs

    _ensure_runtime_dirs(config)


def bootstrap_miniwob_assets(root_dir: str) -> str:
    from src.env.native_browsergym_task import bootstrap_miniwob_assets as _bootstrap_miniwob_assets

    return _bootstrap_miniwob_assets(root_dir)


def run_named_phase(name: str, context: dict[str, Any]) -> dict[str, Any]:
    from src.pipeline.run_all import run_named_phase as _run_named_phase

    return _run_named_phase(name, context)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one browser-agent pipeline phase.")
    parser.add_argument("--phase", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--profile", default="smoke", choices=["smoke", "sft_smoke", "full"])
    parser.add_argument("--artifact-root")
    return parser


def apply_profile(config: dict[str, Any], profile: str) -> dict[str, Any]:
    config = dict(config)
    if profile == "smoke":
        config["default_eval_episodes"] = 5
        config["target_successes"] = 5
        config["max_steps_per_episode"] = 6
        config["manual_smoke_episodes"] = 2
    elif profile == "sft_smoke":
        config["default_eval_episodes"] = 2
        config["target_successes"] = 2
        config["max_steps_per_episode"] = 6
    elif profile != "full":
        raise ValueError(f"Unknown profile: {profile}")
    return config


def apply_artifact_root(config: dict[str, Any], artifact_root: str | None) -> dict[str, Any]:
    config = dict(config)
    if artifact_root is None:
        return config

    root = Path(artifact_root)
    drive_paths = prepare_drive_artifact_root(root)
    config.update({key: str(path) for key, path in drive_paths.items()})
    config["artifact_root"] = str(root)
    config["miniwob_root"] = str(root / "browsergym-assets")
    config["grpo_output_dir"] = str(drive_paths["grpo_output_dir"])
    return config


def prepare_runtime(config: dict[str, Any]) -> dict[str, Any]:
    config = dict(config)
    ensure_runtime_dirs(config)
    if config.get("env_backend", "native_browsergym") == "native_browsergym":
        miniwob_root = config.get("miniwob_root")
        if miniwob_root:
            config["miniwob_url"] = bootstrap_miniwob_assets(str(miniwob_root))
    return config


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.phase not in list_pipeline_phases():
        raise SystemExit(f"Unknown phase: {args.phase}")
    config = load_config(args.config)
    config = apply_profile(config, args.profile)
    config = apply_artifact_root(config, args.artifact_root)
    config = prepare_runtime(config)
    result = run_named_phase(args.phase, context={"config": config})
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
