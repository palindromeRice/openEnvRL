from __future__ import annotations

from pathlib import Path
import gc
from typing import Any

from src.artifacts import DEFAULT_DRIVE_ARTIFACT_PREFIX, write_run_manifest
from src.agents.manual_click_checkboxes_policy import ManualClickCheckboxesPolicy
from src.analysis.report import write_report
from src.config import ensure_runtime_dirs, load_config
from src.data.build_sft_dataset import write_step_dataset
from src.data.storage import read_jsonl
from src.env.native_browsergym_task import NativeMiniWoBClickCheckboxesEnv, NativeMiniWoBTaskConfig
from src.env.native_miniwob_probe import sample_probe_payload
from src.env.openenv_task import BrowserGymTaskConfig, OpenEnvClickCheckboxesEnv
from src.eval.run_eval import evaluate_policy
from src.pipeline.collect import run_collection_loop, validate_successful_episodes
from src.train.grpo import build_grpo_job_config
from src.train.sft import build_sft_job_config, run_sft_training, validate_sft_exports


def list_pipeline_phases() -> list[str]:
    return [
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


def _build_env_factory(config: dict[str, Any]) -> Any:
    backend = config.get("env_backend", "native_browsergym")
    if backend == "native_browsergym":
        task_config = NativeMiniWoBTaskConfig(
            task_name=config["task_id"],
            headless=True,
            miniwob_root=config.get("miniwob_root"),
            miniwob_url=config.get("miniwob_url"),
            use_worker_thread=bool(config.get("use_worker_thread", False)),
        )
        return lambda: NativeMiniWoBClickCheckboxesEnv(config=task_config)

    task_config = BrowserGymTaskConfig(
        base_url=config["base_url"],
        benchmark=config["benchmark"],
        task_name=config["task_id"],
    )
    return lambda: OpenEnvClickCheckboxesEnv(config=task_config)


def _build_policy(config: dict[str, Any], checkpoint_path: str | None = None) -> Any:
    from src.agents.policy import BrowserAgentPolicy, PolicyConfig

    return BrowserAgentPolicy(
        PolicyConfig(
            model_id=config["model_id"],
            checkpoint_path=checkpoint_path,
        )
    )


def run_native_probe_phase() -> dict[str, Any]:
    return sample_probe_payload()


def run_openenv_probe_phase(config: dict[str, Any]) -> dict[str, Any]:
    env = OpenEnvClickCheckboxesEnv(
        config=BrowserGymTaskConfig(
            base_url=config["base_url"],
            benchmark=config["benchmark"],
            task_name=config["task_id"],
        )
    )
    try:
        return env.reset()
    finally:
        env.close()


def run_baseline_eval_phase(config: dict[str, Any], output_name: str = "baseline_metrics.json") -> dict[str, Any]:
    env_factory = _build_env_factory(config)
    policy = _build_policy(config)
    return evaluate_policy(
        env_factory=env_factory,
        policy=policy,
        num_episodes=config["default_eval_episodes"],
        max_steps=config["max_steps_per_episode"],
        max_retries_per_episode=config.get("env_retry_attempts", 2),
        output_path=f"{config['results_dir']}/{output_name}",
    )


def run_manual_smoke_phase(
    config: dict[str, Any],
    output_name: str = "manual_smoke_metrics.json",
) -> dict[str, Any]:
    env_factory = _build_env_factory(config)
    policy = ManualClickCheckboxesPolicy()
    return evaluate_policy(
        env_factory=env_factory,
        policy=policy,
        num_episodes=config.get("manual_smoke_episodes", 3),
        max_steps=config["max_steps_per_episode"],
        max_retries_per_episode=config.get("env_retry_attempts", 2),
        output_path=f"{config['results_dir']}/{output_name}",
    )


def run_collection_phase(config: dict[str, Any]) -> list[dict[str, Any]]:
    env_factory = _build_env_factory(config)
    policy = _build_policy(config)
    return run_collection_loop(
        env_factory=env_factory,
        policy=policy,
        target_successes=config["target_successes"],
        max_steps=config["max_steps_per_episode"],
        max_retries_per_episode=config.get("env_retry_attempts", 2),
        output_path=f"{config['data_dir']}/successful_trajectories.jsonl",
    )


def run_sft_dataset_phase(
    config: dict[str, Any], episodes: list[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    if episodes is None:
        episodes = read_jsonl(f"{config['data_dir']}/successful_trajectories.jsonl")
    validate_successful_episodes(episodes, config.get("target_successes", len(episodes)))
    rows = write_step_dataset(episodes, path=config["sft_dataset_path"])
    expected_row_count = sum(len(episode["steps"]) for episode in episodes)
    if len(rows) != expected_row_count:
        raise RuntimeError("SFT dataset row count does not match source trajectory steps")
    return rows


def run_sft_phase(config: dict[str, Any]) -> dict[str, Any]:
    job_config = build_sft_job_config(
        dataset_path=config["sft_dataset_path"],
        output_dir=config["sft_output_dir"],
        adapter_dir=config["sft_adapter_dir"],
    )
    return run_sft_training(job_config)


def run_sft_eval_phase(config: dict[str, Any]) -> dict[str, Any]:
    sft_config = {
        **config,
        "output_dir": config["sft_output_dir"],
        "adapter_dir": config["sft_adapter_dir"],
    }
    validate_sft_exports(sft_config)
    env_factory = _build_env_factory(config)
    policy = _build_policy(config, checkpoint_path=config["sft_output_dir"])
    return evaluate_policy(
        env_factory=env_factory,
        policy=policy,
        num_episodes=config["default_eval_episodes"],
        max_steps=config["max_steps_per_episode"],
        max_retries_per_episode=config.get("env_retry_attempts", 2),
        output_path=f"{config['results_dir']}/sft_metrics.json",
    )


def run_sft_manifest_phase(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config.get("artifact_root", "")).resolve()
    prefix = DEFAULT_DRIVE_ARTIFACT_PREFIX.resolve()
    try:
        root.relative_to(prefix)
    except ValueError as error:
        raise ValueError("SFT manifest requires a Google Drive artifact root") from error
    paths = {
        "data_dir": config["data_dir"],
        "results_dir": config["results_dir"],
        "models_dir": config["models_dir"],
        "sft_dataset_path": config["sft_dataset_path"],
        "sft_output_dir": config["sft_output_dir"],
        "sft_adapter_dir": config["sft_adapter_dir"],
    }
    return write_run_manifest(paths, config)


def run_grpo_phase(config: dict[str, Any]) -> dict[str, Any]:
    from src.train.grpo import run_grpo_training, build_grpo_job_config
    
    job_config = build_grpo_job_config(
        init_model_path=config["sft_output_dir"],
        output_dir=config["grpo_output_dir"],
    )
    return run_grpo_training(job_config)


def run_grpo_eval_phase(config: dict[str, Any], checkpoint_step: int) -> dict[str, Any]:
    from src.agents.grpo_tool_policy import GRPOToolPolicy

    checkpoint_path = Path(config["grpo_output_dir"]) / f"checkpoint-{checkpoint_step}"
    if not checkpoint_path.is_dir():
        raise ValueError(f"Missing GRPO checkpoint: {checkpoint_path}")
    policy = GRPOToolPolicy(str(checkpoint_path))
    try:
        return evaluate_policy(
            env_factory=_build_env_factory(config),
            policy=policy,
            num_episodes=config["default_eval_episodes"],
            max_steps=config["max_steps_per_episode"],
            max_retries_per_episode=config.get("env_retry_attempts", 2),
            output_path=f"{config['results_dir']}/grpo_eval_step_{checkpoint_step}.json",
        )
    finally:
        policy.close()
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def run_analysis_phase(config: dict[str, Any], metrics: dict[str, dict[str, Any]]) -> str:
    return write_report(metrics, path=f"{config['results_dir']}/report.md")


def run_named_phase(name: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
    if name not in list_pipeline_phases():
        raise ValueError(f"Unknown phase: {name}")
    context = dict(context or {})
    config = context.get("config")
    if config is None:
        config_path = context.get("config_path", "configs/default.yaml")
        config = load_config(config_path)
        ensure_runtime_dirs(config)
        context["config"] = config

    if name == "native_probe":
        return {"phase": name, "result": run_native_probe_phase()}
    if name == "openenv_probe":
        return {"phase": name, "result": run_openenv_probe_phase(config)}
    if name == "manual_smoke":
        return {"phase": name, "result": run_manual_smoke_phase(config)}
    if name == "baseline_eval":
        return {"phase": name, "result": run_baseline_eval_phase(config)}
    if name == "collect_successes":
        episodes = run_collection_phase(config)
        context["successful_episodes"] = episodes
        return {"phase": name, "num_successes": len(episodes)}
    if name == "build_sft_dataset":
        episodes = context.get("successful_episodes")
        rows = run_sft_dataset_phase(config, episodes)
        return {"phase": name, "num_rows": len(rows)}
    if name == "sft_train":
        return {"phase": name, "result": run_sft_phase(config)}
    if name == "sft_eval":
        return {"phase": name, "result": run_sft_eval_phase(config)}
    if name == "sft_manifest":
        return {"phase": name, "result": run_sft_manifest_phase(config)}
    if name == "grpo_train":
        return {"phase": name, "result": run_grpo_phase(config)}
    if name == "grpo_eval":
        checkpoint_step = context.get("checkpoint_step")
        if not isinstance(checkpoint_step, int):
            raise ValueError("grpo_eval requires integer checkpoint_step")
        return {"phase": name, "result": run_grpo_eval_phase(config, checkpoint_step)}
    if name == "analysis":
        return {"phase": name, "result": "Requires collected metrics JSON files"}
    return {"phase": name, "status": "unhandled"}


def run_full_pipeline(context: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    shared_context = dict(context or {})
    outputs = []
    for name in list_pipeline_phases():
        outputs.append(run_named_phase(name, context=shared_context))
    return outputs
