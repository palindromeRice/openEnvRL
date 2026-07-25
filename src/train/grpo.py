from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

from src.env.native_browsergym_task import NativeMiniWoBTaskConfig, NativeMiniWoBClickCheckboxesEnv


def build_grpo_job_config(
    init_model_path: str,
    output_dir: str,
    *,
    max_steps: int = 50,
    save_steps: int = 10,
    resume_from_checkpoint: str | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    config = {
        "init_model_path": init_model_path,
        "output_dir": output_dir,
        "trainer": "unsloth_grpo",
        "online_env_interaction": True,
        "reward_source": "native_env_reward",
        "max_prompt_length": 512,
        "max_completion_length": 512,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "num_generations": 2,
        "max_steps": max_steps,
        "learning_rate": 5e-6,
        "logging_steps": 1,
        "save_steps": save_steps,
        "task_name": "click-checkboxes",
        "benchmark": "miniwob",
        "miniwob_root": "/content/browsergym-assets",
    }
    if resume_from_checkpoint is not None:
        config["resume_from_checkpoint"] = resume_from_checkpoint
    config.update(overrides)
    return config


def ensure_grpo_artifact_dirs(config: dict[str, Any]) -> None:
    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)


class BrowserGymToolEnv:
    """TRL environment_factory wrapper for native BrowserGym.

    TRL interface:
      - reset(**kwargs) -> str | None  — called at start of each rollout
      - get_reward() -> float          — called after rollout to score
      - other public methods (click, noop) exposed as tools to the model
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        merged = build_grpo_job_config("models/sft_model", "models/grpo_model")
        if config:
            merged.update(config)
        self.config = merged
        self.env = NativeMiniWoBClickCheckboxesEnv(
            config=NativeMiniWoBTaskConfig(
                task_name=self.config["task_name"],
                miniwob_root=self.config.get("miniwob_root"),
            )
        )
        self.reward = 0.0
        self.done = False
        self.last_observation: dict[str, Any] = {}

    def reset(self, **kwargs: Any) -> str:
        """Reset the environment for a new episode. Called by TRL at rollout start."""
        self.reward = 0.0
        self.done = False
        self.last_observation = self.env.reset()
        return self.last_observation.get("observation", "Environment ready.")

    def get_reward(self) -> float:
        """Return the cumulative reward for this episode. Called by TRL after rollout."""
        return self.reward

    def click(self, target: str) -> str:
        """Click on an element in the browser.

        Args:
            target: The browsergym ID of the element to click on, e.g. '12' or 'a3'.
        """
        action_text = f"click('{target}')"
        obs, reward, done, info = self.env.step(action_text)
        self.reward = reward
        self.done = done
        self.last_observation = obs | {"info": info}
        return obs["observation"]

    def noop(self) -> str:
        """Do nothing and return the current browser observation."""
        self.reward = 0.0
        return self.last_observation.get("observation", "")

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None


def reward_func(completions: list[Any], **kwargs: Any) -> list[float]:
    environments = kwargs.get("environments", [])
    rewards: list[float] = []
    for index, _completion in enumerate(completions):
        if index < len(environments):
            rewards.append(float(getattr(environments[index], "reward", 0.0)))
        else:
            rewards.append(0.0)
    return rewards


def build_grpo_prompt_dataset(num_rows: int = 32) -> Any:
    from datasets import Dataset

    system_msg = {"role": "system", "content": "You are a helpful assistant with access to tools."}
    user_msg = {
        "role": "user", 
        "content": "You are a browser agent solving the click-checkboxes task. Use the available browser tools to complete the task."
    }
    rows = [{"prompt": [system_msg, user_msg]} for _ in range(num_rows)]
    return Dataset.from_list(rows)


def build_grpo_trainer(config: dict[str, Any]) -> Any:
    from trl import GRPOConfig, GRPOTrainer

    training_args = GRPOConfig(
        learning_rate=config["learning_rate"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        num_generations=config["num_generations"],
        generation_batch_size=config.get("generation_batch_size", config["num_generations"]),
        max_completion_length=config["max_completion_length"],
        max_steps=config["max_steps"],
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        output_dir=config["output_dir"],
        report_to="none",
        use_vllm=False,
    )
    dataset = build_grpo_prompt_dataset()
    environments: list[BrowserGymToolEnv] = []

    def environment_factory() -> BrowserGymToolEnv:
        environment = BrowserGymToolEnv(config=config)
        environments.append(environment)
        return environment

    trainer = GRPOTrainer(
        model=config["init_model_path"],
        reward_funcs=reward_func,
        train_dataset=dataset,
        args=training_args,
        environment_factory=environment_factory,
    )
    trainer._openenvrl_environments = environments
    return trainer


def run_grpo_training(config: dict[str, Any]) -> Any:
    ensure_grpo_artifact_dirs(config)
    resume = config.get("resume_from_checkpoint")
    if resume is not None and not Path(resume).is_dir():
        raise ValueError(f"GRPO resume checkpoint does not exist: {resume}")
    trainer = build_grpo_trainer(config)
    try:
        trainer_stats = trainer.train(resume_from_checkpoint=resume)
        trainer.save_model(config["output_dir"])
        return trainer_stats
    finally:
        for environment in getattr(trainer, "_openenvrl_environments", []):
            environment.close()
        del trainer
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
