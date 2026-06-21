from __future__ import annotations

from pathlib import Path
from typing import Any

from src.env.openenv_task import BrowserGymTaskConfig, OpenEnvClickCheckboxesEnv


def build_grpo_job_config(init_model_path: str, output_dir: str) -> dict[str, Any]:
    return {
        "init_model_path": init_model_path,
        "output_dir": output_dir,
        "trainer": "unsloth_grpo",
        "online_env_interaction": True,
        "reward_source": "native_env_reward",
        "max_prompt_length": 512,
        "max_completion_length": 64,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "num_generations": 2,
        "max_steps": 50,
        "learning_rate": 5e-6,
        "logging_steps": 1,
        "save_steps": 10,
        "task_name": "click-checkboxes",
        "benchmark": "miniwob",
        "base_url": "https://openenv-browsergym-env.hf.space",
    }


def ensure_grpo_artifact_dirs(config: dict[str, Any]) -> None:
    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)


class BrowserGymToolEnv:
    """TRL environment_factory wrapper for BrowserGym/OpenEnv."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        merged = build_grpo_job_config("models/sft_model", "models/grpo_model")
        if config:
            merged.update(config)
        self.config = merged
        self.env = OpenEnvClickCheckboxesEnv(
            config=BrowserGymTaskConfig(
                base_url=self.config["base_url"],
                benchmark=self.config["benchmark"],
                task_name=self.config["task_name"],
            )
        )
        self.reward = 0.0
        self.done = False
        self.last_observation = self.env.reset()

    def click(self, target: str) -> str:
        action_text = f"click('{target}')"
        obs, reward, done, info = self.env.step(action_text)
        self.reward = reward
        self.done = done
        self.last_observation = obs | {"info": info}
        return obs["observation"]

    def noop(self) -> str:
        self.reward = 0.0
        return self.last_observation["observation"]

    def close(self) -> None:
        self.env.close()


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

    prompt = (
        "You are a browser agent solving the click-checkboxes task. "
        "Use the available browser tools to complete the task."
    )
    rows = [{"prompt": [{"role": "user", "content": prompt}]} for _ in range(num_rows)]
    return Dataset.from_list(rows)


def build_grpo_trainer(config: dict[str, Any]) -> Any:
    from trl import GRPOConfig, GRPOTrainer

    training_args = GRPOConfig(
        learning_rate=config["learning_rate"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        num_generations=config["num_generations"],
        max_prompt_length=config["max_prompt_length"],
        max_completion_length=config["max_completion_length"],
        max_steps=config["max_steps"],
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        output_dir=config["output_dir"],
        report_to="none",
        use_vllm=False,
    )
    dataset = build_grpo_prompt_dataset()
    return GRPOTrainer(
        model=config["init_model_path"],
        reward_funcs=reward_func,
        train_dataset=dataset,
        args=training_args,
        environment_factory=lambda: BrowserGymToolEnv(config=config),
    )


def run_grpo_training(config: dict[str, Any]) -> Any:
    ensure_grpo_artifact_dirs(config)
    trainer = build_grpo_trainer(config)
    trainer_stats = trainer.train()
    trainer.save_model(config["output_dir"])
    return trainer_stats
