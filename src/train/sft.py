from __future__ import annotations

from pathlib import Path
from typing import Any

from src.agents.prompting import build_action_prompt
from src.data.storage import read_jsonl


def build_sft_job_config(
    dataset_path: str, output_dir: str, adapter_dir: str | None = None
) -> dict[str, Any]:
    return {
        "model_id": "Qwen/Qwen3.5-0.8B",
        "dataset_path": dataset_path,
        "output_dir": output_dir,
        "adapter_dir": adapter_dir or str(Path(output_dir) / "adapter"),
        "trainer": "unsloth_sft",
        "lora": True,
        "max_seq_length": 1024,
        "load_in_4bit": False,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "warmup_steps": 5,
        "max_steps": 60,
        "learning_rate": 2e-4,
        "logging_steps": 1,
        "optim": "adamw_8bit",
        "seed": 3407,
    }


def ensure_sft_artifact_dirs(config: dict[str, Any]) -> None:
    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["adapter_dir"]).mkdir(parents=True, exist_ok=True)


def format_sft_text(row: dict[str, Any], tokenizer: Any) -> str:
    prompt = build_action_prompt(instruction=row["instruction"], observation=row["observation"])
    chat_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return chat_prompt + row["target_action"] + tokenizer.eos_token


def load_sft_dataset(dataset_path: str, tokenizer: Any) -> Any:
    rows = read_jsonl(dataset_path)
    formatted_rows = [{"text": format_sft_text(row, tokenizer)} for row in rows]

    from datasets import Dataset

    return Dataset.from_list(formatted_rows)


def build_sft_trainer(config: dict[str, Any]) -> Any:
    from unsloth import FastLanguageModel
    from trl import SFTConfig, SFTTrainer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model_id"],
        max_seq_length=config["max_seq_length"],
        load_in_4bit=config["load_in_4bit"],
        load_in_16bit=not config["load_in_4bit"],
        full_finetuning=False,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config["seed"],
        max_seq_length=config["max_seq_length"],
    )
    dataset = load_sft_dataset(config["dataset_path"], tokenizer)
    args = SFTConfig(
        max_seq_length=config["max_seq_length"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        warmup_steps=config["warmup_steps"],
        max_steps=config["max_steps"],
        learning_rate=config["learning_rate"],
        logging_steps=config["logging_steps"],
        output_dir=config["output_dir"],
        optim=config["optim"],
        seed=config["seed"],
        dataset_num_proc=1,
    )
    return SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=config["max_seq_length"],
        packing=False,
        args=args,
    )


def run_sft_training(config: dict[str, Any]) -> Any:
    ensure_sft_artifact_dirs(config)
    trainer = build_sft_trainer(config)
    trainer_stats = trainer.train()

    trainer.model.save_pretrained(config["adapter_dir"])
    trainer.processing_class.save_pretrained(config["adapter_dir"])
    trainer.model.save_pretrained_merged(
        config["output_dir"],
        trainer.processing_class,
        save_method="merged_16bit",
    )

    return trainer_stats


def _load_sft_tokenizer(path: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path)


def _load_sft_causal_lm(path: str) -> Any:
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(path)


def _load_sft_base_causal_lm(model_id: str) -> Any:
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(model_id)


def _load_sft_adapter(base_model: Any, path: str) -> Any:
    from peft import PeftModel

    return PeftModel.from_pretrained(base_model, path)


def validate_sft_exports(config: dict[str, Any]) -> dict[str, Any]:
    merged_dir = config["output_dir"]
    adapter_dir = config["adapter_dir"]
    _load_sft_tokenizer(merged_dir)
    merged_model = _load_sft_causal_lm(merged_dir)
    base_model = _load_sft_base_causal_lm(config["model_id"])
    adapter_model = _load_sft_adapter(base_model, adapter_dir)
    return {"merged_model": merged_model, "adapter_model": adapter_model}
