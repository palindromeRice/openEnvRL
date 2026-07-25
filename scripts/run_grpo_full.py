from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.artifacts import (
    grpo_boundaries,
    is_complete_grpo_boundary,
    prepare_drive_artifact_root,
    write_grpo_run_manifest,
)
from src.config import load_config
from src.pipeline.run_all import run_grpo_eval_phase
from src.train.grpo import build_grpo_job_config, run_grpo_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run resumable Drive-backed GRPO training.")
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--sft-model-path", required=True)
    parser.add_argument("--total-steps", type=int, default=200)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--config", default="configs/default.yaml")
    return parser


def _validate_sft_model(path: Path) -> None:
    if not path.is_dir():
        raise ValueError(f"SFT model path does not exist: {path}")
    from transformers import AutoConfig, AutoTokenizer

    AutoConfig.from_pretrained(path, trust_remote_code=True)
    AutoTokenizer.from_pretrained(path, trust_remote_code=True)


def run_boundaries(config: dict, paths: dict[str, Path], resume_from_checkpoint: str | None = None) -> dict:
    boundaries = grpo_boundaries(config)
    resume = resume_from_checkpoint
    for step in boundaries:
        checkpoint = paths["grpo_output_dir"] / f"checkpoint-{step}"
        if is_complete_grpo_boundary(paths, step):
            resume = str(checkpoint)
            continue
        metric_path = paths["results_dir"] / f"grpo_eval_step_{step}.json"
        checkpoint_valid = checkpoint.is_dir() and all(
            (checkpoint / name).is_file()
            for name in ("config.json", "tokenizer_config.json", "optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json")
        )
        if not checkpoint_valid:
            job = build_grpo_job_config(
                config["sft_model_path"],
                str(paths["grpo_output_dir"]),
                max_steps=step,
                save_steps=config["checkpoint_every"],
                resume_from_checkpoint=resume,
                miniwob_root=config.get("miniwob_root", "/content/browsergym-assets"),
            )
            run_grpo_training(job)
        if not checkpoint.is_dir():
            raise RuntimeError(f"GRPO did not save checkpoint {checkpoint}")
        run_grpo_eval_phase(config, checkpoint_step=step)
        resume = str(checkpoint)
    return write_grpo_run_manifest(paths, config)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sft_model_path = Path(args.sft_model_path)
    _validate_sft_model(sft_model_path)
    config = load_config(args.config)
    config.update(
        {
            "total_steps": args.total_steps,
            "checkpoint_every": args.checkpoint_every,
            "default_eval_episodes": args.eval_episodes,
            "sft_model_path": str(sft_model_path),
        }
    )
    paths = prepare_drive_artifact_root(Path(args.artifact_root))
    config.update({key: str(value) for key, value in paths.items()})
    config["artifact_root"] = args.artifact_root
    config["miniwob_root"] = str(Path(args.artifact_root) / "browsergym-assets")
    manifest = run_boundaries(config, paths, args.resume_from_checkpoint)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
