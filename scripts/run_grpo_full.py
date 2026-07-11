import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ensure_runtime_dirs, load_config
from src.train.grpo import build_grpo_job_config, run_grpo_training

def main() -> int:
    config = load_config("configs/default.yaml")
    ensure_runtime_dirs(config)
    
    job_config = build_grpo_job_config(
        init_model_path=config["sft_output_dir"],
        output_dir=config["grpo_output_dir"],
    )
    
    print("Starting GRPO training...")
    stats = run_grpo_training(job_config)
    print("GRPO training complete.")
    print(stats)
    return 0

if __name__ == "__main__":
    sys.exit(main())
