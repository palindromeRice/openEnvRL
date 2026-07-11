import sys
import json
import os
import subprocess



PROJECT_ROOT = '/content/drive/MyDrive/openEnvRL'
if os.path.exists(PROJECT_ROOT):
    os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print("Installing dependencies...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "git+https://huggingface.co/spaces/openenv/browsergym_env"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "nest_asyncio"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "playwright"])
subprocess.check_call(["playwright", "install", "chromium"])
subprocess.check_call(["playwright", "install-deps"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "flash-attn", "--no-build-isolation"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "causal-conv1d", "--no-build-isolation"])
print("Dependencies installed.")

import nest_asyncio
nest_asyncio.apply()

from src.cli.run_phase import apply_profile, apply_artifact_root, prepare_runtime, load_config
from src.pipeline.run_all import run_named_phase

def main():
    config = load_config("configs/default.yaml")
    config = apply_profile(config, "full")
    config = apply_artifact_root(config, "/content/drive/MyDrive/openEnvRL/artifacts/2026-07-12-sft-full")
    config = prepare_runtime(config)

    phases = [
        "collect_successes",
        "build_sft_dataset",
        "sft_train",
        "sft_eval",
        "sft_manifest"
    ]

    context = {"config": config}
    results = {}
    for phase in phases:
        print(f"--- Starting phase: {phase} ---")
        res = run_named_phase(phase, context)
        results[phase] = res
        print(f"--- Finished phase: {phase} ---")
        print(json.dumps(res, indent=2))
        print()
    
    print("Full SFT pipeline completed successfully.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
