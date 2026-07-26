"""Colab entrypoint: re-evaluate r5 checkpoint-20 with corrected tool execution."""

import os
import subprocess
import sys
from pathlib import Path


REPO = "/content/drive/MyDrive/openEnvRL"
ARTIFACT_ROOT = f"{REPO}/artifacts/2026-07-26-grpo-gate-r5"
CHECKPOINT = f"{ARTIFACT_ROOT}/models/grpo_model/checkpoint-20"
MINIWOB_ROOT = "/content/browsergym-assets"
EPISODES = 50

os.chdir(REPO)
sys.path.insert(0, REPO)

log_dir = Path(ARTIFACT_ROOT) / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
log_path = log_dir / "corrected_eval_50.log"
telemetry_dir = Path(ARTIFACT_ROOT) / "telemetry"
telemetry_dir.mkdir(parents=True, exist_ok=True)
telemetry_path = telemetry_dir / "gpu_corrected_eval_50.csv"

print("Installing evaluation dependencies...", flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
subprocess.run([sys.executable, "-m", "playwright", "install-deps"], check=True)
subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True)

child = f"""
import json
import sys
sys.path.insert(0, {REPO!r})
from src.config import load_config
from src.pipeline.run_all import run_grpo_eval_phase

config = load_config("configs/default.yaml")
config.update({{
    "results_dir": {ARTIFACT_ROOT!r} + "/results",
    "grpo_output_dir": {ARTIFACT_ROOT!r} + "/models/grpo_model",
    "miniwob_root": {MINIWOB_ROOT!r},
    "default_eval_episodes": {EPISODES},
}})
metrics = run_grpo_eval_phase(
    config,
    checkpoint_step="20_corrected_50",
    model_path={CHECKPOINT!r},
)
print("CORRECTED_EVAL_METRICS:", json.dumps(metrics), flush=True)
"""

telemetry = subprocess.Popen(
    [
        "nvidia-smi",
        "--query-gpu=timestamp,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
        "--format=csv",
        "-l",
        "10",
        "-f",
        str(telemetry_path),
    ]
)
try:
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, "-u", "-c", child],
            cwd=REPO,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        result = process.wait()
finally:
    telemetry.terminate()

print(f"CORRECTED_EVAL_RESULT: {'PASS' if result == 0 else 'FAIL'}", flush=True)
raise SystemExit(result)
