"""Colab entrypoint: score the SFT model with the GRPO tool-call evaluator.

Without this, GRPO checkpoint scores have no valid control -- the 0.58 SFT
number came from a different evaluator (src/agents/policy.py, CLICK ref=
format) and is not comparable to a GRPOToolPolicy score.

Writes results/grpo_eval_step_sft_control.json under the gate artifact root.
"""

import os
import subprocess
import sys
from pathlib import Path

REPO = "/content/drive/MyDrive/openEnvRL"
os.chdir(REPO)

# Edit these directly -- `colab exec` runs this on the VM, so local shell env
# vars never reach it. Raise EPISODES to 50 for a comparison worth trusting;
# the recorded 0.40 control is only n=10.
ARTIFACT_ROOT = f"{REPO}/artifacts/2026-07-26-grpo-gate-r2"
SFT_MODEL = f"{REPO}/artifacts/2026-07-12-sft-full/models/sft_model"
EPISODES = "10"
MINIWOB_ROOT = "/content/browsergym-assets"

# Child process: Playwright's sync API cannot start inside colab exec's asyncio loop.
CHILD = f"""
import json, sys
sys.path.insert(0, {REPO!r})
from src.config import load_config
from src.pipeline.run_all import run_grpo_eval_phase

config = load_config("configs/default.yaml")
config.update({{
    "results_dir": {ARTIFACT_ROOT!r} + "/results",
    "grpo_output_dir": {ARTIFACT_ROOT!r} + "/models/grpo_model",
    "miniwob_root": {MINIWOB_ROOT!r},
    "default_eval_episodes": int({EPISODES!r}),
}})
metrics = run_grpo_eval_phase(config, checkpoint_step="sft_control", model_path={SFT_MODEL!r})
print("SFT_CONTROL_METRICS:", json.dumps(metrics))
"""

Path(ARTIFACT_ROOT, "results").mkdir(parents=True, exist_ok=True)
log_path = Path(ARTIFACT_ROOT, "logs", "sft_control.log")
log_path.parent.mkdir(parents=True, exist_ok=True)

lines = []
with open(log_path, "w", encoding="utf-8") as log:
    process = subprocess.Popen(
        [sys.executable, "-c", CHILD], cwd=REPO,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    for line in process.stdout:
        log.write(line)
        lines.append(line)
    rc = process.wait()

for line in lines[-200:]:
    print(line, end="", flush=True)
print(f"CONTROL_RESULT: {'PASS' if rc == 0 else 'FAIL'} log -> {log_path}", flush=True)
sys.exit(rc)
