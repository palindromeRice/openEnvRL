"""Colab entrypoint: score Base and SFT with the corrected GRPO tool-call evaluator.

The published Base 0.30 and SFT 0.58 came from src/agents/policy.py (CLICK ref=
format), not GRPOToolPolicy, so they are not comparable to the corrected GRPO
0.88. This reruns both controls, 50 episodes each, under the same evaluator.
"""

import os
import subprocess
import sys
from pathlib import Path


REPO = "/content/drive/MyDrive/openEnvRL"
ARTIFACT_ROOT = f"{REPO}/artifacts/2026-10-22-same-evaluator-controls"
SFT_MODEL = f"{REPO}/artifacts/2026-07-12-sft-full/models/sft_model"
# Re-scored on this VM too: the Colab stack moved (Python 3.13, torch 2.11) since July.
GRPO_CHECKPOINT = f"{REPO}/artifacts/2026-07-26-grpo-gate-r5/models/grpo_model/checkpoint-20"
BASE_MODEL_ID = "Qwen/Qwen3.5-0.8B"
BASE_MODEL_DIR = "/content/base_model"
MINIWOB_ROOT = "/content/browsergym-assets"
EPISODES = 50

os.chdir(REPO)
sys.path.insert(0, REPO)

log_dir = Path(ARTIFACT_ROOT) / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
Path(ARTIFACT_ROOT, "results").mkdir(parents=True, exist_ok=True)
log_path = log_dir / "controls_eval_50.log"

print("Installing evaluation dependencies...", flush=True)
# ponytail: skip the HF Space git dep -- its partial clone fails ("expected 'packfile'")
# and the native_browsergym eval backend never imports it.
eval_reqs = Path("/content/eval_requirements.txt")
eval_reqs.write_text(
    "".join(line for line in Path("requirements.txt").read_text().splitlines(True)
            if "huggingface.co/spaces" not in line and "browsergym-miniwob" not in line)
)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(eval_reqs)], check=True)
# browsergym-core 0.14.3 pins playwright==1.44 -> greenlet 3.0.3, which has no
# Python 3.13 wheel and fails to build. Install a current playwright, then
# browsergym without its pin.
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "playwright", "beautifulsoup4",
     "gymnasium", "lxml", "mcp[cli]", "numpy", "pillow", "pyparsing"],
    check=True,
)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--no-deps",
     "browsergym-core==0.14.3", "browsergym-miniwob==0.14.3"],
    check=True,
)
subprocess.run([sys.executable, "-m", "playwright", "install-deps"], check=True)
subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True)

# GRPOToolPolicy loads from a local directory, so materialize the base model first.
from huggingface_hub import snapshot_download

snapshot_download(BASE_MODEL_ID, local_dir=BASE_MODEL_DIR)

# Child process: Playwright's sync API cannot start inside colab exec's asyncio loop.
child = f"""
import json
import sys
sys.path.insert(0, {REPO!r})
from src.config import load_config
from src.pipeline.run_all import run_grpo_eval_phase

config = load_config("configs/default.yaml")
config.update({{
    "results_dir": {ARTIFACT_ROOT!r} + "/results",
    "grpo_output_dir": {ARTIFACT_ROOT!r} + "/models/unused",
    "miniwob_root": {MINIWOB_ROOT!r},
    "default_eval_episodes": {EPISODES},
}})
for step, path in [
    ("sft_control_50", {SFT_MODEL!r}),
    ("base_control_50", {BASE_MODEL_DIR!r}),
    ("grpo_r5_step20_rescore_50", {GRPO_CHECKPOINT!r}),
]:
    metrics = run_grpo_eval_phase(config, checkpoint_step=step, model_path=path)
    print(f"CONTROL_METRICS {{step}}:", json.dumps(metrics), flush=True)
"""

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

print(f"CONTROLS_RESULT: {'PASS' if result == 0 else 'FAIL'} log -> {log_path}", flush=True)
raise SystemExit(result)
