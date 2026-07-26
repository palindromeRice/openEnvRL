"""Colab entrypoint: GRPO smoke gate after the missing-task-instruction fix.

Runs deps -> prompt sanity check -> GPU telemetry -> short GRPO run + eval.
Args come from GATE_* env vars so the same file serves the 200-step run.
"""

import os
import subprocess
import sys
from pathlib import Path

REPO = "/content/drive/MyDrive/openEnvRL"
os.chdir(REPO)
sys.path.insert(0, REPO)

# Edit these constants directly. `colab exec` runs this file on the VM, so local
# shell env vars never reach it -- an env-var override here would silently no-op.
#
# ARTIFACT_ROOT must be a FRESH directory every run. run_boundaries() skips any
# boundary that already has a complete checkpoint plus a valid metric, so
# pointing at a finished root trains nothing and still reports success.
ARTIFACT_ROOT = f"{REPO}/artifacts/2026-07-26-grpo-gate-r5"
SFT_MODEL = f"{REPO}/artifacts/2026-07-12-sft-full/models/sft_model"
TOTAL_STEPS = "20"
CHECKPOINT_EVERY = "20"
EVAL_EPISODES = "10"
MINIWOB_ROOT = "/content/browsergym-assets"

LOG_DIR = Path(ARTIFACT_ROOT) / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def run_logged(cmd, name, tail=250):
    """Run a child process, tee to a Drive log, echo through Python.

    IPython captures the kernel's stdout, not a subprocess's raw fd, so a plain
    subprocess.run() loses every child line -- including tracebacks.
    """
    log_path = LOG_DIR / f"{name}.log"
    lines = []
    with open(log_path, "w", encoding="utf-8") as log:
        process = subprocess.Popen(
            cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in process.stdout:
            log.write(line)
            log.flush()
            lines.append(line)
        rc = process.wait()
    for line in lines[-tail:]:
        print(line, end="", flush=True)
    print(f"[{name}] exit={rc} full log -> {log_path}", flush=True)
    return rc


print("Installing dependencies...", flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
subprocess.run([sys.executable, "-m", "playwright", "install-deps"], check=True)
subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True)

# Gate item 1: the model's initial context must contain the dynamic task text.
# Must run in a child process -- `colab exec` hosts an IPython kernel with a
# running asyncio loop, and Playwright's sync API refuses to start inside one.
print("=" * 60, flush=True)
print("Prompt sanity check", flush=True)
PROBE = f"""
import sys
sys.path.insert(0, {REPO!r})
from src.train.grpo import BrowserGymToolEnv, build_grpo_job_config

probe = BrowserGymToolEnv(config={{"miniwob_root": {MINIWOB_ROOT!r}}})
try:
    prompt = probe.reset()
finally:
    probe._close()
print(prompt[:1500])
first = prompt.split("\\n")[0]
assert first.startswith("Task: ") and len(first) > len("Task: "), \\
    "reset() prompt is missing the dynamic task instruction"
print("OK: dynamic task instruction present in initial context.")
print("config:", {{k: build_grpo_job_config("x", "y")[k] for k in
      ("num_generations", "per_device_train_batch_size",
       "gradient_accumulation_steps", "max_prompt_length",
       "max_completion_length", "optim")}})
"""
if run_logged([sys.executable, "-c", PROBE], "probe") != 0:
    print("GATE_RESULT: FAIL (prompt sanity check)", flush=True)
    sys.exit(1)

telemetry_dir = Path(ARTIFACT_ROOT) / "telemetry"
telemetry_dir.mkdir(parents=True, exist_ok=True)
telemetry_log = telemetry_dir / "gpu.csv"
process_memory_log = telemetry_dir / "process_gpu_memory.csv"
print(f"GPU telemetry -> {telemetry_log}", flush=True)
telemetry = subprocess.Popen(
    ["nvidia-smi",
     "--query-gpu=timestamp,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
     "--format=csv", "-l", "10", "-f", str(telemetry_log)]
)
process_memory = subprocess.Popen(
    ["nvidia-smi",
     "--query-compute-apps=pid,process_name,used_gpu_memory",
     "--format=csv", "-l", "10", "-f", str(process_memory_log)]
)

print("=" * 60, flush=True)
print(f"GRPO: {TOTAL_STEPS} steps, eval {EVAL_EPISODES} episodes -> {ARTIFACT_ROOT}", flush=True)
try:
    rc = run_logged(
        [sys.executable, "-u", "scripts/run_grpo_full.py",
         "--artifact-root", ARTIFACT_ROOT,
         "--sft-model-path", SFT_MODEL,
         "--total-steps", TOTAL_STEPS,
         "--checkpoint-every", CHECKPOINT_EVERY,
         "--eval-episodes", EVAL_EPISODES,
         "--miniwob-root", MINIWOB_ROOT],
        "train",
    )
finally:
    telemetry.terminate()
    process_memory.terminate()

# `colab exec` reports 0 even when the kernel cell raises, so never trust its
# exit status -- grep the run output for this marker instead.
print(f"GATE_RESULT: {'PASS' if rc == 0 else 'FAIL'}", flush=True)

# Peak/avg VRAM + GPU util. Low util here means decode is serial and the next
# lever is vLLM, not a bigger batch.
rows = [line.split(",") for line in telemetry_log.read_text().splitlines()[1:] if "," in line]
if rows:
    util = [int(r[1].strip().split()[0]) for r in rows]
    used = [int(r[2].strip().split()[0]) for r in rows]
    print(f"VRAM MiB: peak={max(used)} avg={sum(used) // len(used)}", flush=True)
    print(f"GPU util %: peak={max(util)} avg={sum(util) // len(util)} samples={len(util)}", flush=True)

sys.exit(rc)
