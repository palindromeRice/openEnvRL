# GRPO Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable execution of the GRPO training phase using the recovered SFT model.

**Architecture:** Create a dedicated runner script `scripts/run_grpo_full.py` and update the existing pipeline router in `src/pipeline/run_all.py` to correctly trigger GRPO training.

**Tech Stack:** Python, TRL (Unsloth GRPO)

## Global Constraints

- Execution will primarily happen via `scripts/run_grpo_full.py` in the Colab CLI environment.
- The `init_model_path` must point to the SFT output directory.

---

### Task 1: Update Pipeline Router for GRPO

**Files:**
- Modify: `src/pipeline/run_all.py`
- Test: `tests/test_run_all_syntax.py` (synthetic test for syntax validation)

**Interfaces:**
- Consumes: `config` dictionary from the pipeline runner.
- Produces: Updates `run_grpo_phase` to call `run_grpo_training(job_config)`.

- [ ] **Step 1: Write the syntax/import test**

```python
# Create tests/test_run_all_syntax.py
def test_grpo_phase_imports():
    from src.pipeline.run_all import run_grpo_phase
    assert callable(run_grpo_phase)
```

- [ ] **Step 2: Run test to verify it executes (might pass already if no syntax errors, but ensures we don't break it)**

```bash
pytest tests/test_run_all_syntax.py -v
```

- [ ] **Step 3: Write minimal implementation**

Modify `src/pipeline/run_all.py` (around line 188):

```python
def run_grpo_phase(config: dict[str, Any]) -> dict[str, Any]:
    from src.train.grpo import run_grpo_training, build_grpo_job_config
    
    job_config = build_grpo_job_config(
        init_model_path=config["sft_output_dir"],
        output_dir=config["grpo_output_dir"],
    )
    # Merge base config if needed, but build_grpo_job_config provides defaults
    return run_grpo_training(job_config)
```

- [ ] **Step 4: Run test to verify it passes syntax checks**

```bash
pytest tests/test_run_all_syntax.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/run_all.py tests/test_run_all_syntax.py
git commit -m "feat: update run_grpo_phase to execute training"
```

---

### Task 2: Create Dedicated GRPO Runner Script

**Files:**
- Create: `scripts/run_grpo_full.py`
- Test: `tests/test_run_grpo_full_syntax.py`

**Interfaces:**
- Consumes: `src.train.grpo.run_grpo_training` and `build_grpo_job_config`.
- Produces: A standalone executable script.

- [ ] **Step 1: Write the failing test**

```python
# Create tests/test_run_grpo_full_syntax.py
import subprocess

def test_script_syntax():
    result = subprocess.run(["python", "-m", "py_compile", "scripts/run_grpo_full.py"], capture_output=True)
    assert result.returncode == 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_run_grpo_full_syntax.py -v
```
Expected: FAIL (file not found)

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/run_grpo_full.py
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_run_grpo_full_syntax.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/run_grpo_full.py tests/test_run_grpo_full_syntax.py
git commit -m "feat: add dedicated GRPO runner script"
```
