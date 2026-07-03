# Browser Agent RL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Colab-first, one-task browser-agent training pipeline that runs baseline evaluation, successful trajectory collection, SFT, GRPO, and final comparison for `click-checkboxes` using `Qwen/Qwen3.5-0.8B` with final runtime through OpenEnv.

**Architecture:** Put all logic in focused Python modules under `src/`, with one thin orchestrator notebook in `notebooks/` that installs dependencies, configures paths, and calls phase entrypoints. Start by inspecting native BrowserGym/MiniWoB to lock the prompt input shape and canonical action schema, then run baseline, data collection, SFT, RL, evaluation, and analysis through shared interfaces.

**Tech Stack:** Python, OpenEnv, BrowserGym, MiniWoB, Unsloth, Hugging Face Transformers, PEFT/LoRA, PyTorch, pytest, matplotlib/pandas, Google Colab on T4.

---

## File Structure

### Files to Create

- `requirements.txt` — pinned Python dependencies for Colab runtime
- `README.md` — project overview, Colab workflow, and run instructions
- `configs/default.yaml` — single runtime config for paths, budgets, model id, and task id
- `src/__init__.py` — package marker
- `src/config.py` — config loading helpers
- `src/utils/logging.py` — run logger and JSON helpers
- `src/env/native_miniwob_probe.py` — phase-1 native BrowserGym/MiniWoB inspection helpers
- `src/env/openenv_task.py` — final OpenEnv environment wrapper for `click-checkboxes`
- `src/env/action_schema.py` — canonical action parse/format helpers chosen after probe
- `src/agents/prompting.py` — prompt builder from instruction + observation
- `src/agents/policy.py` — model loading and generation wrapper for base/SFT/GRPO checkpoints
- `src/agents/rollout.py` — shared episode rollout loop
- `src/pipeline/collect.py` — success-only trajectory collection entrypoint with `stop after 50 successes`
- `src/pipeline/run_all.py` — thin phase orchestrator callable from notebook cells
- `src/data/trajectory_schema.py` — typed trajectory record helpers
- `src/data/storage.py` — JSONL save/load helpers
- `src/data/build_sft_dataset.py` — convert successful trajectories into step-wise SFT rows
- `src/train/sft.py` — Unsloth SFT entrypoint
- `src/train/grpo.py` — Unsloth GRPO entrypoint
- `src/eval/metrics.py` — success/reward/step metrics
- `src/eval/run_eval.py` — shared evaluation entrypoint
- `src/analysis/plots.py` — plot helpers
- `src/analysis/report.py` — final comparison table and analysis artifact writer
- `notebooks/orchestrator.ipynb` — single Colab orchestrator notebook
- `tests/test_config.py`
- `tests/env/test_native_probe.py`
- `tests/env/test_action_schema.py`
- `tests/agents/test_prompting.py`
- `tests/agents/test_rollout.py`
- `tests/pipeline/test_collect.py`
- `tests/pipeline/test_run_all.py`
- `tests/data/test_storage.py`
- `tests/data/test_build_sft_dataset.py`
- `tests/eval/test_metrics.py`
- `tests/test_sft_config.py`
- `tests/test_grpo_config.py`
- `tests/analysis/test_report.py`

### Runtime Artifact Paths to Create

- `data/successful_trajectories.jsonl`
- `data/sft_dataset.jsonl`
- `models/sft_model/`
- `models/grpo_model/`
- `results/baseline_metrics.json`
- `results/sft_metrics.json`
- `results/grpo_metrics.json`
- `results/figures/`
- `results/report.md`

### Files to Modify Later

- `docs/superpowers/specs/2026-06-22-browser-agent-rl-design.md` — only if implementation reveals spec-level mismatch

## Task 1: Scaffold Project Skeleton and Config

**Files:**
- Create: `requirements.txt`
- Create: `README.md`
- Create: `configs/default.yaml`
- Create: `src/__init__.py`
- Create: `src/config.py`
- Create: `src/utils/logging.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write failing config test**

```python
from src.config import load_config


def test_load_config_reads_model_and_task(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("model_id: Qwen/Qwen3.5-0.8B\ntask_id: click-checkboxes\n")
    cfg = load_config(path)
    assert cfg["model_id"] == "Qwen/Qwen3.5-0.8B"
    assert cfg["task_id"] == "click-checkboxes"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError` or missing `load_config`

- [ ] **Step 3: Create minimal project files**

Implement:
- `load_config(path)` in `src/config.py`
- minimal `configs/default.yaml`
- minimal dependency list in `requirements.txt`
- README stub with project purpose and planned phases
- JSON helper/logger stubs in `src/utils/logging.py`

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add requirements.txt README.md configs/default.yaml src/__init__.py src/config.py src/utils/logging.py tests/test_config.py
git commit -m "feat: scaffold project config and docs"
```

## Task 2: Build Native MiniWoB Probe for Phase-1 Exploration

**Files:**
- Create: `src/env/native_miniwob_probe.py`
- Create: `tests/env/test_native_probe.py`

- [ ] **Step 1: Write failing probe normalization test**

```python
from src.env.native_miniwob_probe import summarize_probe_result


def test_summarize_probe_result_extracts_required_keys():
    raw = {
        "utterance": "Select Apple",
        "fields": {"target 0": "Apple"},
        "dom_elements": [{"ref": 1, "text": "Apple"}],
    }
    summary = summarize_probe_result(raw)
    assert summary["instruction"] == "Select Apple"
    assert "fields" in summary
    assert "dom_refs" in summary
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/env/test_native_probe.py -v`
Expected: FAIL because probe module does not exist

- [ ] **Step 3: Write minimal probe helpers**

Implement:
- environment factory stub for native `click-checkboxes`
- `summarize_probe_result(raw_obs)`
- helper that records one reset payload and one step payload for manual notebook inspection

- [ ] **Step 4: Add notebook-facing manual check snippet to README**

Document exact command or import path that prints:
- raw observation keys
- instruction field
- fields payload
- one sample action attempt result

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/env/test_native_probe.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/env/native_miniwob_probe.py tests/env/test_native_probe.py README.md
git commit -m "feat: add native MiniWoB probe helpers"
```

## Task 3: Lock Canonical Action Schema and OpenEnv Wrapper

**Files:**
- Create: `src/env/openenv_task.py`
- Create: `src/env/action_schema.py`
- Create: `tests/env/test_action_schema.py`

- [ ] **Step 1: Write failing action schema tests**

```python
from src.env.action_schema import format_action, parse_model_action


def test_parse_model_action_accepts_canonical_click():
    action = parse_model_action("CLICK ref=12")
    assert action["type"] == "CLICK"
    assert action["ref"] == 12


def test_format_action_returns_env_payload():
    payload = format_action({"type": "CLICK", "ref": 12})
    assert payload is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/env/test_action_schema.py -v`
Expected: FAIL because parser/formatter not implemented

- [ ] **Step 3: Implement minimal canonical schema**

Implement after phase-1 inspection:
- one canonical model-output action syntax
- parser from model text to structured action
- formatter from structured action to OpenEnv step payload
- OpenEnv wrapper with `reset()` and `step()` returning normalized instruction/observation/reward/done/info
- one explicit success contract used everywhere: an episode is successful only when the environment terminates and wrapper-normalized `info["success"] == True`

- [ ] **Step 4: Add phase-1 exit criterion check**

Write a small helper or README checklist that is complete only when:
- one concrete prompt input shape is written down
- one concrete action schema is written down
- one successful OpenEnv `reset -> action -> step` example is recorded

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/env/test_action_schema.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/env/openenv_task.py src/env/action_schema.py tests/env/test_action_schema.py README.md
git commit -m "feat: add OpenEnv wrapper and canonical action schema"
```

## Task 4: Build Prompting and Policy Wrapper

**Files:**
- Create: `src/agents/prompting.py`
- Create: `src/agents/policy.py`
- Create: `tests/agents/test_prompting.py`

- [ ] **Step 1: Write failing prompt test**

```python
from src.agents.prompting import build_action_prompt


def test_build_action_prompt_includes_instruction_and_observation():
    prompt = build_action_prompt(
        instruction="Select Apple",
        observation="Apple checkbox visible",
    )
    assert "Select Apple" in prompt
    assert "Apple checkbox visible" in prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/agents/test_prompting.py -v`
Expected: FAIL because prompting module does not exist

- [ ] **Step 3: Implement minimal prompt and model wrapper**

Implement:
- deterministic prompt builder
- model loader for `Qwen/Qwen3.5-0.8B`
- generation wrapper that returns short action text only
- checkpoint path override so same wrapper can load base, SFT, or GRPO model

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/agents/test_prompting.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/agents/prompting.py src/agents/policy.py tests/agents/test_prompting.py
git commit -m "feat: add prompting and policy wrapper"
```

## Task 5: Implement Shared Rollout Loop

**Files:**
- Create: `src/agents/rollout.py`
- Create: `src/data/trajectory_schema.py`
- Create: `tests/agents/test_rollout.py`

- [ ] **Step 1: Write failing rollout test**

```python
from src.agents.rollout import run_episode


class DummyEnv:
    def reset(self):
        return {"instruction": "Select Apple", "observation": "Apple"}

    def step(self, action):
        return {"instruction": "Select Apple", "observation": "Done"}, 1.0, True, {"success": True}


class DummyPolicy:
    def act(self, instruction, observation):
        return "CLICK ref=1"


def test_run_episode_collects_steps():
    episode = run_episode(DummyEnv(), DummyPolicy(), max_steps=3)
    assert len(episode["steps"]) == 1
    assert episode["steps"][0]["action"] == "CLICK ref=1"
    assert episode["success"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/agents/test_rollout.py -v`
Expected: FAIL because rollout loop does not exist

- [ ] **Step 3: Implement minimal rollout loop**

Implement:
- `run_episode(env, policy, max_steps)`
- trajectory step records with instruction, observation, action, reward, done
- success computation only from wrapper-normalized `info["success"]`

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/agents/test_rollout.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/agents/rollout.py src/data/trajectory_schema.py tests/agents/test_rollout.py
git commit -m "feat: add shared rollout loop"
```

## Task 6: Add Trajectory Storage and SFT Dataset Builder

**Files:**
- Create: `src/data/storage.py`
- Create: `src/data/build_sft_dataset.py`
- Create: `tests/data/test_storage.py`
- Create: `tests/data/test_build_sft_dataset.py`

- [ ] **Step 1: Write failing storage and dataset tests**

```python
from src.data.storage import write_jsonl, read_jsonl
from src.data.build_sft_dataset import build_step_dataset


def test_write_and_read_jsonl_roundtrip(tmp_path):
    path = tmp_path / "traj.jsonl"
    rows = [{"instruction": "Select Apple", "action": "CLICK ref=1"}]
    write_jsonl(path, rows)
    loaded = read_jsonl(path)
    assert loaded == rows


def test_build_step_dataset_expands_episode_steps():
    episodes = [{
        "steps": [
            {"instruction": "Select Apple", "observation": "Apple", "action": "CLICK ref=1"},
            {"instruction": "Select Apple", "observation": "Submit", "action": "CLICK ref=2"},
        ]
    }]
    rows = build_step_dataset(episodes)
    assert len(rows) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/data/test_storage.py tests/data/test_build_sft_dataset.py -v`
Expected: FAIL because functions do not exist

- [ ] **Step 3: Implement minimal storage and dataset builder**

Implement:
- JSONL read/write helpers
- successful trajectory save path
- step-wise dataset expansion
- output rows with prompt input and target action fields expected by SFT
- write built step-wise dataset to `data/sft_dataset.jsonl`

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/data/test_storage.py tests/data/test_build_sft_dataset.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/storage.py src/data/build_sft_dataset.py tests/data/test_storage.py tests/data/test_build_sft_dataset.py
git commit -m "feat: add trajectory storage and SFT dataset builder"
```

## Task 7: Add Success-Only Collection Pipeline Entry Point

**Files:**
- Create: `src/pipeline/collect.py`
- Create: `tests/pipeline/test_collect.py`

- [ ] **Step 1: Write failing collection stop-rule test**

```python
from src.pipeline.collect import collect_successful_trajectories


def test_collect_successful_trajectories_stops_after_target_successes():
    episodes = [
        {"success": True, "steps": [1]},
        {"success": False, "steps": [1]},
        {"success": True, "steps": [1]},
    ]

    def runner():
        for episode in episodes:
            yield episode

    collected = collect_successful_trajectories(runner(), target_successes=2)
    assert len(collected) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/pipeline/test_collect.py -v`
Expected: FAIL because collection entrypoint does not exist

- [ ] **Step 3: Implement collection pipeline**

Implement:
- success-only filtering
- `stop after 50 successes` behavior via config
- trajectory artifact writes to `data/successful_trajectories.jsonl`
- counters for attempted episodes and collected successes

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/pipeline/test_collect.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/collect.py tests/pipeline/test_collect.py
git commit -m "feat: add success-only collection pipeline"
```

## Task 8: Implement Shared Evaluation Metrics and Eval Entry Point

**Files:**
- Create: `src/eval/metrics.py`
- Create: `src/eval/run_eval.py`
- Create: `tests/eval/test_metrics.py`

- [ ] **Step 1: Write failing metrics test**

```python
from src.eval.metrics import aggregate_metrics


def test_aggregate_metrics_computes_success_reward_and_steps():
    episodes = [
        {"success": True, "total_reward": 1.0, "num_steps": 2},
        {"success": False, "total_reward": 0.0, "num_steps": 3},
    ]
    metrics = aggregate_metrics(episodes)
    assert metrics["success_rate"] == 0.5
    assert metrics["average_reward"] == 0.5
    assert metrics["average_steps"] == 2.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/eval/test_metrics.py -v`
Expected: FAIL because metrics helper does not exist

- [ ] **Step 3: Implement minimal evaluator**

Implement:
- metric aggregation helper
- eval runner that executes `N` episodes with chosen policy
- metrics JSON write helper
- support small default eval budget and override for final `500`

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/eval/test_metrics.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/eval/metrics.py src/eval/run_eval.py tests/eval/test_metrics.py
git commit -m "feat: add evaluation metrics and runner"
```

## Task 9: Implement SFT Training Entry Point

**Files:**
- Create: `src/train/sft.py`
- Modify: `configs/default.yaml`
- Modify: `README.md`

- [ ] **Step 1: Write a failing dry-run test or smoke assertion**

```python
from src.train.sft import build_sft_job_config


def test_build_sft_job_config_points_to_dataset_and_output():
    cfg = build_sft_job_config(
        dataset_path="data/sft_dataset.jsonl",
        output_dir="models/sft_model",
    )
    assert cfg["output_dir"] == "models/sft_model"
```

- [ ] **Step 2: Run targeted test to verify it fails**

Run: `pytest tests/test_sft_config.py -v`
Expected: FAIL because helper/test file does not exist

- [ ] **Step 3: Add `tests/test_sft_config.py` and minimal SFT config builder**

Implement:
- SFT config builder
- Unsloth setup entrypoint
- load `data/sft_dataset.jsonl` as trainer input artifact
- save adapter/model under `models/sft_model/`

- [ ] **Step 4: Add smoke-run command to README**

Document exact command for tiny SFT smoke run in Colab.

- [ ] **Step 5: Run targeted test to verify it passes**

Run: `pytest tests/test_sft_config.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/train/sft.py tests/test_sft_config.py configs/default.yaml README.md
git commit -m "feat: add SFT training entrypoint"
```

## Task 10: Implement GRPO Training Entry Point

**Files:**
- Create: `src/train/grpo.py`
- Modify: `configs/default.yaml`
- Modify: `README.md`

- [ ] **Step 1: Write a failing GRPO config test**

```python
from src.train.grpo import build_grpo_job_config


def test_build_grpo_job_config_uses_sft_checkpoint():
    cfg = build_grpo_job_config(
        init_model_path="models/sft_model",
        output_dir="models/grpo_model",
    )
    assert cfg["init_model_path"] == "models/sft_model"
```

- [ ] **Step 2: Run targeted test to verify it fails**

Run: `pytest tests/test_grpo_config.py -v`
Expected: FAIL because helper/test file does not exist

- [ ] **Step 3: Add `tests/test_grpo_config.py` and minimal GRPO entrypoint**

Implement:
- GRPO config builder
- OpenEnv-backed rollout/update loop wiring
- native reward usage only
- save RL checkpoint under `models/grpo_model/`

- [ ] **Step 4: Add smoke-run command to README**

Document tiny online GRPO smoke run with minimal rollout budget.

- [ ] **Step 5: Run targeted test to verify it passes**

Run: `pytest tests/test_grpo_config.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/train/grpo.py tests/test_grpo_config.py configs/default.yaml README.md
git commit -m "feat: add GRPO training entrypoint"
```

## Task 11: Implement Analysis Outputs and Educational Report

**Files:**
- Create: `src/analysis/plots.py`
- Create: `src/analysis/report.py`
- Create: `tests/analysis/test_report.py`

- [ ] **Step 1: Write failing report test**

```python
from src.analysis.report import build_comparison_table


def test_build_comparison_table_formats_three_models():
    table = build_comparison_table({
        "base": {"success_rate": 0.2, "average_reward": 0.3, "average_steps": 4.0},
        "sft": {"success_rate": 0.4, "average_reward": 0.5, "average_steps": 3.0},
        "grpo": {"success_rate": 0.6, "average_reward": 0.7, "average_steps": 2.0},
    })
    assert "base" in table.lower()
    assert "sft" in table.lower()
    assert "grpo" in table.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/analysis/test_report.py -v`
Expected: FAIL because report helper does not exist

- [ ] **Step 3: Implement minimal analysis helpers**

Implement:
- comparison table builder
- metric JSON loader
- plot helpers for success, reward, and steps
- optional GRPO training-curve plot only if logs exist
- report writer that generates `results/report.md` with:
  - what a trajectory is
  - why successful-only filtering is used
  - what behavior cloning is
  - why SFT helps RL
  - what GRPO is
  - how GRPO differs from PPO at high level
  - what reward is optimized
  - why RL can outperform imitation
  - limitations of this setup

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/analysis/test_report.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/analysis/plots.py src/analysis/report.py tests/analysis/test_report.py
git commit -m "feat: add analysis helpers"
```

## Task 12: Build Phase Orchestrator and Colab Notebook

**Files:**
- Create: `src/pipeline/run_all.py`
- Create: `tests/pipeline/test_run_all.py`
- Create: `notebooks/orchestrator.ipynb`
- Modify: `README.md`

- [ ] **Step 1: Write failing phase-order test**

```python
from src.pipeline.run_all import list_pipeline_phases


def test_list_pipeline_phases_matches_spec_order():
    phases = list_pipeline_phases()
    assert phases == [
        "native_probe",
        "openenv_probe",
        "baseline_eval",
        "collect_successes",
        "build_sft_dataset",
        "sft_train",
        "sft_eval",
        "grpo_train",
        "grpo_eval",
        "analysis",
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/pipeline/test_run_all.py -v`
Expected: FAIL because orchestrator module does not exist

- [ ] **Step 3: Implement thin phase orchestrator module**

Implement:
- named phase entrypoints
- a top-level runner that calls existing modules in spec order
- no business logic duplicated from env/train/eval modules
- notebook-facing helpers for running one phase or full pipeline

- [ ] **Step 4: Draft notebook cell plan in markdown first**

Create notebook sections for:
- environment setup
- dependency install
- config load
- phase-1 native probe
- OpenEnv probe
- baseline eval
- trajectory collection
- SFT
- SFT eval
- GRPO
- GRPO eval
- analysis/report

- [ ] **Step 5: Generate notebook with thin orchestration cells only**

Each code cell should call module functions or CLI entrypoints from `src/`, not re-implement logic inline.

- [ ] **Step 6: Add notebook smoke assertions**

At minimum:
- config loaded
- paths created
- one probe call works
- one eval call returns metrics dict

- [ ] **Step 7: Update README with Colab-first run order**

Document notebook as primary entrypoint.

- [ ] **Step 8: Commit**

```bash
git add src/pipeline/run_all.py tests/pipeline/test_run_all.py notebooks/orchestrator.ipynb README.md
git commit -m "feat: add pipeline orchestrator and Colab notebook"
```

## Task 13: End-to-End Smoke Verification

**Files:**
- Modify: `README.md`
- Modify: `configs/default.yaml`

- [ ] **Step 1: Run unit test suite**

Run: `pytest tests -v`
Expected: PASS

- [ ] **Step 2: Run native probe smoke test**

Run documented probe command in Colab/local dev shell.
Expected: observation keys and one step payload printed

- [ ] **Step 3: Run OpenEnv smoke episode**

Run one model-driven episode with tiny step limit.
Expected: one trajectory record written without interface crash

- [ ] **Step 4: Run tiny SFT smoke job**

Expected: checkpoint saved under `models/sft_model/`

- [ ] **Step 5: Run tiny GRPO smoke job**

Expected: checkpoint saved under `models/grpo_model/`

- [ ] **Step 6: Run final smoke analysis**

Expected:
- metrics JSON files exist
- comparison table renders
- at least one plot saved

- [ ] **Step 7: Commit**

```bash
git add README.md configs/default.yaml
git commit -m "test: verify browser agent RL smoke flow"
```

## Notes for Implementers

- Keep `OpenEnv` as final runtime path even if native MiniWoB probe is easier to debug.
- Do not add teacher policies or failed-trajectory training unless spec is revised.
- Do not split back into many notebooks; notebook is control surface only.
- If baseline cannot reach 50 successes in reasonable time, stop and replan instead of silently changing dataset policy.
- Prefer smallest stable action schema confirmed by phase-1 exploration.
- Keep T4 budgets conservative by default.
