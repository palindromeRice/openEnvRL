# GRPO Tool-Use Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tune GRPO hyper-parameters (`max_completion_length`, `num_generations`, batch sizes) to allow the model to successfully emit JSON tool calls without getting cut off, and run a smoke test.

**Architecture:** Modifying `configs/default.yaml` to configure TRL's `GRPOTrainer` generation parameters. Since TRL auto-exposes the environment's public methods as tools via the chat template, no code changes are needed in `grpo.py`—we just need to give the model enough generation length to emit them.

**Tech Stack:** TRL, PyTorch, Colab CLI

## Global Constraints

- Never compile `flash-attn` from source on Colab.
- Run tests on the remote Colab L4 runtime using the provided script.

---

### Task 1: Update Generation Hyperparameters

**Files:**
- Modify: `configs/default.yaml`

**Interfaces:**
- Consumes: N/A
- Produces: Updated `default.yaml` that `run_grpo_full.py` will load.

- [ ] **Step 1: Modify `default.yaml` with aggressive generation parameters**

```yaml
# In configs/default.yaml, update these fields:
# (Leave other fields unchanged)

max_completion_length: 512
num_generations: 8
per_device_train_batch_size: 1
gradient_accumulation_steps: 4
```

- [ ] **Step 2: Commit**

```bash
git add configs/default.yaml
git commit -m "chore: tune GRPO generation hyperparams for tool use"
```

---

### Task 2: Execute Smoke Test & Verify

**Files:**
- Execute: `scripts/_colab_grpo_smoke.py`

**Interfaces:**
- Consumes: `configs/default.yaml`
- Produces: `models/grpo_model` and console output

- [ ] **Step 1: Execute the smoke test on Colab**

```bash
colab exec -s grpo -f "scripts/_colab_grpo_smoke.py" --timeout 3600
```

- [ ] **Step 2: Verify `tools/call_frequency` and `reward` metrics**
Review the console output during training. You MUST verify that:
1. `tools/call_frequency` is greater than 0
2. `reward` or `rewards/BrowserGymToolEnv/mean` is non-zero
(These indicate the model successfully output a tool call JSON that was executed and scored).

- [ ] **Step 3: Document Results**
Update `grpo_smoke_results.md` (or output directly to the user) with the verified metrics. No commit needed for the test run.
