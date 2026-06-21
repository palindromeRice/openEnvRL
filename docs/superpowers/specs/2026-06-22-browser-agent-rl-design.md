# Browser Agent RL Design

Date: 2026-06-22

## Goal

Build a Colab-first educational project that shows how a pretrained language model improves on one browser task through:

1. Baseline evaluation
2. Successful trajectory collection
3. Supervised fine-tuning (SFT / behavior cloning)
4. GRPO reinforcement learning
5. Quantitative evaluation and analysis

Primary outcome is a simple, reproducible agent-training pipeline for learning, not state-of-the-art benchmark results.

## Scope

This spec covers one end-to-end training pipeline for one task:

- Environment family: OpenEnv + BrowserGym + MiniWoB
- Task: `click-checkboxes`
- Model: `Qwen/Qwen3.5-0.8B`
- Runtime: Google Colab on one GPU, assumed to be T4

Out of scope for first version:

- Multi-task training
- Distributed training
- Ray, Kubernetes, or custom cloud infrastructure
- Reward shaping beyond native environment reward
- Rule-based teacher policies
- Human-generated demonstrations

## Design Decisions Locked

- One task only for first version: `click-checkboxes`
- Final runnable pipeline uses `OpenEnv`
- Task mechanics are inspected first through native BrowserGym/MiniWoB before locking final adapter behavior
- One orchestrator Colab notebook controls whole run
- Real logic lives in Python scripts/modules, not duplicated across multiple notebooks
- Dataset comes from model-generated rollouts only
- Only successful trajectories are kept for SFT
- Failed trajectories are discarded in first version
- Trajectory collection stops after `50` successful trajectories
- SFT training format is step-wise: current state to next action
- GRPO uses real online environment interaction
- Default iterative eval is small; final reporting eval may use 500 episodes

## Recommended Approach

Three approaches were considered:

1. OpenEnv-first everywhere
2. Native BrowserGym/MiniWoB for inspection, OpenEnv for final runtime
3. Native-only runtime

Chosen approach: option 2.

Reason:

- Native inspection is simpler for understanding real task mechanics
- Final runtime still matches desired stack and deliverable story
- Debugging becomes easier because task/API issues can be separated from wrapper issues

## High-Level Architecture

### Control Surface

One Colab notebook acts as orchestrator. It installs dependencies, sets configuration, runs each project phase in order, and displays metrics and plots.

Notebook role is control and presentation only. Core logic stays in Python modules so runs are reproducible and easier to rerun.

### Project Shape

```text
browser-agent-rl/
├── notebooks/
│   └── orchestrator.ipynb
├── src/
│   ├── env/
│   ├── agents/
│   ├── data/
│   ├── train/
│   ├── eval/
│   └── analysis/
├── configs/
├── data/
├── models/
├── results/
├── requirements.txt
└── README.md
```

### Module Responsibilities

- `src/env/`
  Native inspection helpers for BrowserGym/MiniWoB and final OpenEnv adapter/runtime.
- `src/agents/`
  Prompting, generation settings, action parsing, rollout loop, and policy wrappers for base/SFT/GRPO models.
- `src/data/`
  Trajectory schema, success filtering, JSONL writing, and SFT dataset construction.
- `src/train/`
  Unsloth SFT and GRPO training entrypoints.
- `src/eval/`
  Shared evaluator and metric aggregation.
- `src/analysis/`
  Plotting, comparison tables, and report helpers.
- `configs/`
  Colab/T4-friendly runtime and training budgets.

### End-to-End Flow

```text
inspect task API
-> baseline evaluation
-> collect 50 successful rollouts
-> build step-wise SFT dataset
-> run SFT
-> evaluate SFT model
-> run GRPO online RL
-> evaluate GRPO model
-> generate analysis outputs
```

## Environment Design

### Environment Stack

Final pipeline uses OpenEnv. Before that, phase 1 inspects native BrowserGym/MiniWoB behavior for `click-checkboxes` to confirm:

- reset/step behavior
- observation schema
- action interface
- reward behavior
- episode termination behavior

This avoids guessing about wrapper-exposed fields too early.

### State Representation

Baseline minimum model input:

- task goal/instruction
- environment observation

No extra LLM call is used to summarize state.

If later exploration shows normalization is needed, any summary must be deterministic Python preprocessing based only on existing environment fields. It is not part of the initial design commitment.

### Action Representation

Canonical action format is intentionally deferred until phase 1 confirms what the chosen stack supports most cleanly in Colab.

Selection rule:

- choose smallest unambiguous action format that OpenEnv accepts reliably for `click-checkboxes`
- use same format across baseline rollouts, stored trajectories, SFT targets, GRPO policy outputs, and evaluation

Action representation is therefore a phase-1 output, not a precommitted assumption.

Phase-1 exit criterion for interface locking:

- one concrete prompt input shape is written down
- one concrete action schema is written down
- one successful `reset -> action -> step` example is recorded for final OpenEnv path

### Reward Mechanism

Use native environment reward only.

No custom reward shaping is added. For `click-checkboxes`, native reward is already meaningful because checkbox selection correctness and task completion are reflected in reward.

### Episode Definition

One episode is one task instance from `reset()` to termination.

Per step:

1. Policy receives current instruction and observation
2. Policy emits next action
3. Environment executes action
4. Environment returns next observation, reward, done flag, and metadata

Evaluation metrics:

- success rate
- average reward
- average steps per episode

### Why `click-checkboxes`

Chosen because it is:

- multi-step but still small
- easy to debug
- reward-rich enough for RL learning signal
- likely feasible on T4 with minimal infrastructure
- simple enough to support a complete educational pipeline

## Baseline Agent Design

### Model

Use `Qwen/Qwen3.5-0.8B`.

This is treated as the starting policy for:

- baseline evaluation
- successful rollout collection
- SFT initialization

### Prompting

Policy receives task goal/instruction and observation in a consistent template. Prompt/output design should bias toward short action outputs, not long explanations.

Exact action formatting remains tied to phase-1 environment findings.

### Baseline Evaluation

Baseline evaluator runs model in environment loop until episode terminates.

Default iterative evaluation uses a small episode budget. Final reporting may use `500` episodes after pipeline is stable.

Baseline outputs:

- metrics JSON
- optional per-episode debug logs for inspection

## Trajectory Collection Design

### Data Source

Trajectories come only from model-generated rollouts.

No rule-based teacher and no human demos are used in first version.

### Success Filtering

Only successful trajectories are kept.

Failed trajectories are discarded rather than used for training or analysis in first version.

### Collection Stop Rule

Collection continues until `50` successful trajectories have been stored.

This is intentionally simple for first version. Total attempted episodes are therefore variable and depend on baseline success rate.

### Stored Schema

Raw rollouts should retain enough information to support dataset building and debugging. At minimum, each step record should include:

- instruction
- observation
- action
- reward
- done

Additional metadata such as episode id, step index, raw environment info, and parsed action can be included if available.

## SFT Design

### Training Objective

Behavior cloning on successful trajectories only.

### Example Format

Use step-wise samples, not full-trajectory sequence supervision.

Each training example teaches:

- input: instruction + current observation
- target: next action

Reason:

- simpler data pipeline
- many samples from few episodes
- better fit for T4 memory limits
- matches inference loop used at runtime

### Training Stack

Use Unsloth with LoRA on `Qwen/Qwen3.5-0.8B`.

Design assumes Unsloth support for this exact model on Colab and avoids designing around unsupported extras such as vLLM-first RL.

### Output

Save SFT adapter/model artifact under `models/sft_model/` or equivalent project path.

## GRPO Design

### Starting Point

Initialize GRPO from SFT-trained model, not directly from base model.

### Interaction Mode

GRPO uses real online environment interaction through OpenEnv.

This is true RL training, not offline replay pretending to be RL.

### Reward

Optimize native environment reward only.

### Budget Strategy

Even though GRPO is online, T4 constraints require:

- small rollout batches
- short update windows
- conservative generation lengths
- low memory settings

These are implementation constraints, not changes to the core learning story.

### Output

Save RL-trained model under `models/grpo_model/` or equivalent project path.

## Evaluation Design

### Shared Evaluator

Use one shared evaluation loop for:

- baseline model
- SFT model
- GRPO model

This keeps metric computation and environment handling identical across comparisons.

### Metrics

- success rate
- average reward
- average steps

### Eval Modes

- default iterative eval: small budget for fast checks
- final eval: optional `500` episodes for final reporting

### Saved Outputs

- `results/baseline_metrics.json`
- `results/sft_metrics.json`
- `results/grpo_metrics.json`

## Analysis Design

Produce:

- comparison table for baseline vs SFT vs GRPO
- success-rate plot
- reward plot
- episode-length plot
- training-curve plot if GRPO logging supports it cleanly

Analysis notebook or script should also explain:

- what a trajectory is
- why successful trajectories are collected
- what behavior cloning is
- why SFT helps RL
- what GRPO is
- how GRPO differs from PPO at high level
- what reward is being optimized
- why RL can improve past imitation-only behavior
- limitations of this simple setup

## Testing and Verification

Before heavy training, project should support smoke checks for:

- one manual episode through native inspection path
- one manual or scripted episode through OpenEnv path
- one model-driven episode
- one trajectory save/load cycle
- one tiny SFT batch on miniature sample
- one tiny GRPO update on minimal rollout budget

These checks exist to catch interface and formatting bugs before longer Colab runs.

## Risks and Constraints

### Main Risks

- base model may succeed too rarely, making 50-success collection slow
- OpenEnv action interface may add friction compared with native task API
- GRPO on T4 may be slow even for small budgets
- action-format mismatch may dominate failures early

If the first implementation shows baseline success is too low to reach 50 successes in reasonable time, that is treated as a replanning trigger. The first fallback discussion should revisit collection budget and data-source strategy, rather than silently changing the training story mid-implementation.

### Constraint Summary

- single Colab notebook orchestration
- single GPU T4 target
- one-task first version
- minimal infrastructure
- educational clarity prioritized over benchmark performance

## Implementation Order

1. Verify native BrowserGym/MiniWoB `click-checkboxes` reset/step and inspect schema
2. Verify final OpenEnv path for same task
3. Lock canonical action representation
4. Build baseline rollout/eval loop
5. Collect 50 successful trajectories
6. Build step-wise SFT dataset
7. Run SFT
8. Evaluate SFT
9. Run GRPO online
10. Evaluate GRPO
11. Generate final analysis outputs

## Open Questions Deferred to Exploration

These are intentionally not forced before phase 1 because they depend on actual environment behavior:

- exact observation fields used in final prompt
- exact canonical action format
- exact OpenEnv adapter details for `click-checkboxes`
- exact batch sizes, sequence lengths, and runtime budgets needed on T4

They are not unknown product requirements. They are environment- and runtime-level details to resolve during implementation.
