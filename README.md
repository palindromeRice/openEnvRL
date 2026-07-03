# Browser Agent RL

Colab-CLI-first educational project for browser-agent training on MiniWoB `click-checkboxes` using `Qwen/Qwen3.5-0.8B`, BrowserGym/MiniWoB locally in Colab, plus OpenEnv-oriented wrappers for later GRPO integration.

Primary execution path:

- local machine uses `google-colab-cli`
- remote Colab runtime executes `scripts/run_phase.py`
- notebook is optional and mainly for inspection or plots

## Project Status

Current scope:

- single-task pipeline for MiniWoB `click-checkboxes`
- single-GPU Google Colab execution on `T4`
- Python scripts as primary runtime, with one thin orchestrator notebook

Completed so far:

- Colab CLI execution flow is wired
- native BrowserGym + MiniWoB manual smoke path works
- baseline evaluation path works on Colab `T4`
- action parser now accepts bracketed model outputs like `CLICK [21]`

Current baseline result:

- profile: `50` evaluation episodes
- success rate: `0.30`
- average reward: `0.30`
- average steps: `4.36`

Pending next tasks:

- run successful trajectory collection smoke on Colab
- inspect and validate `successful_trajectories.jsonl`
- scale collection beyond smoke run
- train LoRA SFT model on successful trajectories
- evaluate SFT against baseline
- run GRPO starting from SFT model
- compare baseline vs SFT vs GRPO in final analysis

Setup approach:

- local machine uses Colab CLI to create and manage remote runtimes
- remote Colab runtime installs project dependencies and browser support
- BrowserGym + MiniWoB provide the real `click-checkboxes` task environment
- OpenEnv remains in project as comparison path and later RL integration path
- Unsloth is planned for SFT and GRPO on the same single-GPU Colab workflow

Planned phases:

1. Native MiniWoB/BrowserGym exploration
2. Optional OpenEnv probe/debugging
3. Manual click-checkboxes smoke test
4. Baseline evaluation
5. Successful trajectory collection
6. SFT training
7. GRPO training
8. Final evaluation and analysis

Phase-1 probe should capture:

- raw observation keys
- instruction field
- fields payload
- one sample action attempt result

Phase-1 exit checklist:

- one concrete prompt input shape is written down
- one concrete action schema is written down
- one successful BrowserGym MiniWoB `reset -> action -> step` example is recorded

Project entrypoints:

- primary runtime entrypoint is `scripts/run_phase.py`
- optional notebook entrypoint is `notebooks/orchestrator.ipynb`
- notebook is meant for orchestration, inspection, plotting, and manual debugging
- Python scripts remain source of truth for execution flow

Runtime notes:

- run the manual smoke phase before baseline; if it fails, the real `click-checkboxes` task wiring is still wrong
- baseline/data collection use local BrowserGym MiniWoB for the real `click-checkboxes` task
- OpenEnv HF Space can silently fall back to its default demo task, so it is probe-only unless task binding is verified
- BrowserGym actions are sent as natural-language tool strings like `click('12')`
- GRPO integration is prepared around OpenEnv tool-style environment usage, not a fake offline loop
