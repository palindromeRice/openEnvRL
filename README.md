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
- successful trajectory collection smoke run completed on Colab `T4` (5/5 successes)
- `successful_trajectories.jsonl` inspected and validated against the episode/step schema contracts (success flags, reward sums, done flags, parseable actions) — 0 errors
- collection scaled to the full `50` successes on Colab `T4`, validated the same way — 0 errors, avg `1.34` steps/episode, reward `1.0` across all episodes
- collection loop now checkpoints `successful_trajectories.jsonl` after every success (not just at the end), so a mid-run Colab disconnect loses at most one in-flight episode instead of the whole batch
- SFT dataset built from `successful_trajectories.jsonl`: 67 step-level rows from the 50 real episodes, 1:1 with no synthetic augmentation
- LoRA SFT trained on Colab `T4` (Qwen3.5-0.8B, unsloth, r=16, 60 steps, ~3.5 epochs)
- SFT evaluated against baseline on the same `50`-episode profile: success rate `0.30` -> `0.58`, average reward `0.30` -> `0.58`, average steps `4.36` -> `1.4`

Current baseline result:

- profile: `50` evaluation episodes
- success rate: `0.30`
- average reward: `0.30`
- average steps: `4.36`

Current SFT result:

- profile: `50` evaluation episodes
- success rate: `0.58`
- average reward: `0.58`
- average steps: `1.42`

Pending next tasks:

- increase `max_completion_length` and add tool-use instructions to system prompt for GRPO
- scale GRPO to full run
- compare baseline vs SFT vs GRPO in final analysis

GRPO smoke test status:

- Successfully executed a 50-step GRPO training pipeline on Colab `L4`.
- Switched to native `BrowserGym` for `click-checkboxes` to match the SFT environment.
- Implemented TRL `GRPOTrainer` `environment_factory` interface (`reset`, `get_reward`, arg docstrings).
- Discovered 0 tool calls during smoke test due to short `max_completion_length` (64) and lack of tool-use prompt; pipeline runs end-to-end otherwise.

SFT recovery status:

- SFT run was successfully rerun with Drive-backed persistence on Colab `L4` using `flash-attn`.
- `successful_trajectories.jsonl`, `sft_dataset.jsonl`, `sft_adapter/`, `sft_model/`, `sft_metrics.json`, and `run_manifest.json` are successfully saved on Drive.
- Metrics match earlier runs: 58% success rate, 1.42 avg steps.

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
