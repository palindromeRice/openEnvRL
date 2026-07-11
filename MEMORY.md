i t# Session Memory & Key Learnings

## Technical Context
- SFT pipeline is fully verified. We successfully trained a LoRA adapter on 50 episodes of `miniwob click-checkboxes` using Qwen3.5-0.8B on a Colab L4 GPU.
- Success rate improved from 30% to 58%, and average steps dropped from 4.36 to 1.42.
- Artifacts are persistently stored on Google Drive at `artifacts/2026-07-12-sft-full/`.

## Infrastructure Notes
- **Fast-Path Kernels**: Training and rollout generation is significantly faster when `flash-attn` and `causal-conv1d` are installed on the Colab instance. They must be included in `requirements.txt`.
- **Colab Execution**: `colab console` handles persistent training scripts well, but running secondary commands via `colab exec` while the console is active can disrupt the connection and crash the session. 
- **Drive Mounting**: Automated mounting via `pexpect` needs high timeouts because user auth takes time. If previously authenticated, it resolves instantly.

## Next Steps for New Session
1. **GRPO Execution**: The `grpo_train` phase is defined in `src/pipeline/run_all.py` but currently only returns a config. It needs to be updated to actually execute `run_grpo_training()` from `src/train/grpo.py`.
2. **GRPO Script**: We need to create a dedicated runner (like `scripts/run_grpo_full.py`) or use `run_phase.py` to trigger the RL phase using the newly recovered SFT model.
3. Compare the baseline, SFT, and GRPO models.
