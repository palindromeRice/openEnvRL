# GRPO 200-Step Run Handoff

## Current state

- Date: 2026-07-26.
- Current branch: `feature/grpo-200-step-run`.
- Working tree before this handoff: clean. This handoff file is currently
  uncommitted.
- Local verification: `pytest -q` -> `78 passed`.
- Remote/Colab verification: not yet run.

## Already completed evidence

- Full SFT artifact exists at
  `/content/drive/MyDrive/openEnvRL/artifacts/2026-07-12-sft-full`.
- Full SFT result: success rate `0.58`, average reward `0.58`, average steps
  `1.42` over 50 episodes.
- Baseline comparison point: success rate/reward `0.30`, average steps `4.36`.
- Prior GRPO smoke completed 50 training steps. Its saved trainer state showed
  non-zero tool calls and non-zero rewards at early steps. This was a smoke
  result, not final evaluation evidence.

## Implemented on this branch

Commits, oldest to newest:

- `21e58d3` fixes stale default-config test expectation.
- `d9854e4` adds configurable GRPO max/save steps, resume path validation,
  Trainer resume support, environment cleanup, and CUDA cleanup.
- `5651bd8` adds Drive GRPO output path, dynamic boundary validation, and
  `grpo_run_manifest.json`.
- `74ec6cc` adds GRPO tool-call evaluation. Qwen `<tool_call>` click output is
  converted to native `click('target')`; malformed output becomes safe `NOOP`.
- `37c5448` adds Drive-backed boundary orchestrator.

Design and plan are committed too:

- `docs/superpowers/specs/2026-07-26-grpo-200-step-run-design.md`
- `docs/superpowers/plans/2026-07-26-grpo-200-step-run.md`

## Approved run contract

Run from full SFT checkpoint for 200 GRPO steps. At steps 50, 100, 150, and
200:

1. save a resumable checkpoint;
2. run 50 fresh `click-checkboxes` evaluation episodes;
3. save `results/grpo_eval_step_<step>.json`.

Final output includes `grpo_run_manifest.json`. A boundary is complete only
when both its complete checkpoint and numeric metric JSON exist. On restart:

- complete checkpoint + valid metric -> skip;
- complete checkpoint + missing/invalid metric -> evaluate only;
- missing/invalid checkpoint -> train that boundary, then evaluate.

## Launch gate and blocker

Do not launch 200 steps first. Launch a 50-step, one-boundary Drive smoke gate
first. It must produce:

- `models/grpo_model/checkpoint-50/`;
- `results/grpo_eval_step_50.json` with numeric metrics;
- final model export;
- `grpo_run_manifest.json`.

Current blocker: local `google-colab-cli` OAuth has expired. The last live
check requested browser authorization, so active Colab-session state is unknown.
Re-authenticate before creating/using a named Colab session. Use CLI-driven
execution and Drive-backed artifacts; do not compile `flash-attn` from source.

## Intended launch inputs

- New Drive artifact root: a fresh timestamped directory under
  `/content/drive/MyDrive/openEnvRL/artifacts/`.
- SFT source:
  `/content/drive/MyDrive/openEnvRL/artifacts/2026-07-12-sft-full/models/sft_model`.
- Smoke gate: `--total-steps 50 --checkpoint-every 50 --eval-episodes 50`.
- Full run after gate: `--total-steps 200 --checkpoint-every 50 --eval-episodes 50`.

Use `scripts/run_grpo_full.py`; it accepts `--artifact-root`,
`--sft-model-path`, `--total-steps`, `--checkpoint-every`, `--eval-episodes`,
and optional `--resume-from-checkpoint`.

## Next-run GPU telemetry

Starting with the next GRPO training or evaluation launch, save timestamped GPU
telemetry under that run's Drive artifact root. Record GPU utilization, used and
total VRAM, power draw, temperature, and process memory throughout the run.
Include a short peak/average summary in the final status. The current corrected
50-episode evaluation was launched without this monitor, so no GPU-usage claim
should be made for it.
