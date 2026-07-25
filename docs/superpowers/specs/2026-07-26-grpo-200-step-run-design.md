# GRPO 200-Step Drive-Backed Run Design

## Goal

Run the next online GRPO experiment for MiniWoB `click-checkboxes` from the
durable full-SFT checkpoint, recover safely from a Colab interruption, and
measure progress at four comparable points.

## Fixed run contract

- Total GRPO training: 200 steps.
- Initial model: supplied absolute Drive path
  `/content/drive/MyDrive/openEnvRL/artifacts/2026-07-12-sft-full/models/sft_model`.
  Before training, load its tokenizer and model configuration successfully.
- New, timestamped Drive artifact root; do not reuse smoke-run output.
- At steps 50, 100, 150, and 200:
  - save a resumable GRPO checkpoint;
  - evaluate that checkpoint on 50 fresh `click-checkboxes` episodes;
  - persist one metric record under the same run root.
- Evaluation configuration matches prior SFT evaluation: native BrowserGym,
  `click-checkboxes`, at most 10 environment steps, retry count 2, and 50
  fresh episodes.
- Final comparison uses the same 50-episode metric shape as existing results:
  baseline success rate 0.30 and SFT success rate 0.58 (average steps 1.42).

## Required runner changes

The current GRPO defaults are smoke-only.  The implementation must add an
explicit run configuration consumed by `scripts/run_grpo_full.py` and
`src/train/grpo.py`, rather than leaving 50 steps and 10-step saves hard-coded.
It sets `max_steps=200`, `save_steps=50`, the absolute SFT input path, and a
new Drive-rooted GRPO output directory.  The trainer is resumed with
`trainer.train(resume_from_checkpoint=...)` when a validated checkpoint path is
supplied.

After every save boundary, runner invokes existing evaluator against that exact
checkpoint, writes `results/grpo_eval_step_<step>.json`, and records the
checkpoint path plus metrics in the run manifest.  The phase router must no
longer return its current GRPO-evaluation placeholder.

The GRPO artifact helper extends the existing Drive-root guard.  Its manifest
contains run config, SFT input path, all checkpoint paths, four evaluation
records, final model files, and final comparison metrics.  It is written only
after paths and metric records validate.

## Recovery behavior

Each checkpoint must contain model weights, optimizer state, scheduler state,
RNG state, and trainer state.  The runner accepts an explicit latest-checkpoint
resume path and validates it before passing it to the trainer.  If the runtime
stops, restart from the newest completed checkpoint, rather than restarting at
step 0.  A failure can lose at most the work since the previous 50-step
checkpoint.

## Evaluation behavior

Training steps and evaluation episodes are different quantities.  Training
updates the model for 200 steps.  Evaluation runs 50 fresh tasks without
updating the model at each 50-step boundary.  This produces four directly
comparable metrics and avoids eight separate evaluations.

## Acceptance gates

Before calling the run complete:

- checkpoints exist at 50, 100, 150, and 200;
- all four 50-episode metric records exist and contain numeric success rate,
  average reward, and average steps;
- final output is saved under the new Drive artifact root with a validated GRPO
  manifest;
- final GRPO result is compared with baseline and SFT;
- a resume test or verified interrupted-run recovery confirms the runner uses
  a supplied checkpoint rather than starting again from step 0.
