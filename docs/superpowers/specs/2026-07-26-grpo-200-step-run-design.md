# GRPO 200-Step Drive-Backed Run Design

## Goal

Run the next online GRPO experiment for MiniWoB `click-checkboxes` from the
durable full-SFT checkpoint, recover safely from a Colab interruption, and
measure progress at four comparable points.

## Fixed run contract

- Total GRPO training: 200 steps.
- Initial model: `artifacts/2026-07-12-sft-full/models/sft_model`.
- New, timestamped Drive artifact root; do not reuse smoke-run output.
- At steps 50, 100, 150, and 200:
  - save a resumable GRPO checkpoint;
  - evaluate that checkpoint on 50 fresh `click-checkboxes` episodes;
  - persist metrics under the same run root.
- Final comparison uses the same 50-episode metric shape as existing results:
  baseline success rate 0.30 and SFT success rate 0.58.

## Recovery behavior

Each checkpoint must contain model weights, optimizer state, scheduler state,
RNG state, and trainer state.  The runner accepts an explicit latest-checkpoint
resume path.  If the runtime stops, restart from the newest completed
checkpoint, rather than restarting at step 0.  A failure can lose at most the
work since the previous 50-step checkpoint.

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
- final output is saved under the new Drive artifact root;
- final GRPO result is compared with baseline and SFT;
- a resume test or verified interrupted-run recovery confirms the runner uses
  a supplied checkpoint rather than starting again from step 0.
