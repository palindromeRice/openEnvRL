# GRPO Missing Task Instruction Handoff

## Decision

Stop the current GRPO run. Do not resume any checkpoint from
`2026-07-26-grpo-full-200-r1`.

The full SFT model is still valid and does not need retraining. After fixing
the GRPO environment prompt, start GRPO again from:

`/content/drive/MyDrive/openEnvRL/artifacts/2026-07-12-sft-full/models/sft_model`

## Current live state

Snapshot taken on 2026-07-26 while Colab session `grpo` was still busy.

- Branch: `feature/grpo-200-step-run`
- Broken run root:
  `/content/drive/MyDrive/openEnvRL/artifacts/2026-07-26-grpo-full-200-r1`
- Saved checkpoints: `checkpoint-50`, `checkpoint-100`
- Step-50 evaluation: success `0.00`, reward `0.00`, average steps `10.0`
- Step-100 evaluation: success `0.00`, reward `0.00`, average steps `10.0`
- Baseline reference: success/reward `0.30`
- Valid SFT reference: success/reward `0.58`
- GPU telemetry is under the broken run's `telemetry/` directory.

The run may have progressed beyond step 100 after this snapshot. Any later
checkpoint from this artifact root remains scientifically invalid because it
uses the same missing-instruction training contract.

To stop the current VM:

```bash
colab stop -s grpo
```

## Confirmed root cause

GRPO training does not show the model the dynamic task instruction.

`NativeMiniWoBClickCheckboxesEnv.reset()` returns a state with separate fields:

```python
{
    "instruction": "Select X, Y and click Submit.",
    "observation": "RootWebArea ... checkbox tree ..."
}
```

But `BrowserGymToolEnv.reset()` in `src/train/grpo.py` currently returns only:

```python
return self.last_observation.get("observation", "Environment ready.")
```

TRL appends the string returned by environment `reset()` to the model's user
message. The static dataset prompt only says to solve a generic
`click-checkboxes` task. Therefore the training model sees the checkbox tree
but never sees which checkbox labels it must select.

This was verified against the TRL `GRPOTrainer` environment contract:
`reset()` may return the initial observation string appended to the user
message.

## Evidence

Step-50 evaluation contains 50 episodes and 500 attempted actions:

- 490 tool targets were the text `Submit`;
- 10 were malformed `[33]button'Submit'`;
- 10 were malformed `[30]checkbox'Lx9'`;
- zero actions contained a valid numeric BrowserGym element ID.

The safe evaluator converted these invalid targets to `NOOP`, so every episode
ran for 10 steps with zero reward.

Training logs prove the environment and reward channel sometimes worked:
there were nonzero tool-call frequencies and a few positive reward batches.
Those successes were chance exploration without the task instruction. Most
batches had identical zero rewards, zero reward variance, zero loss, and zero
gradient. With `num_generations=2`, useful within-group reward contrast was
rare.

## Required fix

Use TDD. First add a failing test proving `BrowserGymToolEnv.reset()` includes
both the dynamic instruction and observation. Then change it to return a
stable prompt such as:

```python
state = self.env.reset()
self.last_observation = state
return (
    f"Task: {state['instruction']}\n"
    f"Observation: {state['observation']}"
)
```

Also verify every subsequent `click()` tool result still returns the updated
observation. The task instruction does not need to be repeated after every
click if it remains in conversation history.

Keep evaluator parsing strict:

- numeric targets such as `click('36')` are valid;
- malformed targets become `NOOP`;
- text `Submit` is not a BrowserGym element ID and must not be treated as a
  successful action.

Do not weaken the parser merely to make the zero score disappear.

## Validation sequence

1. Add the missing-instruction regression test.
2. Apply the minimal `reset()` fix.
3. Run targeted GRPO/environment tests.
4. Run full local suite. Last known state before this fix: `86 passed`.
5. Launch a fresh 10- or 20-step GRPO smoke from the existing SFT model under a
   new Drive artifact root.
6. Inspect training logs and raw tool calls. Gate requires:
   - dynamic task text visible in the initial model context;
   - tool-call frequency greater than zero;
   - valid numeric click targets;
   - at least one nonzero environment reward;
   - checkpoint and GPU telemetry saved.
7. Run a small checkpoint evaluation and inspect raw episode traces.
8. Only after the gate passes, launch a new 200-step run from SFT with
   checkpoints and 50-episode evaluations at steps 50, 100, 150, and 200.

Do not resume `checkpoint-50` or `checkpoint-100` from the broken run. GRPO
must restart from the valid SFT model because those checkpoints learned from
incomplete observations.

## Working tree warning

The branch has uncommitted changes. Preserve them; do not reset or overwrite
the working tree. At snapshot time the modified files included:

- `requirements.txt`
- `src/agents/grpo_tool_policy.py`
- `src/agents/rollout.py`
- `src/env/native_browsergym_task.py`
- `src/eval/run_eval.py`
- `src/pipeline/run_all.py`
- `src/train/grpo.py`
- related tests

The changes include evaluator diagnostics, state-preserving `NOOP`, safe
numeric tool-target parsing, removal of incompatible FlashAttention
installation, and GRPO cleanup fixes. Review the live `git diff` before
editing.

Do not compile FlashAttention on Colab. Use native PyTorch SDPA unless an exact
prebuilt wheel matching Python, PyTorch, and CUDA is available.
