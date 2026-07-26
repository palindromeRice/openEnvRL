# GRPO KL Anchor Handoff

Supersedes `2026-07-26-grpo-missing-task-instruction-handoff.md`. That handoff's
required fix is **done and verified on GPU**. This note covers what came next.

## Status

- Branch: `feature/grpo-200-step-run`, HEAD `37c5448`, working tree dirty.
- Local suite: `pytest -q` -> **91 passed**.
- Colab session `grpo` (L4) was live at handoff time. Drive mounted.
- **Do not launch the 200-step run yet.** One gate rerun is required first.

## What is confirmed fixed

The missing-task-instruction bug is closed. `BrowserGymToolEnv.reset()` in
`src/train/grpo.py` now returns the instruction and observation together, in the
same shape `GRPOToolPolicy.build_prompt` uses. Verified live on the VM:

```
Task: Select qxpL, k5I, iyjzCt2 and click Submit.
Observation: RootWebArea 'Click Checkboxes Task'
[18] checkbox 'qxpL'
[21] checkbox 'k5I'
[24] checkbox 'iyjzCt2'
[27] checkbox '9F'
[30] button 'Submit'
```

Gate run `artifacts/2026-07-26-grpo-gate-r2` (20 steps, 10 eval episodes)
produced real learning signal where the broken run had none:

| metric | broken run | gate r2 |
| --- | --- | --- |
| `tools/call_frequency` | ~0, targets were text `Submit` | 0.75 -> 4.0 |
| `reward` | 0.00 every step | nonzero in 18/20 steps, max 2.0 |
| `reward_std` | 0.0 | mean 0.57 |
| `grad_norm` | 0.0 | 19-64 |
| `loss` | 0.0 | +/- 0.11-0.43 |

## The open problem

GRPO **degrades** the SFT policy. Measured with the *same* evaluator on both
sides (this control did not exist before; the old 0.58 SFT number came from
`src/agents/policy.py`, a different prompt format, and is not comparable):

| model | evaluator | success | avg steps |
| --- | --- | --- | --- |
| SFT | `GRPOToolPolicy` | **0.40** | **3.3** |
| GRPO checkpoint-20 | `GRPOToolPolicy` | **0.20** | **8.2** |

Both n=10, so the success delta alone is weak (4 successes vs 2). The avg-steps
more than doubling corroborates the direction. Treat "GRPO is currently harmful"
as the working hypothesis, not settled fact.

Evidence for the cause, from
`artifacts/2026-07-26-grpo-gate-r2/models/grpo_model/checkpoint-20/trainer_state.json`:

1. **No KL anchor.** `log_history` has no `kl` key. TRL logs it only when
   `beta > 0`, and TRL's default is `0.0`. Nothing held the policy near SFT.
2. **Entropy collapse.** 0.387 at step 1 -> 0.0185 by step 4. ~20x in 4 steps.
3. **Large gradients.** `grad_norm` 19-64 at `learning_rate=5e-6`.
4. **Full finetune of an LoRA-trained model.** `build_grpo_trainer` passes no
   `peft_config`, so all ~0.8B weights move. SFT used LoRA (`src/train/sft.py:96`).

Training reward rose (mean 1.35) while eval fell — consistent with drift toward
the rollout distribution rather than genuine improvement.

## Option A, applied (needs GPU verification)

In `build_grpo_job_config`:

- `beta: 0.04` — KL anchor to the reference policy.
- `learning_rate: 5e-6 -> 1e-6`.

Plus a real bug found while wiring that up: `max_prompt_length` and `beta` were
present in the config dict but **never forwarded to `GRPOConfig`**. TRL silently
used its own defaults, so the earlier `max_prompt_length: 1024` change never took
effect — gate r2 actually ran at 512. Both are now passed through, covered by
`test_build_grpo_trainer_forwards_prompt_length_and_kl`.

**Option A is unverified on GPU.** It is a hypothesis about the regression, not a
demonstrated fix.

## Next step

Rerun the gate and compare against the 0.40 control.

The Colab session from this session is gone (VMs expire). Start a new one:

```bash
colab new --gpu L4 -s grpo
colab drivemount -s grpo          # prompts for browser OAuth, needs a human
colab exec -s grpo -f scripts/_colab_grpo_gate.py --timeout 7200   # background
```

`ARTIFACT_ROOT`, `TOTAL_STEPS`, `CHECKPOINT_EVERY` and `EVAL_EPISODES` are plain
constants at the top of `scripts/_colab_grpo_gate.py`. **Edit them in the file.**
They are not env vars: `colab exec` runs the file on the VM, so a local
`GATE_TOTAL_STEPS=...` never reaches it and would silently no-op.

`ARTIFACT_ROOT` is currently `...-grpo-gate-r3` and **must be bumped to a fresh
directory every run**. `run_boundaries()` skips any boundary that already has a
complete checkpoint plus a valid metric — pointing at a finished root (such as
`gate-r2`) trains nothing, rewrites the manifest, and still prints
`GATE_RESULT: PASS`.

Pass criteria, in order of importance:

1. `kl` key present in `trainer_state.json` `log_history` — proves `beta` landed.
2. Entropy does **not** collapse to ~0.02 within a few steps.
3. Checkpoint eval **>= 0.40**, i.e. at least not worse than SFT.
4. Tool-call frequency > 0 and nonzero rewards, as in gate r2.

If eval is still below 0.40, the next lever is **LoRA for GRPO** (pass a
`peft_config` to `GRPOTrainer` so SFT's weights cannot be overwritten). That was
option B and is untried.

Before trusting any 0.40-vs-X comparison at n=10, re-measure both sides at 50
episodes. Raise `EPISODES` at the top of `scripts/_colab_sft_control_eval.py`
(again a constant, not an env var) and `EVAL_EPISODES` in the gate script.

## Throughput notes (measured, do not re-derive)

- VRAM **peak 12,810 MiB of 23,034** on the L4. ~10 GB spare.
- GPU util **avg 16%, peak 68%** — generation is serial-decode-bound. Bigger
  training batches buy nothing; the backward pass is ~2s of a ~26s step.
- **26.5 s/step.** 200 steps ~= 88 min training, plus ~60-80 min for 4x
  50-episode evals.
- `completions/mean_length` 58.9, max 103, against the 128 cap. Cap is
  right-sized; it never truncated a well-formed tool call. Do **not** drop to 64
  (tried previously, truncated real calls) and do not restore 512 (that is what
  made the broken run and its eval crawl).
- `frac_reward_zero_std` averaged **0.45** — 45% of groups still produce
  identical rewards and contribute no gradient. `num_generations=8` would reduce
  this; costs ~2x wall clock, memory is not the constraint.

### vLLM: do not enable

Checked. vLLM 0.17+ nominally supports Qwen3.5, but
[vllm#38643](https://github.com/vllm-project/vllm/issues/38643) is **open** and
hits exactly our arch (`Qwen3_5ForConditionalGeneration` with `linear_attn`,
18 of 24 layers linear attention): it produces **gibberish output**, with no fix
PR. `--enforce-eager` hangs instead. Gibberish rollouts in GRPO look identical to
the missing-instruction bug — zero parseable tool calls, zero reward — so
enabling it reintroduces an indistinguishable silent-corruption mode.
[vllm#38041](https://github.com/vllm-project/vllm/issues/38041) (V2 runner crash
on mixed attention) is avoidable: never set `VLLM_USE_V2_MODEL_RUNNER=1`.

If someone wants to revisit: load the SFT checkpoint in vLLM with
`--max-model-len 2048` on the default V1 runner, generate on ~20 fixed prompts,
diff against HF greedy `generate()`. ~5 minutes, and it settles it. Do this only
when the correctness question above is closed, so a failure stays attributable.

Untried and cheap: `flash-linear-attention` (pure Triton, pip-installable, no
source build). The training log shows `The fast path is not available ... Falling
back to torch implementation`, so 18 of 24 layers run on the slow path. Do
**not** add `causal-conv1d` — it compiles from source, which `AGENTS.md` forbids.

## Colab harness gotchas (cost 3 failed runs; do not rediscover)

1. **`colab exec` returns exit code 0 even when the kernel cell raises.** Never
   trust it. `scripts/_colab_grpo_gate.py` prints `GATE_RESULT: PASS|FAIL` and
   `_colab_sft_control_eval.py` prints `CONTROL_RESULT:`. Grep for those.
2. **Playwright's sync API cannot start inside `colab exec`'s asyncio loop**
   (`Error: It looks like you are using Playwright Sync API inside the asyncio
   loop`). Anything touching BrowserGym must run in a child process. Both scripts
   do this. `configs/default.yaml` also has a `use_worker_thread` flag and
   `src/env/native_browsergym_task.py` has a `_worker` path — an alternative, untried.
3. **A subprocess's stdout does not reach you.** IPython captures the kernel's
   stdout, not a child's raw fd, so `subprocess.run()` silently swallows child
   tracebacks. `run_logged()` in the gate script pipes it through Python and tees
   to `<artifact_root>/logs/{probe,train}.log` on Drive.
4. **Drive -> VM sync lags ~30-60s** after a local edit. Poll for a marker string
   in the file on the VM before launching, or you will run stale code.
5. Launch long jobs with `run_in_background: true` per `AGENTS.md`.

## Files changed, uncommitted

Nothing is committed. Per user practice, get the commit message approved before
committing, and use no AI co-author trailer (`AGENTS.md`).

Core fixes:

- `src/train/grpo.py` — `reset()` returns instruction+observation; `beta`,
  `learning_rate`, `max_prompt_length`, `max_completion_length`, `optim`,
  `num_generations`, batch/accum; forwards `max_prompt_length` and `beta` to
  `GRPOConfig`.
- `src/pipeline/run_all.py` — `run_grpo_eval_phase(config, checkpoint_step,
  model_path=None)`; `model_path` scores an arbitrary model under the tool-call
  evaluator, which is how the SFT control is measured.
- `scripts/run_grpo_full.py` — `--miniwob-root` flag (default
  `/content/browsergym-assets`; Drive FUSE is slow for miniwob's many small
  files); records `config["grpo_training_args"]` so the manifest states the real
  hyperparameters instead of the stale `configs/default.yaml` values.
- `src/agents/grpo_tool_policy.py` — eval `max_new_tokens` 512 -> 128, matching
  training.
- `requirements.txt` — `bitsandbytes>=0.43.0` for `optim="adamw_8bit"` (SFT got
  it transitively via unsloth; GRPO does not use unsloth).

New scripts:

- `scripts/_colab_grpo_gate.py` — deps, prompt sanity assert, GPU telemetry, gate
  run, VRAM/util summary.
- `scripts/_colab_sft_control_eval.py` — SFT under the tool-call evaluator.

Also modified earlier on this branch, not by this work: `src/agents/rollout.py`,
`src/env/native_browsergym_task.py`, `src/eval/run_eval.py` and their tests
(evaluator diagnostics, state-preserving `NOOP`, safe numeric tool-target
parsing). Review the live `git diff` before editing.

## Do not weaken the evaluator

Malformed targets must stay `NOOP`. `click('36')` is valid; the text `Submit` is
not a BrowserGym element ID and must never count as a successful action. If a
score looks bad, that is data, not a parser bug to paper over.
