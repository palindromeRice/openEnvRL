# From Demonstrations to Reward Learning: A Small GRPO Browser-Agent Experiment

## Summary

This report documents a small, reproducible browser-agent experiment on the MiniWoB++ `click-checkboxes` task. Starting from `Qwen/Qwen3.5-0.8B`, the project first used supervised fine-tuning (SFT) on successful browser trajectories, then applied a short Group Relative Policy Optimization (GRPO) stage using the task's binary success reward.

Scored by one evaluator, 50 fresh episodes per stage, the base model succeeded **30%** of the time, the SFT model **64%**, and the GRPO checkpoint **94% (47 of 50)**. This is encouraging evidence that a short reward-learning stage can improve a small model's ability to carry out multi-click browser actions. It is not evidence of general web-agent competence: the experiment uses one MiniWoB++ task family, a small model, one GRPO training run, and different random evaluation samples per stage.

Getting comparable numbers took two evaluator fixes, described in Section 3.5. The first fix stopped the evaluator from discarding every tool call after the first. The second stopped it from rejecting element IDs written in brackets, such as `[24]`.

*Report updated 2026-10-22: all three stages were re-scored by the same evaluator. The original July numbers (30% / 58% / 88%) came from two different evaluators and are kept in Section 3.6 as history.*

## 1. Research question

Can a small instruction-following model be turned into a reliable browser tool user by combining successful demonstrations of a narrow task with a short online reward-learning stage in the real task environment?

The task was MiniWoB++ `click-checkboxes`. Each episode gives the agent an instruction such as “Select A, B and click Submit,” plus a browser accessibility-tree observation listing checkboxes and their current IDs. The agent must select every requested checkbox and then submit. It receives reward `1` only when the submitted selection is correct; otherwise it receives `0`. Episodes were limited to ten browser actions.

## 2. Experimental setup

| Item | Value |
| --- | --- |
| Base model | `Qwen/Qwen3.5-0.8B` |
| Environment | MiniWoB++ `click-checkboxes`, through BrowserGym |
| Hardware | Google Colab L4 GPU |
| Browser interface | Native BrowserGym environment and accessibility-tree observations |
| Maximum actions per episode | 10 |
| Persistent storage | Google Drive-backed run artifacts |

The implementation represented browser actions as Qwen-style tool calls and converted them into native actions such as `click('21')`. Artifacts were written to Google Drive so a Colab restart would not lose the model, logs, or episode records.

## 3. Methodology

### 3.1 Baseline

Before fine-tuning, the base model was evaluated on 50 `click-checkboxes` episodes. The comparable number, from the shared tool-call evaluator, is **30%** (Section 4).

### 3.2 Successful-trajectory collection

The project collected 50 successful trajectories from the same task family. Each trajectory contains the instruction, current browser observation, and executed action. These trajectories were transformed into 65 SFT rows: one row per observed browser action, not one row per episode.

This matters because a puzzle can require several checkbox clicks followed by Submit. The SFT dataset therefore teaches the action sequence, not only the final answer.

### 3.3 Supervised fine-tuning

The SFT stage trained a LoRA adapter on the 65 action rows, then exported a merged model for inference and GRPO initialization.

| SFT setting | Value |
| --- | --- |
| Training examples | 65 action rows from 50 successful trajectories |
| Optimizer updates | 60 |
| Per-device batch size | 1 |
| Gradient accumulation | 4 |
| Learning rate | `2e-4` |
| LoRA rank / alpha | 16 / 16 |
| Sequence length | 1024 |

The implementation used a fixed number of update steps rather than an explicit epoch count. With four accumulated examples per update, 60 updates exposed the model to roughly 240 example presentations, or about 3.7 passes through the 65-row dataset. This is approximate because the trainer shuffles data and uses update steps as its stopping rule.

Scored by the shared tool-call evaluator, the SFT model succeeded in **64%** of 50 episodes (Section 4).

### 3.4 GRPO reward-learning stage

GRPO started from the merged SFT model. For each training update, the trainer generated four candidate tool-call completions, executed them in separate live browser environments, and used the task reward to compare the candidates. A KL penalty kept the policy close to the SFT starting point.

| GRPO setting | Value |
| --- | --- |
| Starting checkpoint | Merged SFT model |
| Optimizer updates | 20 |
| GRPO generations per update | 4 |
| Approximate online attempts | 80 generated rollouts |
| Per-device batch size | 1 |
| Gradient accumulation | 4 |
| Learning rate | `1e-6` |
| KL coefficient (`beta`) | 0.04 |
| Prompt limit | 1024 tokens |
| Completion limit | 128 tokens |
| Optimizer | `adamw_8bit` |

The trainer progress display ended at epoch `0.625`. That framework counter refers to progress through its 32-prompt synthetic training dataset; it does **not** mean that GRPO ran for 0.625 updates. The actual training budget was 20 optimizer updates.

### 3.5 Evaluation correctness fix

The first post-training evaluation reported 50% success over ten episodes. It was not used as the final result because the evaluator had a real execution bug: it read only the first tool call emitted by the model. A valid output can contain several tool calls—select checkbox A, select checkbox B, then click Submit—so executing only the first call could make a correct plan look like a failure. The evaluator also could not resolve a control named in the model output to the matching visible browser control.

The evaluator was corrected to parse every tool call in order, queue them for successive browser turns, resolve exact visible control names to their current BrowserGym IDs, and turn malformed actions into safe no-ops. A corrected ten-episode sanity check then achieved 10/10 success.

A second parser bug only showed up when the base model went through this evaluator. The base model often wrote element IDs as `[24]`, copying how the observation lists them (`[24] checkbox 'A'`). The parser stripped quotes from the target but not brackets, so these well-formed clicks became no-ops. Under that strict parser the base model scored 4%, and 414 of its 454 no-op steps were bracketed IDs. The parser now accepts `[24]` as ID `24`. The SFT and GRPO models never wrote bracketed IDs, so this fix does not change their scores.

### 3.6 History: the July numbers

The first version of this report compared Base 30%, SFT 58%, and GRPO 88%. Those numbers came from two different evaluators. Base and SFT were scored by the original `CLICK ref=` action evaluator (`src/agents/policy.py`). GRPO was scored by the tool-call evaluator (`GRPOToolPolicy`). A 58-to-88 comparison therefore mixes two scoring rules.

On 2026-10-22, all three models were re-scored by `GRPOToolPolicy` with the parser fix above, in one Colab environment (L4, Python 3.13, torch 2.11, transformers 5.20, current Playwright), at 50 fresh episodes each. Section 4 reports these numbers. The July GRPO evaluation (44/50, 88%) and the re-score (47/50, 94%) are two independent 50-episode samples of the same checkpoint. They ran on different software stacks, so they are not pooled into one number.

## 4. Results

All rows use the same tool-call evaluator, the same software stack, and 50 fresh randomized episodes.

| Stage | Episodes | Successes | Success | 95% interval (Wilson) | Average actions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base model | 50 | 15 | 30% | 19%–44% | 6.58 |
| SFT model | 50 | 32 | 64% | 50%–76% | 3.76 |
| GRPO checkpoint-20 | 50 | 47 | **94%** | 84%–98% | 2.88 |

GRPO improves on SFT by +30 percentage points, and SFT improves on Base by +34. The SFT and GRPO intervals do not overlap. Each stage still used a separately sampled set of puzzles, so this is a descriptive comparison rather than a paired test.

Of the three GRPO failures, one pressed Submit before selecting every requested checkbox, and two selected a wrong checkbox and then submitted. None selected everything correctly and then failed to submit. These categories come from the saved action traces, not from guesses about what the model intended.

Average actions go down from Base to GRPO mostly because failed episodes run long. Base hit the ten-action limit in 30 of 50 episodes, SFT in 8, and GRPO in none. Successful Base episodes averaged 1.47 actions, compared with 2.69 for SFT and 2.74 for GRPO. That suggests the base model mostly solved puzzles that needed very few clicks.

## 5. Interpretation

The experiment shows a useful pattern for narrow browser tasks:

- Demonstrations taught the model the tool-call format and basic action sequence, lifting success from 30% to 64%.
- GRPO then used the environment's binary success signal to improve selection and sequencing, reaching 94% on 50 fresh puzzles after only 20 updates.
- Evaluator details decided the headline. Running only the first tool call, or rejecting bracketed IDs, moved measured success by tens of points without any change to a model.
- The model could emit a complete multi-action plan in one generation, while the evaluator executed its calls one at a time against the changing browser state.

The result is best described as a **strong single-task baseline**. It does not show that the model can generalize to arbitrary websites, unseen task types, or long-horizon web workflows.

## 6. Limitations and threats to validity

- **One task family.** Every result is on randomized variants of `click-checkboxes`; no cross-task generalization was measured.
- **Small evaluation samples.** Fifty episodes give a useful practical signal but are not enough for a precise statistical claim. The 95% Wilson interval for 47/50 is 84% to 98%.
- **Different evaluation samples by stage.** The baseline, SFT, and GRPO evaluations used fresh randomized episodes rather than a single fixed test set. Their scores are informative but not a controlled paired comparison.
- **Evaluator changes.** The initial 50% GRPO measurement and the strict-parser Base 4% were caused by evaluator bugs. Neither should be reported as model performance.
- **Single GRPO training run.** This report documents one GRPO run. It does not show how much results vary across training seeds.
- **Software stack.** The Section 4 numbers were produced on a newer Colab stack than the July training runs. They are consistent with each other, but not directly comparable to July evaluations.
- **No throughput conclusion.** GPU telemetry was saved for training and final evaluation, but this report makes no claim about cost, utilization, or production serving throughput.

## 7. Reproducibility and artifact record

All outputs are Drive-backed under the following paths when mounted in Colab as `/content/drive`:

| Artifact | Path |
| --- | --- |
| Full SFT run and 50-episode SFT result | `/content/drive/MyDrive/openEnvRL/artifacts/2026-07-12-sft-full/` |
| GRPO checkpoint and logs | `/content/drive/MyDrive/openEnvRL/artifacts/2026-07-26-grpo-gate-r5/` |
| Same-evaluator Base/SFT/GRPO results (Section 4) | `/content/drive/MyDrive/openEnvRL/artifacts/2026-10-22-same-evaluator-controls/results/` |
| July GRPO aggregate metrics | `.../results/grpo_eval_step_20_corrected_50.json` |
| July GRPO per-episode records | `.../results/grpo_eval_step_20_corrected_50_episodes.jsonl` |
| Final evaluation log | `.../logs/corrected_eval_50.log` |
| GRPO checkpoint | `.../models/grpo_model/checkpoint-20/` |

Each per-episode JSONL file stores the task instruction, browser observation, executed actions, reward, terminal status, and policy diagnostics for every evaluation episode. Per-episode files exist for all three stages in the same-evaluator results. This makes every pass and failure auditable instead of relying only on an aggregate percentage.

The core implementation is in:

- `src/train/sft.py` — SFT configuration and trainer;
- `src/train/grpo.py` — online environment, reward function, and GRPO trainer;
- `src/agents/grpo_tool_policy.py` — evaluation-time parsing and action queue;
- `scripts/run_grpo_full.py` — Drive-backed checkpoint/evaluation orchestration;
- `scripts/_colab_grpo_r5_eval.py` — July corrected GRPO evaluation launcher;
- `scripts/_colab_same_evaluator_controls.py` — same-evaluator Base/SFT/GRPO launcher (Section 4).

## 8. Suggested wording for a public post

> Starting with Qwen3.5-0.8B, I trained a browser agent for MiniWoB++ `click-checkboxes` in two stages: supervised fine-tuning on 50 successful demonstrations, followed by 20 online GRPO updates using only the task's binary success reward. Scored by one evaluator on 50 fresh randomized puzzles each, the base model succeeded 30% of the time, the SFT model 64%, and the GRPO model 94% (47/50). This is a narrow single-task result from one training run; it is not a claim of general web-agent capability.

## 9. Next steps, if the project resumes

1. Use one fixed, seeded held-out set for base, SFT, and GRPO comparisons.
2. Run at least two additional GRPO training seeds before making a stronger quantitative claim.
3. Extend evaluation to other MiniWoB++ tasks only after preserving this `click-checkboxes` baseline.
