# From Demonstrations to Reward Learning: A Small GRPO Browser-Agent Experiment

## Summary

This report documents a small, reproducible browser-agent experiment on the MiniWoB++ `click-checkboxes` task. Starting from `Qwen/Qwen3.5-0.8B`, the project first used supervised fine-tuning (SFT) on successful browser trajectories, then applied a short Group Relative Policy Optimization (GRPO) stage using the task's binary success reward.

The final GRPO checkpoint solved **44 of 50 fresh evaluation puzzles (88%)**. The earlier SFT model recorded **58% success** on its own 50-episode evaluation. This is encouraging evidence that a short reward-learning stage can improve a small model's ability to carry out multi-click browser actions. It is not evidence of general web-agent competence: the experiment uses one MiniWoB++ task family, a small model, and different random evaluation samples for the SFT and GRPO measurements.

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

Before fine-tuning, the base model was evaluated on 50 `click-checkboxes` episodes.

| Model | Evaluation episodes | Success | Average reward | Average actions |
| --- | ---: | ---: | ---: | ---: |
| Base Qwen model | 50 | 30% | 0.30 | 4.36 |

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

The saved 50-episode SFT evaluation produced:

| Model | Evaluation episodes | Success | Average reward | Average actions |
| --- | ---: | ---: | ---: | ---: |
| SFT model | 50 | 58% | 0.58 | 1.42 |

The low SFT action count should not be read as better efficiency than GRPO. Failed episodes can stop early, reducing their action count. Success rate is the more meaningful comparison here.

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

The evaluator was corrected to parse every tool call in order, queue them for successive browser turns, resolve exact visible control names to their current BrowserGym IDs, and turn malformed actions into safe no-ops. A corrected ten-episode sanity check then achieved 10/10 success. The final measurement below uses a fresh 50-episode evaluation run with this corrected evaluator.

## 4. Results

| Stage | Evaluation episodes | Success | Average reward | Average actions |
| --- | ---: | ---: | ---: | ---: |
| Base model | 50 | 30% | 0.30 | 4.36 |
| SFT model | 50 | 58% | 0.58 | 1.42 |
| GRPO model, corrected sanity check | 10 | 100% | 1.00 | 2.90 |
| GRPO model, final fresh evaluation | 50 | **88%** | **0.88** | **3.32** |

The final GRPO evaluation contains 50 individually saved episodes: 44 passed and 6 failed. This represents a +30 percentage-point improvement over the SFT measurement and a +58 percentage-point improvement over the base-model measurement. These differences are descriptive rather than a formal statistical comparison because each stage used a separately sampled set of randomized puzzles.

## 5. Interpretation

The experiment shows a useful pattern for narrow browser tasks:

- Demonstrations taught the model the tool-call format and basic action sequence, lifting success from 30% to 58%.
- GRPO then used the environment's binary success signal to improve selection and sequencing, reaching 88% on 50 fresh puzzles after only 20 updates.
- The model could emit a complete multi-action plan in one generation, while the evaluator executed its calls one at a time against the changing browser state.

The result is best described as a **strong single-task baseline**. It does not show that the model can generalize to arbitrary websites, unseen task types, or long-horizon web workflows.

## 6. Limitations and threats to validity

- **One task family.** Every result is on randomized variants of `click-checkboxes`; no cross-task generalization was measured.
- **Small evaluation samples.** Fifty episodes give a useful practical signal but are not enough for a precise statistical claim. The approximate 95% binomial confidence interval for 44/50 is 76% to 95%.
- **Different evaluation samples by stage.** The baseline, SFT, and GRPO evaluations used fresh randomized episodes rather than a single fixed test set. Their scores are informative but not a controlled paired comparison.
- **Evaluator change.** The initial 50% GRPO measurement was invalidated by an evaluator bug and must not be reported as model performance. Only the corrected 10- and 50-episode results should be used.
- **No independent rerun.** This report documents one successful GRPO run. A second seed and another held-out 50-episode evaluation would improve confidence.
- **No throughput conclusion.** GPU telemetry was saved for training and final evaluation, but this report makes no claim about cost, utilization, or production serving throughput.

## 7. Reproducibility and artifact record

All outputs are Drive-backed under the following paths when mounted in Colab as `/content/drive`:

| Artifact | Path |
| --- | --- |
| Full SFT run and 50-episode SFT result | `/content/drive/MyDrive/openEnvRL/artifacts/2026-07-12-sft-full/` |
| GRPO checkpoint and logs | `/content/drive/MyDrive/openEnvRL/artifacts/2026-07-26-grpo-gate-r5/` |
| Final aggregate GRPO metrics | `.../results/grpo_eval_step_20_corrected_50.json` |
| Final per-episode records | `.../results/grpo_eval_step_20_corrected_50_episodes.jsonl` |
| Final evaluation log | `.../logs/corrected_eval_50.log` |
| GRPO checkpoint | `.../models/grpo_model/checkpoint-20/` |

The per-episode JSONL file stores the task instruction, browser observation, executed actions, reward, terminal status, and policy diagnostics for each of the 50 final evaluation episodes. This makes every pass and failure auditable instead of relying only on an aggregate percentage.

The core implementation is in:

- `src/train/sft.py` — SFT configuration and trainer;
- `src/train/grpo.py` — online environment, reward function, and GRPO trainer;
- `src/agents/grpo_tool_policy.py` — evaluation-time parsing and action queue;
- `scripts/run_grpo_full.py` — Drive-backed checkpoint/evaluation orchestration;
- `scripts/_colab_grpo_r5_eval.py` — corrected final-evaluation launcher.

## 8. Suggested wording for a public post

> Starting with Qwen3.5-0.8B, I trained a browser agent for MiniWoB++ `click-checkboxes` in two stages: supervised fine-tuning on 50 successful demonstrations, followed by 20 online GRPO updates using only the task's binary success reward. The SFT model achieved 58% success in its 50-episode evaluation. After GRPO, the corrected evaluator measured 44/50 success (88%) on a fresh set of randomized puzzles. This is a narrow but reproducible single-task result; it is not a claim of general web-agent capability.

## 9. Next steps, if the project resumes

1. Evaluate the saved GRPO checkpoint on another independently sampled 50-episode set.
2. Use one fixed held-out set for base, SFT, and GRPO comparisons.
3. Run at least two additional seeds before making a stronger quantitative claim.
4. Extend evaluation to other MiniWoB++ tasks only after preserving this `click-checkboxes` baseline.
