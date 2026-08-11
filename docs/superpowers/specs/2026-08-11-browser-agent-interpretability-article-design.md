# Browser-Agent Training Interpretability Article Design

## Goal

Create an engineering/technical blog post that makes the completed `click-checkboxes` browser-agent run inspectable without training new models or collecting new experiments. The article should explain both the behavioral progression from Base to SFT to GRPO and the concrete tool-use behavior that produces each episode outcome.

The central claim is behavioral and procedural, not neuron-level interpretability:

> The improvement is observable as a change in generated tool calls, executed browser transitions, and terminal rewards, and each public claim can be traced to a saved artifact.

## Scope and non-goals

In scope:

- Analyze existing local/Drive-backed artifact copies only.
- Build publication-ready figures and a concise technical narrative.
- Use the Base, full SFT, and corrected final GRPO results as the public comparison.
- Explain the evaluator/action queue that turns generated calls into browser actions.
- Produce reproducibility and provenance notes for every figure.

Out of scope:

- Any new training, fine-tuning, GRPO updates, or model modification.
- New evaluation episodes or additional Colab sessions.
- Claims about general web-agent competence.
- Neuron-level, attention-level, or causal mechanistic interpretability.
- Public discussion or charting of the invalid intermediate GRPO result.

## Article spine

1. Introduce the question: can a small model learn a narrow browser task?
2. Show the behavior journey: Base `30%` → SFT `58%` → corrected final GRPO `88%`.
3. Expand one representative GRPO episode from instruction and accessibility-tree observation through queued tool calls, browser transitions, and terminal reward.
4. Compare behavioral patterns across stages using aggregate evidence plus clearly labeled qualitative traces.
5. Explain the evaluator/action-flow contract: parse all calls, queue them, resolve visible controls to current IDs, execute sequentially, and record reward.
6. Close with the artifact lineage and the limits of the result.

## Visual set

### Figure 1: Stage progression

A simple three-stage bar or dot chart:

| Stage | Success | Source |
| --- | ---: | --- |
| Base | 30% | saved comparison result / report record |
| SFT | 58% | `artifacts/2026-07-12-sft-full/results/sft_metrics.json` |
| Corrected final GRPO | 88% | `artifacts/2026-07-26-grpo-gate-r5/results/grpo_eval_step_20_corrected_50.json` |

The caption must state that the stages use separate randomized evaluations, so the chart is descriptive rather than a paired statistical comparison.

### Figure 2: Tool-use episode timeline

Render one representative final GRPO episode as a horizontal sequence:

`instruction → accessibility tree → raw multi-call completion → queued calls → browser state transitions → Submit → reward 1`

The visual should distinguish generated calls from executed calls. The source is `artifacts/2026-07-26-grpo-gate-r5/results/grpo_eval_step_20_corrected_50_episodes.jsonl`. Each step stores `steps[].policy_diagnostics.raw_completion` and `steps[].policy_diagnostics.parsed_action`, while `steps[].action`, `steps[].observation`, `steps[].reward`, and `steps[].done` show execution. The queued-call sequence is reconstructed from repeated raw completions and sequential parsed actions; it is a derived interpretation, not a directly persisted queue field.

### Figure 3: Demonstration-to-evaluation behavior cards

Use a small number of representative SFT training demonstrations alongside final GRPO evaluation traces. Durable Base and SFT model-evaluation episode traces are not available in the repository, so do not present Base/SFT examples as model-behavior evaluation traces or claim cross-stage action distributions. Show the action schema and termination behavior, and label each card as either `SFT training demonstration` or `GRPO final evaluation trace`. The SFT trace format (`CLICK ref=...`) and GRPO format (`click('...')`) should be shown honestly rather than normalized in a way that hides the difference.

### Figure 4: Evaluator/action-flow diagram

Show the procedural path:

`model completion → parse every tool call → queue remaining calls → resolve visible names/IDs → execute one action per environment turn → collect reward and terminal state`

This figure explains what tool-use success means operationally and why raw generation alone is not the same as environment behavior. The queue is runtime evaluator behavior reconstructed from the saved traces, not a separately persisted artifact field.

### Figure 5: Final-episode behavior summary

Derive from all 50 final GRPO episode records:

- 44 successes and 6 failures;
- episode-length distribution;
- action-type/termination counts;
- malformed or rejected action diagnostics, if present in the records;
- success versus failure trace patterns.

Any category requiring heuristic classification must include the rule in the caption or accompanying methods note.

### Figure 6: Artifact lineage

Use a compact lineage diagram:

`successful trajectories → SFT dataset → SFT model → GRPO checkpoint-20 → corrected per-episode evaluation → aggregate metrics/report`

Include manifests and logs as provenance sidecars, not as additional experimental stages. The SFT metric is authoritative from `artifacts/2026-07-12-sft-full/results/sft_metrics.json`, but its corresponding per-episode evaluation artifact is not available locally. The Base `30%` comparison is authoritative from the results table in `docs/notes/2026-07-27-click-checkboxes-sft-grpo-technical-report.md`; it has no durable per-episode trace source in this checkout.

### Figure 7: Capability-profile radar

Add a radar-style visual for the Base/SFT/GRPO comparison, but plot only shared measured dimensions supported by the authoritative artifacts, such as success rate and average reward. Do not turn unsupported dimensions like format robustness, tool coverage, error recovery, or self-verification into invented Base/SFT scores. Those dimensions may appear as an explicitly unavailable state or in a separate trace-backed SFT-demonstration versus GRPO-final-evaluation profile. Average steps must carry the existing caveat that failed episodes can terminate early; it should not be labeled as an uncomplicated efficiency score.

## Evidence and claim rules

- Aggregate metrics support the headline comparison only: Base `0.30`, SFT `0.58`, final corrected GRPO `0.88`.
- The final GRPO per-episode JSONL at `artifacts/2026-07-26-grpo-gate-r5/results/grpo_eval_step_20_corrected_50_episodes.jsonl` is the source for quantitative behavior visualizations.
- The full SFT artifact supplies trajectories, the 65-row dataset, merged model, metrics, and manifest.
- The baseline is used as a comparison value from `docs/notes/2026-07-27-click-checkboxes-sft-grpo-technical-report.md`; do not invent per-episode baseline details.
- The SFT value is used from `artifacts/2026-07-12-sft-full/results/sft_metrics.json`; do not invent per-episode SFT evaluation details.
- Selected traces are qualitative illustrations, not statistical evidence.
- The invalid intermediate GRPO result is excluded from all public prose, tables, figures, and narrative comparisons.
- Claims must distinguish generated output, parsed action, executed action, and environment reward.
- The article must state that separate randomized evaluations prevent paired per-puzzle behavioral comparisons across stages.
- Use “behavioral trace interpretability” or “procedural transparency,” not mechanistic interpretability.
- State the single-task scope, small sample size, separate randomized evaluations, and lack of independent rerun.

## Data flow

1. Read existing JSON/JSONL artifacts and manifests.
2. Validate required fields and identify the authoritative metric files.
3. Normalize records into analysis tables without changing source values.
4. Compute only descriptive summaries and explicitly documented classifications.
5. Generate figures with captions containing source paths and evaluation boundaries.
6. Assemble article-ready markdown and a reproducibility appendix.

No stage writes to or modifies model/checkpoint directories. Analysis outputs should live under a separate report/figure output directory.

## Error handling and integrity checks

- Fail if an authoritative metric file is missing or has nonnumeric required fields.
- Fail if the final per-episode count is not 50 or if aggregate success does not agree with the episode records.
- Produce a schema-validation report listing record counts, required fields, and the exact locations of raw completions and policy diagnostics.
- Preserve source episode IDs and never silently drop malformed records; report them separately.
- Keep raw action strings alongside normalized action categories.
- Mark unavailable baseline fields as unavailable rather than reconstructing them from averages.
- Exclude diagnostic/smoke artifacts from final-result figures unless explicitly labeled as such.

## Verification

Before publication, verify:

- Figure 1 values equal the authoritative metric files/report record.
- Figure 5 counts reconcile with the 50 final GRPO JSONL records.
- At least one Figure 2 trace can be followed from `raw_completion` through each executed step to terminal reward.
- Figure 3 distinguishes SFT training demonstrations from GRPO evaluation traces and makes no Base/SFT action-distribution claim.
- The schema-validation report names `steps[].policy_diagnostics.raw_completion`, `steps[].policy_diagnostics.parsed_action`, and `steps[].{action,observation,reward,done}`.
- The Base comparison cites the technical-report path and the SFT comparison cites the SFT metric JSON path.
- Every figure has a source path and a scope/provenance caption.
- The article contains no new-training claim and no mention of the invalid intermediate GRPO score in public-facing content.
- Existing unrelated worktree changes, including the modified `pytest.ini`, remain untouched.
