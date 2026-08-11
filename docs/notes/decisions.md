# Decisions

Short log. Add newest decision at top.

## 2026-08-12 — Final-GRPO reliability visual redesign

- Keep article Figures 1, 5, and 7 only; remove opaque tool-trace, process, and lineage diagrams.
- Replace Figure 5 with a 50-episode final-GRPO reliability audit: outcome, deterministic failure behavior, and steps by outcome.
- Failure categories are behavioral trace classifications, not model-intent or mechanistic claims.
- Evidence: corrected final GRPO episode JSONL only; no new training/evaluation/model loading.
- Execution: design approved; implementation pending plan and review.

## 2026-08-11 — End-to-end verification

- CLI ran against real SFT + corrected final GRPO artifacts.
- Output: seven figures, article, figure manifest, normalized summary, schema reports.
- Validation: 50 records, 44 successes, aggregate reconciled, no invalid records.
- Input integrity diff empty before/after CLI.
- Full suite: `111 passed`.
- Figure 2 widened/wrapped after visual QA; final image readable.

## 2026-08-11 — Article CLI implementation

- Added artifact-only `scripts/build_article_analysis.py`.
- Output contains article, seven figures, figure manifest, normalized summary, schema JSON/Markdown.
- CLI rejects non-empty output directory.
- Recursive output scan blocks forbidden invalid-result markers.
- No model loading, training, or evaluation in article path.

## 2026-08-11 — Figure implementation

- Added seven deterministic figures plus caption sidecars.
- Radar uses shared metrics only; reward duplication and step caveat stated.
- Figure metadata records evidence type and source paths.

## 2026-08-11 — Trace extraction implementation

- Added summaries for 50 final GRPO episodes.
- Added trace view preserving raw completion, parsed action, executed action, reward, done.
- Queue flag explicitly marked derived, not persisted.
- SFT examples remain labeled training demonstrations.

## 2026-08-11 — Artifact validation implementation

- Added `src/analysis/artifact_analysis.py`.
- Final GRPO validation requires 50 records and reconciles 44/50 aggregate result.
- Raw tool diagnostics preserved at step level.
- Malformed records use diagnostic `invalid_records`; no silent dropping.
- Source artifacts remain read-only.

## 2026-08-11 — Trace-doc design locked

- `decisions.md`: short, newest-first, dated bullets; record choices, evidence limits, execution status.
- `flow.md`: small separated Mermaid views; tables for contracts; explicit planned/current boundaries.
- Goal: human visibility + traceability. Avoid giant tangled graph.

## 2026-08-11 — Flow readability

- Dense graph replaced with separate views: entry, stages, evaluation, training, planned analysis.
- Keep diagrams small. Use tables for contracts. Prioritize traceability over node count.

## 2026-08-11 — Article analysis

- Audience: engineering/technical blog.
- Story: Base → SFT → corrected final GRPO.
- Interpretability: behavioral traces + tool-use flow. No neuron-level claims.
- Data: existing artifacts only. No new training. No new evaluation.
- Public GRPO result: corrected final `44/50 = 88%` only.
- Invalid intermediate GRPO result: exclude from public prose, charts, metadata.
- Stage comparison: Base `30%`, SFT `58%`, final GRPO `88%`.
- Trace limits: durable Base/SFT model-eval traces unavailable. Do not invent cross-stage action distributions.
- Radar: use measured axes only. Mark unsupported dimensions unavailable.
- Source of truth: final GRPO per-episode JSONL + aggregate JSON + manifests.
- Safety: analysis reads artifacts; never edits model/checkpoint/source artifacts.
- Execution: plan exists; implementation waits for explicit user approval.
