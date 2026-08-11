# Final GRPO Reliability Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace opaque article figures with an artifact-only final-GRPO reliability audit explaining 44 successes and six observed failure patterns across 50 episodes.

**Architecture:** Add deterministic failure classification beside existing episode parsing. Replace Figure 5 with three evidence panels: outcome, failure taxonomy, and steps by outcome. Update article assembly and manifest generation to emit only Figures 1, 5, and 7 while retaining provenance. Existing artifacts remain read-only.

**Tech Stack:** Python 3.11+, pathlib, json, matplotlib, pytest, Markdown.

---

## File map

- Modify: `src/analysis/artifact_analysis.py` — typed failure classification and summary fields.
- Modify: `src/analysis/plots.py` — Figure 5 reliability audit renderer.
- Modify: `src/analysis/report.py` — approved figure set and reliability text.
- Modify: `scripts/build_article_analysis.py` — render approved figure set.
- Modify: `tests/analysis/test_artifact_analysis.py`, `tests/analysis/test_plots.py`, `tests/analysis/test_report.py` — focused regression coverage.
- Modify: `docs/notes/decisions.md`, `docs/notes/flow.md` — execution trace.
- Runtime only: `results/article_analysis/` — regenerated outputs.

Do not modify `pytest.ini`, source artifacts, model directories, checkpoints, or training/evaluation code.

## Task 1: Add deterministic failure classification

- [ ] Test six failed production records: two `no_submit_after_correct_selection`, three `premature_submit`, one `wrong_selection_then_submit`.
- [ ] Test priority and failure on unclassifiable records.
- [ ] Implement `FailureClassification`, `classify_failure(record)`, and `summarize_failure_behavior(records)` in `artifact_analysis.py`.
- [ ] Use only instruction labels, initial accessibility-tree mapping, executed actions, and terminal state.
- [ ] Run `pytest tests/analysis/test_artifact_analysis.py -q`.

## Task 2: Replace Figure 5

- [ ] Test non-empty output and metadata for outcome counts, three failure categories, and steps grouped by outcome.
- [ ] Implement `plot_episode_reliability_audit(summary, output_path)` in `plots.py`.
- [ ] Use human labels; exclude DOM IDs, raw completions, queues, and inferred intentions.
- [ ] Keep two ten-step failed episodes visible as outliers.
- [ ] Caption source: corrected final GRPO episode JSONL; state deterministic behavioral classification and final-only scope.
- [ ] Run `pytest tests/analysis/test_plots.py -q`.

## Task 3: Prune article outputs

- [ ] Require article figure set `{figure1, figure5, figure7}`; reject Figures 2/3/4/6 in article body and manifest.
- [ ] Update report helpers and CLI to render/write only approved figures.
- [ ] Add `44/50`, six failures, and category counts to article text.
- [ ] Preserve invalid-result exclusion and Base/SFT trace-unavailable boundary.
- [ ] Run `pytest tests/analysis/test_report.py -q`.

## Task 4: Trace docs and end-to-end verification

- [ ] Update `decisions.md` and `flow.md` with implementation status and classifier edge.
- [ ] Run `pytest tests/analysis -q`.
- [ ] Run CLI against existing artifacts in a clean temporary output directory, then regenerate `results/article_analysis/`.
- [ ] Verify Figure 1 shows 30/58/88, Figure 5 shows 44/50 and 2/3/1 categories, Figure 7 remains, and old figures are absent from manifest.
- [ ] Run `pytest -q tests` and `git diff --check`.
- [ ] Confirm only pre-existing `pytest.ini` remains unrelated.

## Commit

```bash
git add src/analysis tests/analysis scripts/build_article_analysis.py docs/notes/decisions.md docs/notes/flow.md docs/superpowers/plans/2026-08-12-final-grpo-reliability-audit.md
git commit -m "feat: add final GRPO reliability audit"
```

Generated `results/article_analysis/` remains runtime output unless explicitly requested for version control.
