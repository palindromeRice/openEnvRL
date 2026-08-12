# Final GRPO Reliability Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace opaque article figures with an artifact-only final-GRPO reliability audit explaining 44 successes and six observed failure patterns across 50 episodes.

**Architecture:** Add deterministic failure classification beside existing episode parsing. Render Figure 5 as a clean 50-episode outcome strip and Figure 6 as separate failure-mode bars. Update article assembly and manifest generation to emit Figures 1, 5, 6, and 7 while retaining provenance. Existing artifacts remain read-only.

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

- [x] Test six failed production records: two `no_submit_after_correct_selection`, two `premature_submit`, two `wrong_selection_then_submit`.
- [x] Test priority and failure on unclassifiable records.
- [x] Implement `FailureClassification`, `classify_failure(record)`, and `summarize_failure_behavior(records)` in `artifact_analysis.py`.
- [x] Use only instruction labels, initial accessibility-tree mapping, executed actions, and terminal state.
- [x] Run `pytest tests/analysis/test_artifact_analysis.py -q`.

## Task 2: Replace Figure 5

- [x] Test non-empty output and metadata for outcome strip and three failure categories.
- [x] Implement `plot_episode_reliability_summary` and `plot_episode_failure_modes` in `plots.py`.
- [x] Use human labels; exclude DOM IDs, raw completions, queues, and inferred intentions.
- [x] Keep outcome and failure-mode messages separate; omit steps-by-outcome.
- [x] Caption source: corrected final GRPO episode JSONL; state deterministic behavioral classification and final-only scope.
- [x] Run `pytest tests/analysis/test_plots.py -q`.

## Task 3: Prune article outputs

- [x] Require article figure set `{figure1, figure5, figure7}`; reject Figures 2/3/4/6 in article body and manifest.
- [x] Update report helpers and CLI to render/write only approved figures.
- [x] Add `44/50`, six failures, and category counts to article text.
- [x] Preserve invalid-result exclusion and Base/SFT trace-unavailable boundary.
- [x] Run `pytest tests/analysis/test_report.py -q`.

## Task 4: Trace docs and end-to-end verification

- [x] Update `decisions.md` and `flow.md` with implementation status and classifier edge.
- [x] Run `pytest tests/analysis -q`.
- [x] Run CLI against existing artifacts in a clean temporary output directory, then regenerate `results/article_analysis/`.
- [x] Verify Figure 1 shows 30/58/88, Figure 5 shows 44/50, Figure 6 shows 2/2/2 categories, Figure 7 remains, and old figures are absent from manifest.
- [x] Run `pytest -q tests` and `git diff --check`.
- [x] Confirm only pre-existing `pytest.ini` remains unrelated.

## Commit

```bash
git add src/analysis tests/analysis scripts/build_article_analysis.py docs/notes/decisions.md docs/notes/flow.md docs/superpowers/plans/2026-08-12-final-grpo-reliability-audit.md
git commit -m "feat: add final GRPO reliability audit"
```

Generated `results/article_analysis/` remains runtime output unless explicitly requested for version control.
