# Browser-Agent Interpretability Article Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an artifact-only analysis pipeline that validates the completed OpenEnvRL runs, generates publication-ready behavioral/tool-use visualizations, and writes an engineering-blog report with explicit provenance and limitations.

**Architecture:** Add a focused artifact-analysis module that loads authoritative JSON/JSONL files, validates their schemas and cross-file counts, and returns typed plain-Python records. Extend the existing plotting and report helpers to consume those validated records, keeping source artifacts read-only. Add one CLI script that assembles the analysis output directory, figures, schema-validation report, and article markdown deterministically from the existing Base, SFT, and final corrected GRPO evidence.

**Tech Stack:** Python 3.11+, `json`, `pathlib`, pandas, matplotlib, pytest, Markdown, existing OpenEnvRL artifact directories.

---

## File map

- Create: `src/analysis/artifact_analysis.py` — authoritative paths, JSON/JSONL loading, schema validation, descriptive summaries, trace extraction, and provenance metadata.
- Modify: `src/analysis/plots.py` — deterministic Figure 1–7 plot/diagram renderers; no artifact loading here.
- Modify: `src/analysis/report.py` — article-ready Markdown sections, captions, evidence labels, and artifact lineage text.
- Create: `scripts/build_article_analysis.py` — CLI entrypoint that wires loading, validation, plots, and report generation.
- Modify: `tests/analysis/test_report.py` — preserve existing table coverage and add report/provenance assertions.
- Create: `tests/analysis/test_artifact_analysis.py` — loader, schema, reconciliation, trace, and failure-path tests.
- Create: `tests/analysis/test_plots.py` — figure creation and output-file assertions using temporary directories.
- Create: `tests/fixtures/analysis/final_grpo_episodes.jsonl` — minimal synthetic final-GRPO fixture with raw completion and step diagnostics.
- Create: `tests/fixtures/analysis/final_grpo_metrics.json` — matching synthetic final-GRPO aggregate fixture.
- Create: `tests/fixtures/analysis/final_grpo_manifest.json` — minimal GRPO provenance/config fixture.
- Create: `tests/fixtures/analysis/sft_metrics.json` — minimal authoritative SFT metric fixture.
- Create: `tests/fixtures/analysis/sft_trajectories.jsonl` — minimal SFT training-demonstration fixture.
- Create: `tests/fixtures/analysis/sft_manifest.json` — minimal SFT provenance/config fixture.
- Create: `tests/fixtures/analysis/base_report.md` — minimal technical-report fixture containing the Base comparison row.
- Create at runtime only: `results/article_analysis/` — figures, per-figure caption/source sidecars, `figure_manifest.json`, normalized summaries, schema-validation report, and article Markdown. Do not commit generated outputs unless explicitly requested later.

The existing modified `pytest.ini` must remain untouched. No model, checkpoint, telemetry, or source artifact directory is modified.

## Authoritative inputs

- Base comparison: `docs/notes/2026-07-27-click-checkboxes-sft-grpo-technical-report.md`, the documented Base `0.30` comparison.
- SFT metrics: `artifacts/2026-07-12-sft-full/results/sft_metrics.json`.
- SFT demonstrations: `artifacts/2026-07-12-sft-full/data/successful_trajectories.jsonl`.
- Final GRPO aggregate: `artifacts/2026-07-26-grpo-gate-r5/results/grpo_eval_step_20_corrected_50.json`.
- Final GRPO per-episode traces: `artifacts/2026-07-26-grpo-gate-r5/results/grpo_eval_step_20_corrected_50_episodes.jsonl`.
- GRPO manifest: `artifacts/2026-07-26-grpo-gate-r5/grpo_run_manifest.json`.
- SFT manifest: `artifacts/2026-07-12-sft-full/run_manifest.json`.

The invalid intermediate GRPO result is not an input to the public analysis pipeline.

### Task 1: Add artifact loading and validation

**Files:**
- Create: `src/analysis/artifact_analysis.py`
- Create: `tests/analysis/test_artifact_analysis.py`
- Create: `tests/fixtures/analysis/final_grpo_episodes.jsonl`
- Create: `tests/fixtures/analysis/final_grpo_metrics.json`
- Create: `tests/fixtures/analysis/final_grpo_manifest.json`
- Create: `tests/fixtures/analysis/sft_trajectories.jsonl`
- Create: `tests/fixtures/analysis/sft_metrics.json`
- Create: `tests/fixtures/analysis/sft_manifest.json`
- Create: `tests/fixtures/analysis/base_report.md`

- [x] **Step 1: Write failing tests for authoritative metric loading.**

Test that the loader returns Base, SFT, and GRPO records with `success_rate`, `average_reward`, `average_steps`, source path, and evidence type. Test that SFT and GRPO values match `sft_metrics.json` and `final_grpo_metrics.json` exactly, that manifest fixtures provide provenance metadata, and that SFT demonstration extraction reads `sft_trajectories.jsonl`.

- [x] **Step 2: Run the focused tests and verify failure.**

Run: `pytest tests/analysis/test_artifact_analysis.py -q`

Expected: FAIL because the new analysis loader is not implemented.

- [x] **Step 3: Implement metric and Base-report loaders.**

Add explicit path constants and small functions such as `load_stage_metrics(...)` and `load_base_metric_from_report(...)`. Parse only the documented Base row; reject missing or nonnumeric values. Return source metadata alongside values so the report can cite exact paths.

- [x] **Step 4: Add final-GRPO episode schema validation.**

Require exactly 50 records for the production input, with the fixture explicitly containing 44 successes and 6 failures, unique `episode_id`, top-level `steps`, and each step’s `instruction`, `observation`, `action`, `reward`, `done`, and optional `policy_diagnostics`. When diagnostics exist, validate `raw_completion` and `parsed_action`. Preserve raw strings and source IDs.

- [x] **Step 5: Add aggregate reconciliation and schema report generation.**

Compute episode success count, success rate, reward mean, step mean, and record/field counts. Compare computed final-GRPO values with the authoritative aggregate JSON within exact numeric equality for the saved values. Emit a structured validation result that names `steps[].policy_diagnostics.raw_completion`, `steps[].policy_diagnostics.parsed_action`, and `steps[].{action,observation,reward,done}`. Define malformed-record output as `invalid_records`, each containing source line number, episode ID when available, field path, error code, and message; valid records remain available for diagnostics only when validation is run in report/diagnostic mode.

- [x] **Step 6: Test failure paths.**

Cover missing files, malformed JSON, duplicate episode IDs, wrong episode count, missing required fields, and aggregate mismatch. Assert that errors identify the source path and field, and that diagnostic-mode validation reports malformed records in `invalid_records` rather than silently dropping them.

- [x] **Step 7: Run the focused tests and commit.**

Run: `pytest tests/analysis/test_artifact_analysis.py -q`

Expected: PASS.

Commit: `git add src/analysis/artifact_analysis.py tests/analysis/test_artifact_analysis.py tests/fixtures/analysis && git commit -m "feat: validate article analysis artifacts"`

### Task 2: Add descriptive behavior and trace extraction

**Files:**
- Modify: `src/analysis/artifact_analysis.py`
- Modify: `tests/analysis/test_artifact_analysis.py`

- [x] **Step 1: Write failing tests for final-episode summaries.**

Test 44 successes/6 failures, episode-length values, terminal reward, action strings, and the presence of raw completions using `final_grpo_episodes.jsonl` and `final_grpo_metrics.json`. Test that a representative raw completion containing multiple tool calls is preserved while the executed `steps` remain one action per environment turn.

- [x] **Step 2: Implement descriptive summaries.**

Add functions that return plain records for success/failure counts, episode-length distribution, terminal-action counts, action strings, and diagnostic availability. Keep classifications conservative; do not infer hidden model intent.

- [x] **Step 3: Implement representative trace extraction.**

Return a trace object containing instruction, first accessibility observation, each raw completion, parsed action, executed action, reward, done flag, and final success. Mark the queue as `derived_from_trace=True`; do not claim it is a persisted queue field.

- [x] **Step 4: Add SFT demonstration extraction.**

Load one or more successful SFT training trajectories from `sft_trajectories.jsonl` for Figure 3 and label them `SFT training demonstration`. Do not expose them as SFT model-evaluation traces. Preserve their original `CLICK ref=...` action format.

- [x] **Step 5: Run tests and commit.**

Run: `pytest tests/analysis/test_artifact_analysis.py -q`

Expected: PASS.

Commit: `git add src/analysis/artifact_analysis.py tests/analysis/test_artifact_analysis.py && git commit -m "feat: extract browser-agent behavior traces"`

### Task 3: Implement publication figures

**Files:**
- Modify: `src/analysis/plots.py`
- Create: `tests/analysis/test_plots.py`

- [x] **Step 1: Write failing plot tests.**

Using temporary directories and fixture records, test that each renderer creates a non-empty PNG/SVG at a requested path, does not mutate input records, and uses the expected stage labels/colors.

- [x] **Step 2: Implement Figure 1 stage progression.**

Render Base `30%`, SFT `58%`, and corrected final GRPO `88%` only. Include a caption/source metadata hook and a visible note that evaluations are separate randomized samples.

- [x] **Step 3: Implement Figure 2 tool-use timeline.**

Render instruction, accessibility observation, raw multi-call completion, derived queued-call interpretation, executed actions, and terminal reward. Clearly separate generated versus executed content.

- [x] **Step 4: Implement Figure 3 demonstration-to-evaluation cards.**

Render SFT training demonstrations beside final GRPO evaluation traces. Do not render Base/SFT model-evaluation action distributions. Label each card with its evidence type.

- [x] **Step 5: Implement Figure 4 evaluator/action-flow diagram.**

Render parse → queue → resolve → execute → reward. Add a note that the queue is runtime/derived behavior reconstructed from saved traces, not a persisted artifact field.

- [x] **Step 6: Implement Figure 5 final-episode summary.**

Render success/failure counts and episode-length/action summaries from all 50 final GRPO records. If a malformed-action category is included, derive it from an explicit rule and include the rule in the returned metadata/caption.

- [x] **Step 7: Implement Figure 6 artifact lineage.**

Render trajectories → SFT dataset → SFT model → GRPO checkpoint-20 → corrected per-episode evaluation → aggregate metrics/report, with manifests/logs as provenance sidecars.

- [x] **Step 8: Implement Figure 7 capability-profile radar.**

Add a radar-style visual inspired by the proposed reference image. The shared Base/SFT/GRPO radar has three explicitly defined axes: (1) task success = `success_rate * 100`, higher is better; (2) average reward = `average_reward * 100`, higher is better but visually duplicates task success because the task reward is binary; and (3) relative terminal-step index = min-max normalization of `average_steps` across the three stages, inverted so fewer steps plots outward, explicitly labeled as descriptive rather than an independent capability score because failed episodes can terminate early. Do not convert unsupported tool-use dimensions into invented Base/SFT scores. Format robustness, tool coverage, error recovery, and self-verification may appear only in a separate trace-backed SFT-demonstration versus GRPO-final-evaluation profile, with unavailable states where a rule cannot be supported.

- [x] **Step 9: Run plot tests and commit.**

Run: `pytest tests/analysis/test_plots.py -q`

Expected: PASS and all figure files non-empty.

Commit: `git add src/analysis/plots.py tests/analysis/test_plots.py && git commit -m "feat: render browser-agent analysis figures"`

### Task 4: Generate the article and validation report

**Files:**
- Modify: `src/analysis/report.py`
- Create: `scripts/build_article_analysis.py`
- Modify: `tests/analysis/test_report.py`

- [x] **Step 1: Write failing report tests.**

Test that generated Markdown contains the three headline stages, exact source paths, evidence labels, separate-randomized-evaluation limitation, behavioral/procedural interpretability wording, and no invalid intermediate GRPO result.

- [x] **Step 2: Implement report sections.**

Extend report helpers to write a blog-ready Markdown document with: question, pipeline overview, stage comparison, tool-use walkthrough, evaluator flow, final-episode summary, artifact lineage, reproducibility instructions, and limitations. Use captions generated from source metadata.

- [x] **Step 3: Implement the schema-validation report.**

Write `schema_validation.json` and a human-readable `schema_validation.md` under the output directory. Include source paths, record counts, required-field checks, aggregate reconciliation, diagnostic-field locations, unavailable Base/SFT per-episode evidence, and an `invalid_records` section with line number, episode ID, field path, error code, and message for any malformed records encountered in diagnostic mode.

- [x] **Step 4: Implement the CLI.**

Add `scripts/build_article_analysis.py` with arguments for `--output-dir`, `--sft-root`, `--grpo-root`, and optional `--base-report`. Default paths must point to the existing local artifacts. Refuse to overwrite model/checkpoint roots; create only the requested output directory. Make the run deterministic and fail nonzero on validation errors.

- [x] **Step 5: Add report tests and run them.**

Run: `pytest tests/analysis/test_report.py -q`

Expected: PASS. Assert that the generated article, validation JSON/Markdown, captions, normalized summaries, and figure metadata contain no reference to the invalid intermediate GRPO result and only use the explicit authoritative input allowlist. Write `figure_manifest.json` containing every generated figure path, caption, evidence type, and source path; test every manifest entry and every corresponding figure sidecar for forbidden invalid-result markers. Recursively scan all generated text, JSON, Markdown, sidecar, and figure-metadata files for forbidden source paths/labels; binary pixel contents are not treated as textual evidence.

- [x] **Step 6: Commit the report/CLI implementation.**

Commit: `git add src/analysis/report.py scripts/build_article_analysis.py tests/analysis/test_report.py && git commit -m "feat: generate interpretable browser-agent report"`

### Task 5: End-to-end artifact-only verification

**Files:**
- Modify only if verification exposes a defect in the implementation files above.

- [x] **Step 1: Run the focused analysis suite.**

Run: `pytest tests/analysis -q`

Expected: all analysis tests pass.

- [x] **Step 2: Run the CLI against the existing artifacts.**

Run: `python scripts/build_article_analysis.py --output-dir results/article_analysis`

Expected: successful completion with seven figures, article Markdown, normalized summaries, and schema-validation reports. If `--output-dir` already exists and is non-empty, the CLI must fail without modifying it; an explicit `--overwrite` flag may be added only if later approved. No model/checkpoint files change.

- [x] **Step 3: Snapshot authoritative artifacts before and after CLI execution.**

Run a reproducible SHA-256/file-list snapshot before the CLI, run the CLI, then repeat it and compare the files byte-for-byte. The snapshot must cover the SFT and GRPO model/checkpoint trees plus every authoritative source file: the two manifests, SFT metrics/trajectories, GRPO aggregate/per-episode JSONL, and Base technical-report source. Example command shape: `find <explicit-input-paths> -type f -print0 | sort -z | xargs -0 shasum -a 256 > /tmp/openenvrl-analysis-inputs.before.sha256`; repeat to `.after.sha256` and compare with `diff -u`. The comparison must be empty.

- [x] **Step 4: Inspect generated outputs.**

Check that Figure 1 reads `30%`, `58%`, `88%`; Figure 5 reconciles to 44/50; Figure 2 shows raw completion versus executed steps; and the article does not mention the invalid intermediate GRPO score.

- [x] **Step 5: Run the complete test suite.**

Run: `pytest -q tests`

Expected: existing suite plus new analysis tests pass. Do not modify `pytest.ini` as part of this work.

- [x] **Step 6: Check the worktree and commit verification metadata only if needed.**

Run: `git status --short --branch` and `git diff --check`.

Expected: only the pre-existing `pytest.ini` modification remains outside the implementation commits; generated `results/article_analysis/` remains untracked or ignored according to repository policy.

## Handoff

After implementation, report the output directory, figure filenames, validation summary, exact test results, and any unavailable evidence. Do not claim a new model result, new evaluation, or general web-agent capability.
