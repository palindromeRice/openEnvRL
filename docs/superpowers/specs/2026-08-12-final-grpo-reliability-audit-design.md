# Final GRPO Reliability Audit Design

## Goal

Replace opaque tool-trace and process diagrams with one publication-ready
behavioral audit of the corrected final GRPO evaluation. It must help a reader
understand reliability and the observed forms of failure, using existing saved
artifacts only.

## Approved article visual set

Keep:

1. Figure 1 — Base `30%` → SFT `58%` → corrected final GRPO `88%`.
2. Figure 5 — final GRPO reliability audit, `n=50`.
3. Figure 7 — capability radar, retaining its measured-axis caveats.

Remove Figures 2, 3, 4, and 6 from article output and article narrative. They
are procedural or provenance diagrams rather than reader-useful evidence.

## Figure 5: final GRPO reliability audit

Title: `Final GRPO reliability audit — 50 held-out episodes`.

Three panels, in reading order:

1. Outcome: large `44 / 50` success result and `6 / 50` failures.
2. Observed failure behavior, using only deterministic classifications from
   saved actions and terminal state:
   - correct selections, no `Submit`: 2;
   - premature `Submit` before all requested checkboxes: 3;
   - wrong-checkbox selections then `Submit`: 1.
3. Executed steps by outcome: show successful episodes and failed episodes;
   retain 10-step no-Submit outliers visibly.

Panel 2 labels behavioral patterns, not inferred model intentions. The figure
must not expose DOM IDs, raw completions, queues, or claim internal reasoning.

## Data and classification rules

Source: `artifacts/2026-07-26-grpo-gate-r5/results/grpo_eval_step_20_corrected_50_episodes.jsonl`.

For each failed episode, parse requested checkbox labels from instruction and
the visible control mapping from the initial accessibility-tree observation.
Map executed `click('<id>')` actions to their initial labels.

Priority order:

1. `no_submit_after_correct_selection`: no executed Submit; every non-Submit
   click maps to a requested label.
2. `premature_submit`: Submit occurs and number of requested clicked labels is
   lower than requested-label count, with no wrong-label click.
3. `wrong_selection_then_submit`: Submit occurs and at least one non-Submit
   click maps to a non-requested label.

Fail loudly on a record that cannot be classified, instead of placing it in an
unexplained bucket. Episode 3, 27, 39, 40, 42, and 44 in source order provide
current expected failure classifications.

## Evidence boundary

Use “held-out final evaluation” only if existing evaluator artifacts support
that label; otherwise use “final evaluation.” This audit describes final GRPO
behavior only. Base/SFT per-episode evaluation traces are unavailable, so the
article must not use failure-type comparisons across stages.

No training, new evaluation, model loading, or source-artifact writes.

## Verification

- classification counts reconcile to 6 failed and 50 total records;
- `44/50` matches corrected aggregate metric;
- generated article includes only Figures 1, 5, and 7;
- captions name the source and behavioral-interpretation boundary;
- existing tests plus targeted classification/plot/report tests pass.
