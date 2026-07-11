# Drive-Backed SFT Rerun Design

## Goal

Rerun successful-trajectory collection and LoRA SFT on Colab while writing every recoverable artifact directly to a timestamped Google Drive run directory.

## Scope

This work covers the SFT recovery path only:

1. collect 50 successful `click-checkboxes` episodes;
2. build the step-wise SFT dataset;
3. train and save the LoRA-derived SFT model;
4. evaluate the SFT model; and
5. persist a manifest that identifies the exact run.

GRPO implementation is explicitly out of scope until this run has durable artifacts and its SFT checkpoint is revalidated.

## Storage Layout

Colab executes source code from `/content/openEnvRL`. Artifacts are written directly to Drive using a user-supplied run root, for example:

```text
/content/drive/MyDrive/openEnvRL/artifacts/2026-07-11-sft-rerun/
  data/
    successful_trajectories.jsonl
    sft_dataset.jsonl
  models/
    sft_model/                 # mandatory merged, evaluable model
    sft_adapter/               # mandatory LoRA adapter
  results/
    sft_metrics.json
  run_manifest.json
```

The source checkout must not be used as an artifact store. A new timestamped run root is used for each rerun, preventing accidental overwrite of prior evidence.

An accepted run root must resolve under `/content/drive/MyDrive/openEnvRL/artifacts/` and have a non-empty run-id directory beneath it. Runtime preparation creates the root, writes and removes a `.openenvrl-drive-write-test` sentinel file, and fails if either operation fails. Paths outside this mount are rejected rather than treated as artifact roots.

## Runtime Interface

`scripts/run_phase.py` continues to be the primary entrypoint. Its existing `--artifact-root` option maps `data_dir`, `results_dir`, `models_dir`, the SFT dataset path, and model output paths under the selected run root. A Drive-rooted phase must never fall back to `/content` output paths.

Each named phase is independently runnable. `build_sft_dataset` reads and validates `data/successful_trajectories.jsonl` from the selected run root; it must not depend on in-memory episodes from an earlier CLI invocation. `sft_train` invokes the real trainer rather than returning a configuration dictionary. `sft_eval` loads the mandatory merged model from `models/sft_model/` and writes `results/sft_metrics.json` there.

The Colab notebook remains optional. Its rerun cells must set one explicit Drive artifact root and call the script phases with it. The notebook may call the training function directly only when it passes the exact same Drive-rooted configuration.

## Persistence Contract

- Each new environment reset receives a generated UUID4 `episode_id`; this id is persisted in every trajectory record from that episode. A rerun resumes from an existing `successful_trajectories.jsonl` only after validating every stored record and its unique `episode_id`. It rejects duplicate ids and legacy records without ids, so old collections must use a new run root or be explicitly migrated. It collects only the remaining successes needed to reach the configured target.
- Every collection checkpoint uses a same-directory temporary JSONL file, flushes it, validates the full resulting file, then atomically renames it to `successful_trajectories.jsonl`. A disconnect can therefore lose only the current in-flight episode, never corrupt accepted episodes.
- Dataset creation writes `data/sft_dataset.jsonl` directly to Drive.
- Training saves the LoRA adapter under `models/sft_adapter/` and a merged, evaluation-ready 16-bit model plus tokenizer under `models/sft_model/` on Drive. Both are mandatory deliverables. The adapter directory must include `adapter_config.json` and adapter weights (`adapter_model.safetensors` or `adapter_model.bin`). The merged directory must include `config.json`, tokenizer configuration/files, and model weights (`model.safetensors`, `pytorch_model.bin`, or a recognized sharded-weight index). `sft_eval` loads only `models/sft_model/`; later training/resume work may use the adapter.
- Evaluation writes `results/sft_metrics.json` on Drive.
- A manifest is written only after validation of all required outputs. It records run id, collection target and count, dataset row count and source-step count, required artifact paths, SHA-256 and byte size for JSON/metric files, model directory file list and aggregate byte size, Git revision and dirty state, full runtime config, model identifier, SFT configuration, installed package versions, and observed metric values.

## Validation and Failure Behavior

The Drive smoke run uses its own timestamped root. It performs two successful collections, builds their dataset, runs a bounded SFT job, validates the merged export with `AutoTokenizer.from_pretrained` plus `AutoModelForCausalLM.from_pretrained`, validates the adapter by loading it onto its recorded base model with `PeftModel.from_pretrained`, then evaluates the merged model for two episodes. This verifies Drive write access, both training exports, and the evaluator loader together.

The full rerun uses a fresh root and requires exactly 50 validated successful episodes. The dataset row count need not equal the prior run's 67 rows; it must be positive and exactly equal to the sum of persisted successful episode steps.

The system must fail early if a Drive root is not mounted/writable, collection records are malformed/duplicated, the target count is not reached, the dataset count does not match source steps, either model output is absent, or SFT evaluation cannot load the merged checkpoint. Metrics must contain numeric `success_rate`, `average_reward`, and `average_steps` fields. It must not silently write artifacts to ephemeral `/content` paths when a Drive root was requested.

## Tests

Unit tests cover:

- artifact-root path mapping;
- persisted trajectory loading, resume, duplicate rejection, and atomic-checkpoint behavior;
- executable SFT phase dispatch;
- SFT evaluation checkpoint selection and output path; and
- manifest contents, required-artifact validation, and count consistency.

Colab verification covers one Drive-backed smoke run and the full 50-success SFT rerun. The full run is considered recoverable only when all required Drive artifacts and manifest are present.
