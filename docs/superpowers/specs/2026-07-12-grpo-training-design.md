# GRPO Training Implementation Design

## Purpose
Enable execution of the GRPO (Group Relative Policy Optimization) training phase using the recovered SFT (Supervised Fine-Tuning) model.

## Architecture & Components

### 1. Dedicated Runner Script (`scripts/run_grpo_full.py`)
A standalone entrypoint for the GRPO phase.
- **Responsibility**: Setup the environment, load the config, build the GRPO job config, and trigger the training directly.
- **Data Flow**: 
  1. Imports project root into `sys.path`.
  2. Loads `configs/default.yaml` and ensures runtime directories exist.
  3. Uses `build_grpo_job_config` from `src.train.grpo`, passing the SFT output directory as the `init_model_path`.
  4. Calls `run_grpo_training` with the built config.
  5. Prints the trainer statistics upon completion.

### 2. Pipeline Integration (`src/pipeline/run_all.py`)
The existing main pipeline router needs to be updated to match the new behavior, even though the standalone script will be the primary runner.
- **Changes**: Modify `run_grpo_phase` to execute `run_grpo_training(job_config)` instead of just returning the configuration dictionary.
- **Dependency**: It will import `run_grpo_training` from `src.train.grpo`.

## Error Handling
- The standalone script will rely on standard Python exception bubbling. If the configuration paths (e.g., `sft_output_dir`) do not exist, the `TRL` trainer will naturally throw an error.
- Directory creation for `grpo_output_dir` is already handled internally by `ensure_grpo_artifact_dirs` inside `run_grpo_training`.

## Testing
- **Smoke Testing**: Can be verified by running `python scripts/run_grpo_full.py` in the Colab CLI environment and checking if the `GRPOTrainer` successfully initializes and starts the first training step.
