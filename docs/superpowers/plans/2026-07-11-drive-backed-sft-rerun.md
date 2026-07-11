# Drive-Backed SFT Rerun Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rerun trajectory collection and LoRA SFT on Colab, with every recoverable artifact saved directly under a validated Google Drive run root.

**Architecture:** Add a small artifact module that validates and prepares Drive roots, validates saved outputs, and writes a run manifest. Make the existing CLI phases independently runnable from persisted artifacts: collection resumes safely, dataset building reads saved trajectories, training writes adapter plus merged model, and evaluation loads the merged Drive checkpoint.

**Tech Stack:** Python 3.11, pytest, Google Colab Drive mount, BrowserGym/MiniWoB, Transformers, PEFT, Unsloth, TRL.

**Spec:** `docs/superpowers/specs/2026-07-11-drive-backed-sft-rerun-design.md` (currently in primary workspace until user approves a commit).

---

## File Structure

- Create: `src/artifacts.py` — Drive-root validation, artifact path mapping, required-output validation, manifest generation.
- Modify: `src/cli/run_phase.py` — require/prep a valid Drive root when `--artifact-root` is supplied.
- Modify: `src/data/storage.py` — atomic JSONL checkpoint write and JSONL validation.
- Modify: `src/agents/rollout.py` — attach UUID4 `episode_id` to each completed episode.
- Modify: `src/pipeline/collect.py` — resume persisted successes, reject duplicate/malformed ids, checkpoint atomically.
- Modify: `src/pipeline/run_all.py` — persisted dataset input, executable SFT phase, merged-model SFT evaluation, manifest phase/result.
- Modify: `src/train/sft.py` — save both adapter and merged model/tokenizer; expose artifact validation helpers.
- Modify: `configs/default.yaml` and `notebooks/orchestrator.ipynb` — adapter path and one explicit Drive run root.
- Create/modify tests under `tests/` for each public behavior below.

No commit is authorized by this plan. Commit only after user asks.

### Task 1: Drive Artifact Root Contract

**Files:**
- Create: `tests/test_artifacts.py`
- Create: `src/artifacts.py`
- Modify: `src/cli/run_phase.py`
- Test: `tests/cli/test_run_phase.py`

- [ ] **Step 1: Write failing tests for Drive-root acceptance and rejection**

```python
def test_prepare_drive_artifact_root_maps_run_paths_and_removes_sentinel(tmp_path):
    root = tmp_path / "content/drive/MyDrive/openEnvRL/artifacts/run-1"
    paths = prepare_drive_artifact_root(root, drive_prefix=tmp_path / "content/drive/MyDrive/openEnvRL/artifacts")
    assert paths["sft_output_dir"] == str(root / "models/sft_model")
    assert not (root / ".openenvrl-drive-write-test").exists()

def test_prepare_drive_artifact_root_rejects_non_drive_path(tmp_path):
    with pytest.raises(ValueError, match="Google Drive artifact root"):
        prepare_drive_artifact_root(tmp_path / "content/openEnvRL/run-1")
```

- [ ] **Step 2: Run test to verify red state**

Run: `pytest tests/test_artifacts.py -v`

Expected: FAIL because `src.artifacts` and `prepare_drive_artifact_root` do not exist.

- [ ] **Step 3: Implement minimal Drive-root preparation**

Implement `prepare_drive_artifact_root(root, drive_prefix=Path("/content/drive/MyDrive/openEnvRL/artifacts"))` with resolved-prefix validation, non-empty run-id validation, directory creation, sentinel write/remove, and returned mapping for `data_dir`, `results_dir`, `models_dir`, `sft_dataset_path`, `sft_output_dir`, and `sft_adapter_dir`.

- [ ] **Step 4: Route CLI artifact-root handling through helper**

Replace ad-hoc `apply_artifact_root` mapping in `src/cli/run_phase.py`; retain no-root local behavior for existing unit tests. Add a CLI test asserting a non-Drive artifact root exits before bootstrap/phase dispatch.

- [ ] **Step 5: Run focused green tests**

Run: `pytest tests/test_artifacts.py tests/cli/test_run_phase.py -v`

Expected: PASS.

### Task 2: Atomic, Resumable Trajectory Storage

**Files:**
- Create: `tests/data/test_storage.py`
- Modify: `src/data/storage.py`
- Modify: `src/agents/rollout.py`
- Modify: `src/pipeline/collect.py`
- Test: `tests/agents/test_rollout.py`
- Test: `tests/pipeline/test_collect.py`

- [ ] **Step 1: Write failing tests for episode identity and atomic writes**

```python
def test_run_episode_assigns_uuid_episode_id():
    episode = run_episode(DummyEnv(), DummyPolicy(), max_steps=1)
    assert UUID(episode["episode_id"]).version == 4

def test_write_jsonl_atomic_replaces_only_valid_complete_file(tmp_path):
    path = tmp_path / "successful_trajectories.jsonl"
    write_jsonl_atomic(path, [{"episode_id": "id-1"}])
    assert read_jsonl(path) == [{"episode_id": "id-1"}]
    assert not list(tmp_path.glob("*.tmp"))
```

- [ ] **Step 2: Run red tests**

Run: `pytest tests/data/test_storage.py tests/agents/test_rollout.py -v`

Expected: FAIL because `episode_id` and `write_jsonl_atomic` do not exist.

- [ ] **Step 3: Implement UUID and atomic JSONL helper**

Generate `str(uuid.uuid4())` once per `run_episode`. Write JSONL to a same-directory temporary file, flush and `os.fsync`, parse all rows back for validation, then `Path.replace()` onto destination. Keep `write_jsonl` as compatibility wrapper only if existing callers need it.

- [ ] **Step 4: Write failing resume/duplicate tests**

```python
def test_run_collection_loop_resumes_valid_saved_successes(tmp_path):
    output = tmp_path / "successes.jsonl"
    write_jsonl_atomic(output, [successful_episode(str(uuid.uuid4()))])
    episodes = run_collection_loop(..., target_successes=2, output_path=str(output))
    assert len(episodes) == 2
    assert len(created_envs) == 1

def test_run_collection_loop_rejects_duplicate_saved_episode_ids(tmp_path):
    duplicate_id = str(uuid.uuid4())
    write_jsonl_atomic(output, [successful_episode(duplicate_id), successful_episode(duplicate_id)])
    with pytest.raises(ValueError, match="duplicate episode_id"):
        run_collection_loop(...)

def test_run_collection_loop_rejects_legacy_non_uuid_episode_id(tmp_path):
    write_jsonl_atomic(output, [successful_episode("legacy-id")])
    with pytest.raises(ValueError, match="UUID4"):
        run_collection_loop(...)
```

- [ ] **Step 5: Run red tests**

Run: `pytest tests/pipeline/test_collect.py -v`

Expected: FAIL because collection always starts from an empty list and does not validate ids.

- [ ] **Step 6: Implement resume contract**

Add `validate_successful_episodes(rows, target_successes)` that requires string UUID4 `episode_id`, successful flag, numeric reward, non-empty steps, unique ids, and count no greater than target. Validate full proposed collection before every atomic checkpoint; add test proving malformed newly accepted row cannot replace prior valid file. In `run_collection_loop`, load/validate existing output when present, collect only remaining successes, and atomically checkpoint after each accepted new episode. Reject legacy rows without ids and pre-existing collections above target rather than silently mixing/truncating them.

- [ ] **Step 7: Run focused green tests**

Run: `pytest tests/data/test_storage.py tests/agents/test_rollout.py tests/pipeline/test_collect.py -v`

Expected: PASS.

### Task 3: Persisted Dataset and Executable SFT Phases

**Files:**
- Modify: `tests/data/test_build_sft_dataset.py`
- Modify: `tests/pipeline/test_run_all.py`
- Modify: `src/pipeline/run_all.py`
- Modify: `src/train/sft.py`
- Create: `tests/train/test_sft_artifacts.py`

- [ ] **Step 1: Write failing test for a standalone dataset phase**

```python
def test_build_sft_dataset_phase_reads_drive_trajectory_file(tmp_path):
    write_jsonl_atomic(tmp_path / "data/successful_trajectories.jsonl", [successful_episode(str(uuid.uuid4()))])
    result = run_named_phase("build_sft_dataset", {"config": drive_config(tmp_path)})
    assert result["num_rows"] == 1
    assert (tmp_path / "data/sft_dataset.jsonl").exists()
```

- [ ] **Step 2: Run red test**

Run: `pytest tests/pipeline/test_run_all.py -v`

Expected: FAIL because `build_sft_dataset` currently uses empty in-memory context in a new CLI process.

- [ ] **Step 3: Implement persisted dataset input**

Make `run_sft_dataset_phase` read and validate `data/successful_trajectories.jsonl` when explicit episode context is absent. Assert output row count equals summed source episode steps. Keep in-memory input only for existing same-process tests.

- [ ] **Step 4: Write failing phase-execution tests**

```python
def test_sft_train_phase_calls_run_sft_training(monkeypatch, drive_config):
    monkeypatch.setattr("src.pipeline.run_all.run_sft_training", lambda config: {"train_loss": 0.1})
    assert run_named_phase("sft_train", {"config": drive_config})["result"]["train_loss"] == 0.1

def test_sft_eval_phase_loads_merged_model_and_writes_drive_metrics(monkeypatch, drive_config):
    ...
    assert (Path(drive_config["results_dir"]) / "sft_metrics.json").exists()
```

- [ ] **Step 5: Run red tests**

Run: `pytest tests/pipeline/test_run_all.py tests/train/test_sft_artifacts.py -v`

Expected: FAIL because SFT phase returns config and SFT eval is a placeholder.

- [ ] **Step 6: Implement executable training, dual export, and evaluation**

`run_sft_phase` must build config then call `run_sft_training`. Add `sft_adapter_dir` to SFT config; save PEFT adapter plus tokenizer there before `save_pretrained_merged` creates mandatory `sft_model`. Add explicit `validate_sft_exports(config)` helpers: load merged export with `AutoTokenizer.from_pretrained` and `AutoModelForCausalLM.from_pretrained`; load adapter with its recorded base model and `PeftModel.from_pretrained`. Write failing unit tests with monkeypatched loaders asserting each exact directory is selected. Call checks before SFT evaluation. Add `run_sft_eval_phase` using `_build_policy(config, checkpoint_path=config["sft_output_dir"])`, existing native evaluator, and `results/sft_metrics.json`.

- [ ] **Step 7: Run focused green tests**

Run: `pytest tests/data/test_build_sft_dataset.py tests/pipeline/test_run_all.py tests/train/test_sft_artifacts.py -v`

Expected: PASS.

### Task 4: Required Artifact Validation and Manifest

**Files:**
- Modify: `tests/test_artifacts.py`
- Modify: `src/artifacts.py`
- Modify: `src/pipeline/run_all.py`
- Modify: `src/cli/run_phase.py`

- [ ] **Step 1: Write failing manifest validation test**

```python
def test_write_run_manifest_requires_configured_success_count_and_consistent_dataset(tmp_path):
    paths = make_complete_artifact_tree(tmp_path, successes=2, dataset_rows=3, source_steps=3)
    manifest = write_run_manifest(paths, config={"target_successes": 2, "model_id": "Qwen/Qwen3.5-0.8B"})
    assert manifest["run_id"] == tmp_path.name
    assert manifest["artifact_paths"]["sft_model"] == str(tmp_path / "models/sft_model")
    assert manifest["model_directories"]["sft_model"]["total_bytes"] > 0
    assert manifest["collection"]["count"] == 2
    assert manifest["metrics"]["success_rate"] == 0.5
    assert (tmp_path / "run_manifest.json").exists()
```

- [ ] **Step 2: Run red test**

Run: `pytest tests/test_artifacts.py -v`

Expected: FAIL because manifest and required-artifact validation do not exist.

- [ ] **Step 3: Implement deterministic validation and manifest generation**

Validate exactly configured `target_successes`, positive dataset rows equal source steps, JSON metric keys/numeric values, merged HF files, adapter HF files, and no missing paths. Add separate tests for smoke target `2` and full target `50`. Record explicit run id, required-artifact path mapping, SHA-256/bytes for JSON files, recursively sorted model file lists plus aggregate byte sizes per model directory, current Git revision/dirty state, Python/package versions, full sanitized config, model id, SFT config, and observed metrics. Write manifest atomically only after checks pass.

- [ ] **Step 4: Add manifest phase and CLI test**

Add `sft_manifest` after `sft_eval` in phase list, or add `--write-manifest` to `sft_eval`; choose one and update tests/notebook consistently. Require the Drive-root config. A missing model, bad metric schema, or count mismatch must fail before manifest write.

- [ ] **Step 5: Run focused green tests**

Run: `pytest tests/test_artifacts.py tests/pipeline/test_run_all.py tests/cli/test_run_phase.py -v`

Expected: PASS.

### Task 5: Config, Notebook, and Colab Runbook

**Files:**
- Modify: `configs/default.yaml`
- Modify: `notebooks/orchestrator.ipynb`
- Modify: `README.md`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write failing config test**

```python
def test_default_config_declares_sft_adapter_output():
    config = load_config("configs/default.yaml")
    assert config["sft_adapter_dir"] == "models/sft_adapter"
```

- [ ] **Step 2: Run red test**

Run: `pytest tests/test_config.py::test_default_config_declares_sft_adapter_output -v`

Expected: FAIL because no adapter output configuration exists.

- [ ] **Step 3: Implement config and notebook Drive path**

Add `sft_adapter_dir`. In notebook, mount Drive then define one fresh timestamped `ARTIFACT_ROOT = /content/drive/MyDrive/openEnvRL/artifacts/<run-id>`. Show script-first phase calls in order: `collect_successes`, `build_sft_dataset`, `sft_train`, `sft_eval`, manifest. Keep code execution from `/content/openEnvRL`; never copy artifacts back from ephemeral disk.

- [ ] **Step 3b: Add bounded SFT smoke profile**

Add `sft_smoke` as a CLI profile with exactly `target_successes=2`, `default_eval_episodes=2`, and existing short episode limit. Add tests proving `--profile sft_smoke` passes those values through Drive-rooted phases. Use this profile for Drive smoke; preserve existing `smoke` profile behavior for earlier pipeline tests.

- [ ] **Step 4: Update README status only**

Keep README command-free. State that SFT reruns are Drive-backed, successful trajectories checkpoint per success, and required saved deliverables are trajectory file, dataset, adapter, merged model, SFT metrics, and manifest. Do not claim a rerun result until Colab evidence exists.

- [ ] **Step 5: Run config green test**

Run: `pytest tests/test_config.py -v`

Expected: PASS.

### Task 6: Full Local Regression and Colab Verification

**Files:**
- No production files expected unless verification finds an issue.

- [ ] **Step 1: Run full local regression**

Run: `pytest -q`

Expected: all tests pass.

- [ ] **Step 2: Start named Colab session and smoke root**

Use named session `openenvrl-sft-smoke` with `--profile sft_smoke`. Run with fresh Drive root such as `/content/drive/MyDrive/openEnvRL/artifacts/2026-07-11-sft-smoke`. Verify 2 saved trajectory records, non-empty dataset, direct merged-model and adapter load validation, 2-episode SFT metrics, and manifest.

- [ ] **Step 3: Run full recovery experiment**

Use named session `openenvrl-sft-rerun`. Use a new root such as `/content/drive/MyDrive/openEnvRL/artifacts/2026-07-11-sft-rerun`. Collect exactly 50 successes, train LoRA SFT, evaluate 50 episodes, then create manifest.

- [ ] **Step 4: Inspect Drive artifacts before ending Colab session**

Verify `successful_trajectories.jsonl`, `sft_dataset.jsonl`, `sft_adapter/`, `sft_model/`, `sft_metrics.json`, and `run_manifest.json` exist under the full root. Confirm manifest counts equal actual JSONL counts and it records numeric metrics. Only then stop the session.
