import importlib.util
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "run_grpo_full", Path(__file__).parents[1] / "scripts" / "run_grpo_full.py"
)
run_grpo_full = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(run_grpo_full)


def test_parser_defaults_to_approved_run_shape():
    args = run_grpo_full.build_parser().parse_args(
        ["--artifact-root", "/content/drive/MyDrive/openEnvRL/artifacts/run", "--sft-model-path", "/content/sft"]
    )
    assert (args.total_steps, args.checkpoint_every, args.eval_episodes) == (200, 50, 50)


def test_run_boundaries_skips_complete_boundary(monkeypatch, tmp_path):
    paths = {"grpo_output_dir": tmp_path / "models", "results_dir": tmp_path / "results", "models_dir": tmp_path / "models"}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    calls = []
    monkeypatch.setattr(run_grpo_full, "is_complete_grpo_boundary", lambda paths, step: step == 50)
    monkeypatch.setattr(run_grpo_full, "run_grpo_training", lambda config: calls.append(("train", config["max_steps"])))
    monkeypatch.setattr(run_grpo_full, "run_grpo_eval_phase", lambda config, checkpoint_step: calls.append(("eval", checkpoint_step)))
    monkeypatch.setattr(run_grpo_full, "write_grpo_run_manifest", lambda paths, config: {"ok": True})
    # Pretend remaining checkpoint exists after mocked trainer.
    monkeypatch.setattr(Path, "is_dir", lambda self: True)
    assert run_grpo_full.run_boundaries({"total_steps": 100, "checkpoint_every": 50, "sft_model_path": "/sft"}, paths) == {"ok": True}
    assert calls == [("train", 100), ("eval", 100)]
