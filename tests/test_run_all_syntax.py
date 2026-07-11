def test_grpo_phase_imports():
    from src.pipeline.run_all import run_grpo_phase
    assert callable(run_grpo_phase)
