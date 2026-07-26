import subprocess
from pathlib import Path

def test_script_syntax():
    result = subprocess.run(["python", "-m", "py_compile", "scripts/run_grpo_full.py"], capture_output=True)
    assert result.returncode == 0


def test_colab_gate_streams_child_output_to_drive_log():
    source = (Path(__file__).parents[1] / "scripts" / "_colab_grpo_gate.py").read_text()

    assert '[sys.executable, "-u", "scripts/run_grpo_full.py"' in source
    assert "log.flush()" in source


def test_corrected_r5_evaluation_launcher_targets_fifty_fresh_tasks():
    source = (Path(__file__).parents[1] / "scripts" / "_colab_grpo_r5_eval.py").read_text()

    assert "checkpoint-20" in source
    assert "EPISODES = 50" in source
    assert 'checkpoint_step="20_corrected_50"' in source
