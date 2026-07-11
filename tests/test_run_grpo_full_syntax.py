import subprocess

def test_script_syntax():
    result = subprocess.run(["python", "-m", "py_compile", "scripts/run_grpo_full.py"], capture_output=True)
    assert result.returncode == 0
