import subprocess
import sys
import os


print("Starting wrapped execution...")
try:
    os.chdir("/content/drive/MyDrive/openEnvRL")
    subprocess.check_call([sys.executable, "scripts/run_grpo_full.py"])
except subprocess.CalledProcessError as e:
    print(f"Execution failed with code {e.returncode}")
    sys.exit(e.returncode)
