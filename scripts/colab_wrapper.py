import subprocess
import sys
import os

print("Starting wrapped execution...")
try:
    subprocess.check_call([sys.executable, "scripts/run_sft_full.py"])
except subprocess.CalledProcessError as e:
    print(f"Execution failed with code {e.returncode}")
    sys.exit(e.returncode)
