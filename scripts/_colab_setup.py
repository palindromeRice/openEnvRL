import subprocess
import sys
import os

os.chdir("/content/drive/MyDrive/openEnvRL")
print("Installing requirements...")
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
    capture_output=True, text=True
)
print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr[-2000:])
    print(f"pip install failed with code {result.returncode}")
else:
    print("Dependencies installed successfully!")
