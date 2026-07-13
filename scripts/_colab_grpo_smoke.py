import subprocess, sys, os, time

os.chdir("/content/drive/MyDrive/openEnvRL")

# Force Drive sync by touching the file
print("Waiting for Drive sync...")
time.sleep(5)

# Verify the click docstring has Args section
with open("src/train/grpo.py") as f:
    content = f.read()
if "target: The browsergym ID" not in content:
    print("ERROR: Drive sync hasn't propagated yet. File is stale.")
    print("Relevant section:")
    idx = content.find("def click")
    if idx >= 0:
        print(content[idx:idx+300])
    sys.exit(1)
else:
    print("File is up to date.")

# Install dependencies
print("Installing dependencies...")
subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)

# Install Playwright system dependencies
print("Installing Playwright system dependencies...")
subprocess.run([sys.executable, "-m", "playwright", "install-deps"], check=True)
print("System deps ready.")

print("Installing Playwright chromium browser...")
subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True)
print("Playwright ready.")

# Run training as a subprocess
print("=" * 60)
print("Launching GRPO smoke test as subprocess...")
print("=" * 60)

result = subprocess.run(
    [sys.executable, "scripts/run_grpo_full.py"],
    cwd="/content/drive/MyDrive/openEnvRL",
)

if result.returncode != 0:
    print(f"\nProcess exited with code {result.returncode}")
else:
    print("\nGRPO training completed successfully!")
