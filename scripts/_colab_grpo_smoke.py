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

# Install Playwright system dependencies
print("Installing Playwright system dependencies...")
subprocess.run(["playwright", "install-deps"], check=True)
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
    text=True,
    capture_output=True,
)

print(result.stdout[-5000:] if len(result.stdout) > 5000 else result.stdout)
if result.stderr:
    print("STDERR (last 3000 chars):")
    print(result.stderr[-3000:])

if result.returncode != 0:
    print(f"\nProcess exited with code {result.returncode}")
else:
    print("\nGRPO training completed successfully!")
