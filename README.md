# Browser Agent RL

Colab-CLI-first educational project for browser-agent training on MiniWoB `click-checkboxes` using `Qwen/Qwen3.5-0.8B`, BrowserGym/MiniWoB locally in Colab, plus OpenEnv-oriented wrappers for later GRPO integration.

Primary execution path:

- local machine uses `google-colab-cli`
- remote Colab runtime executes `scripts/run_phase.py`
- notebook is optional and mainly for inspection or plots

Install the local Colab CLI on macOS or Linux:

```bash
uv tool install google-colab-cli
```

or:

```bash
pip install google-colab-cli
```

Colab install notes:

```bash
pip install -q -r requirements.txt
pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install -q "openenv-browsergym @ git+https://huggingface.co/spaces/openenv/browsergym_env"
pip install -q playwright
playwright install chromium
```

Then bootstrap MiniWoB assets for the real `click-checkboxes` task:

```python
from src.env.native_browsergym_task import bootstrap_miniwob_assets

miniwob_url = bootstrap_miniwob_assets("/content/browsergym-assets")
print(miniwob_url)
```

This follows current BrowserGym + MiniWoB + Unsloth Colab usage, while keeping the OpenEnv client path available for comparison and later harness work.

Planned phases:

1. Native MiniWoB/BrowserGym exploration
2. Optional OpenEnv probe/debugging
3. Manual click-checkboxes smoke test
4. Baseline evaluation
5. Successful trajectory collection
6. SFT training
7. GRPO training
8. Final evaluation and analysis

Native probe smoke command:

```bash
python -c "from src.env.native_miniwob_probe import sample_probe_payload; import json; print(json.dumps(sample_probe_payload(), indent=2))"
```

This should print:

- raw observation keys
- instruction field
- fields payload
- one sample action attempt result

Phase-1 exit checklist:

- one concrete prompt input shape is written down
- one concrete action schema is written down
- one successful BrowserGym MiniWoB `reset -> action -> step` example is recorded

SFT smoke command:

```bash
python -c "from src.train.sft import build_sft_job_config; import json; print(json.dumps(build_sft_job_config('data/sft_dataset.jsonl', 'models/sft_model'), indent=2))"
```

GRPO smoke command:

```bash
python -c "from src.train.grpo import build_grpo_job_config; import json; print(json.dumps(build_grpo_job_config('models/sft_model', 'models/grpo_model'), indent=2))"
```

Primary Colab CLI entrypoint:

- `scripts/run_phase.py`

Recommended command order:

```bash
colab run --gpu T4 scripts/run_phase.py --phase manual_smoke --profile smoke
colab run --gpu T4 scripts/run_phase.py --phase baseline_eval --profile smoke
colab run --gpu T4 scripts/run_phase.py --phase collect_successes --profile smoke
```

Useful flags:

- `--config configs/default.yaml`
- `--artifact-root /tmp/browser-agent-rl-artifacts`

Optional notebook entrypoint:

- `notebooks/orchestrator.ipynb`
- use it for inspection, plots, or manual debugging after CLI runs

Runtime notes:

- run the manual smoke phase before baseline; if it fails, the real `click-checkboxes` task wiring is still wrong
- baseline/data collection use local BrowserGym MiniWoB for the real `click-checkboxes` task
- OpenEnv HF Space can silently fall back to its default demo task, so it is probe-only unless task binding is verified
- BrowserGym actions are sent as natural-language tool strings like `click('12')`
- GRPO integration is prepared around OpenEnv tool-style environment usage, not a fake offline loop
