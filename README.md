# Browser Agent RL

Colab-first educational project for browser-agent training on MiniWoB `click-checkboxes` using `Qwen/Qwen3.5-0.8B`, OpenEnv, SFT, and GRPO.

Colab install notes:

```bash
pip install -q -r requirements.txt
pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install -q "openenv-browsergym @ git+https://huggingface.co/spaces/openenv/browsergym_env"
```

This follows current OpenEnv + Unsloth docs for Colab/T4 usage.

Planned phases:

1. Native MiniWoB/BrowserGym exploration
2. OpenEnv task wrapper
3. Baseline evaluation
4. Successful trajectory collection
5. SFT training
6. GRPO training
7. Final evaluation and analysis

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
- one successful OpenEnv `reset -> action -> step` example is recorded

SFT smoke command:

```bash
python -c "from src.train.sft import build_sft_job_config; import json; print(json.dumps(build_sft_job_config('data/sft_dataset.jsonl', 'models/sft_model'), indent=2))"
```

GRPO smoke command:

```bash
python -c "from src.train.grpo import build_grpo_job_config; import json; print(json.dumps(build_grpo_job_config('models/sft_model', 'models/grpo_model'), indent=2))"
```

Primary Colab entrypoint:

- `notebooks/orchestrator.ipynb`
- run cells in order: setup -> probe -> baseline -> collection -> SFT -> GRPO -> analysis

OpenEnv runtime notes:

- baseline/data collection use direct `BrowserGymEnv` `reset()` / `step()` calls
- BrowserGym actions are sent as natural-language tool strings like `click('12')`
- GRPO integration is prepared around OpenEnv tool-style environment usage, not a fake offline loop
