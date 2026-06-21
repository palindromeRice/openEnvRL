# Browser Agent RL

Colab-first educational project for browser-agent training on MiniWoB `click-checkboxes` using `Qwen/Qwen3.5-0.8B`, OpenEnv, SFT, and GRPO.

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
