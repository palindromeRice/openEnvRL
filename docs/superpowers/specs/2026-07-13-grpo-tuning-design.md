# GRPO Tool-Use Tuning Design

## Objective
Tune the GRPO training pipeline so the Qwen model successfully emits tool calls (JSON schemas) and receives non-zero learning signals (rewards) from the environment.

## Context
During the initial GRPO smoke test, the pipeline ran successfully end-to-end, but `tools/call_frequency` and `reward` were exactly 0. This was caused by a very short `max_completion_length` (64 tokens) cutting off tool generation, and a lack of tool-use instructions in the system prompt.

## Approach

### 1. Tool Exposure (Prompting)
- **Method:** Rely on TRL's built-in chat template rendering.
- **Action:** Explicitly pass the tool methods into the `tools` argument of `GRPOTrainer` or `build_grpo_prompt_dataset`. Wait, actually, TRL `GRPOTrainer` uses `environment_factory` which already exposes tools. But we need to make sure the chat template renders them. We'll update `build_grpo_prompt_dataset` to include the tools in the prompt generation or rely on TRL's internals.
- **Why:** Qwen3.5 has specific `<tool_call>` tags it expects for JSON schema outputs. TRL will automatically render our environment methods into the correct system prompt syntax for Qwen.

### 2. Generation Hyperparameters
- **max_completion_length:** `512`
  - *Reasoning:* Qwen's tool calls are verbose (they include thinking tags, tool name, and JSON arguments). 64 tokens is not enough; 512 gives ample room.
- **num_generations:** `8`
  - *Reasoning:* GRPO requires a group of responses to calculate relative advantages. 8 is a standard size that provides a strong variance signal.
- **per_device_train_batch_size:** `1`
  - *Reasoning:* 8 generations of 512 tokens for an 0.8B model will consume significant memory. We drop batch size to 1 to prevent OOM on a 24GB L4 GPU.
- **gradient_accumulation_steps:** `4`
  - *Reasoning:* Compensates for the small device batch size to maintain a stable effective batch size.

### 3. Execution & Verification
- **Dataset:** 32 dummy prompt rows ("You are a browser agent...").
- **Steps:** 50 gradient steps.
- **Runner:** `scripts/_colab_grpo_smoke.py`
- **Success Criteria:** 
  - Pipeline completes without OOM.
  - `tools/call_frequency` > 0.
  - `reward` metrics > 0.
