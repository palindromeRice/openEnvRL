# Agent Instructions

- **Branch Management**: Never delete a branch after merging it locally (overrides default behavior from `finishing-a-development-branch` skill). Always keep branches intact unless explicitly asked by the user to discard or delete them.
- **Colab Dependencies**: NEVER compile `flash-attn` or similar packages from source on Colab. To use them, you MUST provide the exact pre-compiled `.whl` URL in `requirements.txt` (matching the environment's Python, PyTorch, and CUDA versions). If an exact wheel URL is unknown, omit the package entirely and fall back to native PyTorch SDPA.
