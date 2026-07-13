# Agent Instructions

- **Branch Management**: Never delete a branch after merging it locally (overrides default behavior from `finishing-a-development-branch` skill). Always keep branches intact unless explicitly asked by the user to discard or delete them.
- **Colab Dependencies**: NEVER compile `flash-attn` or similar packages from source on Colab. To use them, you MUST provide the exact pre-compiled `.whl` URL in `requirements.txt` (matching the environment's Python, PyTorch, and CUDA versions). If an exact wheel URL is unknown, omit the package entirely and fall back to native PyTorch SDPA.
- **Colab Workflow**: Launch long jobs with `colab exec -s <session> -f <script> --timeout <seconds>` run via the Bash tool's `run_in_background: true` (or `colab run --session <name> --gpu <type> --keep` for a fresh VM) — the harness notifies on completion, no manual `nohup`/log-polling needed. `colab run` does NOT auto-mount Drive; run `colab drivemount -s <session>` first if the session was created fresh via `colab new`.
- **Git Commits**: Do not add a `Co-Authored-By: Claude` (or any AI attribution) line to commit messages in this repo.
