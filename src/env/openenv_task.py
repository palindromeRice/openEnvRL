from __future__ import annotations

from typing import Any


class OpenEnvClickCheckboxesEnv:
    """Thin wrapper contract for later OpenEnv integration."""

    def __init__(self, backend: Any | None = None) -> None:
        self.backend = backend

    def reset(self) -> dict[str, Any]:
        if self.backend is None:
            return {"instruction": "", "observation": "", "info": {"success": False}}
        result = self.backend.reset()
        return self._normalize_reset(result)

    def step(self, action: dict[str, Any]) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        if self.backend is None:
            return {"instruction": "", "observation": ""}, 0.0, True, {"success": False}
        result = self.backend.step(action)
        return self._normalize_step(result)

    def _normalize_reset(self, result: Any) -> dict[str, Any]:
        return {
            "instruction": getattr(result.observation, "goal", ""),
            "observation": getattr(result.observation, "text", ""),
            "info": {"success": False},
        }

    def _normalize_step(self, result: Any) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        observation = {
            "instruction": getattr(result.observation, "goal", ""),
            "observation": getattr(result.observation, "text", ""),
        }
        reward = float(getattr(result, "reward", 0.0))
        done = bool(getattr(result, "done", False))
        raw_info = getattr(result, "info", {}) or {}
        success = bool(done and raw_info.get("success") is True)
        return observation, reward, done, {"success": success, "raw_info": raw_info}
