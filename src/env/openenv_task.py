from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Any

from src.env.action_schema import format_action, parse_model_action


@dataclass
class BrowserGymTaskConfig:
    base_url: str = "https://openenv-browsergym-env.hf.space"
    benchmark: str = "miniwob"
    task_name: str = "click-checkboxes"
    headless: bool = True
    timeout_ms: int = 10000
    viewport_width: int = 1280
    viewport_height: int = 720
    extra_environment: dict[str, str] = field(default_factory=dict)

    def environment(self) -> dict[str, str]:
        payload = {
            "BROWSERGYM_BENCHMARK": self.benchmark,
            "BROWSERGYM_TASK_NAME": self.task_name,
            "BROWSERGYM_HEADLESS": str(self.headless).lower(),
            "BROWSERGYM_TIMEOUT": str(self.timeout_ms),
            "BROWSERGYM_VIEWPORT_WIDTH": str(self.viewport_width),
            "BROWSERGYM_VIEWPORT_HEIGHT": str(self.viewport_height),
        }
        payload.update(self.extra_environment)
        return payload

    def reset_kwargs(self) -> dict[str, Any]:
        return {"task_name": self.task_name}


class OpenEnvTransientError(RuntimeError):
    """Raised when the remote BrowserGym/OpenEnv session drops transiently."""


class OpenEnvClickCheckboxesEnv:
    """Direct BrowserGym/OpenEnv client for baseline, eval, and collection."""

    def __init__(self, config: BrowserGymTaskConfig | None = None, backend: Any | None = None) -> None:
        self.config = config or BrowserGymTaskConfig()
        self.backend = backend or self._create_backend()
        self.last_result: Any | None = None

    @staticmethod
    def _resolve_maybe_awaitable(value: Any) -> Any:
        if not inspect.isawaitable(value):
            return value

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(value)

        try:
            import nest_asyncio
        except ImportError as exc:
            raise RuntimeError(
                "Async BrowserGym client detected inside a running notebook loop. "
                "Install nest_asyncio or run this code in a plain Python process."
            ) from exc

        nest_asyncio.apply()
        return loop.run_until_complete(value)

    def _create_backend(self) -> Any:
        try:
            from browsergym_env import BrowserGymEnv
        except ImportError as exc:
            raise ImportError(
                "browsergym_env is required. Install with "
                "\"openenv-browsergym @ git+https://huggingface.co/spaces/openenv/browsergym_env\"."
            ) from exc

        try:
            return BrowserGymEnv(
                base_url=self.config.base_url,
                environment=self.config.environment(),
            )
        except TypeError:
            # Some client versions only accept base_url and use server-side task defaults.
            return BrowserGymEnv(base_url=self.config.base_url)

    def _reset_backend(self) -> Any:
        reset_kwargs = self.config.reset_kwargs()
        try:
            return self.backend.reset(**reset_kwargs)
        except TypeError:
            return self.backend.reset()

    def _validate_task_binding(self, normalized_reset: dict[str, Any]) -> None:
        instruction = normalized_reset.get("instruction", "").lower()
        observation = normalized_reset.get("observation", "").lower()
        expected = self.config.task_name

        # Prevent silent fallback to the OpenEnv default click-test task.
        if expected == "click-checkboxes":
            looks_like_click_test = "click the button" in instruction or "click test task" in observation
            if looks_like_click_test:
                raise RuntimeError(
                    "OpenEnv did not bind to click-checkboxes. It appears to have fallen back to the default "
                    "click-test task. Verify the remote BrowserGym space supports task_name on reset()."
                )

    @staticmethod
    def _is_transient_connection_error(exc: Exception) -> bool:
        class_name = exc.__class__.__name__
        message = str(exc).lower()
        return class_name in {
            "ConnectionClosed",
            "ConnectionClosedError",
            "ConnectionClosedOK",
        } or any(
            snippet in message
            for snippet in (
                "keepalive ping timeout",
                "no close frame received",
                "connection closed",
                "sent 1011",
                "received 1011",
            )
        )

    def reset(self) -> dict[str, Any]:
        try:
            result = self._resolve_maybe_awaitable(self._reset_backend())
        except Exception as exc:
            if self._is_transient_connection_error(exc):
                raise OpenEnvTransientError("OpenEnv reset failed due to a dropped remote session.") from exc
            raise
        self.last_result = result
        normalized = self._normalize_reset(result)
        self._validate_task_binding(normalized)
        return normalized

    def step(self, action: dict[str, Any] | str) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        browsergym_action = self._build_action(action)
        try:
            result = self._resolve_maybe_awaitable(self.backend.step(browsergym_action))
        except Exception as exc:
            if self._is_transient_connection_error(exc):
                raise OpenEnvTransientError("OpenEnv step failed due to a dropped remote session.") from exc
            raise
        self.last_result = result
        return self._normalize_step(result)

    def close(self) -> None:
        close_fn = getattr(self.backend, "close", None)
        if callable(close_fn):
            try:
                self._resolve_maybe_awaitable(close_fn())
            except Exception as exc:
                if not self._is_transient_connection_error(exc):
                    raise

    def _build_action(self, action: dict[str, Any] | str) -> Any:
        try:
            from browsergym_env import BrowserGymAction
        except ImportError as exc:
            raise ImportError(
                "browsergym_env is required. Install with "
                "\"openenv-browsergym @ git+https://huggingface.co/spaces/openenv/browsergym_env\"."
            ) from exc

        structured = parse_model_action(action) if isinstance(action, str) else action
        action_payload = format_action(structured)
        return BrowserGymAction(action_str=action_payload["action_str"])

    def _normalize_reset(self, result: Any) -> dict[str, Any]:
        observation = getattr(result, "observation", None)
        return {
            "instruction": getattr(observation, "goal", ""),
            "observation": getattr(observation, "text", ""),
            "axtree_txt": getattr(observation, "axtree_txt", ""),
            "pruned_html": getattr(observation, "pruned_html", ""),
            "metadata": getattr(observation, "metadata", {}) or {},
            "info": {"success": False},
        }

    def _normalize_step(self, result: Any) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        observation_obj = getattr(result, "observation", None)
        observation = {
            "instruction": getattr(observation_obj, "goal", ""),
            "observation": getattr(observation_obj, "text", ""),
            "axtree_txt": getattr(observation_obj, "axtree_txt", ""),
            "pruned_html": getattr(observation_obj, "pruned_html", ""),
            "metadata": getattr(observation_obj, "metadata", {}) or {},
        }
        reward = float(getattr(result, "reward", 0.0))
        done = bool(getattr(result, "done", False))
        raw_info = dict(getattr(result, "info", {}) or {})
        success = bool(done and (raw_info.get("success") is True or reward > 0.0))
        raw_info["success"] = success
        return observation, reward, done, {"success": success, "raw_info": raw_info}
