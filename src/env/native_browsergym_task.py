from __future__ import annotations

import asyncio
import os
import subprocess
import threading
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.env.action_schema import format_action, parse_model_action


MINIWOB_COMMIT = "7fd85d71a4b60325c6585396ec4f48377d049838"


@dataclass
class NativeMiniWoBTaskConfig:
    task_name: str = "click-checkboxes"
    headless: bool = True
    miniwob_root: str | None = None
    miniwob_url: str | None = None
    use_worker_thread: bool = False


def bootstrap_miniwob_assets(root_dir: str | Path) -> str:
    """Clone MiniWoB++ assets once and return a BrowserGym-compatible file URL."""

    root = Path(root_dir)
    repo_dir = root / "miniwob-plusplus"
    if not repo_dir.exists():
        subprocess.run(
            ["git", "clone", "https://github.com/Farama-Foundation/miniwob-plusplus.git", str(repo_dir)],
            check=True,
        )
    subprocess.run(["git", "-C", str(repo_dir), "fetch", "--all", "--tags"], check=True)
    subprocess.run(["git", "-C", str(repo_dir), "checkout", MINIWOB_COMMIT], check=True)
    miniwob_url = f"file://{repo_dir / 'miniwob' / 'html' / 'miniwob'}/"
    os.environ["MINIWOB_URL"] = miniwob_url
    return miniwob_url


def ensure_playwright_browser(browser_name: str = "chromium") -> None:
    """Install the Playwright browser binary needed by BrowserGym in Colab."""

    cache_key = f"OPENENVRL_PLAYWRIGHT_{browser_name.upper()}_READY"
    if os.environ.get(cache_key) == "1":
        return
    subprocess.run(["playwright", "install", browser_name], check=True)
    os.environ[cache_key] = "1"


def _axtree_prop_value(node: dict[str, Any], prop_name: str) -> str | None:
    for prop in node.get("properties", []):
        if prop.get("name") == prop_name:
            value = prop.get("value")
            if isinstance(value, dict):
                return value.get("value")
            return value
    return None


def serialize_axtree_object(axtree_object: dict[str, Any]) -> str:
    nodes = axtree_object.get("nodes", [])
    root_lines: list[str] = []
    checkbox_lines: list[str] = []
    button_lines: list[str] = []

    for node in nodes:
        role = node.get("role", {}).get("value")
        name = node.get("name", {}).get("value")
        bid = node.get("browsergym_id")

        if role == "RootWebArea" and name:
            root_lines.append(f"RootWebArea '{name}'")
            continue

        if role == "checkbox" and bid and name:
            checked = _axtree_prop_value(node, "checked") == "true"
            suffix = " checked" if checked else ""
            checkbox_lines.append(f"[{bid}] checkbox '{name}'{suffix}")
            continue

        if role == "button" and bid and name:
            button_lines.append(f"[{bid}] button '{name}'")

    return "\n".join(root_lines + checkbox_lines + button_lines)


class _EventLoopWorker:
    def __init__(self) -> None:
        self._ready = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="browsergym-miniwob-loop",
            daemon=True,
        )
        self._thread.start()
        self._ready.wait()

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._ready.set()
        loop.run_forever()
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()

    def call(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        if self._loop is None:
            raise RuntimeError("Worker loop is not initialized.")

        future: Future[Any] = Future()

        def runner() -> None:
            try:
                result = fn(*args, **kwargs)
            except BaseException as exc:  # pragma: no cover - propagated to caller
                future.set_exception(exc)
            else:
                future.set_result(result)

        self._loop.call_soon_threadsafe(runner)
        return future.result()

    def close(self) -> None:
        if self._loop is None:
            return
        loop = self._loop
        self._loop = None
        loop.call_soon_threadsafe(loop.stop)
        self._thread.join()


class NativeMiniWoBClickCheckboxesEnv:
    """Local BrowserGym MiniWoB environment for real click-checkboxes episodes in Colab."""

    def __init__(self, config: NativeMiniWoBTaskConfig | None = None, backend: Any | None = None) -> None:
        self.config = config or NativeMiniWoBTaskConfig()
        self._worker: _EventLoopWorker | None = None
        self._uses_worker_thread = (
            backend is None
            and self.config.use_worker_thread
            and self._has_running_event_loop()
        )
        if backend is not None:
            self.backend = backend
        elif self._uses_worker_thread:
            self._worker = _EventLoopWorker()
            self.backend = self._call_backend(self._create_backend)
        else:
            self.backend = self._create_backend()

    def _has_running_event_loop(self) -> bool:
        try:
            asyncio.get_running_loop()
            return True
        except RuntimeError:
            return False

    def _call_backend(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        if self._worker is None:
            return fn(*args, **kwargs)
        return self._worker.call(fn, *args, **kwargs)

    def _create_backend(self) -> Any:
        try:
            import gymnasium as gym
            import browsergym.miniwob  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "browsergym-miniwob is required for local MiniWoB runs. Install it in Colab and bootstrap assets."
            ) from exc

        if self.config.miniwob_url:
            os.environ["MINIWOB_URL"] = self.config.miniwob_url
        elif self.config.miniwob_root:
            bootstrap_miniwob_assets(self.config.miniwob_root)

        ensure_playwright_browser()
        env_id = f"browsergym/miniwob.{self.config.task_name}"
        return gym.make(env_id)

    def reset(self) -> dict[str, Any]:
        observation, info = self._call_backend(self.backend.reset)
        return self._normalize(observation, info=info, reward=0.0, done=False)

    def step(self, action: dict[str, Any] | str) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        structured = parse_model_action(action) if isinstance(action, str) else action
        if structured.get("type") == "NOOP":
            normalized = self._normalize({}, info={}, reward=0.0, done=False)
            return normalized, 0.0, False, normalized["info"]
        action_text = format_action(structured)["action_str"]
        observation, reward, terminated, truncated, info = self._call_backend(self.backend.step, action_text)
        done = bool(terminated or truncated)
        normalized = self._normalize(observation, info=info, reward=float(reward), done=done)
        return normalized, float(reward), done, normalized["info"]

    def close(self) -> None:
        close_fn = getattr(self.backend, "close", None)
        try:
            if callable(close_fn):
                self._call_backend(close_fn)
        finally:
            if self._worker is not None:
                self._worker.close()
                self._worker = None

    def _normalize(
        self,
        observation: Any,
        info: dict[str, Any] | None,
        reward: float,
        done: bool,
    ) -> dict[str, Any]:
        info = dict(info or {})
        if isinstance(observation, dict):
            goal = str(observation.get("goal", ""))
            axtree = str(observation.get("axtree_txt", ""))
            if not axtree and isinstance(observation.get("axtree_object"), dict):
                axtree = serialize_axtree_object(observation["axtree_object"])
            pruned_html = str(observation.get("pruned_html", ""))
            dom_txt = str(observation.get("dom_txt", ""))
            observation_text = axtree or dom_txt or pruned_html or str(observation)
            metadata = {
                "raw_observation": observation,
                "browsergym_info": info,
            }
        else:
            goal = ""
            axtree = ""
            pruned_html = ""
            observation_text = str(observation)
            metadata = {"raw_observation": observation, "browsergym_info": info}

        success = bool(done and reward > 0.0)
        return {
            "instruction": goal,
            "observation": observation_text,
            "axtree_txt": axtree,
            "pruned_html": pruned_html,
            "metadata": metadata,
            "info": {"success": success, "raw_info": info},
        }
