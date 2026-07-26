import asyncio
import threading

from src.env.native_browsergym_task import (
    NativeMiniWoBClickCheckboxesEnv,
    NativeMiniWoBTaskConfig,
    ensure_playwright_browser,
)


class DummyBackend:
    def __init__(self) -> None:
        self.actions = []
        self.closed = False

    def reset(self):
        return {
            "goal": "Select Apple and Orange",
            "axtree_txt": "RootWebArea\n[1] checkbox 'Apple'\n[2] checkbox 'Orange'",
            "pruned_html": "<input>Apple</input><input>Orange</input>",
        }, {"seed": 1}

    def step(self, action_text):
        self.actions.append(action_text)
        return {
            "goal": "Select Apple and Orange",
            "axtree_txt": "RootWebArea\n[1] checkbox 'Apple' checked\n[2] checkbox 'Orange'",
            "pruned_html": "<input checked>Apple</input><input>Orange</input>",
        }, 1.0, True, False, {"success": True}

    def close(self):
        self.closed = True


class DummyAXTreeBackend:
    def reset(self):
        return {
            "goal": "Select NRFk and click Submit.",
            "axtree_object": {
                "nodes": [
                    {"role": {"value": "RootWebArea"}, "name": {"value": "Click Checkboxes Task"}},
                    {"role": {"value": "button"}, "name": {"value": "Submit"}, "browsergym_id": "36"},
                    {
                        "role": {"value": "checkbox"},
                        "name": {"value": "NRFk"},
                        "browsergym_id": "18",
                        "properties": [{"name": "checked", "value": {"value": "false"}}],
                    },
                    {
                        "role": {"value": "checkbox"},
                        "name": {"value": "vf"},
                        "browsergym_id": "33",
                        "properties": [{"name": "checked", "value": {"value": "true"}}],
                    },
                ]
            },
        }, {"seed": 9}

    def step(self, action_text):
        raise AssertionError("step should not be called in this test")

    def close(self):
        return None


def test_native_env_normalizes_reset_and_step():
    backend = DummyBackend()
    env = NativeMiniWoBClickCheckboxesEnv(
        config=NativeMiniWoBTaskConfig(task_name="click-checkboxes"),
        backend=backend,
    )

    state = env.reset()
    assert state["instruction"] == "Select Apple and Orange"
    assert "checkbox" in state["observation"]

    next_state, reward, done, info = env.step("CLICK ref=1")
    assert backend.actions == ["click('1')"]
    assert reward == 1.0
    assert done is True
    assert info["success"] is True
    assert "checked" in next_state["observation"]

    env.close()
    assert backend.closed is True


def test_native_env_noop_preserves_current_state():
    backend = DummyBackend()
    env = NativeMiniWoBClickCheckboxesEnv(
        config=NativeMiniWoBTaskConfig(task_name="click-checkboxes"),
        backend=backend,
    )
    current_state = env.reset()

    next_state, reward, done, info = env.step("NOOP")

    assert next_state == current_state
    assert reward == 0.0
    assert done is False
    assert info["success"] is False
    assert backend.actions == []


def test_native_env_builds_text_observation_from_axtree_object():
    env = NativeMiniWoBClickCheckboxesEnv(
        config=NativeMiniWoBTaskConfig(task_name="click-checkboxes"),
        backend=DummyAXTreeBackend(),
    )

    state = env.reset()

    assert state["instruction"] == "Select NRFk and click Submit."
    assert state["observation"] == (
        "RootWebArea 'Click Checkboxes Task'\n"
        "[18] checkbox 'NRFk'\n"
        "[33] checkbox 'vf' checked\n"
        "[36] button 'Submit'"
    )


class ThreadCheckingBackend:
    def __init__(self) -> None:
        self.created_on = threading.get_ident()
        self.call_threads = []
        self.closed = False
        try:
            asyncio.get_running_loop()
            self.running_loop_available = True
        except RuntimeError:
            self.running_loop_available = False

    def reset(self):
        self.call_threads.append(threading.get_ident())
        return {
            "goal": "Select Apple",
            "axtree_txt": "RootWebArea\n[1] checkbox 'Apple'",
            "pruned_html": "<input>Apple</input>",
        }, {"seed": 7}

    def step(self, action_text):
        self.call_threads.append(threading.get_ident())
        return {
            "goal": "Select Apple",
            "axtree_txt": "RootWebArea\n[1] checkbox 'Apple' checked",
            "pruned_html": "<input checked>Apple</input>",
        }, 1.0, True, False, {"success": True}

    def close(self):
        self.call_threads.append(threading.get_ident())
        self.closed = True


def test_native_env_uses_worker_thread_when_event_loop_is_running(monkeypatch):
    created = {}
    main_thread = threading.get_ident()

    def fake_create_backend(self):
        backend = ThreadCheckingBackend()
        created["backend"] = backend
        return backend

    monkeypatch.setattr(NativeMiniWoBClickCheckboxesEnv, "_create_backend", fake_create_backend)

    async def scenario():
        env = NativeMiniWoBClickCheckboxesEnv(
            config=NativeMiniWoBTaskConfig(task_name="click-checkboxes", use_worker_thread=True)
        )
        state = env.reset()
        assert state["instruction"] == "Select Apple"
        env.step("CLICK ref=1")
        env.close()

    asyncio.run(scenario())

    backend = created["backend"]
    assert backend.created_on != main_thread
    assert backend.running_loop_available is True
    assert backend.closed is True
    assert backend.call_threads
    assert set(backend.call_threads) == {backend.created_on}


def test_native_env_can_disable_worker_thread_even_when_event_loop_is_running(monkeypatch):
    created = {}
    main_thread = threading.get_ident()

    def fake_create_backend(self):
        backend = ThreadCheckingBackend()
        created["backend"] = backend
        return backend

    monkeypatch.setattr(NativeMiniWoBClickCheckboxesEnv, "_create_backend", fake_create_backend)

    async def scenario():
        env = NativeMiniWoBClickCheckboxesEnv(
            config=NativeMiniWoBTaskConfig(task_name="click-checkboxes", use_worker_thread=False)
        )
        state = env.reset()
        assert state["instruction"] == "Select Apple"
        env.step("CLICK ref=1")
        env.close()

    asyncio.run(scenario())

    backend = created["backend"]
    assert backend.created_on == main_thread
    assert backend.call_threads
    assert set(backend.call_threads) == {main_thread}


def test_ensure_playwright_browser_installs_chromium(monkeypatch):
    calls = []

    def fake_run(cmd, check):
        calls.append((cmd, check))

    monkeypatch.setattr("src.env.native_browsergym_task.subprocess.run", fake_run)
    monkeypatch.delenv("OPENENVRL_PLAYWRIGHT_CHROMIUM_READY", raising=False)
    ensure_playwright_browser()
    ensure_playwright_browser()

    assert calls == [(["playwright", "install", "chromium"], True)]
