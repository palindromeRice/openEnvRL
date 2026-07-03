from types import SimpleNamespace

import pytest

from src.env.openenv_task import BrowserGymTaskConfig, OpenEnvClickCheckboxesEnv


class BackendWithTaskReset:
    def __init__(self) -> None:
        self.reset_task_name = None

    def reset(self, task_name=None):
        self.reset_task_name = task_name
        observation = SimpleNamespace(
            goal="Select Apple and Orange",
            text="RootWebArea 'click-checkboxes'",
            axtree_txt="",
            pruned_html="",
            metadata={},
        )
        return SimpleNamespace(observation=observation)


class BackendWithoutTaskReset:
    def reset(self):
        observation = SimpleNamespace(
            goal="Select Apple",
            text="RootWebArea 'click-checkboxes'",
            axtree_txt="",
            pruned_html="",
            metadata={},
        )
        return SimpleNamespace(observation=observation)


def test_reset_passes_task_name_when_supported():
    backend = BackendWithTaskReset()
    env = OpenEnvClickCheckboxesEnv(
        config=BrowserGymTaskConfig(task_name="click-checkboxes"),
        backend=backend,
    )

    state = env.reset()

    assert backend.reset_task_name == "click-checkboxes"
    assert "Select" in state["instruction"]


def test_reset_falls_back_when_backend_has_no_task_name_argument():
    env = OpenEnvClickCheckboxesEnv(
        config=BrowserGymTaskConfig(task_name="click-checkboxes"),
        backend=BackendWithoutTaskReset(),
    )

    state = env.reset()

    assert state["instruction"] == "Select Apple"


def test_reset_raises_if_remote_falls_back_to_click_test():
    backend = BackendWithTaskReset()

    def bad_reset(task_name=None):
        backend.reset_task_name = task_name
        observation = SimpleNamespace(
            goal="Click the button.",
            text="RootWebArea 'Click Test Task', focused",
            axtree_txt="",
            pruned_html="",
            metadata={},
        )
        return SimpleNamespace(observation=observation)

    backend.reset = bad_reset
    env = OpenEnvClickCheckboxesEnv(
        config=BrowserGymTaskConfig(task_name="click-checkboxes"),
        backend=backend,
    )

    with pytest.raises(RuntimeError, match="did not bind to click-checkboxes"):
        env.reset()
