from src.data.storage import read_jsonl
from src.env.openenv_task import OpenEnvTransientError
from src.eval.run_eval import evaluate_policy


class DummyEnv:
    def __init__(self) -> None:
        self.closed = False
        self.done = False

    def reset(self):
        self.done = False
        return {"instruction": "test", "observation": "obs"}

    def step(self, action):
        self.done = True
        return {"instruction": "test", "observation": "obs"}, 1.0, True, {"success": True}

    def close(self):
        self.closed = True


class DummyPolicy:
    def act(self, instruction, observation):
        return "CLICK ref=1"


def test_evaluate_policy_closes_every_env():
    created = []

    def env_factory():
        env = DummyEnv()
        created.append(env)
        return env

    metrics = evaluate_policy(
        env_factory=env_factory,
        policy=DummyPolicy(),
        num_episodes=3,
        max_steps=1,
    )

    assert metrics["success_rate"] == 1.0
    assert len(created) == 3
    assert all(env.closed for env in created)


def test_evaluate_policy_retries_transient_openenv_errors():
    created = []

    class FlakyEnv(DummyEnv):
        def __init__(self, should_fail: bool) -> None:
            super().__init__()
            self.should_fail = should_fail

        def step(self, action):
            if self.should_fail:
                raise OpenEnvTransientError("keepalive ping timeout")
            return super().step(action)

    attempts = iter([True, False])

    def env_factory():
        env = FlakyEnv(next(attempts))
        created.append(env)
        return env

    metrics = evaluate_policy(
        env_factory=env_factory,
        policy=DummyPolicy(),
        num_episodes=1,
        max_steps=1,
        max_retries_per_episode=1,
    )

    assert metrics["success_rate"] == 1.0
    assert len(created) == 2
    assert all(env.closed for env in created)


def test_evaluate_policy_saves_episode_diagnostics(tmp_path):
    class DiagnosticPolicy(DummyPolicy):
        def get_last_diagnostics(self):
            return {
                "raw_completion": "<tool_call>raw</tool_call>",
                "parsed_action": "CLICK ref=1",
            }

    episodes_path = tmp_path / "episodes.jsonl"
    evaluate_policy(
        env_factory=DummyEnv,
        policy=DiagnosticPolicy(),
        num_episodes=1,
        max_steps=1,
        episodes_output_path=str(episodes_path),
    )

    episodes = read_jsonl(episodes_path)
    assert episodes[0]["steps"][0]["policy_diagnostics"] == {
        "raw_completion": "<tool_call>raw</tool_call>",
        "parsed_action": "CLICK ref=1",
    }
