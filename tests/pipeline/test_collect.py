from src.env.openenv_task import OpenEnvTransientError
from src.pipeline.collect import run_collection_loop


class DummyEnv:
    def __init__(self) -> None:
        self.closed = False

    def reset(self):
        return {"instruction": "test", "observation": "obs"}

    def step(self, action):
        return {"instruction": "test", "observation": "obs"}, 1.0, True, {"success": True}

    def close(self):
        self.closed = True


class DummyPolicy:
    def act(self, instruction, observation):
        return "CLICK ref=1"


def test_run_collection_loop_closes_envs(tmp_path):
    created = []

    def env_factory():
        env = DummyEnv()
        created.append(env)
        return env

    episodes = run_collection_loop(
        env_factory=env_factory,
        policy=DummyPolicy(),
        target_successes=2,
        max_steps=1,
        output_path=str(tmp_path / "success.jsonl"),
    )

    assert len(episodes) == 2
    assert len(created) == 2
    assert all(env.closed for env in created)


def test_run_collection_loop_retries_transient_openenv_errors(tmp_path):
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

    episodes = run_collection_loop(
        env_factory=env_factory,
        policy=DummyPolicy(),
        target_successes=1,
        max_steps=1,
        max_retries_per_episode=1,
        output_path=str(tmp_path / "success.jsonl"),
    )

    assert len(episodes) == 1
    assert len(created) == 2
    assert all(env.closed for env in created)
