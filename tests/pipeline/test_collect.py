from src.data.storage import read_jsonl
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


def test_run_collection_loop_writes_output_after_each_success(tmp_path):
    output_path = tmp_path / "success.jsonl"
    observed_lengths = []

    class ObservingEnv(DummyEnv):
        def reset(self):
            if output_path.exists():
                observed_lengths.append(len(read_jsonl(output_path)))
            else:
                observed_lengths.append(0)
            return super().reset()

    episodes = run_collection_loop(
        env_factory=ObservingEnv,
        policy=DummyPolicy(),
        target_successes=3,
        max_steps=1,
        output_path=str(output_path),
    )

    assert len(episodes) == 3
    # each episode should see the file already containing every prior success,
    # proving results are persisted incrementally rather than only at the end.
    assert observed_lengths == [0, 1, 2]
    assert len(read_jsonl(output_path)) == 3


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
