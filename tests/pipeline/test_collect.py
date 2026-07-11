from uuid import uuid4

import pytest

from src.data.storage import read_jsonl, write_jsonl_atomic
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


def successful_episode(episode_id: str) -> dict:
    return {
        "episode_id": episode_id,
        "success": True,
        "total_reward": 1.0,
        "steps": [{"action": "CLICK ref=1"}],
    }


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


def test_run_collection_loop_resumes_valid_saved_successes(tmp_path):
    output_path = tmp_path / "success.jsonl"
    write_jsonl_atomic(output_path, [successful_episode(str(uuid4()))])
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
        output_path=str(output_path),
    )

    assert len(episodes) == 2
    assert len(created) == 1
    assert len(read_jsonl(output_path)) == 2


def test_run_collection_loop_rejects_duplicate_saved_episode_ids(tmp_path):
    output_path = tmp_path / "success.jsonl"
    duplicate_id = str(uuid4())
    write_jsonl_atomic(
        output_path,
        [successful_episode(duplicate_id), successful_episode(duplicate_id)],
    )

    with pytest.raises(ValueError, match="duplicate episode_id"):
        run_collection_loop(
            env_factory=DummyEnv,
            policy=DummyPolicy(),
            target_successes=2,
            max_steps=1,
            output_path=str(output_path),
        )


def test_run_collection_loop_rejects_legacy_non_uuid4_episode_id(tmp_path):
    output_path = tmp_path / "success.jsonl"
    write_jsonl_atomic(output_path, [successful_episode("legacy-id")])

    with pytest.raises(ValueError, match="UUID4"):
        run_collection_loop(
            env_factory=DummyEnv,
            policy=DummyPolicy(),
            target_successes=2,
            max_steps=1,
            output_path=str(output_path),
        )


def test_run_collection_loop_rejects_saved_collection_above_target(tmp_path):
    output_path = tmp_path / "success.jsonl"
    write_jsonl_atomic(
        output_path,
        [successful_episode(str(uuid4())), successful_episode(str(uuid4()))],
    )

    with pytest.raises(ValueError, match="exceeds target_successes"):
        run_collection_loop(
            env_factory=DummyEnv,
            policy=DummyPolicy(),
            target_successes=1,
            max_steps=1,
            output_path=str(output_path),
        )


def test_run_collection_loop_rejects_saved_non_successful_episode(tmp_path):
    output_path = tmp_path / "success.jsonl"
    episode = successful_episode(str(uuid4()))
    episode["success"] = False
    write_jsonl_atomic(output_path, [episode])

    with pytest.raises(ValueError, match="success=True"):
        run_collection_loop(
            env_factory=DummyEnv,
            policy=DummyPolicy(),
            target_successes=2,
            max_steps=1,
            output_path=str(output_path),
        )


def test_run_collection_loop_rejects_saved_non_numeric_reward(tmp_path):
    output_path = tmp_path / "success.jsonl"
    episode = successful_episode(str(uuid4()))
    episode["total_reward"] = "1.0"
    write_jsonl_atomic(output_path, [episode])

    with pytest.raises(ValueError, match="numeric total_reward"):
        run_collection_loop(
            env_factory=DummyEnv,
            policy=DummyPolicy(),
            target_successes=2,
            max_steps=1,
            output_path=str(output_path),
        )


def test_run_collection_loop_rejects_saved_episode_without_steps(tmp_path):
    output_path = tmp_path / "success.jsonl"
    episode = successful_episode(str(uuid4()))
    episode["steps"] = []
    write_jsonl_atomic(output_path, [episode])

    with pytest.raises(ValueError, match="nonempty steps"):
        run_collection_loop(
            env_factory=DummyEnv,
            policy=DummyPolicy(),
            target_successes=2,
            max_steps=1,
            output_path=str(output_path),
        )


def test_run_collection_loop_rejects_non_object_saved_row(tmp_path):
    output_path = tmp_path / "success.jsonl"
    write_jsonl_atomic(output_path, ["not-an-episode"])

    with pytest.raises(ValueError, match="object"):
        run_collection_loop(
            env_factory=DummyEnv,
            policy=DummyPolicy(),
            target_successes=2,
            max_steps=1,
            output_path=str(output_path),
        )


def test_run_collection_loop_does_not_replace_valid_file_for_malformed_new_success(
    tmp_path, monkeypatch
):
    output_path = tmp_path / "success.jsonl"
    original = successful_episode(str(uuid4()))
    write_jsonl_atomic(output_path, [original])
    malformed = successful_episode("legacy-id")

    monkeypatch.setattr(
        "src.pipeline.collect.run_episode", lambda *args, **kwargs: malformed
    )

    with pytest.raises(ValueError, match="UUID4"):
        run_collection_loop(
            env_factory=DummyEnv,
            policy=DummyPolicy(),
            target_successes=2,
            max_steps=1,
            output_path=str(output_path),
        )

    assert read_jsonl(output_path) == [original]
