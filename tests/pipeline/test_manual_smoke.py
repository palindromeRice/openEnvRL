from src.pipeline import run_all


class DummyEnv:
    def __init__(self) -> None:
        self.state_index = 0
        self.closed = False
        self.states = [
            {"instruction": "Select Apple and Orange", "observation": "RootWebArea\n[1] checkbox 'Apple'\n[2] checkbox 'Orange'\n[3] button 'Submit'"},
            {"instruction": "Select Apple and Orange", "observation": "RootWebArea\n[1] checkbox 'Apple' checked\n[2] checkbox 'Orange'\n[3] button 'Submit'"},
            {"instruction": "Select Apple and Orange", "observation": "RootWebArea\n[1] checkbox 'Apple' checked\n[2] checkbox 'Orange' checked\n[3] button 'Submit'"},
        ]

    def reset(self):
        self.state_index = 0
        return self.states[0]

    def step(self, action):
        self.state_index += 1
        done = self.state_index >= 2
        next_state = self.states[min(self.state_index, 2)]
        reward = 1.0 if done else 0.0
        return next_state, reward, done, {"success": done}

    def close(self):
        self.closed = True


def test_run_manual_smoke_phase_uses_manual_policy_and_writes_metrics(tmp_path, monkeypatch):
    created = []

    def fake_env_factory():
        env = DummyEnv()
        created.append(env)
        return env

    monkeypatch.setattr(run_all, "_build_env_factory", lambda config: fake_env_factory)
    config = {
        "results_dir": str(tmp_path),
        "manual_smoke_episodes": 2,
        "max_steps_per_episode": 5,
        "env_retry_attempts": 0,
    }

    metrics = run_all.run_manual_smoke_phase(config)

    assert metrics["success_rate"] == 1.0
    assert metrics["average_reward"] == 1.0
    assert metrics["average_steps"] == 2.0
    assert len(created) == 2
    assert all(env.closed for env in created)
    assert (tmp_path / "manual_smoke_metrics.json").exists()
