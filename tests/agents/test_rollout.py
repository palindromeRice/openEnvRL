from src.agents.rollout import run_episode


class DummyEnv:
    def reset(self):
        return {"instruction": "Select Apple", "observation": "Apple"}

    def step(self, action):
        return {"instruction": "Select Apple", "observation": "Done"}, 1.0, True, {"success": True}


class DummyPolicy:
    def act(self, instruction, observation):
        return "CLICK ref=1"


def test_run_episode_collects_steps():
    episode = run_episode(DummyEnv(), DummyPolicy(), max_steps=3)
    assert len(episode["steps"]) == 1
    assert episode["steps"][0]["action"] == "CLICK ref=1"
    assert episode["success"] is True
