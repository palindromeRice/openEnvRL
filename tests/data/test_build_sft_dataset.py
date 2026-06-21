from src.data.build_sft_dataset import build_step_dataset


def test_build_step_dataset_expands_episode_steps():
    episodes = [{
        "steps": [
            {"instruction": "Select Apple", "observation": "Apple", "action": "CLICK ref=1"},
            {"instruction": "Select Apple", "observation": "Submit", "action": "CLICK ref=2"},
        ]
    }]
    rows = build_step_dataset(episodes)
    assert len(rows) == 2
