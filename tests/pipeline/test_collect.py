from src.pipeline.collect import collect_successful_trajectories


def test_collect_successful_trajectories_stops_after_target_successes():
    episodes = [
        {"success": True, "steps": [1]},
        {"success": False, "steps": [1]},
        {"success": True, "steps": [1]},
    ]

    def runner():
        for episode in episodes:
            yield episode

    collected = collect_successful_trajectories(runner(), target_successes=2)
    assert len(collected) == 2
