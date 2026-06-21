from src.eval.metrics import aggregate_metrics


def test_aggregate_metrics_computes_success_reward_and_steps():
    episodes = [
        {"success": True, "total_reward": 1.0, "num_steps": 2},
        {"success": False, "total_reward": 0.0, "num_steps": 3},
    ]
    metrics = aggregate_metrics(episodes)
    assert metrics["success_rate"] == 0.5
    assert metrics["average_reward"] == 0.5
    assert metrics["average_steps"] == 2.5
