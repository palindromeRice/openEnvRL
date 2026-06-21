from src.analysis.report import build_comparison_table


def test_build_comparison_table_formats_three_models():
    table = build_comparison_table({
        "base": {"success_rate": 0.2, "average_reward": 0.3, "average_steps": 4.0},
        "sft": {"success_rate": 0.4, "average_reward": 0.5, "average_steps": 3.0},
        "grpo": {"success_rate": 0.6, "average_reward": 0.7, "average_steps": 2.0},
    })
    assert "base" in table.lower()
    assert "sft" in table.lower()
    assert "grpo" in table.lower()
