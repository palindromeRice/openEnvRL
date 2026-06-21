from __future__ import annotations

from pathlib import Path
from typing import Any


def build_comparison_table(metrics: dict[str, dict[str, Any]]) -> str:
    lines = [
        "| Model | Success Rate | Avg Reward | Avg Steps |",
        "| --- | --- | --- | --- |",
    ]
    for model_name, values in metrics.items():
        lines.append(
            f"| {model_name} | {values['success_rate']:.3f} | {values['average_reward']:.3f} | {values['average_steps']:.3f} |"
        )
    return "\n".join(lines)


def build_analysis_sections() -> list[str]:
    return [
        "What is a trajectory? A trajectory is one full episode record made of step-by-step observations, actions, rewards, and termination state.",
        "Why successful-only filtering? This first version keeps only successful trajectories so SFT learns from clear action targets instead of mixed-quality supervision.",
        "What is behavior cloning? Behavior cloning is supervised learning where the model predicts the next action from recorded successful states.",
        "Why does SFT help RL? SFT gives the policy a better starting point so online RL spends less time exploring obviously bad actions.",
        "What is GRPO? GRPO is a policy-optimization method that improves the policy from reward-ranked rollouts without needing a separate value head in the same way PPO setups often do.",
        "How does GRPO differ from PPO at a high level? PPO usually uses clipped policy updates with an explicit advantage/value estimation pipeline, while GRPO uses grouped relative preference-style reward comparisons across sampled rollouts.",
        "What reward is optimized? Native environment reward from the `click-checkboxes` task only.",
        "Why can RL outperform imitation? RL can discover action choices that produce higher reward than the limited demonstrations seen during SFT.",
        "What are the limitations? One task only, tiny data budget, T4 constraints, and simplified environment wrappers mean results are educational rather than benchmark-grade.",
    ]


def write_report(metrics: dict[str, dict[str, Any]], path: str = "results/report.md") -> str:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n\n".join(
        [
            "# Browser Agent RL Report",
            build_comparison_table(metrics),
            *build_analysis_sections(),
        ]
    )
    report_path.write_text(content, encoding="utf-8")
    return content
