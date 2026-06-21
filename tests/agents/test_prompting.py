from src.agents.prompting import build_action_prompt


def test_build_action_prompt_includes_instruction_and_observation():
    prompt = build_action_prompt(
        instruction="Select Apple",
        observation="Apple checkbox visible",
    )
    assert "Select Apple" in prompt
    assert "Apple checkbox visible" in prompt
