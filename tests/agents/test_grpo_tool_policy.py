from src.agents.grpo_tool_policy import GRPOToolPolicy


def test_grpo_tool_policy_converts_qwen_tool_call_to_native_action():
    assert GRPOToolPolicy.parse_completion(
        "<tool_call>\n<function=click>\n<parameter=target>\n21\n</parameter>\n</function>\n</tool_call>"
    ) == "click('21')"


def test_grpo_tool_policy_returns_noop_for_invalid_completion():
    assert GRPOToolPolicy.parse_completion("not a tool call") == "NOOP"
