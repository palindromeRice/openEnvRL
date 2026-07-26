from src.agents.grpo_tool_policy import GRPOToolPolicy


def test_grpo_tool_policy_converts_qwen_tool_call_to_native_action():
    assert GRPOToolPolicy.parse_completion(
        "<tool_call>\n<function=click>\n<parameter=target>\n21\n</parameter>\n</function>\n</tool_call>"
    ) == "click('21')"


def test_grpo_tool_policy_returns_noop_for_invalid_completion():
    assert GRPOToolPolicy.parse_completion("not a tool call") == "NOOP"


def test_grpo_tool_policy_strips_quotes_around_tool_target():
    assert GRPOToolPolicy.parse_completion(
        "<tool_call>\n<function=click>\n<parameter=target>\n24'\n"
        "</parameter>\n</function>\n</tool_call>"
    ) == "click('24')"


def test_grpo_tool_policy_rejects_malformed_non_numeric_tool_target():
    assert GRPOToolPolicy.parse_completion(
        "<tool_call>\n<function=click>\n<parameter=target>\n[27]checkbox'dmb6VH\n"
        "</parameter>\n</function>\n</tool_call>"
    ) == "NOOP"


def test_grpo_tool_policy_parses_every_tool_call_in_order():
    completion = """<tool_call><function=click><parameter=target>18</parameter></function></tool_call>
    <tool_call><function=click><parameter=target>30</parameter></function></tool_call>"""

    assert GRPOToolPolicy.parse_completion_actions(completion, "") == [
        "click('18')",
        "click('30')",
    ]


def test_grpo_tool_policy_resolves_visible_control_names_to_ids():
    completion = """<tool_call><function=click><parameter=target>7A</parameter></function></tool_call>
    <tool_call><function=click><parameter=target>Submit</parameter></function></tool_call>"""
    observation = "[18] checkbox '7A'\n[30] button 'Submit'"

    assert GRPOToolPolicy.parse_completion_actions(completion, observation) == [
        "click('18')",
        "click('30')",
    ]


def test_grpo_tool_policy_returns_queued_action_without_generating_again():
    policy = GRPOToolPolicy("/checkpoint")
    policy._pending_actions = ["click('30')"]

    assert policy.act("Select 7A", "[30] button 'Submit'") == "click('30')"


def test_grpo_tool_policy_renders_click_tool_schema():
    captured = {}

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            captured["messages"] = messages
            captured.update(kwargs)
            return "rendered prompt"

    policy = GRPOToolPolicy("/checkpoint")
    policy.tokenizer = Tokenizer()

    prompt = policy.build_prompt("Select Apple", "[12] checkbox 'Apple'")

    assert prompt == "rendered prompt"
    assert captured["tokenize"] is False
    assert captured["add_generation_prompt"] is True
    assert len(captured["tools"]) == 1
    assert captured["tools"][0].__name__ == "click"


def test_grpo_tool_policy_exposes_raw_completion_diagnostics():
    policy = GRPOToolPolicy("/checkpoint")
    policy.last_completion = "<tool_call>raw</tool_call>"
    policy.last_action = "NOOP"

    assert policy.get_last_diagnostics() == {
        "raw_completion": "<tool_call>raw</tool_call>",
        "parsed_action": "NOOP",
    }
