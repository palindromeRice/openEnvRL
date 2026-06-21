from src.env.action_schema import format_action, parse_model_action


def test_parse_model_action_accepts_canonical_click():
    action = parse_model_action("CLICK ref=12")
    assert action["type"] == "CLICK"
    assert action["ref"] == 12


def test_format_action_returns_env_payload():
    payload = format_action({"type": "CLICK", "ref": 12})
    assert payload is not None
