from src.env.action_schema import format_action, parse_model_action


def test_parse_model_action_accepts_canonical_click():
    action = parse_model_action("CLICK ref=12")
    assert action["type"] == "CLICK"
    assert action["ref"] == 12


def test_format_action_returns_env_payload():
    payload = format_action({"type": "CLICK", "ref": 12})
    assert payload is not None


def test_parse_model_action_accepts_browsergym_click_text():
    action = parse_model_action("click('Apple')")
    assert action == {"type": "CLICK_TEXT", "target": "Apple"}


def test_parse_model_action_accepts_natural_language_click():
    action = parse_model_action("Click the button 'Click Me!'.")
    assert action == {"type": "CLICK_TEXT", "target": "Click Me!"}


def test_parse_model_action_accepts_bracketed_click_ref():
    action = parse_model_action("CLICK [21]")
    assert action == {"type": "CLICK", "ref": 21}


def test_parse_model_action_accepts_noop():
    assert parse_model_action("NOOP") == {"type": "NOOP"}
    assert format_action({"type": "NOOP"})["canonical"] == "NOOP"
