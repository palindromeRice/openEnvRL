import pytest

from src.agents.manual_click_checkboxes_policy import ManualClickCheckboxesPolicy


def test_manual_policy_clicks_first_unchecked_target():
    policy = ManualClickCheckboxesPolicy()

    action = policy.act(
        instruction="Select Apple and Orange",
        observation=(
            "RootWebArea\n"
            "[1] checkbox 'Apple'\n"
            "[2] checkbox 'Banana'\n"
            "[3] checkbox 'Orange'\n"
            "[4] button 'Submit'"
        ),
    )

    assert action == "CLICK ref=1"


def test_manual_policy_clicks_remaining_target_then_submit():
    policy = ManualClickCheckboxesPolicy()

    second_action = policy.act(
        instruction="Select Apple and Orange",
        observation=(
            "RootWebArea\n"
            "[1] checkbox 'Apple' checked\n"
            "[2] checkbox 'Banana'\n"
            "[3] checkbox 'Orange'\n"
            "[4] button 'Submit'"
        ),
    )
    submit_action = policy.act(
        instruction="Select Apple and Orange",
        observation=(
            "RootWebArea\n"
            "[1] checkbox 'Apple' checked\n"
            "[2] checkbox 'Banana'\n"
            "[3] checkbox 'Orange' checked\n"
            "[4] button 'Submit'"
        ),
    )

    assert second_action == "CLICK ref=3"
    assert submit_action == "CLICK ref=4"


def test_manual_policy_rejects_observation_without_target_checkbox():
    policy = ManualClickCheckboxesPolicy()

    with pytest.raises(ValueError, match="Could not find target checkbox"):
        policy.act(
            instruction="Select Apple",
            observation=(
                "RootWebArea\n"
                "[2] checkbox 'Banana'\n"
                "[4] button 'Submit'"
            ),
        )


def test_manual_policy_submits_immediately_when_instruction_is_select_nothing():
    policy = ManualClickCheckboxesPolicy()

    action = policy.act(
        instruction="Select nothing",
        observation=(
            "RootWebArea\n"
            "[1] checkbox 'Apple'\n"
            "[2] checkbox 'Banana'\n"
            "[4] button 'Submit'"
        ),
    )

    assert action == "CLICK ref=4"


def test_manual_policy_ignores_trailing_click_submit_clause():
    policy = ManualClickCheckboxesPolicy()

    first_action = policy.act(
        instruction="Select NRFk and click Submit.",
        observation=(
            "RootWebArea 'Click Checkboxes Task'\n"
            "[18] checkbox 'NRFk'\n"
            "[33] checkbox 'vf' checked\n"
            "[36] button 'Submit'"
        ),
    )
    second_action = policy.act(
        instruction="Select NRFk and click Submit.",
        observation=(
            "RootWebArea 'Click Checkboxes Task'\n"
            "[18] checkbox 'NRFk' checked\n"
            "[33] checkbox 'vf' checked\n"
            "[36] button 'Submit'"
        ),
    )

    assert first_action == "CLICK ref=18"
    assert second_action == "CLICK ref=36"


def test_manual_policy_submits_when_instruction_is_select_nothing_and_click_submit():
    policy = ManualClickCheckboxesPolicy()

    action = policy.act(
        instruction="Select nothing and click Submit.",
        observation=(
            "RootWebArea 'Click Checkboxes Task'\n"
            "[18] checkbox 'NRFk'\n"
            "[33] checkbox 'vf' checked\n"
            "[36] button 'Submit'"
        ),
    )

    assert action == "CLICK ref=36"
