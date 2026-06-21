def build_action_prompt(instruction: str, observation: str) -> str:
    return (
        "You are a browser agent.\n"
        f"Task: {instruction}\n"
        f"Observation: {observation}\n"
        "Respond with one action only."
    )
