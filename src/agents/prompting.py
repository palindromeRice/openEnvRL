def build_action_prompt(instruction: str, observation: str) -> str:
    return (
        "You are a browser agent.\n"
        f"Task: {instruction}\n"
        f"Observation: {observation}\n"
        "Respond with one action only.\n"
        "Valid formats:\n"
        "- CLICK ref=<number>\n"
        "- click('<visible text>')\n"
        "Do not explain. Do not write a sentence."
    )
