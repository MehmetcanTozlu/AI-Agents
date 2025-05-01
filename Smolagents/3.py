from smolagents import (
    CodeAgent,
    tool,
    HfApiModel,
    TransformersModel,
    )


model = TransformersModel(
    model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    max_new_tokens=4096,
    device_map="cuda",
)


@tool
def suggest_menu(occasion: str) -> str:
    """
    Suggest a menu based on the occasion.

    Args:
        occasion(str): The type of occasion for the party. Allowed values are:
                        - "casual": Menu for casual party.
                        - "formal": Menu for formal party.
                        - "superhero": Menu for superhero party.
                        - "custom": Menu for custom party.
    """

    if occasion == "casual":
        return "Pizza, snacks and drinks."
    elif occasion == "formal":
        return "3-course dinner with wine and dessert."
    elif occasion == "superhero":
        return "Buffet with high-energy and healthy food."
    else:
        return "Custom menu for the butler."


agent = CodeAgent(tools=[suggest_menu], model=model)

agent.run("Prepare a superhero menu for the party.")
