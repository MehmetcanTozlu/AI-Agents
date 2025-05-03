from smolagents import (
    CodeAgent,
    TransformersModel,
    tool,
)


@tool
def catering_service_tool(query: str) -> str:
    """
    This tool returns the highest-rated catering service in Gotham City.

    Args:
        query: A search term for finding catering services.
    """

    services = {
        "Gotham Catering Co.": 4.5,
        "Wayne Manor Catering": 4.6,
        "Gotham City Events": 3.89,
    }

    best_services = max(services, key=services.get)

    return best_services


model = TransformersModel(
    model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    max_new_tokens=4096,
    device_map="cuda",
)

agent = CodeAgent(tools=[catering_service_tool], model=model)

result = agent.run("Can you give me the name of the highest-rated catering service in Gotham City?")

print(result)
