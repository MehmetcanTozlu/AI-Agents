from smolagents import (
    ToolCallingAgent,
    TransformersModel,
    DuckDuckGoSearchTool,
)


model = TransformersModel(
    model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    max_new_tokens=4096,
    device_map="cuda"
)

# Using ToolCallingAgent instead of CodeAgent
agent = ToolCallingAgent(tools={DuckDuckGoSearchTool()}, model=model)

agent.run("Search for the best music recommendation for a party at the Wayne's mansion.")
