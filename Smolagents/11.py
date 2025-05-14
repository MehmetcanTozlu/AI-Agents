from smolagents import CodeAgent, DuckDuckGoSearchTool, TransformersModel


search_tool = DuckDuckGoSearchTool()

model = TransformersModel(
    model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    max_new_tokens=8192,
    device_map="cuda",
)

agent = CodeAgent(
    tools=[search_tool],
    model=model,
)

response = agent.run("Search for luxury superhero-themed party ideas, including decorations, entartainment, and catering.")

print(response)
