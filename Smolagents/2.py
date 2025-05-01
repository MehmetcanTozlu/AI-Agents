from smolagents import (
    CodeAgent,
    DuckDuckGoSearchTool,
    HfApiModel,
    InferenceClientModel,
    TransformersModel,
    )
from huggingface_hub import login


login(token="HUGGINGFACE-TOKEN-KEY")


# from local
model = TransformersModel(
    model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    max_new_tokens=4096,
    device_map="cuda",
)

# Other choice
# model = InferenceClientModel(
#     model_id="deepseek-ai/DeepSeek-R1",
#     provider="together",
# )

# Other choice
# agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=HfApiModel())

agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=model)

agent.run("Search for the best music recommendations for a party at the Wayne's mansion.")
