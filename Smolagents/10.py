from smolagents import (
    CodeAgent,
    TransformersModel,
    Tool,
)
from PIL import Image


image_generation_tool = Tool.from_space(
    space_id="black-forest-labs/FLUX.1-dev",
    name="image_generator",
    description="Generate an image from prompt",
)

model = TransformersModel(
    model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
)

agent = CodeAgent(tools=[image_generation_tool], model=model)

prompt = "A grand superhero-themed party at Wayne Manor, with Alfred overseeing a luxurious gala"

result = agent.run(
    "Improve this prompt, then generate an image of it.",
    additional_args={"user_prompt": f"{prompt}"},
    max_steps=30,
)

image = Image.open(result)
image.save("YOUR_OUTPUT_PATH")
