import requests
from PIL import Image
from io import BytesIO


image_urls = {
    "https://upload.wikimedia.org/wikipedia/commons/e/e8/The_Joker_at_Wax_Museum_Plus.jpg", # Joker image
    "https://upload.wikimedia.org/wikipedia/en/9/98/Joker_%28DC_Comics_character%29.jpg" # Joker image
}

images = []
for url in image_urls:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    images.append(image)

from smolagents import CodeAgent, TransformersModel, OpenAIServerModel, InferenceClientModel

model = InferenceClientModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct")

# model = TransformersModel(
#     model_id="/mnt/dev-s1/Qwen2.5-VL-7B-Instruct",
#     max_new_tokens=4096,
#     device_map="cuda",
# )

agent = CodeAgent(
    model=model,
    tools=[],
    max_steps=20,
    verbosity_level=2,
)

response = agent.run(
    """
    Describe the costume and makeup that the comic character in these photos is wearing and return the description.
    Tell me if the guest is The Joker or Wonder Woman.
    """,
    images=images,
)
print(response)

