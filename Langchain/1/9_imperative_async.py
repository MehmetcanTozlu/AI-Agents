from langchain_community.llms import HuggingFaceHub
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain
import asyncio

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{question}"),
])

model = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    model_kwargs={"temperature": 0.5},
    huggingfacehub_api_token="<YOUR_HUGGING_FACE_API_KEY>",
)

@chain
async def hf_chatbot(values):
    prompt = await template.ainvoke(values)
    return await model.ainvoke(prompt)

async def main():
    return await hf_chatbot.ainvoke({"question": "Which model providers offer LLMs?"})

if __name__ == "__main__":
    print(asyncio.run(main()))