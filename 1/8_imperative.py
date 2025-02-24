from langchain_community.llms import HuggingFaceHub
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{question}"),
])

model = HuggingFaceHub(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    model_kwargs={"temperature": 0.5},
    huggingfacehub_api_token="<YOUR_HUGGING_FACE_API_KEY>",
)

@chain
def hf_chatbot(values):
    prompt = template.invoke(values)
    return model.invoke(prompt)

print(hf_chatbot.invoke({"question": "Which model providers offer LLMs?"}))