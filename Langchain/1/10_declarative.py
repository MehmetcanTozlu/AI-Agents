from langchain_community.llms import HuggingFaceHub
from langchain_core.prompts import ChatPromptTemplate

# Define components
template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{question}"),
])

model = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    model_kwargs={"temperature": 0.5},
    huggingfacehub_api_token="<YOUR_HUGGING_FACE_API_KEY>",
)

# Create chain
chatbot = template | model

# Single invocation
print(chatbot.invoke({"question": "Which model providers offer LLMs?"}))

# Streaming (simulated)
for token in chatbot.stream({"question": "Which model providers offer LLMs?"}):
    print(token, end="", flush=True)