"""
Using Huggingface Transformers
"""
from transformers import pipeline

generator = pipeline("text-generation", model="meta-llama/Llama-3.2-1B-Instruct")

context = "The most recent advancements in NLP are being driven by Large Language Models (LLMs). These models outperform their smaller counterparts and have become invaluable for developers who are creating applications with NLP capabilities. Developers can tap into these models through Hugging Face's `transformers` library, or by utilizing OpenAI and Cohere's offerings through the `openai` and `cohere` libraries, respectively."
question = "Which model providers offer LLMs?"

prompt = f"""Answer the question based on the context below. If the question cannot be answered using the information provided, answer with "I don't know".

Context: {context}

Question: {question}

Answer: """

response = generator(prompt, max_length=200)
print(response[0]['generated_text'], flush=True, end="")


print("############################################################################################################################")


"""
Using LangChain with Huggingface models
"""
from langchain_community.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Load a Hugging Face model via LangChain
llm = HuggingFaceHub(repo_id="meta-llama/Llama-3.2-1B-Instruct", huggingfacehub_api_token="<YOUR_HUGGING_FACE_API_KEY>")

# Define the prompt template
template = """Answer the question based on the context below. If the question cannot be answered using the information provided, answer with "I don't know".

Context: {context}

Question: {question}

Answer: """

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template
)

# Create a chain
chain = LLMChain(llm=llm, prompt=prompt)

# Define the context and question
context = "The most recent advancements in NLP are being driven by Large Language Models (LLMs). These models outperform their smaller counterparts and have become invaluable for developers who are creating applications with NLP capabilities. Developers can tap into these models through Hugging Face's `transformers` library, or by utilizing OpenAI and Cohere's offerings through the `openai` and `cohere` libraries, respectively."
question = "Which model providers offer LLMs?"

# Generate text
response = chain.run(context=context, question=question)
print(response)
