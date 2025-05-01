from transformers import pipeline

# Load a text-generation model
generator = pipeline("text-generation", model="meta-llama/Llama-3.2-1B-Instruct")

# Define the context and question
context = "The most recent advancements in NLP are being driven by Large Language Models (LLMs). These models outperform their smaller counterparts and have become invaluable for developers who are creating applications with NLP capabilities. Developers can tap into these models through Hugging Face's `transformers` library, or by utilizing OpenAI and Cohere's offerings through the `openai` and `cohere` libraries, respectively."
question = "Which model providers offer LLMs?"

# Create the chat-based prompt
prompt = f"""Answer the question based on the context below. If the question cannot be answered using the information provided, answer with "I don't know".

Context: {context}

Question: {question}

Answer: """

# Generate a response
response = generator(prompt, max_length=200, num_return_sequences=1)
print(response[0]['generated_text'])


print("############################################################################################################################")


from langchain_community.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# Load a Hugging Face model via LangChain
llm = HuggingFaceHub(repo_id="meta-llama/Llama-3.2-1B-Instruct", huggingfacehub_api_token="<YOUR_HUGGING_FACE_API_KEY>")

# Define the chat-based prompt template
template = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            'Answer the question based on the context below. If the question cannot be answered using the information provided, answer with "I don\'t know".'
        ),
        HumanMessagePromptTemplate.from_template("Context: {context}"),
        HumanMessagePromptTemplate.from_template("Question: {question}"),
    ]
)

# Create a chain
chain = LLMChain(llm=llm, prompt=template)

# Define the context and question
context = "The most recent advancements in NLP are being driven by Large Language Models (LLMs). These models outperform their smaller counterparts and have become invaluable for developers who are creating applications with NLP capabilities. Developers can tap into these models through Hugging Face's `transformers` library, or by utilizing OpenAI and Cohere's offerings through the `openai` and `cohere` libraries, respectively."
question = "Which model providers offer LLMs?"

# Generate text
response = chain.run(context=context, question=question)
print(response)
