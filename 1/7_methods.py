from langchain_community.llms import HuggingFaceHub

# Single completion
model = HuggingFaceHub(repo_id="meta-llama/Llama-3.2-1B-Instruct", huggingfacehub_api_token="<YOUR_HUGGING_FACE_API_KEY>")
completion = model.invoke("Hi there!")
print(completion)

# Batch processing
completions = model.batch(["Hi there!", "Bye!"])
print(completions)

# Streaming (not directly supported, but can simulate)
for token in model.stream("Bye!"):
    print(token, end="", flush=True)