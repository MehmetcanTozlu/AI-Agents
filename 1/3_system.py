"""
Using Huggingface Transformers
"""
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

system_msg = "You are a helpful assistant that responds to questions with three exclamation marks."

human_msg = "What is the capital of Italy?"

prompt = f"{system_msg}\n\n{human_msg}"

response = generator(prompt, max_length=77, num_return_sequences=1)
print(response[0]["generated_text"])


print() # ############################################################################################################################


"""
Using a custom fine-tuned model
"""
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

system_msg = "You are a helpful assistant that responds to questions with three exclamations marks."
human_msg = "What is the capital of Germany?"

prompt = f"{system_msg}\n\n{human_msg}"

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=77)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
