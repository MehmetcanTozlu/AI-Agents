"""
using transformers.pipeline
"""
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

prompt = "What is the capital of Italy?"

response = generator(prompt, max_length=33, num_return_sequences=1)

print(response[0]["generated_text"])


print() # ############################################################################################################################


"""
Using a custom fine-tuned model
"""
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

inputs = tokenizer("What is the capital of Italy", return_tensors="pt")
outputs = model.generate(**inputs, max_length=33)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
