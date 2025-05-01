"""
Using Huggingface Transformers
"""
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

response = generator("The earth is", max_length=33, num_return_sequences=2)
print(response[0]["generated_text"])


print() # ############################################################################################################################


"""
Using a custom fine-tuned model
"""
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

inputs = tokenizer("The earth is", return_tensors="pt")
outputs = model.generate(**inputs, max_length=33)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
