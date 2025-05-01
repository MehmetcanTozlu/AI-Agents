from transformers import pipeline
from pydantic import BaseModel
import json
import re

class AnswerWithJustification(BaseModel):
    answer: str
    justification: str

# Load a text-generation model capable of JSON output (e.g., Mistral-7B)
generator = pipeline("text-generation", model="/meta-llama/Llama-3.2-1B-Instruct")

# Create a structured prompt
prompt = """[INST]
Generate a JSON response with 'answer' and 'justification' fields. Question: 
What weighs more, a pound of bricks or a pound of feathers?
[/INST]"""

# Generate and parse response
response = generator(prompt, max_length=200, num_return_sequences=1)
raw_output = response[0]['generated_text'].split("[/INST]")[-1].strip()
raw_output = re.sub(r"```json|```", "", raw_output).strip()

try:
    parsed_output = json.loads(raw_output)
    parsed = AnswerWithJustification(**parsed_output)
    print(parsed)
except json.JSONDecodeError as e:
    print("Failed to parse model output:", str(e))
    print("Raw Output:", raw_output)