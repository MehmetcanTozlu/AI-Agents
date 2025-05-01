from smolagents import CodeAgent, TransformersModel, HfApiModel
import numpy as np
import time
import datetime


model = TransformersModel(
    model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    max_new_tokens=40960,
    device_map="cuda:0"
)

agent = CodeAgent(tools=[], model=HfApiModel(), additional_authorized_imports=['datetime'])

# Other choice
#agent = CodeAgent(tools=[], model=model, additional_authorized_imports=['datetime'])

agent.run(
    """
    Alfred needs to prepare for the party. Here are the tasks:
    1. Prepare the drinks - 30 minutes
    2. Decorate the mansion - 60 minutes
    3. Set up the menu - 45 minutes
    4. Prepare the music and playlist - 45 minutes

    If we start right now, at what time will the party be ready?
    """
)

# Push to HF
# Change to your username and repo name
# agent.push_to_hub('xxx/yyy')


# Pull from HF
# Change to your username and repo name
# alfred_agent = agent.from_hub('xxx/yyy', trust_remote_code=True)


# alfred_agent.run("Give me the best playlist for a party at Wayne's mansion. The party idea is a 'villain masquerade' theme")
