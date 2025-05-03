from smolagents import (
    Tool,
    CodeAgent,
    TransformersModel,
    DuckDuckGoSearchTool,
    VisitWebpageTool,
)


model = TransformersModel(
    model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    max_new_tokens=8192,
    device_map="cuda",
)


class SuperheroPartyThemeTool(Tool):
    
    name = "superhero_party_theme_generator"
    description = """
    This tool suggests creative superhero-themed party ideas based on a category.
    It returns a unique party theme idea.
    """

    inputs = {
        "category": {
            "type": "string",
            "description": "The type of superhero party (e.g. 'classic heroes', 'villain masquerade', 'futuristic Gotham')."
        }
    }

    output_type = "string"

    def forward(self, category: str) -> str:
        
        themes = {
            "classic heroes": "Justice League Gala: Guests come dressed as their favorite DC heroes with themed cocktails like 'The Kryptonite Punch'.",
            "villain masquerade": "Gotham Rogues' Ball: A mysterious masquerade where guests dress as classic Batman villains.",
            "futuristic Gotham": "Neo-Gotham Night: A cyberpunk-style party inspired by Batman Beyond, with neon decorations and futuristic gadgets.",
        }

        return themes.get(category.lower(), "Themed party idea not found. Try 'classic heroes', 'villain masquerade', or 'futuristic Gotham'.")


party_theme_tool = SuperheroPartyThemeTool()

agent = CodeAgent(
    tools=[
        party_theme_tool,
        DuckDuckGoSearchTool(),
        VisitWebpageTool(),
    ],
    model=model)

result = agent.run(
    "What would be a good superhero party idea for a 'classic heroes' theme?"
)
print(result)


# party_theme_tool.push_to_hub("{your_username}/party_theme_tool", token="<YOUR_HUGGINGFACEHUB_API_TOKEN>")

