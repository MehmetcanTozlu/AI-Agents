from smolagents import (
    CodeAgent,
    TransformersModel,
    DuckDuckGoSearchTool,
    FinalAnswerTool,
    Tool,
    tool,
    VisitWebpageTool
)


model = TransformersModel(
    model_id="/Qwen/Qwen2.5-Coder-32B-Instruct",
    max_new_tokens=8192,
    device_map="cuda"
)


@tool
def suggest_menu(occasion: str) -> str:
    """
    Suggest a menu based on the occasion.

    Args:
        occasion (str): The type of occasion for the party.
    """
    if occasion == "casual":
        return "Pizza, snacks and drinks"
    elif occasion == "formal":
        return "3-course dinner with wine and dessert."
    elif occasion == "superhero":
        return "Buffet with high-energy with healthy food."
    else:
        return "Custom menu for the butler."


@tool
def catering_service_tool(query: str) -> str:
    """
    This tool returns the highest-rated catering service in Gotham City.

    Args:
        query (str): A search term for finding catering services.
    """

    services = {
        "Gotham Catering Co.": 4.9,
        "Wayne Manor Catering": 4.8,
        "Gotham City Events": 4.7,
    }

    # Find highest-rated catering service
    best_service = max(services, key=services.get)

    return best_service


class SuperheroPartyThemeTool(Tool):

    name = "superhero_party_theme_generator"
    description = """
    This tool suggests creative superhero-themed party ideas based on a category.
    It returns a unique party theme idea."""

    inputs = {
        "category": {
            "type": "string",
            "description": "The type"
        }
    }

    output_type = "string"

    def forward(self, category: str):
        
        themes = {
            "classic heroes": "Justice League Gala: Guests come dressed as their favorite DC heroes with themed cocktails like 'The Kryptonite Punch'.",
            "villain masquerade": "Gotham Rogues' Ball: A mysterious masquerade where guests dress as classic Batman villains.",
            "futuristic Gotham": "Neo-Gotham Night: A cyberpunk-style party inspired by Batman Beyond, with neon decorations and futuristic gadgets."
        }

        return themes.get(category.lower())


agent = CodeAgent(
    tools=[
        DuckDuckGoSearchTool(),
        VisitWebpageTool(),
        suggest_menu,
        catering_service_tool,
        SuperheroPartyThemeTool()
    ],
    model=model,
    max_steps=10,
    verbosity_level=2,
)

agent.run("Give me the best playlist for a party at the Wayne's mansion. The party idea is a 'vilain masquerade' theme")
