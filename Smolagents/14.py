import os
os.environ["SERPER_API_KEY"] = "YOUR_TOKEN"
import math
from PIL import Image
from typing import Optional, Tuple
from smolagents import (
    CodeAgent,
    GoogleSearchTool,
    DuckDuckGoSearchTool,
    TransformersModel,
    VisitWebpageTool, 
    tool  
)


@tool
def calculate_cargo_travel_time(
    origin_coords: Tuple[float, float],
    destination_coords: Tuple[float, float],
    cruising_speed_kmh: Optional[float] = 750.0,
) -> float:
    """
    Calculate the travel time for a cargo plane between two points on Earth using great-circle distance.

    Args:
        origin_coords (Tuple): Tuple of (latitude, longitude) for the starting point
        destination_coords (Tuple): Tuple of (latitude, longitude) for the destination
        cruising_speed_kmh (Optipnal): Optional cruising speed in km/h (defaults to 750km/h for typical cargo planes)

    Returns:
        float: The estimated travel time in hours
    
    Example:
        >>> # Chicago (41.8781° N, 87.6298° W) to Sydney (33.8688° S, 151.2093° E)
        >>> result = calculate_cargo_travel_time((41.8781, -87.6298), (-33.8688, 151.2093))
    """

    def to_radians(degrees: float) -> float:
        
        return degrees * (math.pi / 180)

    # Extract coordinates
    lat1, lon1 = map(to_radians, origin_coords)
    lat2, lon2 = map(to_radians, destination_coords)

    # Earth's radius in kilometers
    EARTH_RADIUS_KM = 6371.0

    # Calculate great-circle distance using the haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        math.sin(dlat / 2) ** 2 +
        math.cos(lat1) * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))
    distance = EARTH_RADIUS_KM * c

    # Add 10% to account for non-direct routes and air traffic controls
    actual_distance = distance * 1.1

    # Calculate flight time
    # Add 1 hour for takeoff and landing procedures
    flight_time = (actual_distance / cruising_speed_kmh) + 1.0

    # Format the results
    return round(flight_time, 2)


model = TransformersModel(
    model_id="model_from_local_path",
    max_new_tokens=8192,
    device_map="cuda",
)

task = """
Find all Batman filming locations in the world, calculate the time to transfer via cargo plane to here (we're in Gotham, 40.7128° N, 74.0060° W), and return them to me as a pandas dataframe.
Also give me some supercar factories with the same cargo plane transfer time.
"""

agent = CodeAgent(
    tools=[
        GoogleSearchTool("serper"),
        VisitWebpageTool(),
        calculate_cargo_travel_time,
    ],
    model=model,
    additional_authorized_imports=["pandas"],
    max_steps=20,
)

result = agent.run(task)

print(result)

print('##########################################################################################################')

# Planning steps allow the agent to think ahead and plan its next steps, which can be useful for more complex tasks.

agent.planning_interval = 4

detailed_report = agent.run(
f"""You're an expert analyst. You make comprehensive reports after visiting many websites.
Don't hesitate to search for many queries at once in a for loop.
For each data point that you find, visit the source url to confirm numbers.

{task}"""
)

print(detailed_report)