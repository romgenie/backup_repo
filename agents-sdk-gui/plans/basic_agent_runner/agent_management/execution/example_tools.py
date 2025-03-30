"""
Example function tools that can be used in the application.
"""

import datetime
from typing import Dict, Any
from agents import function_tool

@function_tool
def get_weather(location: str, units: str) -> str:
    """
    Get the current weather for a location.
    
    Args:
        location: The city and country (e.g., "London,UK")
        units: The units to use (metric or imperial)
        
    Returns:
        A description of the current weather
    """
    # This is a mock implementation - in a real application, you would call a weather API
    
    # Default to metric if units value is not valid
    if not units or (units != "metric" and units != "imperial"):
        print(f"DEBUG: Invalid units value '{units}', defaulting to metric")
        units = "metric"
        
    weather_conditions = ["sunny", "cloudy", "rainy", "snowy", "windy"]
    temperature = 20 if units == "metric" else 68
    
    # Handle empty location
    if not location:
        location = "Unknown location"
        
    condition = weather_conditions[hash(location + str(datetime.date.today())) % len(weather_conditions)]
    
    return f"The weather in {location} is currently {condition} with a temperature of {temperature}°{'C' if units == 'metric' else 'F'}"

@function_tool
def search_news(query: str, max_results: int) -> str:
    """
    Search for news articles matching a query.
    
    Args:
        query: The search query
        max_results: Maximum number of results to return (1-10)
        
    Returns:
        A list of news headlines and snippets
    """
    # This is a mock implementation - in a real application, you would call a news API
    mock_results = [
        {"title": f"Latest developments in {query}", "snippet": f"New research shows promising results related to {query}..."},
        {"title": f"Analysis: The impact of {query} on industry", "snippet": f"Experts weigh in on how {query} is changing the landscape..."},
        {"title": f"Interview with {query} specialist", "snippet": f"We spoke with leading researchers about their work on {query}..."},
        {"title": f"{query} breakthrough announced", "snippet": f"A major discovery in the field of {query} was announced today..."},
    ]
    
    # Ensure max_results is an integer (since we removed the default)
    try:
        max_results_int = int(max_results)
    except (ValueError, TypeError):
        max_results_int = 3  # Fallback if conversion fails
        
    results = mock_results[:min(max_results_int, len(mock_results))]
    formatted_results = "\n\n".join([f"**{r['title']}**\n{r['snippet']}" for r in results])
    
    return f"Found {len(results)} results for '{query}':\n\n{formatted_results}"