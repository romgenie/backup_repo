"""
Tools that can be used by agents in the application.
This module contains function tools for various agent capabilities.
"""

import datetime
import time
from typing import Literal
from pydantic import BaseModel, Field

from agents import function_tool

# =========== Weather Tool Models ===========

class WeatherUnits(BaseModel):
    """Units for weather information"""
    units: Literal["metric", "imperial"] = Field(
        description="The units to use for weather information (metric or imperial)",
    )

class WeatherLocation(BaseModel):
    """Location for weather information"""
    location: str = Field(
        description="The city and country (e.g., 'London,UK')",
    )

class WeatherRequest(WeatherLocation, WeatherUnits):
    """Combined weather request parameters"""
    pass

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
    # Mock implementation
    weather_conditions = ["sunny", "cloudy", "rainy", "snowy", "windy"]
    temperature = 20 if units == "metric" else 68
    condition = weather_conditions[hash(location + str(datetime.date.today())) % len(weather_conditions)]
    
    return f"The weather in {location} is currently {condition} with a temperature of {temperature}°{'C' if units == 'metric' else 'F'}"

# =========== News Tool Models ===========

class NewsQuery(BaseModel):
    """News search query parameters"""
    query: str = Field(
        description="The search query for news articles",
    )
    max_results: int = Field(
        description="Maximum number of results to return (1-10)",
        ge=1,
        le=10,
        default=3
    )

@function_tool
def search_news(query: str, max_results: int = 3) -> str:
    """
    Search for news articles matching a query.
    
    Args:
        query: The search query
        max_results: Maximum number of results to return (1-10)
        
    Returns:
        A list of news headlines and snippets
    """
    # Mock implementation
    mock_results = [
        {"title": f"Latest developments in {query}", "snippet": f"New research shows promising results related to {query}..."},
        {"title": f"Analysis: The impact of {query} on industry", "snippet": f"Experts weigh in on how {query} is changing the landscape..."},
        {"title": f"Interview with {query} specialist", "snippet": f"We spoke with leading researchers about their work on {query}..."},
        {"title": f"{query} breakthrough announced", "snippet": f"A major discovery in the field of {query} was announced today..."},
    ]
    
    results = mock_results[:min(max_results, len(mock_results))]
    formatted_results = "\n\n".join([f"**{r['title']}**\n{r['snippet']}" for r in results])
    
    return f"Found {len(results)} results for '{query}':\n\n{formatted_results}"

# =========== Calendar Tool Models ===========

class CalendarEvent(BaseModel):
    """Calendar event details"""
    title: str = Field(
        description="The title of the event",
    )
    date: str = Field(
        description="The date of the event (YYYY-MM-DD)",
    )
    time: str = Field(
        description="The time of the event (HH:MM)",
        default="12:00"
    )
    duration: int = Field(
        description="The duration of the event in minutes",
        default=60
    )

@function_tool
def add_calendar_event(title: str, date: str, time: str = "12:00", duration: int = 60) -> str:
    """
    Add an event to the calendar.
    
    Args:
        title: The title of the event
        date: The date of the event (YYYY-MM-DD)
        time: The time of the event (HH:MM)
        duration: The duration of the event in minutes
        
    Returns:
        A confirmation message
    """
    # Mock implementation
    event_id = hash(f"{title}{date}{time}{duration}{time.time()}")
    return f"Event '{title}' added to calendar on {date} at {time} for {duration} minutes. (ID: {event_id % 10000})"

# Get all tools as a list
def get_all_tools():
    """Return all available tools"""
    return [get_weather, search_news, add_calendar_event]