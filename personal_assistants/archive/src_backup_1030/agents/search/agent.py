"""
Search agent implementation.
"""
from agents import Agent
from src.agents.search.prompts import BASE_INSTRUCTIONS, VOICE_INSTRUCTIONS
from src.tools.web_search.web_search import get_web_search_tool

def create_search_agent(use_voice_optimization: bool = False) -> Agent:
    """
    Create a search agent that performs web searches.
    
    Args:
        use_voice_optimization: Whether to use voice-optimized instructions
        
    Returns:
        Agent: The initialized search agent
    """
    instructions = VOICE_INSTRUCTIONS if use_voice_optimization else BASE_INSTRUCTIONS
    
    return Agent(
        name="SearchAgent",
        instructions=instructions,
        tools=[get_web_search_tool()],
    )