"""
Knowledge agent implementation.
"""
from agents import Agent
from src.agents.knowledge.prompts import BASE_INSTRUCTIONS, VOICE_INSTRUCTIONS
from src.tools.file_search.file_search import get_file_search_tool

def create_knowledge_agent(use_voice_optimization: bool = False) -> Agent:
    """
    Create a knowledge agent that provides product information.
    
    Args:
        use_voice_optimization: Whether to use voice-optimized instructions
        
    Returns:
        Agent: The initialized knowledge agent
    """
    instructions = VOICE_INSTRUCTIONS if use_voice_optimization else BASE_INSTRUCTIONS
    
    return Agent(
        name="KnowledgeAgent",
        instructions=instructions,
        tools=[get_file_search_tool()],
    )