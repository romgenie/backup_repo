"""
Knowledge agent implementation.
"""
from typing import List
from agents import Agent, WebSearchTool
from src.agents.knowledge.prompts import BASE_INSTRUCTIONS, VOICE_INSTRUCTIONS
from src.tools.file_search.file_search import get_file_search_tool

def create_knowledge_agent(use_voice_optimization: bool = False) -> Agent:
    """
    Create a knowledge agent that provides product information.
    If no vector store is available, falls back to web search.
    
    Args:
        use_voice_optimization: Whether to use voice-optimized instructions
        
    Returns:
        Agent: The initialized knowledge agent
    """
    instructions = VOICE_INSTRUCTIONS if use_voice_optimization else BASE_INSTRUCTIONS
    
    # Get file search tool
    file_search_tool = get_file_search_tool()
    
    # Prepare tools list
    tools: List = []
    
    # If file search tool is available, use it
    if file_search_tool:
        tools.append(file_search_tool)
    else:
        # If no file search tool, add a note to instructions
        fallback_note = (
            "\n\nIMPORTANT: No vector store is available for file search. "
            "You will need to use your built-in knowledge to answer product questions. "
            "If you don't have the information, politely explain that you don't have "
            "access to the product database at the moment."
        )
        instructions += fallback_note
    
    return Agent(
        name="KnowledgeAgent",
        instructions=instructions,
        tools=tools,
    )