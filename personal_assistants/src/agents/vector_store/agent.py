"""
Vector store agent implementation.
"""
from agents import Agent
from src.agents.vector_store.prompts import BASE_INSTRUCTIONS, VOICE_INSTRUCTIONS
from src.tools.vector_store.vector_store_manager import vector_store_manager_tool
from src.tools.file_search.file_search import get_file_search_tool

def create_vector_store_agent(use_voice_optimization: bool = False) -> Agent:
    """
    Create a vector store agent that manages and searches vector stores.
    
    Args:
        use_voice_optimization: Whether to use voice-optimized instructions
        
    Returns:
        Agent: The initialized vector store agent
    """
    # Choose appropriate instructions based on voice optimization
    instructions = VOICE_INSTRUCTIONS if use_voice_optimization else BASE_INSTRUCTIONS
    
    # Prepare the tools list - now just using the single consolidated tool
    tools = [vector_store_manager_tool]
    
    # Add file search tool if available
    file_search_tool = get_file_search_tool()
    if file_search_tool:
        tools.append(file_search_tool)
    
    return Agent(
        name="VectorStoreAgent",
        instructions=instructions,
        tools=tools,
    )