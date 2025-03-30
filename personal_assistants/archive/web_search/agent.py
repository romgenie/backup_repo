"""
Web search agent implementation.
"""
from agents import Agent
from src.agents.web_search.prompts import BASE_INSTRUCTIONS, VOICE_INSTRUCTIONS
from src.tools.web_search.web_search import get_web_search_tool
from src.tools.arxiv.arxiv_tools import (
    search_arxiv,
    download_paper_from_arxiv,
    batch_download_papers,
    search_and_download_papers
)

def create_web_search_agent(use_voice_optimization: bool = False) -> Agent:
    """
    Create a web search agent that performs web searches and academic paper searches.
    
    Args:
        use_voice_optimization: Whether to use voice-optimized instructions
        
    Returns:
        Agent: The initialized web search agent
    """
    instructions = VOICE_INSTRUCTIONS if use_voice_optimization else BASE_INSTRUCTIONS
    
    return Agent(
        name="WebSearchAgent",
        instructions=instructions,
        tools=[
            get_web_search_tool(),
            search_arxiv,
            download_paper_from_arxiv,
            batch_download_papers,
            search_and_download_papers
        ],
    )