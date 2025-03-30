"""
Web search tools using OpenAI's WebSearchTool.
"""
from agents import WebSearchTool

def get_web_search_tool() -> WebSearchTool:
    """
    Return an instance of the WebSearchTool.
    
    Returns:
        WebSearchTool: An initialized web search tool
    """
    return WebSearchTool()