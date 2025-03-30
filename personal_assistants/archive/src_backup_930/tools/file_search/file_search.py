"""
File search tools using OpenAI's FileSearchTool.
"""
from typing import List, Optional
from agents import FileSearchTool
from src.config.settings import VECTOR_STORE_ID

def get_file_search_tool(
    vector_store_ids: Optional[List[str]] = None,
    max_num_results: int = 3
) -> FileSearchTool:
    """
    Return an instance of the FileSearchTool configured with vector store IDs.
    
    Args:
        vector_store_ids: Optional list of vector store IDs to search
        max_num_results: Maximum number of results to return
        
    Returns:
        FileSearchTool: An initialized file search tool
    """
    if vector_store_ids is None:
        vector_store_ids = [VECTOR_STORE_ID]
        
    return FileSearchTool(
        max_num_results=max_num_results,
        vector_store_ids=vector_store_ids,
    )