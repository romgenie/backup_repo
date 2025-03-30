"""
File search tools using OpenAI's FileSearchTool.
"""
from typing import List, Optional
from agents import FileSearchTool
from src.config.settings import VECTOR_STORE_ID

def get_file_search_tool(
    vector_store_ids: Optional[List[str]] = None,
    max_num_results: int = 3
) -> Optional[FileSearchTool]:
    """
    Return an instance of the FileSearchTool configured with vector store IDs.
    
    Args:
        vector_store_ids: Optional list of vector store IDs to search
        max_num_results: Maximum number of results to return
        
    Returns:
        FileSearchTool: An initialized file search tool, or None if no valid vector store ID is available
    """
    # If no vector store IDs provided, use the one from settings
    if vector_store_ids is None:
        # Check if we have a valid vector store ID
        if not VECTOR_STORE_ID or VECTOR_STORE_ID == "your_vector_store_id_here":
            # No valid vector store ID, return None
            return None
        vector_store_ids = [VECTOR_STORE_ID]
    
    # Filter out any None or empty values
    vector_store_ids = [id for id in vector_store_ids if id and id != "your_vector_store_id_here"]
    
    # If no valid vector store IDs, return None
    if not vector_store_ids:
        return None
        
    return FileSearchTool(
        max_num_results=max_num_results,
        vector_store_ids=vector_store_ids,
    )