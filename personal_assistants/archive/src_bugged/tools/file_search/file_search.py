"""
File search tools using OpenAI's FileSearchTool.
"""
from typing import List, Optional
from agents import FileSearchTool
from src.config.settings import VECTOR_STORE_ID
# Import vector store cache utility
try:
    from src.utils.vector_store_cache import get_vector_store_cache
except ImportError:
    # Fallback for backward compatibility
    get_vector_store_cache = None

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
    # If no vector store IDs provided, try to get them from cache first, then settings
    if vector_store_ids is None:
        # Try to get active store from cache first
        if get_vector_store_cache is not None:
            cache = get_vector_store_cache()
            active_id = cache.get("active_vector_store_id")
            if active_id:
                vector_store_ids = [active_id]
        
        # Fallback to settings if cache didn't provide a valid ID
        if not vector_store_ids:
            # Check if we have a valid vector store ID in settings
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