"""
Vector store manager utility.
Provides comprehensive management of vector stores for the application.
"""
import asyncio
from typing import Dict, Any, List, Optional
import os
from glob import glob
from openai import OpenAI
from agents import Runner, trace
# Import delayed to prevent circular import
# from src.agents import create_vector_store_agent
from src.config.settings import VECTOR_STORE_ID, OPENAI_API_KEY

# Direct API functions

def create_vector_store(
    store_name: str,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a new vector store using the OpenAI API directly.
    
    Args:
        store_name: Name of the vector store
        api_key: OpenAI API key (defaults to settings value)
        
    Returns:
        Dictionary with vector store details
    """
    client = OpenAI(api_key=api_key or OPENAI_API_KEY)
    
    try:
        vector_store = client.vector_stores.create(name=store_name)
        details = {
            "id": vector_store.id,
            "name": vector_store.name,
            "created_at": vector_store.created_at,
            "file_count": vector_store.file_counts.completed
        }
        print(f"Vector store created: {details}")
        return details
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return {}

def upload_file(
    file_path: str, 
    vector_store_id: str,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Upload a file to a vector store using the OpenAI API directly.
    
    Args:
        file_path: Path to the file to upload
        vector_store_id: ID of the vector store
        api_key: OpenAI API key (defaults to settings value)
        
    Returns:
        Dictionary with upload status
    """
    file_name = os.path.basename(file_path)
    client = OpenAI(api_key=api_key or OPENAI_API_KEY)
    
    try:
        file_response = client.files.create(
            file=open(file_path, 'rb'),
            purpose="assistants"
        )
        
        attach_response = client.vector_stores.files.create(
            vector_store_id=vector_store_id,
            file_id=file_response.id
        )
        
        return {"file": file_name, "status": "success"}
    except Exception as e:
        print(f"Error with {file_name}: {str(e)}")
        return {"file": file_name, "status": "failed", "error": str(e)}

# Agent-based functions

async def ensure_vector_store_exists(
    store_name: str = "Personal_Knowledge_Base",
    force_create: bool = False
) -> Optional[str]:
    """
    Ensures a vector store exists for the application.
    Only creates a new one if force_create is True or no valid ID exists.
    
    Args:
        store_name: Name for the new vector store if one needs to be created
        force_create: If True, creates a new vector store even if one exists
        
    Returns:
        Vector store ID, or None if no store exists and creation was not forced
    """
    # Use the cache-based check
    from src.utils.vector_store_cache import has_valid_vector_store, get_active_vector_store_id
    
    # Check if we already have a vector store ID and aren't forcing creation
    if not force_create and has_valid_vector_store():
        vector_store_id = get_active_vector_store_id()
        print(f"Using existing vector store ID: {vector_store_id}")
        return vector_store_id
    
    # Import here to avoid circular dependency
    from src.agents import create_vector_store_agent
    
    # Create vector store agent
    vector_store_agent = create_vector_store_agent()
    
    if not force_create:
        # If we're not forcing creation and no valid vector store exists, return None
        if not has_valid_vector_store():
            return None
    
    # Create a new vector store (only reached if force_create=True or we need one)
    with trace("Vector Store Manager"):
        result = await Runner.run(
            vector_store_agent, 
            f"Create a new vector store with the name '{store_name}'"
        )
    
    # Extract the vector store ID from the agent's response
    vector_store_id = None
    try:
        # Try to find the vector store ID in the response
        response_text = result.final_output
        import re
        # Look for ID patterns like "vs_1234abcd..." in the response
        match = re.search(r'vs_[a-zA-Z0-9]+', response_text)
        if match:
            vector_store_id = match.group(0)
    except Exception:
        pass
    
    # If we couldn't extract the ID, ask the agent for it directly
    if not vector_store_id:
        with trace("Vector Store Manager"):
            result = await Runner.run(
                vector_store_agent, 
                f"What is the ID of the vector store '{store_name}'? Just give me the ID."
            )
        response_text = result.final_output
        import re
        match = re.search(r'vs_[a-zA-Z0-9]+', response_text)
        if match:
            vector_store_id = match.group(0)
    
    # Save the vector store ID to the .env file
    if vector_store_id:
        with trace("Vector Store Manager"):
            await Runner.run(
                vector_store_agent, 
                f"Save the vector store ID '{vector_store_id}' to the .env file"
            )
    
    return vector_store_id

async def upload_knowledge_files(
    vector_store_id: str,
    knowledge_dir: str = None
) -> List[Dict[str, Any]]:
    """
    Uploads knowledge files to the vector store.
    
    Args:
        vector_store_id: ID of the vector store
        knowledge_dir: Directory containing knowledge files (uses KNOWLEDGE_DIRECTORY from settings if None)
        
    Returns:
        List of upload results
    """
    # Use knowledge directory from settings if not specified
    if knowledge_dir is None:
        from src.config.settings import KNOWLEDGE_DIRECTORY
        knowledge_dir = KNOWLEDGE_DIRECTORY
    # Create vector store agent
    vector_store_agent = create_vector_store_agent()
    
    # Get list of files
    files = glob(f"{knowledge_dir}/**/*.*", recursive=True)
    results = []
    
    for file_path in files:
        with trace("Vector Store Manager"):
            result = await Runner.run(
                vector_store_agent, 
                f"Upload file '{file_path}' to vector store with ID '{vector_store_id}'"
            )
        results.append({
            "file": file_path,
            "result": result.final_output
        })
    
    return results

async def get_vector_store_info() -> str:
    """
    Get information about configured vector stores.
    
    Returns:
        str: A formatted string with vector store information
    """
    # Check if we have a configured vector store
    has_vector_store = VECTOR_STORE_ID is not None and VECTOR_STORE_ID != "your_vector_store_id_here"
    
    # Import here to avoid circular dependency
    from src.agents import create_vector_store_agent
    
    # Create vector store agent
    vector_store_agent = create_vector_store_agent()
    
    if has_vector_store:
        # We have a vector store, get its details
        with trace("Vector Store Info"):
            result = await Runner.run(
                vector_store_agent, 
                f"Get details for vector store with ID '{VECTOR_STORE_ID}'"
            )
        vector_store_info = result.final_output
    else:
        # No vector store configured yet
        vector_store_info = "No vector store is currently configured."
        
    # Get list of all vector stores
    with trace("Vector Store List"):
        result = await Runner.run(
            vector_store_agent, 
            "List all available vector stores"
        )
    vector_store_list = result.final_output
    
    return f"""
Vector Store Information:
------------------------
{vector_store_info}

Available Vector Stores:
------------------------
{vector_store_list}
"""

async def initialize_vector_store(force_create: bool = False) -> Optional[str]:
    """
    Initialize the vector store with knowledge files.
    Only creates a new store if force_create is True or no valid ID exists.
    
    Args:
        force_create: If True, creates a new vector store even if one exists
    
    Returns:
        Vector store ID, or None if no store exists and creation was not forced
    """
    # Ensure vector store exists
    vector_store_id = await ensure_vector_store_exists(force_create=force_create)
    
    if not vector_store_id:
        # Don't raise an exception, just return None
        return None
    
    # Upload knowledge files
    await upload_knowledge_files(vector_store_id)
    
    return vector_store_id