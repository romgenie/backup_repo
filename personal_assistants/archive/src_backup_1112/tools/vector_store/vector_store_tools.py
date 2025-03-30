"""
Vector store management tools.
"""
from typing import Dict, List, Optional, Any
import os
import json
import glob
from agents import function_tool
from openai import OpenAI
from src.config.settings import OPENAI_API_KEY

@function_tool
def create_new_vector_store(store_name: str) -> Dict[str, Any]:
    """
    Create a new vector store with the given name.
    
    Args:
        store_name: Name of the vector store to create
        
    Returns:
        Dictionary with vector store details including ID
    """
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    try:
        vector_store = client.vector_stores.create(name=store_name)
        details = {
            "id": vector_store.id,
            "name": vector_store.name,
            "created_at": vector_store.created_at,
            "file_count": vector_store.file_counts.completed
        }
        
        # Update the vector store cache
        from src.utils.vector_store_cache import add_or_update_vector_store
        add_or_update_vector_store(details)
        
        return details
    except Exception as e:
        return {"error": str(e)}

@function_tool
def upload_file_to_vector_store(file_path: str, vector_store_id: str) -> Dict[str, Any]:
    """
    Upload a file to a vector store.
    
    Args:
        file_path: Path to the file to upload
        vector_store_id: ID of the vector store
        
    Returns:
        Dictionary with upload status
    """
    client = OpenAI(api_key=OPENAI_API_KEY)
    file_name = os.path.basename(file_path)
    
    # Check if the file exists
    if not os.path.exists(file_path):
        return {
            "file": file_name,
            "status": "failed",
            "error": f"File {file_path} does not exist"
        }
    
    try:
        file_response = client.files.create(
            file=open(file_path, 'rb'),
            purpose="assistants"
        )
        
        attach_response = client.vector_stores.files.create(
            vector_store_id=vector_store_id,
            file_id=file_response.id
        )
        
        # Update the store's file count in the cache
        from src.utils.vector_store_cache import get_vector_store_cache, update_vector_store_cache
        
        cache = get_vector_store_cache()
        
        # Find and update the vector store in the cache
        for i, store in enumerate(cache["vector_stores"]):
            if store["id"] == vector_store_id:
                # Increment file count
                if "file_count" in store:
                    cache["vector_stores"][i]["file_count"] += 1
                else:
                    cache["vector_stores"][i]["file_count"] = 1
                
                # Update timestamp
                from datetime import datetime
                cache["last_updated"] = datetime.now().isoformat()
                
                # Write updated cache
                update_vector_store_cache(cache)
                break
        
        return {
            "file": file_name,
            "status": "success",
            "file_id": file_response.id
        }
    except Exception as e:
        return {
            "file": file_name,
            "status": "failed",
            "error": str(e)
        }

@function_tool
def upload_directory_to_vector_store(
    directory_path: str, 
    vector_store_id: str,
    file_pattern: str
) -> Dict[str, Any]:
    """
    Upload all files in a directory to a vector store.
    
    Args:
        directory_path: Path to the directory containing files to upload
        vector_store_id: ID of the vector store
        file_pattern: Glob pattern to match files (e.g. "*.pdf", "*.txt", "*.*")
        
    Returns:
        Dictionary with upload status for each file
    """
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Check if the directory exists
    if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
        return {
            "status": "failed",
            "error": f"Directory {directory_path} does not exist"
        }
    
    # Find all matching files
    pattern = os.path.join(directory_path, file_pattern)
    files = glob.glob(pattern, recursive=True)
    
    if not files:
        return {
            "status": "warning",
            "message": f"No files matching pattern '{file_pattern}' found in {directory_path}"
        }
    
    # Upload each file
    results = []
    successful_uploads = 0
    
    for file_path in files:
        try:
            if os.path.isfile(file_path):
                file_name = os.path.basename(file_path)
                file_response = client.files.create(
                    file=open(file_path, 'rb'),
                    purpose="assistants"
                )
                
                attach_response = client.vector_stores.files.create(
                    vector_store_id=vector_store_id,
                    file_id=file_response.id
                )
                
                results.append({
                    "file": file_name,
                    "status": "success",
                    "file_id": file_response.id
                })
                successful_uploads += 1
        except Exception as e:
            results.append({
                "file": os.path.basename(file_path),
                "status": "failed",
                "error": str(e)
            })
    
    # Update the store's file count in the cache if there were successful uploads
    if successful_uploads > 0:
        from src.utils.vector_store_cache import get_vector_store_cache, update_vector_store_cache
        
        cache = get_vector_store_cache()
        
        # Find and update the vector store in the cache
        for i, store in enumerate(cache["vector_stores"]):
            if store["id"] == vector_store_id:
                # Increment file count
                if "file_count" in store:
                    cache["vector_stores"][i]["file_count"] += successful_uploads
                else:
                    cache["vector_stores"][i]["file_count"] = successful_uploads
                
                # Update timestamp
                from datetime import datetime
                cache["last_updated"] = datetime.now().isoformat()
                
                # Write updated cache
                update_vector_store_cache(cache)
                break
    
    return {
        "status": "complete",
        "files_processed": len(results),
        "results": results
    }

@function_tool
def list_vector_stores() -> Dict[str, Any]:
    """
    List all available vector stores.
    
    Returns:
        Dictionary with vector store list
    """
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    try:
        stores = client.vector_stores.list()
        store_list = []
        
        # Import cache utilities
        from src.utils.vector_store_cache import add_or_update_vector_store
        from datetime import datetime
        
        for store in stores.data:
            store_details = {
                "id": store.id,
                "name": store.name,
                "created_at": store.created_at,
                "file_count": store.file_counts.completed
            }
            
            # Update the cache with each store we find
            add_or_update_vector_store(store_details)
            
            store_list.append(store_details)
            
        return {"stores": store_list}
    except Exception as e:
        return {"error": str(e)}

@function_tool
def get_vector_store_details(vector_store_id: str) -> Dict[str, Any]:
    """
    Get details about a specific vector store.
    
    Args:
        vector_store_id: ID of the vector store
        
    Returns:
        Dictionary with vector store details
    """
    # Try to get from cache first
    from src.utils.vector_store_cache import get_vector_store_details as get_cached_details
    
    cached_details = get_cached_details(vector_store_id)
    if cached_details:
        return cached_details
    
    # If not in cache, get from API
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    try:
        store = client.vector_stores.retrieve(vector_store_id=vector_store_id)
        details = {
            "id": store.id,
            "name": store.name,
            "created_at": store.created_at,
            "file_count": store.file_counts.completed
        }
        
        # Update cache
        from src.utils.vector_store_cache import add_or_update_vector_store
        add_or_update_vector_store(details)
        
        return details
    except Exception as e:
        return {"error": str(e)}

@function_tool
def list_files_in_vector_store(vector_store_id: str) -> Dict[str, Any]:
    """
    List all files in a vector store.
    
    Args:
        vector_store_id: ID of the vector store
        
    Returns:
        Dictionary with list of files
    """
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    try:
        files = client.vector_stores.files.list(vector_store_id=vector_store_id)
        file_list = []
        
        for file in files.data:
            # Get more details about the file
            try:
                file_details = client.files.retrieve(file_id=file.id)
                file_info = {
                    "id": file.id,
                    "object": file.object,
                    "created_at": file.created_at,
                    "filename": getattr(file_details, "filename", "Unknown"),
                    "bytes": getattr(file_details, "bytes", 0),
                    "purpose": getattr(file_details, "purpose", "Unknown")
                }
            except:
                # Fall back to basic info if details retrieval fails
                file_info = {
                    "id": file.id,
                    "object": file.object,
                    "created_at": file.created_at
                }
                
            file_list.append(file_info)
            
        return {"files": file_list}
    except Exception as e:
        return {"error": str(e)}

@function_tool
def delete_file_from_vector_store(vector_store_id: str, file_id: str) -> Dict[str, Any]:
    """
    Delete a file from a vector store.
    
    Args:
        vector_store_id: ID of the vector store
        file_id: ID of the file to delete
        
    Returns:
        Dictionary with deletion status
    """
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    try:
        # Delete the file from the vector store
        client.vector_stores.files.delete(
            vector_store_id=vector_store_id,
            file_id=file_id
        )
        
        # Update the store's file count in the cache
        from src.utils.vector_store_cache import get_vector_store_cache, update_vector_store_cache
        
        cache = get_vector_store_cache()
        
        # Find and update the vector store in the cache
        for i, store in enumerate(cache["vector_stores"]):
            if store["id"] == vector_store_id and "file_count" in store:
                # Decrement file count if greater than 0
                cache["vector_stores"][i]["file_count"] = max(0, store["file_count"] - 1)
                
                # Update timestamp
                from datetime import datetime
                cache["last_updated"] = datetime.now().isoformat()
                
                # Write updated cache
                update_vector_store_cache(cache)
                break
        
        return {
            "status": "success",
            "message": f"File {file_id} deleted from vector store {vector_store_id}"
        }
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e)
        }

@function_tool
def refresh_vector_store_cache() -> Dict[str, Any]:
    """
    Refresh the vector store cache by fetching the latest information from the API.
    Removes any vector stores from the cache that no longer exist.
    
    Returns:
        Dictionary with refresh status
    """
    try:
        # First list all stores to update the cache
        list_result = list_vector_stores()
        
        # Get the list of existing store IDs
        existing_store_ids = []
        if "stores" in list_result and isinstance(list_result["stores"], list):
            existing_store_ids = [store["id"] for store in list_result["stores"] if "id" in store]
        
        # Clean up stores that no longer exist
        from src.utils.vector_store_cache import clean_nonexistent_stores, get_active_vector_store_id, auto_set_active_store
        clean_nonexistent_stores(existing_store_ids)
        
        # Auto-set an active store if none is set but stores exist
        auto_set_active_store()
        
        # Get the active store ID (might have changed if it was removed or auto-set)
        active_id = get_active_vector_store_id()
        
        # If there's an active store, refresh its details
        if active_id:
            details_result = get_vector_store_details(active_id)
        
        return {
            "status": "success",
            "message": "Vector store cache refreshed",
            "active_vector_store_id": active_id,
            "vector_store_count": len(existing_store_ids)
        }
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e)
        }

@function_tool
def delete_vector_store(vector_store_id: str) -> Dict[str, Any]:
    """
    Delete a vector store.
    
    Args:
        vector_store_id: ID of the vector store to delete
        
    Returns:
        Dictionary with deletion status
    """
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    try:
        # Delete the vector store
        client.vector_stores.delete(vector_store_id=vector_store_id)
        
        # Update the vector store cache
        from src.utils.vector_store_cache import get_vector_store_cache, update_vector_store_cache, get_active_vector_store_id
        
        # Get current cache
        cache = get_vector_store_cache()
        
        # Check if this was the active store
        was_active = cache["active_vector_store_id"] == vector_store_id
        
        # Remove from vector stores list
        cache["vector_stores"] = [vs for vs in cache["vector_stores"] if vs["id"] != vector_store_id]
        
        # Update timestamp
        from datetime import datetime
        cache["last_updated"] = datetime.now().isoformat()
        
        # Clear active store if this was the active one
        if was_active:
            cache["active_vector_store_id"] = None
            
            # Also update the .env file for backward compatibility
            try:
                # Get the project root directory
                import os
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
                env_file_path = os.path.join(project_root, ".env")
                with open(env_file_path, "r") as f:
                    lines = f.readlines()
                
                updated_lines = []
                for line in lines:
                    if line.startswith("VECTOR_STORE_ID="):
                        updated_lines.append("VECTOR_STORE_ID=your_vector_store_id_here\n")
                    else:
                        updated_lines.append(line)
                
                with open(env_file_path, "w") as f:
                    f.writelines(updated_lines)
            except Exception as env_err:
                # Continue even if env file update fails
                pass
                
        # Update cache file
        update_vector_store_cache(cache)
        
        return {
            "status": "success",
            "message": f"Vector store {vector_store_id} deleted successfully"
        }
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e)
        }

@function_tool
def save_vector_store_id_to_env(vector_store_id: str) -> Dict[str, Any]:
    """
    Save the vector store ID to the .env file and cache.
    If vector_store_id is "null" or "none" (case-insensitive), clears the active store.
    
    Args:
        vector_store_id: ID of the vector store
        
    Returns:
        Dictionary with save status
    """
    from src.utils.vector_store_cache import set_active_vector_store, get_vector_store_details, get_vector_store_cache, update_vector_store_cache
    
    # Handle null case
    if not vector_store_id or vector_store_id.lower() in ["null", "none"]:
        # Clear the active store
        cache = get_vector_store_cache()
        cache["active_vector_store_id"] = None
        success = update_vector_store_cache(cache)
        
        # Also update env file
        try:
            env_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), ".env")
            if os.path.exists(env_file_path):
                with open(env_file_path, "r") as f:
                    lines = f.readlines()
                
                updated_lines = []
                for line in lines:
                    if line.startswith("VECTOR_STORE_ID="):
                        updated_lines.append("VECTOR_STORE_ID=your_vector_store_id_here\n")
                    else:
                        updated_lines.append(line)
                
                with open(env_file_path, "w") as f:
                    f.writelines(updated_lines)
        except Exception:
            pass  # Continue even if env file update fails
            
        return {
            "status": "success", 
            "message": "Active vector store cleared"
        }
    
    try:
        # First try to get details about this vector store
        store_details = get_vector_store_details(vector_store_id)
        
        if not store_details:
            # If store isn't in cache yet, get details from API
            details_result = get_vector_store_details(vector_store_id)
            if not isinstance(details_result, dict) or "error" in details_result:
                # If we can't get details, just set the active ID anyway
                pass
        
        # Set as active vector store
        success = set_active_vector_store(vector_store_id)
        
        if success:
            return {
                "status": "success",
                "message": f"Vector store ID {vector_store_id} saved as active vector store"
            }
        else:
            return {
                "status": "failed",
                "message": "Failed to save vector store ID"
            }
            
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e)
        }