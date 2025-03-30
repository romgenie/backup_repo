"""
Vector store cache utility.
Provides a caching mechanism for vector store information.
"""
import os
import json
from typing import Dict, List, Optional, Any

# Path to the cache file
CACHE_FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                              "vector_stores.json")

def get_vector_store_cache() -> Dict[str, Any]:
    """
    Get the current vector store cache.
    
    Returns:
        Dict containing vector store information:
        {
            "active_vector_store_id": "vs_123abc...",
            "vector_stores": [
                {
                    "id": "vs_123abc...",
                    "name": "Store Name",
                    "created_at": "timestamp",
                    "file_count": 5
                },
                ...
            ],
            "last_updated": "timestamp"
        }
    """
    if not os.path.exists(CACHE_FILE_PATH):
        # Return empty cache structure if file doesn't exist
        return {
            "active_vector_store_id": None,
            "vector_stores": [],
            "last_updated": None
        }
    
    try:
        with open(CACHE_FILE_PATH, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        # Return empty cache structure if file is invalid
        return {
            "active_vector_store_id": None,
            "vector_stores": [],
            "last_updated": None
        }

def update_vector_store_cache(cache_data: Dict[str, Any]) -> bool:
    """
    Update the vector store cache.
    
    Args:
        cache_data: The cache data to write
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(CACHE_FILE_PATH), exist_ok=True)
        
        # Write cache data to file
        with open(CACHE_FILE_PATH, 'w') as f:
            json.dump(cache_data, f, indent=2)
        return True
    except Exception:
        return False

def set_active_vector_store(vector_store_id: str) -> bool:
    """
    Set the active vector store ID.
    
    Args:
        vector_store_id: The vector store ID to set as active
        
    Returns:
        bool: True if successful, False otherwise
    """
    cache = get_vector_store_cache()
    cache["active_vector_store_id"] = vector_store_id
    
    # Also update env file for backward compatibility
    try:
        env_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env")
        
        if os.path.exists(env_file_path):
            with open(env_file_path, "r") as f:
                lines = f.readlines()
            
            updated = False
            for i, line in enumerate(lines):
                if line.startswith("VECTOR_STORE_ID="):
                    lines[i] = f"VECTOR_STORE_ID={vector_store_id}\n"
                    updated = True
                    break
            
            if not updated:
                lines.append(f"VECTOR_STORE_ID={vector_store_id}\n")
            
            with open(env_file_path, "w") as f:
                f.writelines(lines)
    except Exception:
        pass  # Continue even if env file update fails
    
    return update_vector_store_cache(cache)

def add_or_update_vector_store(store_details: Dict[str, Any]) -> bool:
    """
    Add or update a vector store in the cache.
    
    Args:
        store_details: Vector store details including id, name, etc.
        
    Returns:
        bool: True if successful, False otherwise
    """
    if "id" not in store_details:
        return False
    
    cache = get_vector_store_cache()
    
    # Check if this store already exists
    store_exists = False
    for i, store in enumerate(cache["vector_stores"]):
        if store["id"] == store_details["id"]:
            # Update existing store
            cache["vector_stores"][i] = store_details
            store_exists = True
            break
    
    # Add new store if it doesn't exist
    if not store_exists:
        cache["vector_stores"].append(store_details)
    
    # Update timestamp
    from datetime import datetime
    cache["last_updated"] = datetime.now().isoformat()
    
    return update_vector_store_cache(cache)

def get_active_vector_store_id() -> Optional[str]:
    """
    Get the currently active vector store ID.
    
    Returns:
        str: The active vector store ID, or None if not set
    """
    cache = get_vector_store_cache()
    return cache["active_vector_store_id"]

def get_vector_store_details(vector_store_id: str) -> Optional[Dict[str, Any]]:
    """
    Get details about a specific vector store.
    
    Args:
        vector_store_id: The vector store ID
        
    Returns:
        Dict: Vector store details, or None if not found
    """
    cache = get_vector_store_cache()
    
    for store in cache["vector_stores"]:
        if store["id"] == vector_store_id:
            return store
    
    return None

def auto_set_active_store() -> Optional[str]:
    """
    If no active vector store is set but there are stores available,
    automatically set the most recently created one as active.
    
    Returns:
        str: The ID of the newly set active store, or None if no change was made
    """
    cache = get_vector_store_cache()
    
    # If we already have an active store, do nothing
    if cache["active_vector_store_id"]:
        return None
        
    # If we have no stores, do nothing
    if not cache["vector_stores"]:
        return None
        
    # Find the most recent store by sorting on created_at
    sorted_stores = sorted(
        cache["vector_stores"], 
        key=lambda store: store.get("created_at", 0), 
        reverse=True
    )
    
    # Set the most recent as active
    if sorted_stores:
        cache["active_vector_store_id"] = sorted_stores[0]["id"]
        update_vector_store_cache(cache)
        
        # Also update env file for backward compatibility
        try:
            env_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env")
            
            if os.path.exists(env_file_path):
                with open(env_file_path, "r") as f:
                    lines = f.readlines()
                
                # Update VECTOR_STORE_ID line if it exists
                updated = False
                for i, line in enumerate(lines):
                    if line.startswith("VECTOR_STORE_ID="):
                        lines[i] = f"VECTOR_STORE_ID={sorted_stores[0]['id']}\n"
                        updated = True
                        break
                
                # Add line if it doesn't exist
                if not updated:
                    lines.append(f"VECTOR_STORE_ID={sorted_stores[0]['id']}\n")
                
                with open(env_file_path, "w") as f:
                    f.writelines(lines)
        except Exception:
            pass  # Continue even if env file update fails
            
        return sorted_stores[0]["id"]
        
    return None

def has_valid_vector_store() -> bool:
    """
    Check if there is a valid vector store configured.
    First attempts to auto-set an active store if none is set.
    
    Returns:
        bool: True if a valid vector store is configured, False otherwise
    """
    # Try to auto-set an active store if needed
    auto_set_active_store()
    
    # Now check if we have an active store
    active_id = get_active_vector_store_id()
    
    if not active_id:
        return False
    
    # Check if we have details for this store
    store_details = get_vector_store_details(active_id)
    return store_details is not None

def clean_nonexistent_stores(existing_store_ids: List[str]) -> bool:
    """
    Remove stores from the cache that no longer exist.
    
    Args:
        existing_store_ids: List of vector store IDs that exist
        
    Returns:
        bool: True if cache was updated, False otherwise
    """
    cache = get_vector_store_cache()
    
    original_count = len(cache["vector_stores"])
    
    # Keep only stores that exist in the provided list
    cache["vector_stores"] = [
        store for store in cache["vector_stores"] 
        if store["id"] in existing_store_ids
    ]
    
    # If the active store no longer exists, clear it
    if (cache["active_vector_store_id"] is not None and 
        cache["active_vector_store_id"] not in existing_store_ids):
        cache["active_vector_store_id"] = None
    
    # Only update if anything changed
    if len(cache["vector_stores"]) != original_count:
        # Update timestamp
        from datetime import datetime
        cache["last_updated"] = datetime.now().isoformat()
        
        # Write updated cache
        return update_vector_store_cache(cache)
    
    return False