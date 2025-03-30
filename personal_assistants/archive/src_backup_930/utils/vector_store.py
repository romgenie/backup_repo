"""
Vector store utilities.
"""
from typing import Dict, List, Any, Optional
import os
from openai import OpenAI

def create_vector_store(
    store_name: str,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a new vector store.
    
    Args:
        store_name: Name of the vector store
        api_key: OpenAI API key
        
    Returns:
        Dictionary with vector store details
    """
    client = OpenAI(api_key=api_key)
    
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
    Upload a file to a vector store.
    
    Args:
        file_path: Path to the file to upload
        vector_store_id: ID of the vector store
        api_key: OpenAI API key
        
    Returns:
        Dictionary with upload status
    """
    file_name = os.path.basename(file_path)
    client = OpenAI(api_key=api_key)
    
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