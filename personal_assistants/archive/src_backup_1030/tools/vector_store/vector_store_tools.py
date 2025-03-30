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
        except Exception as e:
            results.append({
                "file": os.path.basename(file_path),
                "status": "failed",
                "error": str(e)
            })
    
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
        
        for store in stores.data:
            store_list.append({
                "id": store.id,
                "name": store.name,
                "created_at": store.created_at,
                "file_count": store.file_counts.completed
            })
            
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
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    try:
        store = client.vector_stores.retrieve(vector_store_id=vector_store_id)
        return {
            "id": store.id,
            "name": store.name,
            "created_at": store.created_at,
            "file_count": store.file_counts.completed
        }
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
        
        # If this was the configured vector store, update the .env file
        from src.config.settings import VECTOR_STORE_ID
        if VECTOR_STORE_ID == vector_store_id:
            try:
                env_file_path = "/Users/timgregg/Desktop/personal_assistants/.env"
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
                return {
                    "status": "partial",
                    "message": f"Vector store {vector_store_id} deleted, but failed to update .env file: {str(env_err)}"
                }
        
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
    Save the vector store ID to the .env file.
    
    Args:
        vector_store_id: ID of the vector store
        
    Returns:
        Dictionary with save status
    """
    env_file_path = "/Users/timgregg/Desktop/personal_assistants/.env"
    
    try:
        # Check if .env file exists, create it if not
        if not os.path.exists(env_file_path):
            with open(env_file_path, "w") as f:
                f.write(f"VECTOR_STORE_ID={vector_store_id}\n")
            
            return {
                "status": "success",
                "message": f"Created .env file with vector store ID {vector_store_id}"
            }
        
        # Read the current .env file
        with open(env_file_path, "r") as f:
            lines = f.readlines()
        
        # Update or add the VECTOR_STORE_ID line
        updated = False
        for i, line in enumerate(lines):
            if line.startswith("VECTOR_STORE_ID="):
                lines[i] = f"VECTOR_STORE_ID={vector_store_id}\n"
                updated = True
                break
        
        if not updated:
            lines.append(f"VECTOR_STORE_ID={vector_store_id}\n")
        
        # Write back to the .env file
        with open(env_file_path, "w") as f:
            f.writelines(lines)
        
        return {
            "status": "success",
            "message": f"Vector store ID {vector_store_id} saved to .env file"
        }
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e)
        }