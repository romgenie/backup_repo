import os
import glob
import json
from datetime import datetime
from typing import Any, Optional, List, Literal, Dict

import openai
from pydantic import BaseModel
from agents import RunContextWrapper, FunctionTool
from src.config.settings import OPENAI_API_KEY

# Make sure the openai key is set
openai.api_key = OPENAI_API_KEY


class VectorStoreManagerArgs(BaseModel):
    """
    Arguments model for the vector_store_manager tool.

    Attributes:
        operation: The operation to perform on the vector store.
                   Must be one of:
                     - create_store
                     - upload_file
                     - upload_directory
                     - list_stores
                     - get_store_details
                     - list_files
                     - delete_file
                     - refresh_cache
                     - delete_store
                     - set_active_store
        store_name: Name of the vector store to create (used by create_store).
        vector_store_id: ID of the vector store to operate on.
        file_path: Path to a single file (used by upload_file, delete_file, etc).
        directory_path: Path to a directory (used by upload_directory).
        file_pattern: Glob file pattern (used by upload_directory).
        file_id: ID of the file to delete (used by delete_file).
    """

    operation: Literal[
        "create_store",
        "upload_file",
        "upload_directory",
        "list_stores",
        "get_store_details",
        "list_files",
        "delete_file",
        "refresh_cache",
        "delete_store",
        "set_active_store",
    ]
    store_name: Optional[str] = None
    vector_store_id: Optional[str] = None
    file_path: Optional[str] = None
    directory_path: Optional[str] = None
    file_pattern: Optional[str] = None
    file_id: Optional[str] = None


async def _vector_store_manager_invoke(
    ctx: RunContextWrapper[Any],
    args_json: str
) -> str:
    """
    Core driver function that receives JSON-serialized arguments, 
    parses them, and routes to the relevant vector store operation.
    Returns a JSON string result.
    """
    # Enable debug logging to file for troubleshooting agent interactions
    import logging
    logging_enabled = True
    
    if logging_enabled:
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, "vector_store_manager.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger("vector_store_manager")
        logger.debug(f"Received arguments: {args_json}")
    
    # Parse the JSON manually to handle required fields with null values
    try:
        args_dict = json.loads(args_json)
        if logging_enabled:
            logger.debug(f"Parsed arguments: {args_dict}")
    except json.JSONDecodeError as e:
        error_msg = f"Failed to parse JSON arguments: {str(e)}"
        if logging_enabled:
            logger.error(error_msg)
        return json.dumps({"error": error_msg})
    
    # Define sanitization functions for different parameter types
    def sanitize_string_param(value):
        """Sanitize a parameter that should be a string."""
        if value is None or value == "":
            return None
            
        # Common agent value patterns to handle
        if isinstance(value, str) and value.lower() in ["null", "none", "undefined", "n/a"]:
            return None

        # Handle lists (common agent mistake)
        if isinstance(value, list):
            # If list is empty, return None
            if not value:
                return None
            # Try to find a non-empty string in the list
            for item in value:
                if item and isinstance(item, (str, int, float)):
                    return str(item)
            # Fall back to first element
            return str(value[0])
            
        # Handle dictionaries (common when agents misunderstand the schema)
        if isinstance(value, dict):
            # If the dict has an explicit value key, use that
            if "value" in value:
                return sanitize_string_param(value["value"])
            # If it has an id key, use that (common pattern)
            if "id" in value:
                return str(value["id"])
            # Fall back to JSON string of the entire dict
            return json.dumps(value)
            
        # Handle other types by converting to string
        return str(value).strip()
    
    def sanitize_path_param(value):
        """Sanitize a parameter that represents a file or directory path."""
        value = sanitize_string_param(value)
        if value is None:
            return None
            
        # Remove quotes that agents might add
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        if value.startswith("'") and value.endswith("'"):
            value = value[1:-1]
            
        # Sometimes agents add backticks for code formatting
        if value.startswith("`") and value.endswith("`"):
            value = value[1:-1]
            
        # Handle Markdown-style code blocks
        if value.startswith("```") and value.endswith("```"):
            value = value[3:-3].strip()
        
        # Normalize path separators (agent might use Windows style on Unix or vice versa)
        value = value.replace('\\', '/')
        
        # Expand ~ if present
        if value.startswith("~"):
            value = os.path.expanduser(value)
            
        # Normalize double slashes
        while '//' in value:
            value = value.replace('//', '/')
            
        return value
    
    def sanitize_id_param(value):
        """Sanitize an ID parameter."""
        value = sanitize_string_param(value)
        if value is None:
            return None
            
        # Remove any quotes, brackets, or other artifacts that agents might add
        value = value.strip("'\"[](){}<>`")
        
        # Remove common prefixes agents might add 
        prefixes_to_remove = ["id:", "id=", "store_id:", "store_id=", "vector_store_id:", "vector_store_id="]
        for prefix in prefixes_to_remove:
            if isinstance(value, str) and value.lower().startswith(prefix):
                value = value[len(prefix):].strip()
                
        # Some agents might format as VS_1234 or VS-1234 instead of vs_1234
        if isinstance(value, str) and value.lower().startswith("vs-"):
            value = "vs_" + value[3:]
            
        return value
    
    # Santitize the operation parameter first
    if "operation" in args_dict:
        operation = sanitize_string_param(args_dict["operation"])
        # Normalize operation names (sometimes agents use variations)
        operation_mapping = {
            "create": "create_store",
            "create_vector_store": "create_store",
            "new": "create_store",
            "new_store": "create_store",
            "upload": "upload_file",
            "upload_directory": "upload_directory",
            "upload_dir": "upload_directory",
            "list": "list_stores",
            "get_stores": "list_stores",
            "get_store": "get_store_details",
            "store_details": "get_store_details",
            "details": "get_store_details",
            "list_files": "list_files",
            "get_files": "list_files",
            "files": "list_files",
            "delete_file": "delete_file",
            "remove_file": "delete_file",
            "delete": "delete_store",
            "delete_store": "delete_store",
            "remove": "delete_store",
            "remove_store": "delete_store",
            "refresh": "refresh_cache",
            "refresh_cache": "refresh_cache",
            "update_cache": "refresh_cache",
            "set_active": "set_active_store",
            "set_active_store": "set_active_store",
            "activate": "set_active_store",
        }
        args_dict["operation"] = operation_mapping.get(operation.lower(), operation)
    
    # Ensure all required fields are present with defaults if not provided
    # Apply appropriate sanitization to each parameter type
    sanitization_map = {
        "store_name": sanitize_string_param,
        "vector_store_id": sanitize_id_param,
        "file_path": sanitize_path_param,
        "directory_path": sanitize_path_param,
        "file_pattern": sanitize_string_param,
        "file_id": sanitize_id_param,
    }
    
    for field, sanitize_func in sanitization_map.items():
        if field in args_dict:
            args_dict[field] = sanitize_func(args_dict[field])
        else:
            args_dict[field] = None
    
    # Log the sanitized parameters 
    if logging_enabled:
        logger.debug(f"Sanitized parameters: {args_dict}")
        
    # Convert to VectorStoreManagerArgs
    try:
        args = VectorStoreManagerArgs(**args_dict)
        if logging_enabled:
            logger.debug(f"Created VectorStoreManagerArgs: operation={args.operation}")
    except Exception as e:
        error_msg = f"Failed to create VectorStoreManagerArgs: {str(e)}"
        if logging_enabled:
            logger.error(error_msg)
        return json.dumps({"error": error_msg})

    # Create an OpenAI "client" object for vector store operations.
    # This code assumes the openai library is extended to have vector_stores endpoints.
    # If your environment uses a different approach, adapt accordingly.
    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    # Import utility functions only when needed to reduce overhead.
    from src.utils.vector_store_cache import (
        add_or_update_vector_store,
        get_vector_store_cache,
        update_vector_store_cache,
        get_vector_store_details as get_cached_details,
        refresh_cache_manually,
        set_active_vector_store
    )

    # ---------------------------------
    # OPERATION: create_store
    # ---------------------------------
    if args.operation == "create_store":
        """
        Creates a new vector store with the given store_name.
        Example usage:
          {
            "operation": "create_store",
            "store_name": "my_new_store"
          }
        """
        if not args.store_name:
            return json.dumps({"error": "store_name is required for create_store"})

        try:
            vector_store = client.vector_stores.create(name=args.store_name)
            details = {
                "id": vector_store.id,
                "name": vector_store.name,
                "created_at": vector_store.created_at,
                "file_count": vector_store.file_counts.completed
            }
            add_or_update_vector_store(details)
            return json.dumps(details)

        except Exception as e:
            return json.dumps({"error": str(e)})

    # ---------------------------------
    # OPERATION: upload_file
    # ---------------------------------
    elif args.operation == "upload_file":
        """
        Uploads a single file (args.file_path) to the vector store (args.vector_store_id).
        Example usage:
          {
            "operation": "upload_file",
            "vector_store_id": "store_123",
            "file_path": "/path/to/file.pdf"
          }
        """
        if not args.file_path or not args.vector_store_id:
            return json.dumps({
                "error": "file_path and vector_store_id are required for upload_file"
            })

        file_name = os.path.basename(args.file_path)

        # Check existence
        if not os.path.exists(args.file_path):
            return json.dumps({
                "file": file_name,
                "status": "failed",
                "error": f"File {args.file_path} does not exist"
            })

        try:
            # Upload file
            with open(args.file_path, 'rb') as f:
                file_response = client.files.create(file=f, purpose="assistants")

            # Attach file to vector store
            _ = client.vector_stores.files.create(
                vector_store_id=args.vector_store_id,
                file_id=file_response.id
            )

            # Update cache
            cache = get_vector_store_cache()
            for i, store in enumerate(cache["vector_stores"]):
                if store["id"] == args.vector_store_id:
                    # Increment file_count
                    existing_count = store.get("file_count", 0)
                    cache["vector_stores"][i]["file_count"] = existing_count + 1
                    cache["last_updated"] = datetime.now().isoformat()
                    update_vector_store_cache(cache)
                    break

            return json.dumps({
                "file": file_name,
                "status": "success",
                "file_id": file_response.id
            })
        except Exception as e:
            return json.dumps({
                "file": file_name,
                "status": "failed",
                "error": str(e)
            })

    # ---------------------------------
    # OPERATION: upload_directory
    # ---------------------------------
    elif args.operation == "upload_directory":
        """
        Uploads all files in a directory matching the pattern (args.file_pattern)
        to the vector store (args.vector_store_id).
        Example usage:
          {
            "operation": "upload_directory",
            "vector_store_id": "store_123",
            "directory_path": "/path/to/dir",
            "file_pattern": "*.pdf"
          }
        """
        if not args.directory_path or not args.vector_store_id or not args.file_pattern:
            return json.dumps({
                "error": "directory_path, file_pattern, and vector_store_id are required"
                         " for upload_directory"
            })

        if not os.path.exists(args.directory_path) or not os.path.isdir(args.directory_path):
            return json.dumps({
                "status": "failed",
                "error": f"Directory {args.directory_path} does not exist"
            })

        pattern = os.path.join(args.directory_path, args.file_pattern)
        files = glob.glob(pattern, recursive=True)

        if not files:
            return json.dumps({
                "status": "warning",
                "message": f"No files matching pattern '{args.file_pattern}' "
                           f"found in {args.directory_path}"
            })

        results = []
        successful_uploads = 0

        for file_path in files:
            if not os.path.isfile(file_path):
                continue
            file_name = os.path.basename(file_path)
            try:
                with open(file_path, 'rb') as f:
                    file_response = client.files.create(file=f, purpose="assistants")
                _ = client.vector_stores.files.create(
                    vector_store_id=args.vector_store_id,
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
                    "file": file_name,
                    "status": "failed",
                    "error": str(e)
                })

        # Update cache for successful uploads
        if successful_uploads > 0:
            cache = get_vector_store_cache()
            for i, store in enumerate(cache["vector_stores"]):
                if store["id"] == args.vector_store_id:
                    existing_count = store.get("file_count", 0)
                    cache["vector_stores"][i]["file_count"] = existing_count + successful_uploads
                    cache["last_updated"] = datetime.now().isoformat()
                    update_vector_store_cache(cache)
                    break

        return json.dumps({
            "status": "complete",
            "files_processed": len(results),
            "results": results
        })

    # ---------------------------------
    # OPERATION: list_stores
    # ---------------------------------
    elif args.operation == "list_stores":
        """
        Lists all available vector stores in the system. Also updates local cache.
        Example usage:
          {
            "operation": "list_stores"
          }
        """
        try:
            stores = client.vector_stores.list()
            store_list = []

            for store in stores.data:
                details = {
                    "id": store.id,
                    "name": store.name,
                    "created_at": store.created_at,
                    "file_count": store.file_counts.completed
                }
                add_or_update_vector_store(details)
                store_list.append(details)

            return json.dumps({"stores": store_list})
        except Exception as e:
            return json.dumps({"error": str(e)})

    # ---------------------------------
    # OPERATION: get_store_details
    # ---------------------------------
    elif args.operation == "get_store_details":
        """
        Gets details about a specific vector store.
        Example usage:
          {
            "operation": "get_store_details",
            "vector_store_id": "store_123"
          }
        """
        if not args.vector_store_id:
            return json.dumps({"error": "vector_store_id is required for get_store_details"})

        # Check cache first
        cached = get_cached_details(args.vector_store_id)
        if cached:
            return json.dumps(cached)

        # Else fetch from API
        try:
            store = client.vector_stores.retrieve(vector_store_id=args.vector_store_id)
            details = {
                "id": store.id,
                "name": store.name,
                "created_at": store.created_at,
                "file_count": store.file_counts.completed
            }
            add_or_update_vector_store(details)
            return json.dumps(details)
        except Exception as e:
            return json.dumps({"error": str(e)})

    # ---------------------------------
    # OPERATION: list_files
    # ---------------------------------
    elif args.operation == "list_files":
        """
        Lists all files in a given vector store.
        Example usage:
          {
            "operation": "list_files",
            "vector_store_id": "store_123"
          }
        """
        if not args.vector_store_id:
            return json.dumps({"error": "vector_store_id is required for list_files"})

        try:
            files = client.vector_stores.files.list(vector_store_id=args.vector_store_id)
            file_list = []
            for f in files.data:
                try:
                    file_details = client.files.retrieve(file_id=f.id)
                    file_info = {
                        "id": f.id,
                        "object": f.object,
                        "created_at": f.created_at,
                        "filename": getattr(file_details, "filename", "Unknown"),
                        "bytes": getattr(file_details, "bytes", 0),
                        "purpose": getattr(file_details, "purpose", "Unknown")
                    }
                except Exception:
                    # Fall back to partial info
                    file_info = {
                        "id": f.id,
                        "object": f.object,
                        "created_at": f.created_at
                    }
                file_list.append(file_info)

            return json.dumps({"files": file_list})
        except Exception as e:
            return json.dumps({"error": str(e)})

    # ---------------------------------
    # OPERATION: delete_file
    # ---------------------------------
    elif args.operation == "delete_file":
        """
        Deletes a file (args.file_id) from a vector store (args.vector_store_id).
        Example usage:
          {
            "operation": "delete_file",
            "vector_store_id": "store_123",
            "file_id": "file_abc"
          }
        """
        if not args.vector_store_id or not args.file_id:
            return json.dumps({
                "error": "vector_store_id and file_id are required for delete_file"
            })

        try:
            client.vector_stores.files.delete(
                vector_store_id=args.vector_store_id,
                file_id=args.file_id
            )
            # Update cache
            cache = get_vector_store_cache()
            for i, store in enumerate(cache["vector_stores"]):
                if store["id"] == args.vector_store_id and "file_count" in store:
                    cache["vector_stores"][i]["file_count"] = max(0, store["file_count"] - 1)
                    cache["last_updated"] = datetime.now().isoformat()
                    update_vector_store_cache(cache)
                    break

            return json.dumps({
                "status": "success",
                "message": f"File {args.file_id} deleted from store {args.vector_store_id}"
            })
        except Exception as e:
            return json.dumps({"status": "failed", "error": str(e)})

    # ---------------------------------
    # OPERATION: refresh_cache
    # ---------------------------------
    elif args.operation == "refresh_cache":
        """
        Refreshes the vector store cache from the API.
        Example usage:
          {
            "operation": "refresh_cache"
          }
        """
        try:
            status = refresh_cache_manually()
            return json.dumps(status)
        except Exception as e:
            return json.dumps({"error": str(e)})

    # ---------------------------------
    # OPERATION: delete_store
    # ---------------------------------
    elif args.operation == "delete_store":
        """
        Deletes a vector store entirely.
        Example usage:
          {
            "operation": "delete_store",
            "vector_store_id": "store_123"
          }
        """
        if not args.vector_store_id:
            return json.dumps({"error": "vector_store_id is required for delete_store"})

        try:
            client.vector_stores.delete(vector_store_id=args.vector_store_id)

            # Update cache
            cache = get_vector_store_cache()
            was_active = (cache["active_vector_store_id"] == args.vector_store_id)
            cache["vector_stores"] = [
                vs for vs in cache["vector_stores"] 
                if vs["id"] != args.vector_store_id
            ]
            cache["last_updated"] = datetime.now().isoformat()

            if was_active:
                cache["active_vector_store_id"] = None
                # Attempt to update .env file
                try:
                    project_root = os.path.dirname(
                        os.path.dirname(
                            os.path.dirname(os.path.dirname(__file__))
                        )
                    )
                    env_file_path = os.path.join(project_root, ".env")
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
                    pass  # Continue even if .env update fails

            update_vector_store_cache(cache)
            return json.dumps({
                "status": "success",
                "message": f"Vector store {args.vector_store_id} deleted successfully"
            })
        except Exception as e:
            return json.dumps({"status": "failed", "error": str(e)})

    # ---------------------------------
    # OPERATION: set_active_store
    # ---------------------------------
    elif args.operation == "set_active_store":
        """
        Sets the active vector store (writes to cache and .env).
        If vector_store_id is 'null' or 'none' (case-insensitive), clears the active store.
        Example usage:
          {
            "operation": "set_active_store",
            "vector_store_id": "store_123"
          }
        """
        if not args.vector_store_id:
            return json.dumps({"error": "vector_store_id is required for set_active_store"})

        # Handle null/none
        if args.vector_store_id.lower() in ["null", "none"]:
            try:
                cache = get_vector_store_cache()
                cache["active_vector_store_id"] = None
                update_vector_store_cache(cache)

                # Also try to update .env
                env_file_path = os.path.join(
                    os.path.dirname(
                        os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                    ),
                    ".env"
                )
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

                return json.dumps({
                    "status": "success",
                    "message": "Active vector store cleared"
                })
            except Exception as e:
                return json.dumps({"status": "failed", "error": str(e)})

        # If a normal ID
        try:
            # Optional: fetch details to ensure it exists
            # (If it doesn't, this won't fail the set_active_store operation by default.)
            details_result = get_cached_details(args.vector_store_id)
            if not details_result:
                # Attempt an API lookup; ignore errors
                try:
                    _ = client.vector_stores.retrieve(vector_store_id=args.vector_store_id)
                except Exception:
                    pass

            success = set_active_vector_store(args.vector_store_id)
            if success:
                return json.dumps({
                    "status": "success",
                    "message": f"Active store set to {args.vector_store_id}"
                })
            else:
                return json.dumps({
                    "status": "failed",
                    "message": f"Failed to set active store to {args.vector_store_id}"
                })
        except Exception as e:
            return json.dumps({"status": "failed", "error": str(e)})

    # If none of the above operations matched:
    error_result = json.dumps({"error": f"Unknown operation: {args.operation}"})
    if logging_enabled:
        logger.error(f"Unknown operation: {args.operation}")
        logger.debug(f"Returning error result: {error_result}")
    return error_result


# Create the single FunctionTool object for your vector store manager.
# Define a manually constructed schema that we know works with OpenAI
manual_schema = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "operation": {
            "type": "string",
            "enum": [
                "create_store",
                "upload_file",
                "upload_directory",
                "list_stores",
                "get_store_details",
                "list_files",
                "delete_file",
                "refresh_cache",
                "delete_store",
                "set_active_store"
            ],
            "description": "The operation to perform on the vector store"
        },
        "store_name": {
            "type": ["string", "null"],
            "description": "Name of the vector store to create (used by create_store)"
        },
        "vector_store_id": {
            "type": ["string", "null"],
            "description": "ID of the vector store to operate on"
        },
        "file_path": {
            "type": ["string", "null"],
            "description": "Path to a single file (used by upload_file, delete_file, etc)"
        },
        "directory_path": {
            "type": ["string", "null"],
            "description": "Path to a directory (used by upload_directory)"
        },
        "file_pattern": {
            "type": ["string", "null"],
            "description": "Glob file pattern (used by upload_directory)"
        },
        "file_id": {
            "type": ["string", "null"],
            "description": "ID of the file to delete (used by delete_file)"
        }
    },
    "required": ["operation", "store_name", "vector_store_id", "file_path", "directory_path", "file_pattern", "file_id"],
    "additionalProperties": False
}

vector_store_manager_tool = FunctionTool(
    name="vector_store_manager",
    description=(
        "A single tool to handle all vector store operations, including creating stores, "
        "uploading files, listing stores, refreshing caches, and more. "
        "Use the 'operation' field to specify the desired action."
    ),
    params_json_schema=manual_schema,
    on_invoke_tool=_vector_store_manager_invoke,
)
