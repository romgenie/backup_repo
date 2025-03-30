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
    # Parse the JSON manually to handle required fields with null values
    args_dict = json.loads(args_json)
    
    # Ensure all required fields are present with defaults if not provided
    for field in ["store_name", "vector_store_id", "file_path", "directory_path", "file_pattern", "file_id"]:
        if field not in args_dict or args_dict[field] is None:
            args_dict[field] = None
    
    # Convert to VectorStoreManagerArgs
    args = VectorStoreManagerArgs(**args_dict)

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
    return json.dumps({"error": f"Unknown operation: {args.operation}"})


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
