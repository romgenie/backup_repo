"""
Vector store agent implementation.
"""
import re
import logging
from agents import Agent, function_tool
from src.agents.vector_store.prompts import BASE_INSTRUCTIONS, VOICE_INSTRUCTIONS

# Configure logging
logger = logging.getLogger("vector_store_agent")
logger.setLevel(logging.INFO)
from src.tools.vector_store.vector_store_tools import (
    create_new_vector_store,
    upload_file_to_vector_store,
    upload_directory_to_vector_store,
    list_vector_stores,
    get_vector_store_details,
    list_files_in_vector_store,
    delete_file_from_vector_store,
    delete_vector_store,
    save_vector_store_id_to_env,
    refresh_vector_store_cache
)
from src.tools.file_search.file_search import get_file_search_tool
from src.tools.vector_store.download_tool import download_file_from_url

@function_tool
def auto_upload_to_vector_store(file_path: str, vector_store_id: str) -> dict:
    """
    Auto-upload a file to the vector store. Used internally by the VectorStoreAgent
    to ensure files are actually uploaded.
    
    Args:
        file_path: The file path to upload
        vector_store_id: The vector store ID to upload to
        
    Returns:
        dict: The upload result
    """
    # This is a wrapper around the regular function to provide logging
    logger.info(f"Auto-uploading {file_path} to vector store {vector_store_id}")
    
    # Try to get the active vector store ID if none provided
    if not vector_store_id or vector_store_id.lower() in ["null", "none"]:
        from src.utils.vector_store_cache import get_active_vector_store_id
        vector_store_id = get_active_vector_store_id()
        logger.info(f"Using active vector store ID: {vector_store_id}")
        
    # Call the regular upload function
    result = upload_file_to_vector_store(file_path=file_path, vector_store_id=vector_store_id)
    
    # Log the result
    if result.get("status") == "success":
        logger.info(f"Successfully uploaded {file_path}")
    else:
        logger.error(f"Failed to upload {file_path}: {result.get('error', 'Unknown error')}")
        
    return result

def process_handoff_message(message: str) -> list:
    """
    Process a handoff message to extract file paths and upload them to the vector store.
    
    Args:
        message: The message from the handoff
        
    Returns:
        list: List of processed file paths
    """
    # Extract file paths with regex
    file_paths = re.findall(r'/[\w/\-\.]+\.pdf', message)
    logger.info(f"Extracted {len(file_paths)} file paths from handoff message")
    
    # Return the unique file paths
    return list(set(file_paths))

def create_vector_store_agent(use_voice_optimization: bool = False) -> Agent:
    """
    Create a vector store agent that manages and searches vector stores.
    
    Args:
        use_voice_optimization: Whether to use voice-optimized instructions
        
    Returns:
        Agent: The initialized vector store agent
    """
    # Choose appropriate instructions based on voice optimization
    instructions = VOICE_INSTRUCTIONS if use_voice_optimization else BASE_INSTRUCTIONS
    
    # Add special instructions to force the agent to use auto_upload_to_vector_store
    force_upload_instruction = """
    MANDATORY ACTION: When you receive a handoff with file paths, you MUST IMMEDIATELY use auto_upload_to_vector_store() 
    for EACH file path you identify before doing anything else.
    """
    
    enhanced_instructions = instructions + "\n" + force_upload_instruction
    
    # Prepare the tools list
    tools = [
        create_new_vector_store,
        upload_file_to_vector_store,
        upload_directory_to_vector_store,
        list_vector_stores,
        get_vector_store_details,
        list_files_in_vector_store,
        delete_file_from_vector_store,
        delete_vector_store,
        save_vector_store_id_to_env,
        refresh_vector_store_cache,
        # Add auto-upload function
        auto_upload_to_vector_store,
        # Add download tool for multi-step workflows
        download_file_from_url
    ]
    
    # Add file search tool if available
    file_search_tool = get_file_search_tool()
    if file_search_tool:
        tools.append(file_search_tool)
        
    # Create an auto-processing function that the agent will be explicitly instructed to call
    @function_tool
    def process_handoff_automatically(message: str) -> dict:
        """
        FIRST PRIORITY FUNCTION: Always call this function FIRST when you receive a handoff about adding papers.
        
        This function automatically processes a handoff message to extract file paths and upload them.
        
        Args:
            message: The handoff message to process
            
        Returns:
            dict: A dictionary with the results of processing
        """
        logger.info("Auto-processing handoff message")
        
        # Extract file paths
        file_paths = process_handoff_message(message)
        
        if not file_paths:
            logger.warning("No file paths found in handoff message")
            return {"status": "warning", "message": "No file paths found in handoff message"}
        
        logger.info(f"Found {len(file_paths)} file paths to auto-upload")
        
        # Try to get the active vector store ID
        from src.utils.vector_store_cache import get_active_vector_store_id
        vector_store_id = get_active_vector_store_id()
        
        if not vector_store_id:
            logger.warning("No active vector store ID found")
            return {"status": "error", "message": "No active vector store ID found - cannot upload files"}
        
        # Track upload results
        upload_results = []
        
        # Upload each file
        for file_path in file_paths:
            try:
                result = upload_file_to_vector_store(file_path=file_path, vector_store_id=vector_store_id)
                upload_results.append(result)
            except Exception as e:
                logger.error(f"Error auto-uploading {file_path}: {str(e)}")
                upload_results.append({
                    "file": file_path,
                    "status": "failed",
                    "error": str(e)
                })
        
        return {
            "status": "complete",
            "message": f"Processed {len(file_paths)} file paths",
            "vector_store_id": vector_store_id,
            "results": upload_results
        }
    
    # Add the processing function to tools
    tools.append(process_handoff_automatically)
    
    # Force the agent to always call the processing function first
    enhanced_instructions += """
    
    CRITICAL PROCESSING INSTRUCTION:
    YOUR FIRST ACTION upon receiving ANY message containing phrases like 'add to vector store', 'add papers', 
    or references to PDF files MUST BE to call process_handoff_automatically() with the FULL message text.
    
    ABSOLUTELY DO NOT respond to the user or perform any other actions until you have called process_handoff_automatically()
    when you receive messages about adding documents to the vector store.
    
    If process_handoff_automatically() reports success, you MUST mention the specific files that were uploaded in your response.
    """
    
    return Agent(
        name="VectorStoreAgent",
        instructions=enhanced_instructions,
        tools=tools
    )