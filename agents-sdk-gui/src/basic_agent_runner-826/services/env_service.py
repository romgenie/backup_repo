"""
Environment service for the Agent Runner application.
"""

import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from typing import Tuple, Optional


def initialize_environment() -> Tuple[str, str]:
    """
    Initialize the environment variables and return API key and env file path.
    
    Returns:
        Tuple[str, str]: API key and environment file path
    """
    # Load .env file if it exists
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path)
    
    # Get OpenAI API key from environment
    api_key = os.environ.get("OPENAI_API_KEY", "")
    env_file_path = dotenv_path if dotenv_path else str(Path.cwd() / ".env")
    
    return api_key, env_file_path


def update_api_key(api_key: str) -> None:
    """
    Update the API key in the environment.
    
    Args:
        api_key: The API key to set
    """
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key