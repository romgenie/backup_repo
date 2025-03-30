from pathlib import Path
import os
from dotenv import load_dotenv

def load_from_env_file(env_file_path):
    """
    Load API key from .env file
    
    Args:
        env_file_path (str): Path to the .env file
        
    Returns:
        str: The loaded API key or empty string if not found
        bool: True if the file was loaded successfully
    """
    env_path = Path(env_file_path)
    if env_path.exists():
        load_dotenv(env_path, override=True)
        api_key = os.environ.get("OPENAI_API_KEY", "")
        return api_key, True
    return "", False