from pathlib import Path

def save_to_env_file(env_file_path, api_key):
    """
    Save API key to .env file
    
    Args:
        env_file_path (str): Path to the .env file
        api_key (str): API key to save
        
    Returns:
        str: Path to the saved .env file
    """
    env_path = Path(env_file_path)
    
    # Create or update existing .env file
    env_content = {}
    if env_path.exists():
        # Read existing content
        with open(env_path, "r") as file:
            for line in file:
                if "=" in line and not line.startswith("#"):
                    key, value = line.strip().split("=", 1)
                    env_content[key] = value
    
    # Update API key
    env_content["OPENAI_API_KEY"] = api_key
    
    # Write back to file
    with open(env_path, "w") as file:
        for key, value in env_content.items():
            file.write(f"{key}={value}\n")
    
    return str(env_path)