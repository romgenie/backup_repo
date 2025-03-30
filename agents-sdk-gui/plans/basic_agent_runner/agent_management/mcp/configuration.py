"""
Configuration management functions for MCP servers.
"""

import json
from typing import Dict, Any, Optional

def save_configurations_to_file(manager, file_path: str) -> bool:
    """
    Save server configurations to a JSON file
    
    Args:
        manager: The McpManager instance
        file_path: Path to save the configurations
        
    Returns:
        Success status
    """
    try:
        with open(file_path, 'w') as f:
            json.dump(manager.server_configs, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving MCP configurations: {str(e)}")
        return False

def load_configurations_from_file(manager, file_path: str) -> bool:
    """
    Load server configurations from a JSON file
    
    Args:
        manager: The McpManager instance
        file_path: Path to load the configurations from
        
    Returns:
        Success status
    """
    try:
        with open(file_path, 'r') as f:
            configs = json.load(f)
            
        # Close any existing servers
        manager.close_all_servers()
            
        # Update configurations
        manager.server_configs = configs
        return True
    except Exception as e:
        print(f"Error loading MCP configurations: {str(e)}")
        return False