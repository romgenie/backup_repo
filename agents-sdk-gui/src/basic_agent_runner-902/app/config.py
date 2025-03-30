"""
Configuration management for the Agent Runner application.
"""

import os
import json
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from typing import Dict, Any, Optional

def initialize_environment():
    """Initialize the environment variables and paths."""
    # Load .env file if it exists
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path)
    
    # Get OpenAI API key from environment
    api_key = os.environ.get("OPENAI_API_KEY", "")
    env_file_path = dotenv_path if dotenv_path else str(Path.cwd() / ".env")
    
    return api_key, env_file_path

def load_mcp_configurations(config_path: str) -> Dict[str, Any]:
    """Load MCP configurations from a file."""
    configs = {}
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                saved_configs = json.load(f)
                if isinstance(saved_configs, dict):
                    configs = saved_configs
                    print(f"Loaded {len(saved_configs)} MCP server configurations from file")
        except Exception as e:
            print(f"Error loading MCP configurations from file: {str(e)}")
    
    return configs

def initialize_mcp_state():
    """Initialize MCP-related session state variables."""
    # Initialize session state for MCP servers if not present
    if "mcp_servers" not in st.session_state:
        st.session_state.mcp_servers = {}
    
    # Auto-select MCP servers if available but not yet selected
    if "selected_mcp_servers" not in st.session_state:
        st.session_state.selected_mcp_servers = []
        print("DEBUG: Initialized empty selected_mcp_servers list")
    
    # If we have servers but none are selected, auto-select them all
    if len(st.session_state.mcp_servers) > 0 and len(st.session_state.selected_mcp_servers) == 0:
        st.session_state.selected_mcp_servers = list(st.session_state.mcp_servers.keys())
        print(f"DEBUG: Auto-selected MCP servers: {st.session_state.selected_mcp_servers}")