"""
MCP service for the Agent Runner application.
"""

import os
import json
from typing import Dict, Any, List, Optional
import streamlit as st

from agent_management.mcp_manager import MCPManager


def load_mcp_configurations(config_path: str) -> Dict[str, Any]:
    """
    Load MCP configurations from a file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dict of MCP configurations
    """
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


def initialize_mcp_state() -> None:
    """
    Initialize MCP-related session state variables.
    """
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


def sync_mcp_manager_with_session_state(mcp_manager: MCPManager) -> None:
    """
    Sync Streamlit session state with MCP manager.
    
    Args:
        mcp_manager: The MCP manager instance
    """
    for server_id, config in st.session_state.mcp_servers.items():
        mcp_manager.add_server_config(server_id, config)
        print(f"Loaded config from session state: {server_id}")