"""
State management functions for MCP tools.
"""

import streamlit as st
from typing import Dict, Any, Optional

def add_tool_to_session(tool_id: str, tool_info: Dict[str, Any]) -> None:
    """
    Add an MCP tool to the session state
    
    Args:
        tool_id: The unique ID of the tool
        tool_info: Dictionary with tool information
    """
    # Initialize MCP tools dict if not present
    if "mcp_tools" not in st.session_state:
        st.session_state.mcp_tools = {}
    
    # Add or update the tool
    st.session_state.mcp_tools[tool_id] = tool_info
    
    # Initialize selected tools set if not present
    if "selected_mcp_tools" not in st.session_state:
        st.session_state.selected_mcp_tools = set()

def remove_tool_from_session(tool_id: str) -> bool:
    """
    Remove an MCP tool from the session state
    
    Args:
        tool_id: The unique ID of the tool
        
    Returns:
        Boolean indicating if the tool was removed
    """
    # Check if MCP tools dict exists
    if "mcp_tools" not in st.session_state:
        return False
    
    # Check if the tool exists
    if tool_id not in st.session_state.mcp_tools:
        return False
    
    # Remove the tool
    del st.session_state.mcp_tools[tool_id]
    
    # Also remove from selected tools if present
    if "selected_mcp_tools" in st.session_state and tool_id in st.session_state.selected_mcp_tools:
        st.session_state.selected_mcp_tools.remove(tool_id)
    
    return True

def get_tool_from_session(tool_id: str) -> Optional[Dict[str, Any]]:
    """
    Get an MCP tool from the session state
    
    Args:
        tool_id: The unique ID of the tool
        
    Returns:
        Dictionary with tool information or None if not found
    """
    # Check if MCP tools dict exists
    if "mcp_tools" not in st.session_state:
        return None
    
    # Return the tool if it exists
    return st.session_state.mcp_tools.get(tool_id)

def get_selected_tools() -> set:
    """
    Get the set of selected MCP tool IDs
    
    Returns:
        Set of selected tool IDs
    """
    # Initialize if not present
    if "selected_mcp_tools" not in st.session_state:
        st.session_state.selected_mcp_tools = set()
    
    return st.session_state.selected_mcp_tools

def set_selected_tools(tool_ids: set) -> None:
    """
    Set the selected MCP tools
    
    Args:
        tool_ids: Set of tool IDs to select
    """
    st.session_state.selected_mcp_tools = tool_ids