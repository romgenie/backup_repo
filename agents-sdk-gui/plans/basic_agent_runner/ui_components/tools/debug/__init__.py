"""
MCP debugging tools and components.

This module provides UI components for debugging MCP servers, tool calls,
and results. It offers functionality for monitoring server health, managing
tool caches, testing tools, and viewing logs.
"""

import streamlit as st
from typing import Dict, Any, List, Optional

# Import the modular components
from ui_components.tools.debug.server_monitor import (
    render_server_health_status,
    render_cache_management
)
from ui_components.tools.debug.call_inspector import render_tool_tester_section
from ui_components.tools.debug.result_viewer import (
    render_tool_results_viewer,
    render_tool_result_details
)
from ui_components.tools.debug.log_display import render_server_logs

def render_server_debug_section(container, mcp_manager) -> None:
    """
    Render the MCP server debugging section
    
    Args:
        container: Streamlit container to render in
        mcp_manager: MCP manager instance
    """
    with container:
        st.markdown("### MCP Server and Tool Debugging")
        
        # Create tabs for different debug sections
        debug_tabs = st.tabs([
            "Server Health", 
            "Cache Management", 
            "Tool Tester", 
            "Server Logs"
        ])
        
        # Server Health tab
        with debug_tabs[0]:
            render_server_health_status(mcp_manager)
            
        # Cache Management tab
        with debug_tabs[1]:
            render_cache_management(mcp_manager)
            
        # Tool Tester tab
        with debug_tabs[2]:
            render_tool_tester_section()
            
        # Server Logs tab
        with debug_tabs[3]:
            render_server_logs(mcp_manager)