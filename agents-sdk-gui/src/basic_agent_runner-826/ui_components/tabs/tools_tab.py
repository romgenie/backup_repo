import streamlit as st
from typing import List
from agents import Tool

from ui_components.tools.mcp_tools import render_mcp_tools_section

def render_tools_tab() -> None:
    """
    Render the tools tab of the application
    """
    st.header("Tools Configuration")
    
    st.info("The tools functionality has been migrated to the MCP tab. Please use MCP servers to access tools.")
    
    # Display any existing MCP tools that may be available
    if "mcp_tools" in st.session_state:
        render_mcp_tools_section(st.session_state.mcp_tools)
    else:
        st.markdown("""
        ### Using MCP Tools
        
        1. Go to the **MCP** tab to add and configure MCP servers
        2. Test your connections to verify server availability
        3. Select the servers you want to use with your agent
        4. Apply your selection
        5. Create a new agent in the Configure tab to use these tools
        """)