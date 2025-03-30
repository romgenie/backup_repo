import streamlit as st
from typing import List
from agents import Tool

from ui_components.tools import (
    render_custom_tool_form,
    render_example_tools_form,
    render_predefined_tool_form,
    render_mcp_tool_form,
    display_tools_list
)

def render_tools_tab() -> None:
    """
    Render the tools tab of the application
    """
    st.header("Tools Configuration")
    
    # Callback for tool selection
    def on_tool_select(tools: List[Tool]):
        # Store selected tools in session state for the agent
        st.session_state.selected_tools = tools
    
    # Render tools forms and list
    example_tools_added = render_example_tools_form()
    st.divider()
    predefined_tool_added = render_predefined_tool_form()
    st.divider()
    mcp_tool_added = render_mcp_tool_form()
    st.divider()
    custom_tool_created = render_custom_tool_form()
    st.divider()
    tools_applied = display_tools_list(on_tool_select)
    
    # If any tools were added or applied, we might need to rerun to refresh the UI
    if example_tools_added or predefined_tool_added or mcp_tool_added or custom_tool_created or tools_applied:
        # This is handled inside each component now
        pass