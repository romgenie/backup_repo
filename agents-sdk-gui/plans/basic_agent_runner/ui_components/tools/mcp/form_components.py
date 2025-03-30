"""
Form components for MCP tool configuration.
"""

import streamlit as st
import json
from typing import Dict, Any, List, Optional, Callable

def render_tool_selector(
    available_tools: Dict[str, Dict[str, Any]],
    on_selection: Callable[[str], None] = None
) -> Optional[str]:
    """
    Render a selector for MCP tools
    
    Args:
        available_tools: Dictionary of available MCP tools
        on_selection: Callback when a tool is selected
        
    Returns:
        The selected tool ID or None if no selection
    """
    selected_tool = None
    
    # Only proceed if we have tools
    if not available_tools:
        st.markdown("""
        <div class="warning-message">
            No MCP tools are available for selection
        </div>
        """, unsafe_allow_html=True)
        return None
    
    # Create a list of tool names for the selectbox
    tool_options = []
    tool_id_mapping = {}
    
    for tool_id, tool_info in available_tools.items():
        tool_name = tool_info.get("name", "Unknown Tool")
        display_name = f"{tool_name} ({tool_id})"
        tool_options.append(display_name)
        tool_id_mapping[display_name] = tool_id
        
    # Sort options alphabetically
    tool_options.sort()
    
    # Add a blank option at the beginning
    tool_options.insert(0, "-- Select a tool --")
    
    # Create the selectbox
    selected_option = st.selectbox(
        "Select MCP Tool",
        options=tool_options,
        key="mcp_tool_selector"
    )
    
    # If a real tool is selected (not the placeholder)
    if selected_option != "-- Select a tool --":
        selected_tool = tool_id_mapping[selected_option]
        
        # Call the callback if provided
        if on_selection:
            on_selection(selected_tool)
    
    return selected_tool

def render_parameter_form(
    tool_info: Dict[str, Any],
    on_submit: Callable[[Dict[str, Any]], None] = None
) -> bool:
    """
    Render a form for configuring MCP tool parameters
    
    Args:
        tool_info: Dictionary with tool information
        on_submit: Callback when the form is submitted
        
    Returns:
        Boolean indicating if the form was submitted
    """
    submitted = False
    
    # Extract tool parameters
    parameters = tool_info.get("parameters", [])
    tool_name = tool_info.get("name", "Unknown Tool")
    
    # Create a form for the parameters
    with st.form(f"mcp_tool_params_{tool_name}"):
        st.markdown(f"### Configure {tool_name}")
        
        # Store parameter values
        param_values = {}
        
        # Render parameter fields
        if parameters:
            for param in parameters:
                param_name = param.get("name", "unnamed")
                param_desc = param.get("description", "No description")
                param_type = param.get("type", "string")
                param_required = param.get("required", False)
                
                # Label with required indicator
                label = f"{param_name}" + (" *" if param_required else "")
                
                # Render appropriate input field based on parameter type
                if param_type == "boolean":
                    param_values[param_name] = st.checkbox(
                        label,
                        value=False,
                        help=param_desc
                    )
                elif param_type == "integer" or param_type == "number":
                    param_values[param_name] = st.number_input(
                        label,
                        value=0,
                        help=param_desc
                    )
                elif param_type == "object":
                    # For objects, use a JSON text area
                    json_str = st.text_area(
                        label,
                        value="{}",
                        help=f"{param_desc} (Enter as JSON)"
                    )
                    try:
                        param_values[param_name] = json.loads(json_str)
                    except json.JSONDecodeError:
                        st.error(f"Invalid JSON for parameter {param_name}")
                        param_values[param_name] = {}
                else:
                    # Default to string input
                    param_values[param_name] = st.text_input(
                        label,
                        value="",
                        help=param_desc
                    )
        else:
            st.markdown("*This tool has no parameters*")
        
        # Submit button
        submit_button = st.form_submit_button("Submit")
        
        if submit_button:
            submitted = True
            
            # Call the callback if provided
            if on_submit:
                on_submit(param_values)
    
    return submitted