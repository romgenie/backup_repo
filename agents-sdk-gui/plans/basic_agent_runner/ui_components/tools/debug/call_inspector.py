"""
MCP tool call inspection components.
"""

import streamlit as st
import json
import pandas as pd
from typing import Dict, Any, List, Optional

def render_tool_tester_section() -> None:
    """
    Render the MCP tool tester section
    """
    st.markdown("### MCP Tool Tester")
    st.write("This utility helps you test MCP tools with example invocations.")
    
    # Get MCP tools if available
    from ui_components.tabs.tools_tab import get_available_mcp_tools
    
    # Get tools
    mcp_tools = get_available_mcp_tools()
    
    if mcp_tools:
        # Create tool selection
        tool_names = []
        tool_info_map = {}
        
        # Extract tool names and organize by name for selection
        for tool_id, tool_info in mcp_tools.items():
            name = tool_info.get("name", "unnamed")
            server = tool_info.get("server_name", "Unknown Server")
            display_name = f"{name} ({server})"
            
            tool_names.append(display_name)
            tool_info_map[display_name] = tool_info
        
        # Sort tool names
        tool_names.sort()
        
        # Tool selection dropdown
        selected_tool = st.selectbox(
            "Select Tool to Test:",
            options=tool_names,
            index=0 if tool_names else None
        )
        
        if selected_tool:
            # Get the tool info
            tool_info = tool_info_map[selected_tool]
            
            # Display tool details
            st.markdown(f"**Description:** {tool_info.get('description', 'No description available')}")
            
            # Check if saved parameter sets exist in session state
            if "saved_param_sets" not in st.session_state:
                st.session_state.saved_param_sets = {}
            
            # Create a unique key for this tool
            tool_key = f"{tool_info.get('server_name', 'unknown')}_{tool_info.get('name', 'unnamed')}"
            
            # Check if we have saved parameters for this tool
            saved_sets = st.session_state.saved_param_sets.get(tool_key, {})
            
            # Get parameters for the tool
            params = tool_info.get("parameters", [])
            
            # If the tool has parameters, offer saved parameter sets
            if params and saved_sets:
                # Create a selectbox for saved parameter sets
                saved_set_names = list(saved_sets.keys())
                saved_set_names.insert(0, "-- New Parameters --")
                
                selected_set = st.selectbox(
                    "Use Saved Parameter Set",
                    options=saved_set_names,
                    index=0
                )
                
                # Initialize param values based on selection
                if selected_set != "-- New Parameters --":
                    # Use the saved parameter set
                    param_values = saved_sets[selected_set].copy()
                else:
                    # Start with empty values
                    param_values = {}
            else:
                # No saved sets available
                param_values = {}
                selected_set = "-- New Parameters --"
            
            # Add a name field for saving the parameter set (outside the form)
            save_as_name = st.text_input(
                "Save Parameter Set As (optional)",
                value="",
                key="save_param_name",
                help="Enter a name to save this parameter set for future use"
            )
            
            # Create a form for tool parameters
            with st.form("tool_test_form"):
                # Create input fields for parameters
                if params:
                    st.markdown("""
                    <div class="form-section">
                        <h3>Parameters</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown('<hr class="mt-sm mb-md" />', unsafe_allow_html=True)
                    
                    for param in params:
                        param_name = param.get("name", "unnamed")
                        param_type = param.get("type", "string")
                        param_desc = param.get("description", "No description")
                        param_required = param.get("required", False)
                        
                        # Label with required indicator
                        label = f"{param_name}" + (" (Required)" if param_required else "")
                        
                        # Default value from saved params if available
                        default_val = param_values.get(param_name, None)
                        
                        # Wrapper for form elements with consistent styling
                        st.markdown('<div class="form-group mb-sm">', unsafe_allow_html=True)
                        
                        # Create appropriate input field based on type
                        if param_type == "boolean":
                            param_values[param_name] = st.checkbox(
                                label,
                                value=bool(default_val) if default_val is not None else False,
                                help=param_desc
                            )
                        elif param_type == "number" or param_type == "integer":
                            param_values[param_name] = st.number_input(
                                label,
                                value=float(default_val) if default_val is not None else 0,
                                help=param_desc
                            )
                        else:  # Default to string
                            param_values[param_name] = st.text_input(
                                label,
                                value=str(default_val) if default_val is not None else "",
                                help=param_desc
                            )
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="info-message">
                        This tool has no parameters
                    </div>
                    """, unsafe_allow_html=True)
                    
                # Form buttons - side by side
                st.markdown('<div class="form-buttons mt-md">', unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    submit = st.form_submit_button("Generate Tool Call")
                with col2:
                    save_button = st.form_submit_button("Save Parameters")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Handle saving parameter set (outside the form to avoid conflicts)
            if save_button and save_as_name and params:
                # Save the parameter set
                if tool_key not in st.session_state.saved_param_sets:
                    st.session_state.saved_param_sets[tool_key] = {}
                
                # Store the parameter set
                st.session_state.saved_param_sets[tool_key][save_as_name] = param_values.copy()
                
                # Show success message
                st.markdown(f"""
                <div class="success-message">
                    Parameter set '{save_as_name}' saved successfully!
                </div>
                """, unsafe_allow_html=True)
                
                # Force a rerun to update the UI
                st.rerun()
                
            # When form is submitted, show the tool call format
            if submit:
                handle_tool_call_generation(tool_info, params, param_values)
    else:
        st.markdown("""
        <div class="warning-message text-center">
            <strong>No MCP tools available.</strong> Please add and test MCP servers in the MCP tab.
        </div>
        """, unsafe_allow_html=True)

def handle_tool_call_generation(tool_info: Dict[str, Any], params: List[Dict[str, Any]], param_values: Dict[str, Any]) -> None:
    """
    Handle the generation of tool call format

    Args:
        tool_info: Tool information dictionary
        params: List of parameter definitions
        param_values: Parameter values from the form
    """
    tool_name = tool_info.get("name", "unnamed")
    
    # Validate required parameters
    missing_required = []
    for param in params:
        param_name = param.get("name", "unnamed")
        param_required = param.get("required", False)
        param_type = param.get("type", "string")
        
        # Check if required parameter is missing
        if param_required:
            if param_name not in param_values or param_values[param_name] in [None, "", 0] and param_type != "boolean":
                missing_required.append(param_name)
    
    # Show warning if required parameters are missing
    if missing_required:
        st.markdown(f"""
        <div class="warning-message">
            <strong>Missing required parameters:</strong> {', '.join(missing_required)}
        </div>
        <div class="info-message mt-xs">
            Please provide values for all required parameters to generate a valid tool call.
        </div>
        """, unsafe_allow_html=True)
    else:
        # Create args JSON
        
        # Remove empty parameters
        clean_params = {}
        for key, value in param_values.items():
            if value not in [None, ""]:
                clean_params[key] = value
        
        args_json = json.dumps(clean_params, indent=2)
        
        # Format tool call
        tool_call = f"""TOOL_START_TAG{tool_name}TOOL_END_TAG
ARGS_START_TAG{args_json}ARGS_END_TAG"""
        
        # Show the tool call
        st.markdown("""
        <div class="form-section mt-md">
            <h3>Tool Call Format</h3>
        </div>
        """, unsafe_allow_html=True)
        st.code(tool_call)
        
        # Add validation feedback
        st.markdown("""
        <div class="success-message">
            ✅ Valid tool call format generated successfully
        </div>
        """, unsafe_allow_html=True)
        
        # Add a collapsible section with validator notes
        with st.expander("Validation Notes"):
            st.markdown("""
            **Tool Call Validation:**
            - Tool name is valid
            - All required parameters are provided
            - Empty parameters have been removed
            - Arguments are properly formatted as JSON
            
            **Reminder:** This is only a tool call generator, not an actual tool executor. 
            Use this format in the chat with your agent to invoke this tool.
            """)
        
        # Add a copy button
        st.markdown("""
        <div class="flex justify-end mt-sm">
            <button onclick="navigator.clipboard.writeText(`{}`)" 
                style="padding: 8px 16px; 
                      background-color: var(--primary-color); 
                      color: white; 
                      border: none; 
                      border-radius: var(--radius-md); 
                      cursor: pointer;
                      transition: background-color var(--transition-fast);"
                onmouseover="this.style.backgroundColor='var(--primary-dark)'"
                onmouseout="this.style.backgroundColor='var(--primary-color)'">
                Copy to Clipboard
            </button>
        </div>
        """.format(tool_call.replace("`", "\\`")), unsafe_allow_html=True)
        
        # Show example usage in conversation
        st.markdown("### Example Usage in Conversation")
        st.markdown("""
        ```
        User: What's the current time?
        
        Assistant: I'll check the current time for you.
        
        {}
        
        Based on the tool result, the current time is [time from result].
        ```
        """.format(tool_call))