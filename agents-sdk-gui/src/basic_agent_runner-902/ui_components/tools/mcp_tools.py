import streamlit as st
import json
from typing import Dict, Any, List, Optional, Callable

def render_mcp_tools_section(
    mcp_tools: Dict[str, Dict[str, Any]],
    on_tool_select: Optional[Callable[[List[str]], None]] = None
) -> bool:
    """
    Render a section for MCP tools
    
    Args:
        mcp_tools: Dictionary of available MCP tools
        on_tool_select: Callback when tools are selected
        
    Returns:
        Boolean indicating if any changes were made
    """
    # If no MCP tools, show appropriate message
    if not mcp_tools:
        st.info("No MCP tools are available. Add MCP servers in the MCP tab.")
        return False
    
    changes_made = False
    
    # Initialize selection state if not present
    if "selected_mcp_tools" not in st.session_state:
        st.session_state.selected_mcp_tools = set()
    
    # Group tools by server for better organization
    tools_by_server = {}
    for tool_id, tool_info in mcp_tools.items():
        server_name = tool_info.get("server_name", "Unknown")
        if server_name not in tools_by_server:
            tools_by_server[server_name] = {}
        tools_by_server[server_name][tool_id] = tool_info
    
    # Display tools grouped by server
    for server_name, server_tools in tools_by_server.items():
        with st.expander(f"{server_name} ({len(server_tools)} tools)", expanded=True):
            # Header row
            col1, col2, col3 = st.columns([1, 3, 2])
            with col1:
                st.write("**Use**")
            with col2:
                st.write("**Tool**")
            with col3:
                st.write("**Description**")
            
            # List tools
            for tool_id, tool_info in server_tools.items():
                col1, col2, col3 = st.columns([1, 3, 2])
                
                with col1:
                    # Checkbox to select the tool
                    is_selected = tool_id in st.session_state.selected_mcp_tools
                    selected = st.checkbox(
                        "", 
                        value=is_selected, 
                        key=f"select_mcp_tool_{tool_id}",
                        help="Select this tool for use with agents"
                    )
                    
                    # Update selection state
                    if selected and tool_id not in st.session_state.selected_mcp_tools:
                        st.session_state.selected_mcp_tools.add(tool_id)
                        changes_made = True
                    elif not selected and tool_id in st.session_state.selected_mcp_tools:
                        st.session_state.selected_mcp_tools.remove(tool_id)
                        changes_made = True
                
                with col2:
                    st.markdown(f"**{tool_info.get('name', 'Unnamed Tool')}**")
                
                with col3:
                    st.markdown(tool_info.get("description", "*No description available*"))
                
                # Show tool details in an expander
                with st.expander("Tool Details"):
                    # Parameters section if available
                    if "parameters" in tool_info:
                        st.markdown("**Parameters:**")
                        for param in tool_info["parameters"]:
                            param_name = param.get("name", "unnamed")
                            param_type = param.get("type", "unknown")
                            param_desc = param.get("description", "No description")
                            param_required = param.get("required", False)
                            
                            required_text = " (Required)" if param_required else ""
                            st.markdown(f"- **{param_name}** ({param_type}){required_text}: {param_desc}")
                    
                    # Return type if available
                    if "return_type" in tool_info:
                        st.markdown(f"**Returns:** {tool_info['return_type']}")
    
    # Apply button if callback provided
    if on_tool_select and st.session_state.selected_mcp_tools:
        if st.button("Apply Selected MCP Tools"):
            selected_tools = list(st.session_state.selected_mcp_tools)
            on_tool_select(selected_tools)
            st.success(f"Selected {len(selected_tools)} MCP tools")
            changes_made = True
    
    return changes_made

def display_mcp_tool_calls(container, tool_calls: List[Dict[str, Any]]):
    """
    Display MCP tool calls with a specific format
    
    Args:
        container: Streamlit container to display in
        tool_calls: List of MCP tool call dictionaries
    """
    with container:
        if not tool_calls:
            st.info("No MCP tool calls in this conversation.")
            return
        
        st.markdown("### MCP Tool Calls")
        
        for i, call in enumerate(tool_calls):
            # Determine status color/icon
            status = call.get("status", "unknown")
            if status == "success":
                status_icon = ":green[✓]"
            elif status == "error" or status == "processing_error":
                status_icon = ":red[✗]"
            elif status == "no_result":
                status_icon = ":orange[⚠]"
            else:
                status_icon = ":grey[?]"
            
            # Create expander with status indicator
            with st.expander(f"{call.get('name', 'Tool')} {status_icon} ({call.get('time', '')})", expanded=i==0):
                # Tool call information
                st.markdown(f"**Server:** {call.get('server', 'Unknown')}")
                st.markdown(f"**Status:** {status}")
                
                # Arguments
                st.markdown("**Arguments:**")
                st.json(call.get("args", {}))
                
                # Result handling with more detailed diagnostics
                if "result" in call:
                    st.markdown("**Result:**")
                    
                    # Determine how to display the result based on type
                    result = call["result"]
                    
                    # Show result type for debugging
                    result_type = type(result).__name__
                    st.info(f"Result type: {result_type}")
                    
                    if result is None:
                        st.info("No result returned (null/None)")
                    elif isinstance(result, dict):
                        if len(result) == 0:
                            st.info("Empty dictionary result {}")
                        else:
                            st.json(result)
                    elif isinstance(result, str):
                        if not result.strip():  # Empty or whitespace-only string
                            st.info("Empty string result")
                        else:
                            # Check if it's JSON
                            try:
                                json_result = json.loads(result)
                                st.json(json_result)
                            except json.JSONDecodeError:
                                # Regular text
                                st.markdown(f"```\n{result}\n```")
                    elif isinstance(result, (list, tuple)):
                        if len(result) == 0:
                            st.info(f"Empty {result_type} result")
                        else:
                            st.json(result)
                    else:
                        st.write(f"Result (type: {result_type}):")
                        st.write(result)
                        # Show raw representation for debugging
                        st.code(repr(result), language="python")
                elif status == "no_result":
                    st.warning("Tool executed but returned no result")
                
                # Timing information
                if "timing" in call:
                    st.markdown(f"**Duration:** {call['timing']} ms")
                
                # Error information if present
                if "error" in call:
                    st.error(f"**Error:** {call['error']}")
                    
                    # Show stack trace if available
                    if "stack" in call:
                        with st.expander("Error Details"):
                            st.code(call["stack"], language="python")