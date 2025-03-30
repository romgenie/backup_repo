"""
MCP tool result viewer components.
"""

import streamlit as st
import json
import pandas as pd
from typing import Dict, Any, List, Optional

def render_tool_results_viewer(tool_calls: List[Dict[str, Any]]) -> None:
    """
    Render the MCP tool result viewer section
    
    Args:
        tool_calls: List of tool call data from the session
    """
    st.markdown("### MCP Tool Results")
    
    if not tool_calls:
        st.info("No tool calls have been executed yet in this session")
        return
    
    # Group tools by server for better organization
    tools_by_server = {}
    for tool in tool_calls:
        server_name = tool.get("server_name", "Unknown Server")
        if server_name not in tools_by_server:
            tools_by_server[server_name] = []
        tools_by_server[server_name].append(tool)
    
    # Create tabs for each server
    server_names = list(tools_by_server.keys())
    
    if len(server_names) > 1:
        tabs = st.tabs(["All Calls"] + server_names)
        
        # All calls tab
        with tabs[0]:
            _display_tool_calls(tool_calls)
        
        # Per-server tabs
        for i, server_name in enumerate(server_names):
            with tabs[i+1]:
                _display_tool_calls(tools_by_server[server_name])
    else:
        # Just one server, no need for tabs
        _display_tool_calls(tool_calls)

def _display_tool_calls(tool_calls: List[Dict[str, Any]]) -> None:
    """
    Display a list of tool calls with details
    
    Args:
        tool_calls: List of tool call data
    """
    # Sort by time (newest first)
    sorted_calls = sorted(tool_calls, key=lambda x: x.get("time", 0), reverse=True)
    
    # Display each call as an expander
    for i, call in enumerate(sorted_calls):
        tool_name = call.get("name", "Unknown Tool")
        timestamp = call.get("time", 0)
        server_name = call.get("server_name", "Unknown Server")
        
        # Format timestamp if available
        time_display = ""
        if timestamp:
            from datetime import datetime
            time_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
            time_display = f" at {time_str}"
        
        # Create a title with tool name and timestamp
        title = f"{tool_name} ({server_name}{time_display})"
        
        # Determine status icon
        has_error = "error" in call and call["error"]
        status_icon = "❌" if has_error else "✅"
        
        # Add status icon to title
        title = f"{status_icon} {title}"
        
        # Create an expander for this call
        with st.expander(title):
            # Show the input arguments
            args = call.get("args", {})
            
            st.markdown("**Arguments:**")
            if args:
                st.json(args)
            else:
                st.markdown("*No arguments provided*")
            
            # Show divider
            st.divider()
            
            # Show the result
            result = call.get("result", None)
            error = call.get("error", None)
            
            if error:
                st.markdown("**Error:**")
                st.error(error)
            elif result is not None:
                st.markdown("**Result:**")
                
                # Try to format the result based on its type
                if isinstance(result, dict) or (isinstance(result, str) and result.strip().startswith("{")):
                    try:
                        # If it's a string that looks like JSON, parse it
                        if isinstance(result, str):
                            result_obj = json.loads(result)
                        else:
                            result_obj = result
                        
                        # Format as JSON
                        st.json(result_obj)
                    except:
                        # Fall back to plain text if JSON parsing fails
                        st.markdown(f"```\n{result}\n```")
                else:
                    # Format as plain text
                    st.markdown(f"```\n{result}\n```")
            else:
                st.info("*No result returned*")
            
            # Show debug info in collapsible section
            with st.expander("Debug Information"):
                # Prepare debug info
                debug_info = {
                    "Tool ID": call.get("id", "Unknown"),
                    "Server": call.get("server_name", "Unknown Server"),
                    "Execution Time": f"{call.get('execution_time', 0):.3f} seconds" if call.get('execution_time') is not None else "Unknown",
                    "Status": "Error" if has_error else "Success"
                }
                
                # Show as table
                debug_df = pd.DataFrame([debug_info])
                st.dataframe(debug_df, use_container_width=True)

def render_tool_result_details(tool_call: Dict[str, Any]) -> None:
    """
    Render detailed information about a single tool call
    
    Args:
        tool_call: Tool call data
    """
    tool_name = tool_call.get("name", "Unknown Tool")
    
    st.markdown(f"## Tool Call: {tool_name}")
    
    # Create tabs for different aspects of the tool call
    tabs = st.tabs(["Summary", "Arguments", "Result", "Raw Data"])
    
    # Summary tab
    with tabs[0]:
        server_name = tool_call.get("server_name", "Unknown Server")
        timestamp = tool_call.get("time", 0)
        execution_time = tool_call.get("execution_time", None)
        has_error = "error" in tool_call and tool_call["error"]
        
        # Format timestamp
        time_str = "Unknown"
        if timestamp:
            from datetime import datetime
            time_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
        
        # Create info grid
        col1, col2 = st.columns(2)
        col1.metric("Tool Name", tool_name)
        col2.metric("Server", server_name)
        
        col1, col2 = st.columns(2)
        col1.metric("Timestamp", time_str)
        if execution_time is not None:
            col2.metric("Execution Time", f"{execution_time:.3f} seconds")
        
        col1, col2 = st.columns(2)
        status = "Error" if has_error else "Success"
        status_emoji = "❌" if has_error else "✅"
        col1.metric("Status", f"{status_emoji} {status}")
        
        # Show description if available
        description = tool_call.get("description", "")
        if description:
            st.markdown("**Description:**")
            st.markdown(description)
    
    # Arguments tab
    with tabs[1]:
        args = tool_call.get("args", {})
        if args:
            st.json(args)
        else:
            st.info("No arguments provided for this tool call")
    
    # Result tab
    with tabs[2]:
        result = tool_call.get("result", None)
        error = tool_call.get("error", None)
        
        if error:
            st.error("**Error:**")
            st.error(error)
        elif result is not None:
            # Try to format the result based on its type
            if isinstance(result, dict) or (isinstance(result, str) and result.strip().startswith("{")):
                try:
                    # If it's a string that looks like JSON, parse it
                    if isinstance(result, str):
                        result_obj = json.loads(result)
                    else:
                        result_obj = result
                    
                    # Format as JSON
                    st.json(result_obj)
                except:
                    # Fall back to plain text if JSON parsing fails
                    st.markdown(f"```\n{result}\n```")
            else:
                # Format as plain text
                st.markdown(f"```\n{result}\n```")
        else:
            st.info("No result was returned from this tool call")
    
    # Raw Data tab
    with tabs[3]:
        # Show the raw tool call data
        st.json(tool_call)