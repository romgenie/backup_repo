"""
Tool integration components for the chat interface.
"""

import streamlit as st
from typing import Dict, List, Any

def render_tool_calls_area():
    """
    Render the collapsible tool calls area.
    
    Returns:
        The tool calls container
    """
    with st.expander("Tool Calls", expanded=False):
        tool_calls_container = st.container()
        
    return tool_calls_container

def display_tool_calls(container, tool_calls: List[Dict[str, Any]]):
    """
    Display tool calls in the provided container.
    
    Args:
        container: Streamlit container to render in
        tool_calls: List of tool call data
    """
    from ui_components import display_tool_calls
    
    # Use the existing utility function to display the tool calls
    display_tool_calls(container, tool_calls)

def create_tool_call_summary(tool_calls: List[Dict[str, Any]]) -> str:
    """
    Create a summary of tool calls for display.
    
    Args:
        tool_calls: List of tool call data
        
    Returns:
        Summary text
    """
    if not tool_calls:
        return "No tools were used"
    
    # Create summary
    total_calls = len(tool_calls)
    unique_tools = set(tc.get("name", "Unknown") for tc in tool_calls)
    
    summary = f"{total_calls} tool call{'s' if total_calls != 1 else ''} using {len(unique_tools)} unique tool{'s' if len(unique_tools) != 1 else ''}"
    
    # Add details of which tools were used
    tool_list = ", ".join(sorted(unique_tools))
    summary += f": {tool_list}"
    
    return summary

def extract_tool_calls_from_result(result: Any) -> List[Dict[str, Any]]:
    """
    Extract tool calls from a run result object.
    
    Args:
        result: Run result object
        
    Returns:
        List of tool call data
    """
    tool_calls = []
    
    # Extract from steps
    if hasattr(result, 'steps'):
        for step in result.steps:
            # Extract standard tool calls
            if hasattr(step, 'tool_calls') and step.tool_calls:
                for tc in step.tool_calls:
                    tool_call = {
                        "name": tc.name,
                        "args": tc.args if hasattr(tc, 'args') else {},
                        "response": tc.response if hasattr(tc, 'response') else None,
                        "type": "standard"
                    }
                    tool_calls.append(tool_call)
            
            # Extract MCP tool calls
            if hasattr(step, 'mcp_tool_calls') and step.mcp_tool_calls:
                for tc in step.mcp_tool_calls:
                    tool_call = {
                        "name": tc.name,
                        "args": tc.arguments if hasattr(tc, 'arguments') else {},
                        "response": tc.result if hasattr(tc, 'result') else None,
                        "type": "mcp"
                    }
                    tool_calls.append(tool_call)
    
    return tool_calls