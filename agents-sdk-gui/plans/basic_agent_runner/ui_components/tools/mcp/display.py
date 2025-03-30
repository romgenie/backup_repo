"""
Display components for MCP tools.
"""

import streamlit as st
import json
from typing import Dict, Any, List, Optional, Callable, Set

from .categorization import is_mcp_tool, get_tool_icon

def render_mcp_tools(
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
        st.markdown("""
        <div class="info-message">
            No MCP tools are available. Add MCP servers in the MCP tab.
        </div>
        """, unsafe_allow_html=True)
        return False
    
    changes_made = False
    
    # Display MCP tools section header
    st.markdown("""
    <div class="form-section">
        <h3>MCP Tools</h3>
        <div class="text-sm text-neutral-600 mb-xs">Tools provided by MCP servers</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize selection state if not present
    if "selected_mcp_tools" not in st.session_state:
        st.session_state.selected_mcp_tools = set()
    
    # Initialize display options if not present
    if "mcp_tools_display_options" not in st.session_state:
        st.session_state.mcp_tools_display_options = {
            "show_descriptions": True,
            "show_parameters": False,
            "compact_view": False,
            "last_updated": ""
        }
    
    # Options for display
    with st.expander("Display Options", expanded=False):
        st.markdown('<div class="form-section">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            # Show descriptions toggle
            st.markdown('<div class="form-group">', unsafe_allow_html=True)
            st.session_state.mcp_tools_display_options["show_descriptions"] = st.checkbox(
                "Show Descriptions",
                value=st.session_state.mcp_tools_display_options["show_descriptions"],
                key="mcp_tools_show_descriptions",
                help="Display tool descriptions inline"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Show parameters toggle
            st.markdown('<div class="form-group">', unsafe_allow_html=True)
            st.session_state.mcp_tools_display_options["show_parameters"] = st.checkbox(
                "Show Parameters",
                value=st.session_state.mcp_tools_display_options["show_parameters"],
                key="mcp_tools_show_parameters",
                help="Display tool parameters inline"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            # Compact view toggle
            st.markdown('<div class="form-group">', unsafe_allow_html=True)
            st.session_state.mcp_tools_display_options["compact_view"] = st.checkbox(
                "Compact View",
                value=st.session_state.mcp_tools_display_options["compact_view"],
                key="mcp_tools_compact_view",
                help="Use a more compact layout"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Show selection guide
        st.markdown("""
        <div class="neutral-message">
            <strong>Tip:</strong> Use checkboxes to select tools that you want to use with your agent.
            Click "Apply Selected MCP Tools" after making your selection.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Group tools by category for better organization
    tools_by_category = {}
    
    # Create a copy of the dictionary to avoid modification during iteration
    mcp_tools_copy = dict(mcp_tools)
    for tool_id, tool_info in mcp_tools_copy.items():
        tool_name = tool_info.get("name", "Unknown Tool")
        _, category = is_mcp_tool(tool_name)
        
        if not category:
            category = "general"
            
        if category not in tools_by_category:
            tools_by_category[category] = []
            
        tools_by_category[category].append((tool_id, tool_info))
    
    # Create select all checkbox
    st.checkbox(
        "Select All MCP Tools",
        key="select_all_mcp_tools",
        on_change=_handle_select_all_change,
        args=(mcp_tools,),
        help="Select all available MCP tools"
    )
    
    # Display tools by category
    for category, tools in tools_by_category.items():
        icon = get_tool_icon(category)
        with st.expander(f"{icon} {category.title()} ({len(tools)})", expanded=True):
            # Create a container for this category
            container = st.container()
            
            with container:
                # Display each tool in this category
                for tool_id, tool_info in tools:
                    changes_made |= display_mcp_tool(tool_id, tool_info)
    
    # Apply selection button if there are tools to select
    if mcp_tools and on_tool_select:
        # Initialize state variables for tool selection
        if "applying_tool_selection" not in st.session_state:
            st.session_state.applying_tool_selection = False
        
        apply_button = st.button("Apply Selected MCP Tools", type="primary", key="apply_mcp_tools_button")
        if apply_button:
            # Set flag to apply on next rerun
            st.session_state.applying_tool_selection = True
            # Force a rerun to immediately process the selection
            st.rerun()
            
        # If the flag is set, apply the selection
        if st.session_state.applying_tool_selection:
            # Reset the flag
            st.session_state.applying_tool_selection = False
            
            # Convert set to list for callback
            selected_tools = list(st.session_state.selected_mcp_tools)
            
            on_tool_select(selected_tools)
            st.markdown(f"""
            <div class="success-message">
                Selected {len(selected_tools)} MCP tools for use with agent
            </div>
            """, unsafe_allow_html=True)
            changes_made = True
    
    return changes_made

def _handle_select_all_change(mcp_tools: Dict[str, Dict[str, Any]]):
    """
    Handle the "Select All" checkbox change
    
    Args:
        mcp_tools: Dictionary of available MCP tools
    """
    if st.session_state.select_all_mcp_tools:
        # Select all tools
        st.session_state.selected_mcp_tools = set(mcp_tools.keys())
    else:
        # Deselect all tools
        st.session_state.selected_mcp_tools = set()

def display_mcp_tool(tool_id: str, tool_info: Dict[str, Any]) -> bool:
    """
    Display a single MCP tool with selection capability
    
    Args:
        tool_id: The unique ID of the tool
        tool_info: Dictionary with tool information
        
    Returns:
        Boolean indicating if the selection state changed
    """
    changes_made = False
    compact_view = st.session_state.mcp_tools_display_options["compact_view"]
    show_descriptions = st.session_state.mcp_tools_display_options["show_descriptions"]
    show_parameters = st.session_state.mcp_tools_display_options["show_parameters"]
    
    # Use different layouts based on display options
    if compact_view:
        col1, col2 = st.columns([1, 5])
        
        with col1:
            # Checkbox for tool selection
            is_selected = tool_id in st.session_state.selected_mcp_tools
            selected = st.checkbox(
                "",
                value=is_selected,
                key=f"select_mcp_tool_{tool_id}",
                help=f"Select {tool_info.get('name', 'this tool')}"
            )
            
            # Update selection state
            if selected and tool_id not in st.session_state.selected_mcp_tools:
                st.session_state.selected_mcp_tools.add(tool_id)
                changes_made = True
            elif not selected and tool_id in st.session_state.selected_mcp_tools:
                st.session_state.selected_mcp_tools.remove(tool_id)
                changes_made = True
        
        with col2:
            # Tool name and details
            st.markdown(f"**{tool_info.get('name', 'Unnamed Tool')}**")
            
            if show_descriptions and 'description' in tool_info:
                st.markdown(f"*{tool_info['description']}*")
                
            if show_parameters and 'parameters' in tool_info:
                params = tool_info['parameters']
                if params:
                    st.markdown("**Parameters:**")
                    for param in params:
                        param_name = param.get('name', 'unnamed')
                        param_desc = param.get('description', 'No description')
                        st.markdown(f"- **{param_name}**: {param_desc}")
    else:
        # Non-compact view
        tool_name = tool_info.get('name', 'Unnamed Tool')
        
        # Checkbox for tool selection
        is_selected = tool_id in st.session_state.selected_mcp_tools
        selected = st.checkbox(
            tool_name,
            value=is_selected,
            key=f"select_mcp_tool_{tool_id}",
            help=f"Select {tool_name}"
        )
        
        # Update selection state
        if selected and tool_id not in st.session_state.selected_mcp_tools:
            st.session_state.selected_mcp_tools.add(tool_id)
            changes_made = True
        elif not selected and tool_id in st.session_state.selected_mcp_tools:
            st.session_state.selected_mcp_tools.remove(tool_id)
            changes_made = True
        
        # Additional details in indented section
        if show_descriptions or show_parameters:
            with st.container():
                if show_descriptions and 'description' in tool_info:
                    st.markdown(f"*{tool_info['description']}*")
                    
                if show_parameters and 'parameters' in tool_info:
                    params = tool_info['parameters']
                    if params:
                        st.markdown("**Parameters:**")
                        for param in params:
                            param_name = param.get('name', 'unnamed')
                            param_desc = param.get('description', 'No description')
                            st.markdown(f"- **{param_name}**: {param_desc}")
    
    return changes_made

def display_mcp_tool_details(tool_info: Dict[str, Any]):
    """
    Display detailed information about an MCP tool
    
    Args:
        tool_info: Dictionary with tool information
    """
    st.markdown(f"## {tool_info.get('name', 'Unnamed Tool')}")
    
    if 'description' in tool_info:
        st.markdown(f"*{tool_info['description']}*")
    
    if 'server_id' in tool_info:
        st.markdown(f"**Server ID:** {tool_info['server_id']}")
    
    st.markdown("### Parameters")
    
    if 'parameters' in tool_info and tool_info['parameters']:
        for param in tool_info['parameters']:
            param_name = param.get('name', 'unnamed')
            param_desc = param.get('description', 'No description')
            param_type = param.get('type', 'unknown')
            param_required = param.get('required', False)
            
            st.markdown(f"**{param_name}**")
            st.markdown(f"- Type: {param_type}")
            st.markdown(f"- Required: {'Yes' if param_required else 'No'}")
            st.markdown(f"- Description: {param_desc}")
    else:
        st.markdown("*No parameters*")
    
    if 'return_type' in tool_info and tool_info['return_type']:
        st.markdown("### Return Type")
        st.markdown(f"```{json.dumps(tool_info['return_type'], indent=2)}```")

# Alias for backward compatibility
render_mcp_tools_section = render_mcp_tools