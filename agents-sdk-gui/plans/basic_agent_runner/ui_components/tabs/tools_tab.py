import streamlit as st
from typing import List, Dict, Any
from agents import Tool

from ui_components.tools.mcp import render_mcp_tools_section

def get_available_mcp_tools() -> Dict[str, Dict[str, Any]]:
    """
    Collect available MCP tools from all configured servers
    
    Returns:
        Dictionary of tool info by tool ID
    """
    tools_collection = {}
    
    # Check if we can access the app instance from session state
    app = None
    if "_app" in st.session_state:
        app = st.session_state._app
    
    # Skip if there's no app instance or MCP manager
    if not app or not hasattr(app, 'mcp_manager'):
        return tools_collection
    
    # Get the MCP manager
    mcp_manager = app.mcp_manager
    
    # Collect tools from all servers in the cache
    for server_id, tools in mcp_manager.server_tools_cache.items():
        # Skip if there are no tools for this server
        if not tools:
            continue
            
        # Get the server name from configurations
        server_name = "Unknown Server"
        if server_id in mcp_manager.server_configs:
            server_name = mcp_manager.server_configs[server_id].get("name", "Unnamed Server")
            
        # Add each tool to the collection with server details
        for tool in tools:
            tool_id = f"{server_id}_{tool['name']}"
            tools_collection[tool_id] = {
                **tool,  # Copy all tool properties
                "server_id": server_id,
                "server_name": server_name,
                "is_selected": server_id in st.session_state.get("selected_mcp_servers", [])
            }
    
    return tools_collection

def render_tools_tab() -> None:
    """
    Render the tools tab of the application
    """
    st.markdown("""
    <div class="form-section">
        <h2>Tools Configuration</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Collect all available MCP tools
    mcp_tools = get_available_mcp_tools()
    
    # Section for MCP tools from all configured servers
    if mcp_tools:
        st.markdown("""
        <div class="form-section mt-md">
            <h3>Available MCP Tools</h3>
        </div>
        """, unsafe_allow_html=True)
        st.write(f"These tools are available from your configured MCP servers.")
        
        # Store in session state for use by other components
        st.session_state.mcp_tools = mcp_tools
        
        # Add filter options in a styled container
        st.markdown('<div class="form-group p-sm" style="background-color: var(--neutral-200); border-radius: var(--radius-md);">', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown('<div class="form-group">', unsafe_allow_html=True)
            view_mode = st.radio(
                "View By:",
                ["Server", "Category", "All Tools"],
                key="mcp_tools_view_mode",
                index=0
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            # Search filter
            st.markdown('<div class="form-group">', unsafe_allow_html=True)
            search_query = st.text_input(
                "Search Tools:",
                key="mcp_tools_search",
                placeholder="Filter by name or description"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
            
        # If there's a search query, filter the tools
        if search_query:
            filtered_tools = {}
            for tool_id, tool_info in mcp_tools.items():
                # Search in tool name and description
                name = tool_info.get("name", "").lower()
                description = tool_info.get("description", "").lower()
                
                if search_query.lower() in name or search_query.lower() in description:
                    filtered_tools[tool_id] = tool_info
            
            # Update the tools to display
            display_tools = filtered_tools
            
            # Show filtered count
            st.markdown(f"""
            <div class="info-message">
                Found {len(filtered_tools)} tools matching '{search_query}'
            </div>
            """, unsafe_allow_html=True)
        else:
            display_tools = mcp_tools
        
        # Organize tools by the selected view mode
        if view_mode == "Server":
            # Group tools by server
            server_grouped_tools = {}
            for tool_id, tool_info in display_tools.items():
                server_name = tool_info.get("server_name", "Unknown Server")
                if server_name not in server_grouped_tools:
                    server_grouped_tools[server_name] = {}
                server_grouped_tools[server_name][tool_id] = tool_info
                
            # Display each server group
            for server_name, server_tools in server_grouped_tools.items():
                with st.expander(f"{server_name} ({len(server_tools)} tools)", expanded=True):
                    # Display tools for this server
                    render_mcp_tools_section(
                        server_tools,
                        on_tool_select=None  # We'll use server selection from MCP tab instead
                    )
                    
        elif view_mode == "Category":
            # Import tool categorization function
            from ui_components.tools.mcp import is_mcp_tool, get_tool_icon
            
            # Group tools by category
            category_grouped_tools = {}
            for tool_id, tool_info in display_tools.items():
                tool_name = tool_info.get("name", "")
                # Get the category
                _, category = is_mcp_tool(tool_name)
                if not category:
                    category = "other"
                    
                if category not in category_grouped_tools:
                    category_grouped_tools[category] = {}
                category_grouped_tools[category][tool_id] = tool_info
            
            # Define category display order and friendly names
            category_order = [
                ("time", "Time Services"),
                ("web", "Web & Browsing"),
                ("filesystem", "File System"),
                ("memory", "Memory & Storage"),
                ("github", "GitHub Integration"),
                ("thinking", "Sequential Thinking"),
                ("general", "General Utilities"),
                ("other", "Other Tools")
            ]
            
            # Display each category group
            for category_key, category_name in category_order:
                if category_key in category_grouped_tools:
                    category_tools = category_grouped_tools[category_key]
                    icon = get_tool_icon(category_key)
                    
                    with st.expander(f"{icon} {category_name} ({len(category_tools)} tools)", expanded=True):
                        # Display tools for this category
                        render_mcp_tools_section(
                            category_tools,
                            on_tool_select=None
                        )
        else:
            # Display all tools without grouping
            render_mcp_tools_section(
                display_tools,
                on_tool_select=None  # We'll use server selection from MCP tab instead
            )
        
        # Show note about server selection
        st.markdown("""
        <div class="info-message mt-md">
            <strong>Note:</strong> To use these tools with your agent, go to the <strong>MCP</strong> tab and select the servers
            that provide them. Then click "Apply Selected Servers" before creating your agent.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-message">
            No MCP tools available. Configure MCP servers in the MCP tab to access tools.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="form-section mt-md">
            <h3>Using MCP Tools</h3>
        </div>
        
        <ol class="mt-sm">
            <li>Go to the <strong>MCP</strong> tab to add and configure MCP servers</li>
            <li>Test your connections to verify server availability</li>
            <li>Select the servers you want to use with your agent</li>
            <li>Apply your selection</li>
            <li>Create a new agent in the Configure tab to use these tools</li>
        </ol>
        """, unsafe_allow_html=True)