"""
MCP Servers tab - Configures Model Context Protocol servers.
"""

import streamlit as st
from typing import Dict, Any, Callable, Optional, List

from .server_management import render_mcp_server_form
from .server_display import display_servers
from .predefined_servers import render_predefined_mcp_servers, PREDEFINED_MCP_SERVERS

def render_mcp_tab(
    on_server_add: Callable[[str, Dict[str, Any]], None] = None,
    on_server_remove: Callable[[str], None] = None,
    on_server_test: Callable[[Dict[str, Any]], Dict[str, Any]] = None,
    on_server_select: Callable[[List[str]], None] = None
) -> bool:
    """
    Render the MCP Servers tab of the application
    
    Args:
        on_server_add: Callback when a server is added
        on_server_remove: Callback when a server is removed
        on_server_test: Callback to test server connection
        on_server_select: Callback when servers are selected for use
        
    Returns:
        Boolean indicating if any changes were made
    """
    # Debug info
    print("\nSession state debugging:")
    print(f"mcp_servers: {type(st.session_state.get('mcp_servers', {}))}")
    print(f"Keys in mcp_servers: {list(st.session_state.get('mcp_servers', {}).keys())}")
    print(f"selected_mcp_servers: {st.session_state.get('selected_mcp_servers', [])}")
    print(f"agent_mcp_servers: {type(st.session_state.get('agent_mcp_servers', []))}")
    print(f"Length of agent_mcp_servers: {len(st.session_state.get('agent_mcp_servers', []))}")
    
    st.markdown("""
    <div class="form-section">
        <h2>MCP Servers</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("Configure Model Context Protocol (MCP) servers to provide tools for your agent.")
    
    # Display helper text
    st.markdown("""
    <div class="info-message">
        MCP is an open protocol that allows agents to access external tools and data sources.
    </div>
    """, unsafe_allow_html=True)
    
    
    # Initialize session state for MCP servers if not present
    if "mcp_servers" not in st.session_state:
        st.session_state.mcp_servers = {}
    
    if "selected_mcp_servers" not in st.session_state:
        st.session_state.selected_mcp_servers = []
    
    # Track changes
    changes_made = False
    
    # Add predefined MCP servers section
    with st.expander("Add Predefined MCP Servers", expanded=len(st.session_state.mcp_servers) == 0):
        changes_made |= render_predefined_mcp_servers(on_server_add)
    
    # Add custom server section
    with st.expander("Add Custom MCP Server", expanded=False):
        changes_made |= render_mcp_server_form(on_server_add)
    
    # Display existing servers
    if st.session_state.mcp_servers:
        changes_made |= display_servers(on_server_remove, on_server_test, on_server_select)
    
    # Display help information
    with st.expander("About MCP Servers", expanded=False):
        st.markdown("""
        ## What are MCP Servers?
        
        The Model Context Protocol (MCP) is an open protocol that standardizes how applications provide context and tools to language models.
        
        MCP servers can provide various capabilities to your agents:
        
        - File system access (Filesystem)
        - Web automation (Puppeteer)
        - GitHub interactions (GitHub)
        - Persistent memory (Memory)
        - Time-based functions (Time)
        - Web fetching (Fetch)
        - Sequential thinking (Sequential Thinking)
        - And more!
        
        ## Server Types
        
        1. **stdio servers** run locally as a subprocess
            - Example: Docker-based MCP servers like the ones provided
            - Requires Docker to be installed and running
            
        2. **SSE servers** run remotely and connect via HTTP
            - Connect to services over the network
            - Requires a URL and potentially headers for authentication
            
        ## Using Multiple MCP Servers
        
        1. Add servers in this tab from the predefined list or create custom ones
        2. Test your connections to verify the servers are working
        3. Select multiple servers to use together (use "Select All" for maximum capabilities)
        4. Apply your selection using the "Apply Selected Servers" button
        5. Create a new agent in the Configure tab
        
        ## Predefined Docker Servers
        
        The predefined Docker-based MCP servers provide a range of capabilities:
        
        - **Sequential Thinking**: Advanced reasoning capabilities
        - **Puppeteer**: Web browsing and automation
        - **GitHub**: Access to GitHub repositories and features
        - **Memory**: Store and retrieve information across conversations
        - **Time**: Access to time-related functions (date, time, timezones)
        - **Fetch**: Retrieve content from web URLs
        - **Filesystem**: Access and manipulate files on the host system
        
        These servers run as Docker containers and require Docker to be installed.
        """)
    
    return changes_made