"""
Server management form for adding new MCP servers.
"""

import streamlit as st
import uuid
from typing import Dict, Any, Callable, Optional

def render_mcp_server_form(on_server_add: Optional[Callable[[str, Dict[str, Any]], None]] = None) -> bool:
    """
    Render the form for adding a custom MCP server
    
    Args:
        on_server_add: Callback when a server is added
        
    Returns:
        Boolean indicating if a server was added
    """
    server_added = False
    
    # Using forms in Streamlit actually handles state between reruns automatically
    # So we don't need to add the same kind of state tracking as with buttons
    with st.form("mcp_server_form"):
        st.markdown("""
        <div class="form-section">
            <h4>Server Details</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Server name
        st.markdown('<div class="form-group">', unsafe_allow_html=True)
        server_name = st.text_input(
            "Server Name", 
            "MCP Server",
            help="A descriptive name for this server"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Server type selection
        st.markdown('<div class="form-group">', unsafe_allow_html=True)
        server_type = st.selectbox(
            "Server Type", 
            ["stdio", "sse"],
            help="stdio servers run locally, SSE servers run remotely"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Fields specific to server type
        st.markdown("""
        <div class="form-section mt-sm">
            <h4>Connection Details</h4>
        </div>
        """, unsafe_allow_html=True)
        
        if server_type == "stdio":
            st.markdown('<div class="form-group">', unsafe_allow_html=True)
            command = st.text_input(
                "Command", 
                "npx",
                help="The command to execute (e.g., npx, node, python)"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="form-group">', unsafe_allow_html=True)
            args = st.text_input(
                "Arguments", 
                "-y @modelcontextprotocol/server-filesystem ./files",
                help="Command arguments (e.g., path to script, options)"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Create server config
            server_config = {
                "type": "stdio",
                "name": server_name,
                "command": command,
                "args": args,
            }
        else:  # sse
            st.markdown('<div class="form-group">', unsafe_allow_html=True)
            url = st.text_input(
                "Server URL", 
                "https://example.com/mcp",
                help="The URL of the SSE server"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="form-group">', unsafe_allow_html=True)
            headers = st.text_area(
                "Headers (JSON)", 
                "{}",
                help="Optional JSON object with headers (e.g., for authentication)"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Create server config
            server_config = {
                "type": "sse",
                "name": server_name,
                "url": url,
                "headers": headers,
            }
        
        # Common options
        cache_tools = st.checkbox(
            "Cache Tools List", 
            True,
            help="Cache the list of tools to reduce latency (disable if tools change frequently)"
        )
        server_config["cache_tools"] = cache_tools
        
        # Submit button
        st.markdown('<div class="form-buttons mt-md">', unsafe_allow_html=True)
        submitted = st.form_submit_button("Add MCP Server")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if submitted:
            # Generate a unique ID for this server
            server_id = f"mcp_server_{uuid.uuid4().hex[:8]}"
            
            # Store in session state
            st.session_state.mcp_servers[server_id] = server_config
            
            # Call the callback if provided
            if on_server_add:
                on_server_add(server_id, server_config)
            
            st.markdown(f"""
            <div class="success-message">
                Added {server_type} server: {server_name}
            </div>
            """, unsafe_allow_html=True)
            server_added = True
    
    return server_added