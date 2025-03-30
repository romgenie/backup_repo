import streamlit as st
import json
from typing import Dict, Any, Callable, Optional, List
import uuid
import os

# Predefined MCP servers that are known to work
PREDEFINED_MCP_SERVERS = {
    "sequentialthinking": {
        "name": "Sequential Thinking",
        "description": "Advanced thinking capabilities for step-by-step reasoning",
        "type": "stdio",
        "command": "docker",
        "args": "run --rm -i mcp/sequentialthinking",
    },
    "puppeteer": {
        "name": "Puppeteer",
        "description": "Web automation and browsing capabilities",
        "type": "stdio",
        "command": "docker",
        "args": "run -i --rm --init -e DOCKER_CONTAINER=true mcp/puppeteer",
    },
    "github": {
        "name": "GitHub",
        "description": "GitHub API integration with repository access",
        "type": "stdio",
        "command": "docker",
        "args": "run -i --rm -e GITHUB_PERSONAL_ACCESS_TOKEN mcp/github",
        "env": {
            "GITHUB_PERSONAL_ACCESS_TOKEN": "{{GITHUB_TOKEN}}"
        }
    },
    "memory": {
        "name": "Memory",
        "description": "Persistent memory capabilities for agents",
        "type": "stdio",
        "command": "docker",
        "args": "run -i --rm mcp/memory",
    },
    "time": {
        "name": "Time",
        "description": "Time-related functions and utilities",
        "type": "stdio",
        "command": "docker",
        "args": "run -i --rm mcp/time --local-timezone America/New_York",
    },
    "fetch": {
        "name": "Fetch",
        "description": "Web content fetching capabilities",
        "type": "stdio",
        "command": "docker",
        "args": "run -i --rm mcp/fetch",
    },
    "filesystem": {
        "name": "Filesystem",
        "description": "File system operations with volume mounts",
        "type": "stdio",
        "command": "docker",
        "args": "run -i --rm --mount type=bind,src=/Users/timgregg/mcp,dst=/projects/mcp --mount type=bind,src=/Users/timgregg/Desktop/mcp,dst=/projects/Desktop/mcp mcp/filesystem /projects",
    }
}

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
    st.header("MCP Servers")
    st.write("Configure Model Context Protocol (MCP) servers to provide tools for your agent.")
    
    # Display helper text
    st.info("MCP is an open protocol that allows agents to access external tools and data sources.")
    
    
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
        st.subheader("Configured Servers")
        
        # Create columns for the server list header
        col1, col2, col3, col4, col5 = st.columns([1, 3, 1, 2, 2])
        with col1:
            st.write("**Use**")
        with col2:
            st.write("**Name**")
        with col3:
            st.write("**Status**")
        with col4:
            st.write("**Type**")
        with col5:
            st.write("**Actions**")
            
        # Create a scrollable container for the servers
        server_container = st.container(height=300)
        
        with server_container:
            # Display each server
            for server_id, server_config in st.session_state.mcp_servers.items():
                col1, col2, col3, col4, col5 = st.columns([1, 3, 1, 2, 2])
                
                with col1:
                    # Checkbox to select this server
                    is_selected = server_id in st.session_state.selected_mcp_servers
                    selected = st.checkbox(
                        "", 
                        value=is_selected, 
                        key=f"select_server_{server_id}",
                        help="Select this server for use with agents"
                    )
                    
                    # Update selection state
                    if selected and server_id not in st.session_state.selected_mcp_servers:
                        st.session_state.selected_mcp_servers.append(server_id)
                        changes_made = True
                        print(f"Added server {server_id} to selection. Current selection: {st.session_state.selected_mcp_servers}")
                    elif not selected and server_id in st.session_state.selected_mcp_servers:
                        st.session_state.selected_mcp_servers.remove(server_id)
                        changes_made = True
                        print(f"Removed server {server_id} from selection. Current selection: {st.session_state.selected_mcp_servers}")
                
                with col2:
                    # Display server name
                    st.write(server_config.get("name", "Unnamed Server"))
                    
                with col3:
                    # Display server health status
                    # Direct reference to the app instance in session state
                    if "agent_mcp_servers" in st.session_state:
                        # Try to get the MCP manager from the current context or session state
                        mcp_manager = None
                        
                        # Check if we can access the app instance
                        if "app" in globals():
                            mcp_manager = app.mcp_manager
                        elif "_app" in st.session_state:
                            mcp_manager = st.session_state._app.mcp_manager
                            
                        if mcp_manager is not None:
                            # Now we can access the server health
                            health_status = mcp_manager.server_health.get(server_id, {}).get("status", "unknown")
                            
                            if health_status == "healthy" or health_status == "connected":
                                st.markdown(":green[✓]")
                            elif health_status == "error" or health_status == "connection_error" or health_status == "creation_failed":
                                st.markdown(":red[✗]")
                            else:
                                st.markdown(":grey[?]")
                        else:
                            st.markdown(":grey[?]")
                    else:
                        st.markdown(":grey[?]")
                
                with col4:
                    # Display server type
                    server_type = server_config.get("type", "unknown")
                    st.write(server_type)
                
                with col5:
                    # Test and Remove buttons
                    test_col, remove_col = st.columns(2)
                    
                    with test_col:
                        if st.button("Test", key=f"test_{server_id}"):
                            if on_server_test:
                                try:
                                    with st.spinner("Testing server connection..."):
                                        result = on_server_test(server_config)
                                        
                                    if result.get("success", False):
                                        st.success("Connection successful!")
                                        # Show available tools
                                        if "tools" in result:
                                            st.write(f"Found {len(result['tools'])} tools:")
                                            for tool in result["tools"]:
                                                st.write(f"- {tool}")
                                    else:
                                        st.error(f"Connection failed: {result.get('error', 'Unknown error')}")
                                        # Display traceback in an expander if available
                                        if "traceback" in result:
                                            with st.expander("Error Details"):
                                                st.code(result["traceback"], language="python")
                                except Exception as e:
                                    st.error(f"Test failed with error: {str(e)}")
                                    import traceback
                                    with st.expander("Error Details"):
                                        st.code(traceback.format_exc(), language="python")
                    
                    with remove_col:
                        if st.button("Remove", key=f"remove_{server_id}"):
                            if on_server_remove:
                                on_server_remove(server_id)
                            
                            # Remove from session state
                            if server_id in st.session_state.mcp_servers:
                                del st.session_state.mcp_servers[server_id]
                            
                            # Remove from selected servers
                            if server_id in st.session_state.selected_mcp_servers:
                                st.session_state.selected_mcp_servers.remove(server_id)
                            
                            changes_made = True
                            st.rerun()
                
                # Display server details in an expander
                with st.expander("Server Details"):
                    if server_type == "stdio":
                        st.write(f"**Command:** {server_config.get('command', 'npx')}")
                        st.write(f"**Arguments:** {server_config.get('args', '')}")
                    elif server_type == "sse":
                        st.write(f"**URL:** {server_config.get('url', '')}")
                        headers = server_config.get('headers', '{}')
                        if headers:
                            try:
                                header_dict = json.loads(headers) if isinstance(headers, str) else headers
                                st.write("**Headers:**")
                                for key, value in header_dict.items():
                                    st.write(f"- {key}: {value}")
                            except json.JSONDecodeError:
                                st.write("**Headers:** Invalid JSON")
                    
                    st.write(f"**Cache Tools:** {'Yes' if server_config.get('cache_tools', True) else 'No'}")
        
        # Apply selection button
        if st.session_state.mcp_servers and on_server_select:
            if st.button("Apply Selected Servers", type="primary"):
                # Debug what we're about to select
                print(f"Selecting servers: {st.session_state.selected_mcp_servers}")
                print(f"Available server configs: {list(st.session_state.mcp_servers.keys())}")
                
                on_server_select(st.session_state.selected_mcp_servers)
                st.success(f"Selected {len(st.session_state.selected_mcp_servers)} servers for use with agents")
                changes_made = True
                
            # Show a checkbox to select all servers at once
            if st.checkbox("Select All Servers", key="select_all_servers"):
                # Make sure mcp_servers is a dictionary
                if isinstance(st.session_state.mcp_servers, dict):
                    st.session_state.selected_mcp_servers = list(st.session_state.mcp_servers.keys())
                else:
                    st.warning("MCP servers data structure is invalid. Please refresh the page.")
                changes_made = True
    
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

def render_predefined_mcp_servers(on_server_add: Optional[Callable[[str, Dict[str, Any]], None]] = None) -> bool:
    """
    Render the section for adding predefined MCP servers
    
    Args:
        on_server_add: Callback when a server is added
        
    Returns:
        Boolean indicating if a server was added
    """
    servers_added = False
    
    st.subheader("Quick Add MCP Servers")
    st.markdown("Select from these predefined Docker-based MCP servers:")
    
    # Create a 3-column layout for the predefined servers
    cols = st.columns(3)
    
    # Display each predefined server with an "Add" button
    for i, (server_id, server_config) in enumerate(PREDEFINED_MCP_SERVERS.items()):
        col = cols[i % 3]
        
        with col:
            st.markdown(f"**{server_config['name']}**")
            st.markdown(f"*{server_config['description']}*")
            
            # Check if this server is already added
            is_added = any(
                s.get('name') == server_config['name'] 
                for s in st.session_state.mcp_servers.values()
            )
            
            if is_added:
                st.success("Added ✓")
            else:
                if st.button(f"Add {server_config['name']}", key=f"add_{server_id}"):
                    # Create a copy of the config with a unique ID
                    new_server_id = f"mcp_server_{uuid.uuid4().hex[:8]}"
                    
                    # Create a copy of the config to avoid modifying the original
                    server_config_copy = server_config.copy()
                    
                    # Process environment variables if any
                    env_vars = server_config_copy.get('env', {})
                    for key, value in env_vars.items():
                        # Handle placeholder tokens
                        if value == "{{GITHUB_TOKEN}}":
                            # Try to get from environment
                            env_value = os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN", "")
                            # If not in environment, prompt user
                            if not env_value:
                                env_value = st.text_input(
                                    f"Enter {key}", 
                                    type="password",
                                    key=f"env_{server_id}_{key}"
                                )
                            server_config_copy['args'] = server_config_copy['args'].replace(
                                f"-e {key}", 
                                f"-e {key}={env_value}"
                            )
                    
                    # Store in session state
                    st.session_state.mcp_servers[new_server_id] = server_config_copy
                    
                    # Call the callback if provided
                    if on_server_add:
                        on_server_add(new_server_id, server_config_copy)
                    
                    st.success(f"Added {server_config['name']}")
                    servers_added = True
                    st.rerun()
    
    return servers_added

def render_mcp_server_form(on_server_add: Optional[Callable[[str, Dict[str, Any]], None]] = None) -> bool:
    """
    Render the form for adding a custom MCP server
    
    Args:
        on_server_add: Callback when a server is added
        
    Returns:
        Boolean indicating if a server was added
    """
    server_added = False
    
    with st.form("mcp_server_form"):
        # Server name
        server_name = st.text_input(
            "Server Name", 
            "MCP Server",
            help="A descriptive name for this server"
        )
        
        # Server type selection
        server_type = st.selectbox(
            "Server Type", 
            ["stdio", "sse"],
            help="stdio servers run locally, SSE servers run remotely"
        )
        
        # Fields specific to server type
        if server_type == "stdio":
            command = st.text_input(
                "Command", 
                "npx",
                help="The command to execute (e.g., npx, node, python)"
            )
            
            args = st.text_input(
                "Arguments", 
                "-y @modelcontextprotocol/server-filesystem ./files",
                help="Command arguments (e.g., path to script, options)"
            )
            
            # Create server config
            server_config = {
                "type": "stdio",
                "name": server_name,
                "command": command,
                "args": args,
            }
        else:  # sse
            url = st.text_input(
                "Server URL", 
                "https://example.com/mcp",
                help="The URL of the SSE server"
            )
            
            headers = st.text_area(
                "Headers (JSON)", 
                "{}",
                help="Optional JSON object with headers (e.g., for authentication)"
            )
            
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
        submitted = st.form_submit_button("Add MCP Server")
        
        if submitted:
            # Generate a unique ID for this server
            server_id = f"mcp_server_{uuid.uuid4().hex[:8]}"
            
            # Store in session state
            st.session_state.mcp_servers[server_id] = server_config
            
            # Call the callback if provided
            if on_server_add:
                on_server_add(server_id, server_config)
            
            st.success(f"Added {server_type} server: {server_name}")
            server_added = True
    
    return server_added