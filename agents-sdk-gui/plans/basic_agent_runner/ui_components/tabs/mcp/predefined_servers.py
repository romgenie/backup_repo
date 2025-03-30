"""
Predefined MCP servers configuration and display.
"""

import streamlit as st
import uuid
import os
from typing import Dict, Any, Callable, Optional

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
                st.markdown("""
                <div class="success-message text-center">
                    Added ✓
                </div>
                """, unsafe_allow_html=True)
            else:
                # Initialize state variable for adding this server
                if f"adding_{server_id}" not in st.session_state:
                    st.session_state[f"adding_{server_id}"] = False
                
                add_server_button = st.button(f"Add {server_config['name']}", key=f"add_{server_id}")
                if add_server_button:
                    # Set flag to add on next rerun
                    st.session_state[f"adding_{server_id}"] = True
                    # Force a rerun to immediately process
                    st.rerun()
                
                # If the flag is set, add this server
                if st.session_state[f"adding_{server_id}"]:
                    # Reset the flag
                    st.session_state[f"adding_{server_id}"] = False
                    
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
                    
                    st.markdown(f"""
                    <div class="success-message">
                        Added {server_config['name']}
                    </div>
                    """, unsafe_allow_html=True)
                    servers_added = True
    
    return servers_added