"""
MCP settings components for settings tab.
"""

import streamlit as st
import json
import os
from typing import Dict, Any, Optional, Callable

def render_mcp_settings() -> None:
    """Render MCP settings section"""
    st.markdown("""
    <div class="form-section">
        <h3>MCP Settings</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize MCP settings if not present
    if "mcp_settings" not in st.session_state:
        st.session_state.mcp_settings = {
            "auto_connect": True,
            "cache_tool_lists": True,
            "tool_timeout": 30,
            "prefer_local_servers": True
        }
    
    # MCP settings
    mcp_settings = st.session_state.mcp_settings
    
    # Auto-connect toggle
    mcp_settings["auto_connect"] = st.checkbox(
        "Auto-connect to MCP servers",
        value=mcp_settings.get("auto_connect", True),
        help="Automatically connect to MCP servers on startup"
    )
    
    # Cache tool lists toggle
    mcp_settings["cache_tool_lists"] = st.checkbox(
        "Cache tool lists",
        value=mcp_settings.get("cache_tool_lists", True),
        help="Cache the list of available tools for better performance"
    )
    
    # Tool timeout slider
    mcp_settings["tool_timeout"] = st.slider(
        "Tool timeout (seconds)",
        min_value=5,
        max_value=120,
        value=mcp_settings.get("tool_timeout", 30),
        step=5,
        help="Maximum time to wait for tool execution before timeout"
    )
    
    # Prefer local servers toggle
    mcp_settings["prefer_local_servers"] = st.checkbox(
        "Prefer local servers",
        value=mcp_settings.get("prefer_local_servers", True),
        help="Prioritize local MCP servers over remote ones"
    )
    
    # Advanced MCP settings
    with st.expander("Advanced MCP Settings", expanded=False):
        # Docker settings for local MCP servers
        st.markdown("""
        <div class="form-section">
            <h4>Docker Settings</h4>
        </div>
        """, unsafe_allow_html=True)
        
        if "docker_settings" not in st.session_state:
            st.session_state.docker_settings = {
                "enabled": True,
                "volume_mounts": "",
                "network": "host",
                "prune_on_exit": False
            }
        
        docker_settings = st.session_state.docker_settings
        
        # Docker enabled toggle
        docker_settings["enabled"] = st.checkbox(
            "Enable Docker for MCP servers",
            value=docker_settings.get("enabled", True),
            help="Use Docker for running MCP servers"
        )
        
        # Only show these options if Docker is enabled
        if docker_settings["enabled"]:
            # Volume mounts
            docker_settings["volume_mounts"] = st.text_area(
                "Volume mounts",
                value=docker_settings.get("volume_mounts", ""),
                help="List of volume mounts for Docker containers, one per line (format: src:dst)"
            )
            
            # Network selection
            network_options = ["host", "bridge", "none"]
            network_index = network_options.index(docker_settings.get("network", "host")) if docker_settings.get("network") in network_options else 0
            docker_settings["network"] = st.selectbox(
                "Docker network",
                options=network_options,
                index=network_index,
                help="Network mode for Docker containers"
            )
            
            # Prune on exit toggle
            docker_settings["prune_on_exit"] = st.checkbox(
                "Prune containers on exit",
                value=docker_settings.get("prune_on_exit", False),
                help="Remove stopped containers when the application exits"
            )
            
            # Docker command preview
            if docker_settings["enabled"]:
                example_cmd = "docker run -i --rm"
                
                if docker_settings["network"] != "bridge":
                    example_cmd += f" --network={docker_settings['network']}"
                
                if docker_settings["volume_mounts"]:
                    mounts = docker_settings["volume_mounts"].strip().split("\n")
                    for mount in mounts:
                        if mount.strip():
                            example_cmd += f" --mount type=bind,src={mount.split(':')[0]},dst={mount.split(':')[1]}" if ":" in mount else f" -v {mount}"
                
                example_cmd += " mcp/example-server"
                
                st.code(example_cmd, language="bash")
        
        # SSE server settings
        st.markdown("""
        <div class="form-section mt-md">
            <h4>Remote Server Settings</h4>
        </div>
        """, unsafe_allow_html=True)
        
        if "remote_settings" not in st.session_state:
            st.session_state.remote_settings = {
                "timeout": 60,
                "retry_count": 3,
                "use_auth": False,
                "auth_token": ""
            }
        
        remote_settings = st.session_state.remote_settings
        
        # Connection timeout
        remote_settings["timeout"] = st.slider(
            "Connection timeout (seconds)",
            min_value=10,
            max_value=120,
            value=remote_settings.get("timeout", 60),
            step=10,
            help="Maximum time to wait when connecting to remote servers"
        )
        
        # Retry count
        remote_settings["retry_count"] = st.slider(
            "Retry count",
            min_value=0,
            max_value=10,
            value=remote_settings.get("retry_count", 3),
            step=1,
            help="Number of times to retry connecting to a remote server before giving up"
        )
        
        # Authentication toggle
        remote_settings["use_auth"] = st.checkbox(
            "Use authentication",
            value=remote_settings.get("use_auth", False),
            help="Enable authentication for remote servers"
        )
        
        # Only show auth token if authentication is enabled
        if remote_settings["use_auth"]:
            remote_settings["auth_token"] = st.text_input(
                "Authentication token",
                value=remote_settings.get("auth_token", ""),
                type="password",
                help="Token to use for authentication with remote servers"
            )
    
    # MCP config file
    with st.expander("MCP Configuration File", expanded=False):
        st.markdown("""
        <div class="info-message">
            The MCP configuration file stores server settings that will be loaded on startup.
        </div>
        """, unsafe_allow_html=True)
        
        mcp_config_path = st.text_input(
            "MCP Config Path",
            value=os.path.join(os.getcwd(), "mcp_config.json"),
            help="Path to the MCP configuration file"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Save Current Config"):
                try:
                    # Get all server configs
                    if "mcp_servers" in st.session_state:
                        server_configs = st.session_state.mcp_servers
                        
                        # Write to file
                        with open(mcp_config_path, "w") as f:
                            json.dump(server_configs, f, indent=2)
                            
                        st.success(f"Configuration saved to {mcp_config_path}")
                    else:
                        st.warning("No MCP servers configured to save")
                except Exception as e:
                    st.error(f"Error saving configuration: {str(e)}")
        
        with col2:
            if st.button("Load Config File"):
                try:
                    # Read from file
                    if os.path.exists(mcp_config_path):
                        with open(mcp_config_path, "r") as f:
                            server_configs = json.load(f)
                            
                        # Update session state
                        if not isinstance(st.session_state.get("mcp_servers"), dict):
                            st.session_state.mcp_servers = {}
                            
                        st.session_state.mcp_servers.update(server_configs)
                        st.success(f"Loaded {len(server_configs)} server configurations")
                    else:
                        st.warning(f"Config file not found: {mcp_config_path}")
                except Exception as e:
                    st.error(f"Error loading configuration: {str(e)}")
    
    # Apply MCP settings button
    if st.button("Apply MCP Settings"):
        st.success("MCP settings applied successfully")
        # In a real implementation, we would apply these settings to the MCP manager