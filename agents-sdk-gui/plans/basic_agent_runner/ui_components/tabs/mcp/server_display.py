"""
Server display management for MCP servers.
"""

import streamlit as st
import json
from typing import Dict, Any, Callable, Optional, List

def display_server_status(server_id: str):
    """
    Display the status indicator for a server
    
    Args:
        server_id: The ID of the server
    """
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

def display_server_details(server_config: Dict[str, Any]):
    """
    Display detailed information about a server
    
    Args:
        server_config: The server configuration dictionary
    """
    server_type = server_config.get("type", "unknown")
    
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

def display_test_results(result: Dict[str, Any]):
    """
    Display the results of a server connection test
    
    Args:
        result: The test result dictionary
    """
    if result.get("success", False):
        st.markdown("""
        <div class="success-message">
            Connection successful!
        </div>
        """, unsafe_allow_html=True)
        # Show available tools
        if "tools" in result:
            st.write(f"Found {len(result['tools'])} tools:")
            for tool in result["tools"]:
                st.write(f"- {tool}")
    else:
        st.markdown(f"""
        <div class="error-message">
            Connection failed: {result.get('error', 'Unknown error')}
        </div>
        """, unsafe_allow_html=True)
        # Display traceback in an expander if available
        if "traceback" in result:
            with st.expander("Error Details"):
                st.code(result["traceback"], language="python")

def display_servers(
    on_server_remove: Optional[Callable[[str], None]] = None,
    on_server_test: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    on_server_select: Optional[Callable[[List[str]], None]] = None
) -> bool:
    """
    Display all configured MCP servers
    
    Args:
        on_server_remove: Callback when a server is removed
        on_server_test: Callback to test server connection
        on_server_select: Callback when servers are selected for use
        
    Returns:
        Boolean indicating if any changes were made
    """
    changes_made = False
    
    st.markdown("""
    <div class="form-section mt-md">
        <h3>Configured Servers</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Create styled table header
    st.markdown("""
    <div class="server-list-header" style="display: grid; grid-template-columns: 1fr 3fr 1fr 2fr 2fr; margin-bottom: 8px; font-weight: 600;">
        <div>Use</div>
        <div>Name</div>
        <div>Status</div>
        <div>Type</div>
        <div>Actions</div>
    </div>
    """, unsafe_allow_html=True)
        
    # Create a scrollable container for the servers
    server_container = st.container(height=300)
    
    with server_container:
        # Create a copy of the dictionary to avoid modification during iteration
        servers_to_display = dict(st.session_state.mcp_servers)
        
        # Display each server
        for server_id, server_config in servers_to_display.items():
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
                display_server_status(server_id)
            
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
                                    
                                display_test_results(result)
                            except Exception as e:
                                st.markdown(f"""
                                <div class="error-message">
                                    Test failed with error: {str(e)}
                                </div>
                                """, unsafe_allow_html=True)
                                import traceback
                                with st.expander("Error Details"):
                                    st.code(traceback.format_exc(), language="python")
                
                with remove_col:
                    # Initialize state variable for removing this server
                    if f"removing_{server_id}" not in st.session_state:
                        st.session_state[f"removing_{server_id}"] = False
                    
                    remove_button = st.button("Remove", key=f"remove_{server_id}")
                    if remove_button:
                        # Set flag to remove on next rerun
                        st.session_state[f"removing_{server_id}"] = True
                        # Force a rerun to immediately process
                        st.rerun()
                    
                    # If the flag is set, remove this server
                    if st.session_state[f"removing_{server_id}"]:
                        # Reset the flag
                        st.session_state[f"removing_{server_id}"] = False
                        
                        if on_server_remove:
                            on_server_remove(server_id)
                        
                        # Remove from session state
                        if server_id in st.session_state.mcp_servers:
                            del st.session_state.mcp_servers[server_id]
                        
                        # Remove from selected servers
                        if server_id in st.session_state.selected_mcp_servers:
                            st.session_state.selected_mcp_servers.remove(server_id)
                        
                        changes_made = True
            
            # Display server details in an expander
            with st.expander("Server Details"):
                display_server_details(server_config)
    
    # Apply selection button
    if st.session_state.mcp_servers and on_server_select:
        # Initialize state variables for server selection
        if "applying_server_selection" not in st.session_state:
            st.session_state.applying_server_selection = False
            st.session_state.temp_selected_servers = []
            
        apply_button = st.button("Apply Selected Servers", type="primary", key="apply_servers_button")
        if apply_button:
            # Set flag to apply on next rerun
            st.session_state.applying_server_selection = True
            # Store the current selection
            st.session_state.temp_selected_servers = st.session_state.selected_mcp_servers.copy()
            # Force a rerun to immediately process the selection
            st.rerun()
            
        # If the flag is set, apply the selection
        if st.session_state.applying_server_selection:
            # Reset the flag
            st.session_state.applying_server_selection = False
            
            # Get the stored selection
            selected_servers = st.session_state.temp_selected_servers
            
            # Debug what we're about to select
            print(f"Selecting servers: {selected_servers}")
            print(f"Available server configs: {list(st.session_state.mcp_servers.keys())}")
            
            on_server_select(selected_servers)
            st.markdown(f"""
        <div class="success-message">
            Selected {len(selected_servers)} servers for use with agents
        </div>
        """, unsafe_allow_html=True)
            changes_made = True
            
        # Show a checkbox to select all servers at once
        if "select_all_checked" not in st.session_state:
            st.session_state.select_all_checked = False
            
        select_all = st.checkbox("Select All Servers", 
                                 key="select_all_servers", 
                                 value=st.session_state.select_all_checked)
        
        # Only apply changes if the checkbox state has changed
        if select_all != st.session_state.select_all_checked:
            st.session_state.select_all_checked = select_all
            
            if select_all:
                # Make sure mcp_servers is a dictionary
                if isinstance(st.session_state.mcp_servers, dict):
                    st.session_state.selected_mcp_servers = list(st.session_state.mcp_servers.keys())
                else:
                    st.markdown("""
                    <div class="warning-message">
                        MCP servers data structure is invalid. Please refresh the page.
                    </div>
                    """, unsafe_allow_html=True)
                changes_made = True
    
    return changes_made