"""
Callback handlers for the Agent Runner application.
"""

import os
import uuid
import streamlit as st
from typing import Dict, Any, List, Optional
import asyncio
import traceback

from sys_utils import save_to_env_file, load_from_env_file


def on_agent_create(app, agent, run_config):
    """Callback when an agent is created"""
    print("DEBUG: Creating agent...")
    
    # Check agent properties before storing
    print(f"DEBUG: Agent type: {type(agent).__name__}")
    print(f"DEBUG: Agent model: {agent.model}")
    print(f"DEBUG: Agent has {len(agent.tools) if hasattr(agent, 'tools') else 'unknown'} tools")
    
    # Check MCP servers
    if hasattr(agent, 'mcp_servers'):
        print(f"DEBUG: Agent has {len(agent.mcp_servers)} MCP servers")
        for i, server in enumerate(agent.mcp_servers):
            print(f"DEBUG: Agent MCP server {i+1} type: {type(server).__name__}")
            if hasattr(server, 'name'):
                print(f"DEBUG: Agent MCP server {i+1} name: {server.name}")
    else:
        print("DEBUG: Agent has no mcp_servers attribute")
        
    # Store in session state
    st.session_state.agent = agent
    st.session_state.run_config = run_config
    
    print("DEBUG: Agent created and stored in session state")


def on_api_key_save(app, api_key: str):
    """Callback to save API key"""
    # Store API key for future use
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        app.api_key = api_key
    
    # Save to env file
    return save_to_env_file(app.env_file_path, api_key)


def on_api_key_load(app, env_file_path: str):
    """Callback to load API key"""
    loaded_api_key, success = load_from_env_file(env_file_path)
    if success:
        app.api_key = loaded_api_key
    
    return loaded_api_key, success


def on_mcp_server_add(app, server_id: Optional[str], server_config: Dict[str, Any]):
    """Callback when an MCP server is added"""
    # Use the provided server ID or generate a new one if missing
    server_id = server_id or f"mcp_server_{uuid.uuid4().hex[:8]}"
    
    # Add the server to the MCP manager
    app.mcp_manager.add_server_config(server_id, server_config)
    
    # Save configurations to file for persistence
    try:
        # Check if the session state has the configuration
        if server_id in st.session_state.mcp_servers:
            # Get all configurations and save them
            app.mcp_manager.save_configurations(app.mcp_config_path)
            print(f"Saved MCP configurations to {app.mcp_config_path}")
    except Exception as e:
        print(f"Error saving MCP configurations: {str(e)}")
    
    # Print confirmation
    print(f"Added server {server_id} to MCP manager")
    
    
def on_mcp_server_remove(app, server_id: str):
    """Callback when an MCP server is removed"""
    # Remove the server from the MCP manager
    app.mcp_manager.remove_server_config(server_id)
    
    # Save updated configurations to file
    try:
        app.mcp_manager.save_configurations(app.mcp_config_path)
        print(f"Saved updated MCP configurations to {app.mcp_config_path}")
    except Exception as e:
        print(f"Error saving MCP configurations after removal: {str(e)}")
    

def on_mcp_server_test(app, server_config: Dict[str, Any]):
    """Callback to test an MCP server connection"""
    try:
        # Create a temporary ID for testing
        server_id = f"test_{uuid.uuid4().hex[:8]}"
        
        # Add the config temporarily
        app.mcp_manager.add_server_config(server_id, server_config)
        
        # Test the server
        result = app.mcp_manager.test_server(server_id)
        
        # Clean up the temporary config
        app.mcp_manager.remove_server_config(server_id)
        
        return result
    except Exception as e:
        error_trace = traceback.format_exc()
        return {
            "success": False,
            "error": str(e),
            "traceback": error_trace
        }
    

def on_mcp_server_select(app, server_ids: List[str]):
    """Callback when MCP servers are selected for use"""
    try:
        # Clear any existing agent MCP servers instances
        if "agent_mcp_servers" in st.session_state:
            del st.session_state.agent_mcp_servers
        
        if not server_ids:
            st.warning("No MCP servers selected. Please select at least one server.")
            return
            
        # First, make sure MCP manager has the latest configurations
        for server_id, config in st.session_state.mcp_servers.items():
            app.mcp_manager.add_server_config(server_id, config)
            
        # Get the configured server IDs
        print(f"Server configurations available: {list(app.mcp_manager.get_server_configs().keys())}")
        
        # Ensure server_ids are correct
        validated_ids = []
        for sid in server_ids:
            if sid in app.mcp_manager.get_server_configs():
                validated_ids.append(sid)
            else:
                print(f"Warning: Server ID {sid} not found in configurations")
                
        if not validated_ids:
            st.warning("No valid MCP servers were found. Please reconfigure the servers.")
            return
            
        # Get the MCP servers using validated IDs
        mcp_servers = app.mcp_manager.get_servers_for_agent(validated_ids)
        
        # Store in session state for agent creation
        st.session_state.agent_mcp_servers = mcp_servers
        
        # Debug: What are we actually storing?
        print(f"Storing {len(mcp_servers)} MCP servers in session state")
        for i, server in enumerate(mcp_servers):
            print(f"  Server {i+1}: {type(server).__name__}")
            if hasattr(server, 'name'):
                print(f"    Name: {server.name}")
            if hasattr(server, 'params'):
                print(f"    Params: {server.params}")
        
        # Store the IDs for reference
        st.session_state.selected_mcp_server_ids = server_ids
        
        # Print debug info
        print(f"Successfully configured {len(mcp_servers)} MCP servers: {server_ids}")
        
        # Debug information about the MCP servers
        for i, server in enumerate(mcp_servers):
            print(f"MCP Server {i+1}:")
            print(f"  Name: {getattr(server, 'name', 'Unnamed')}")
            print(f"  Type: {type(server).__name__}")
            if hasattr(server, 'list_tools'):
                try:
                    # Import asyncio for running async function
                    # Create a new event loop
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    # Run the list_tools method
                    tools = loop.run_until_complete(server.list_tools())
                    print(f"  Available tools: {[t.name for t in tools]}")
                    # Close the loop
                    loop.close()
                except Exception as e:
                    print(f"  Error listing tools: {str(e)}")
        
        st.success(f"Successfully configured {len(mcp_servers)} MCP servers for use with agents")
    except Exception as e:
        st.error(f"Error configuring MCP servers: {str(e)}")
        print(f"MCP server selection error: {str(e)}")
        print(traceback.format_exc())