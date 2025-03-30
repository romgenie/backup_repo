"""
Main application class for the Agent Runner application.
"""

import streamlit as st
from typing import Dict, Any
from pathlib import Path
import traceback
import os

from agent_management.mcp import McpManager
from agent_management import init_conversation
from ui_components.tabs import (
    render_chat_tab, render_configure_tab, render_tools_tab,
    render_history_tab, render_settings_tab, render_mcp_tab
)

from .config import initialize_environment, load_mcp_configurations, initialize_mcp_state
from .callbacks import (
    on_agent_create, on_api_key_save, on_api_key_load,
    handle_mcp_server_add, handle_mcp_server_remove, handle_mcp_server_test, on_mcp_server_select
)

class AgentRunnerApp:
    def __init__(self):
        self.running_agents: Dict[str, Any] = {}
        
        # Initialize environment
        self.api_key, self.env_file_path = initialize_environment()
        
        # Initialize conversation state
        init_conversation()
        
        # Initialize MCP manager
        self.mcp_manager = McpManager()
        
        # Define MCP configuration file path
        self.mcp_config_path = str(Path.cwd() / "mcp_config.json")
        
        # Initialize MCP state
        initialize_mcp_state()
        
        # Load MCP configurations
        mcp_configs = load_mcp_configurations(self.mcp_config_path)
        if mcp_configs and "mcp_servers" in st.session_state:
            st.session_state.mcp_servers.update(mcp_configs)
        
        # Sync Streamlit session state with MCP manager
        for server_id, config in st.session_state.mcp_servers.items():
            self.mcp_manager.add_server_config(server_id, config)
            print(f"Loaded config from session state: {server_id}")
        
        # Auto-configure MCP servers if selected
        if hasattr(st.session_state, 'selected_mcp_servers') and len(st.session_state.selected_mcp_servers) > 0:
            self._auto_configure_mcp_servers()
    
    def _auto_configure_mcp_servers(self):
        """Auto-configure MCP servers for agent use."""
        try:
            print(f"DEBUG: Getting servers for agent use with IDs: {st.session_state.selected_mcp_servers}")
            print(f"DEBUG: Available configs: {list(self.mcp_manager.get_server_configs().keys())}")
            
            # Debug server configs
            for sid, config in self.mcp_manager.get_server_configs().items():
                print(f"DEBUG: Server config {sid}: {config}")
            
            mcp_servers = self.mcp_manager.get_servers_for_agent(st.session_state.selected_mcp_servers)
            
            print(f"DEBUG: Got {len(mcp_servers)} MCP servers for agent use")
            for i, server in enumerate(mcp_servers):
                print(f"DEBUG: Server {i+1} type: {type(server).__name__}")
                print(f"DEBUG: Server {i+1} dir: {dir(server)}")
                if hasattr(server, 'name'):
                    print(f"DEBUG: Server {i+1} name: {server.name}")
            
            st.session_state.agent_mcp_servers = mcp_servers
            print(f"DEBUG: Set agent_mcp_servers in session state with {len(mcp_servers)} servers")
            
            # Check if it was actually set
            if "agent_mcp_servers" in st.session_state:
                print(f"DEBUG: Confirmed agent_mcp_servers in session state: {len(st.session_state.agent_mcp_servers)} servers")
            else:
                print("DEBUG: Failed to set agent_mcp_servers in session state!")
        except Exception as e:
            print(f"ERROR: Auto-configuring MCP servers failed: {str(e)}")
            print(f"ERROR TRACE: {traceback.format_exc()}")
    
    def run(self):
        """Run the application UI."""
        st.title("Agent Runner")
        
        # Store the app instance in session state for access from other components
        st.session_state._app = self
        
        # Create tabs to organize the UI
        chat_tab, config_tab, mcp_tab, history_tab, settings_tab = st.tabs([
            "Chat", "Configure", "MCP", "History", "Settings"
        ])
        
        # Render content in each tab
        with chat_tab:
            # Render chat tab
            agent = st.session_state.get("agent", None)
            run_config = st.session_state.get("run_config", None)
            render_chat_tab(
                agent=agent,
                run_config=run_config,
                running_agents=self.running_agents,
                on_clear_chat=lambda: setattr(self, "running_agents", {})
            )
        
        with config_tab:
            # Render the configuration tab
            render_configure_tab(
                api_key=self.api_key,
                env_file_path=self.env_file_path,
                on_agent_create=lambda agent, run_config: on_agent_create(self, agent, run_config),
                on_api_key_save=lambda api_key: on_api_key_save(self, api_key),
                on_api_key_load=lambda env_file_path: on_api_key_load(self, env_file_path)
            )
        
        with mcp_tab:
            # Render the MCP tab
            render_mcp_tab(
                on_server_add=lambda server_id, config: handle_mcp_server_add(self, server_id, config),
                on_server_remove=lambda server_id: handle_mcp_server_remove(self, server_id),
                on_server_test=lambda config: handle_mcp_server_test(self, config),
                on_server_select=lambda server_ids: on_mcp_server_select(self, server_ids)
            )
        
        with history_tab:
            # Render the history tab
            render_history_tab()
        
        with settings_tab:
            # Render the settings tab
            render_settings_tab(
                api_key=self.api_key,
                env_file_path=self.env_file_path,
                on_api_key_save=lambda api_key: on_api_key_save(self, api_key),
                on_api_key_load=lambda env_file_path: on_api_key_load(self, env_file_path)
            )