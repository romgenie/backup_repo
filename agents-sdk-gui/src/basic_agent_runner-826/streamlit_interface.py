"""
Streamlit interface for the Agent Runner application.
Main entry point for the application.
"""

import streamlit as st
from pathlib import Path
import os

# Import our application components
from app import AgentRunnerApp
from app.callbacks import (
    on_agent_create, on_api_key_save, on_api_key_load,
    on_mcp_server_add, on_mcp_server_remove, on_mcp_server_test, on_mcp_server_select
)
from ui import render_main_ui

class StreamlitInterface:
    """Streamlit interface for the Agent Runner application."""
    
    def __init__(self):
        """Initialize the Streamlit interface."""
        # Create the main application instance
        self.app = AgentRunnerApp()
    
    def run(self):
        """Run the Streamlit interface."""
        # Setup callback dictionary
        callbacks = {
            "on_agent_create": lambda agent, run_config: on_agent_create(self.app, agent, run_config),
            "on_api_key_save": lambda api_key: on_api_key_save(self.app, api_key),
            "on_api_key_load": lambda env_file_path: on_api_key_load(self.app, env_file_path),
            "on_mcp_server_add": lambda server_id, config: on_mcp_server_add(self.app, server_id, config),
            "on_mcp_server_remove": lambda server_id: on_mcp_server_remove(self.app, server_id),
            "on_mcp_server_test": lambda config: on_mcp_server_test(self.app, config),
            "on_mcp_server_select": lambda server_ids: on_mcp_server_select(self.app, server_ids),
            "on_clear_chat": lambda: setattr(self.app, "running_agents", {})
        }
        
        # Render the main UI
        render_main_ui(
            app=self.app,
            api_key=self.app.api_key,
            env_file_path=self.app.env_file_path,
            running_agents=self.app.running_agents,
            callbacks=callbacks
        )

if __name__ == "__main__":
    interface = StreamlitInterface()
    interface.run()